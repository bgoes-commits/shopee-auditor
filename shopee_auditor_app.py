import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Shopee Ads Auditor", layout="wide")

# =============================
# Helpers
# =============================
def read_shopee_csv(uploaded_file):
    raw = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
    header_idx = 0
    for i, line in enumerate(raw):
        if line.startswith("#,"):
            header_idx = i
            break
    df = pd.read_csv(uploaded_file, skiprows=header_idx)
    df.columns = [c.strip() for c in df.columns]
    return df

def to_number(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0)
    s = s.astype(str).str.replace(".", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0)

def brl(v):
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def pct(v):
    return f"{v*100:.2f}%"

# =============================
# UI
# =============================
st.title("Shopee Ads Auditor")

with st.sidebar:
    st.header("Uploads")
    general_file = st.file_uploader("📊 Dados Gerais de Anúncios (CSV)", type="csv")
    group_files = st.file_uploader("📦 Dados do Grupo de Anúncios (CSV)", type="csv", accept_multiple_files=True)
    sales_prev = st.file_uploader("📈 Vendas - Mês anterior (XLSX)", type="xlsx")
    sales_curr = st.file_uploader("📉 Vendas - Mês atual (XLSX)", type="xlsx")

tab1, tab2, tab3 = st.tabs([
    "1️⃣ Resumo Ads (Campanhas)",
    "2️⃣ Operacional (Produtos / Anúncios)",
    "3️⃣ Comparação de Vendas"
])

# =============================
# TAB 1 — CAMPANHAS
# =============================
with tab1:
    if not general_file:
        st.info("Envie o arquivo **Dados Gerais de Anúncios**.")
    else:
        df = read_shopee_csv(general_file)

        for c in ["Impressões", "Cliques", "Despesas", "Conversões Diretas", "GMV"]:
            df[c] = to_number(df[c]) if c in df.columns else 0

        imp, clk = df["Impressões"].sum(), df["Cliques"].sum()
        spend, orders, gmv = df["Despesas"].sum(), df["Conversões Diretas"].sum(), df["GMV"].sum()
        ctr = clk / imp if imp else 0

        a,b,c,d,e,f = st.columns(6)
        a.metric("Impressões", f"{int(imp):,}".replace(",", "."))
        b.metric("Cliques", f"{int(clk):,}".replace(",", "."))
        c.metric("CTR", pct(ctr))
        d.metric("Investimento", brl(spend))
        e.metric("Pedidos", int(orders))
        f.metric("GMV", brl(gmv))

        st.dataframe(df, use_container_width=True)

# =============================
# TAB 2 — PRODUTOS
# =============================
with tab2:
    if not group_files:
        st.info("Envie os arquivos **Dados do Grupo de Anúncios**.")
    else:
        df = pd.concat([read_shopee_csv(f) for f in group_files])

        for c in ["Impressões", "Cliques", "Despesas", "Conversões Diretas", "GMV"]:
            df[c] = to_number(df[c]) if c in df.columns else 0

        df["CTR"] = np.where(df["Impressões"] > 0, df["Cliques"] / df["Impressões"], 0)
        df["CVR"] = np.where(df["Cliques"] > 0, df["Conversões Diretas"] / df["Cliques"], 0)
        df["CPC"] = np.where(df["Cliques"] > 0, df["Despesas"] / df["Cliques"], 0)
        df["CPA"] = np.where(df["Conversões Diretas"] > 0, df["Despesas"] / df["Conversões Diretas"], np.nan)

        st.dataframe(df, use_container_width=True)

# =============================
# TAB 3 — COMPARAÇÃO
# =============================
with tab3:
    if not sales_prev or not sales_curr:
        st.info("Envie os dois arquivos de vendas para comparar.")
    else:
        prev = pd.read_excel(sales_prev)
        curr = pd.read_excel(sales_curr)

        key = "ID do Produto"
        if key not in prev.columns or key not in curr.columns:
            st.error("Coluna **ID do Produto** não encontrada.")
        else:
            for c in ["Impressões", "Cliques", "Pedidos"]:
                prev[c] = to_number(prev[c]) if c in prev.columns else 0
                curr[c] = to_number(curr[c]) if c in curr.columns else 0

            prev = prev.groupby(key, as_index=False).sum()
            curr = curr.groupby(key, as_index=False).sum()

            df = prev.merge(curr, on=key, how="outer", suffixes=("_prev", "_curr")).fillna(0)

            def diagnose(r):
                if r["Impressões_curr"] < r["Impressões_prev"]:
                    return "Colocar ADS"
                if r["Cliques_curr"] < r["Cliques_prev"]:
                    return "Ajustar CTR (preço/imagem)"
                if r["Pedidos_curr"] < r["Pedidos_prev"]:
                    return "Ajustar conversão (copy)"
                return "OK"

            df["Ação recomendada"] = df.apply(diagnose, axis=1)
            st.dataframe(df, use_container_width=True)
