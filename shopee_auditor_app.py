import re
from io import BytesIO, StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Ads & Vendas Auditor", layout="wide")

# ============================
# 1) Regras (lei)
# ============================
CTR_OTIMA_MIN = 0.04
CTR_OTIMA_MAX = 0.06
CTR_BOA_MIN = 0.03
CTR_RUIM_MAX = 0.015

CVR_OTIMA_MIN = 0.03
CVR_BOA_MIN = 0.02
CVR_RUIM_MAX = 0.015


def classify_ctr(ctr: float) -> str:
    if pd.isna(ctr):
        return "n/a"
    if CTR_OTIMA_MIN <= ctr <= CTR_OTIMA_MAX:
        return "ótima"
    if CTR_BOA_MIN <= ctr < CTR_OTIMA_MIN:
        return "boa"
    if ctr <= CTR_RUIM_MAX:
        return "ruim"
    return "média"


def classify_cvr(cvr: float) -> str:
    if pd.isna(cvr):
        return "n/a"
    if cvr >= CVR_OTIMA_MIN:
        return "ótima"
    if cvr >= CVR_BOA_MIN:
        return "boa"
    if cvr <= CVR_RUIM_MAX:
        return "ruim"
    return "média"


# ============================
# 2) Parsing robusto (CSV Shopee + XLSX Shopee)
# ============================

def _to_str(x) -> str:
    if x is None:
        return ""
    return str(x).strip()


def parse_percent(x) -> float:
    """Aceita '2.21%' ou '2,21%' -> 0.0221"""
    s = _to_str(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan
    s = s.replace("%", "").strip()
    # decide separador decimal
    if "," in s and "." in s:
        # usa o ultimo separador como decimal
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def parse_number_any_locale(x) -> float:
    """Converte numero que pode vir com:
    - CSV: 32649.43 (decimal ponto)
    - XLSX BR: 612.085 (milhar ponto)
    - BR moeda: 1.234,56
    - pode vir com 'R$'
    """
    s = _to_str(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "")
    if s.endswith("%"):
        return np.nan

    # remove caracteres estranhos mantendo separadores
    s = re.sub(r"[^0-9,\.\-]", "", s)

    # se tem ambos, decide decimal pelo ultimo separador
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        # decimal com virgula
        s = s.replace(".", "")  # se veio milhar com ponto
        s = s.replace(",", ".")
    else:
        # so ponto ou nada
        if s.count(".") >= 2:
            # muitos pontos -> milhares
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            # caso XLSX BR: 612.085 (milhar)
            if len(right) == 3 and left.isdigit() and right.isdigit():
                s = left + right
            # caso decimal ponto: 32649.43 -> mantem

    try:
        return float(s)
    except Exception:
        return np.nan


def detect_csv_header_row(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines[:120]):
        if line.startswith("#,"):
            return i
    for i, line in enumerate(lines[:200]):
        if "Impress" in line and "Cliques" in line and "," in line:
            return i
    return 0


def read_shopee_ads_csv(uploaded_file) -> tuple[pd.DataFrame, dict]:
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="replace")
    header_row = detect_csv_header_row(text)

    meta_lines = text.splitlines()[:header_row]
    meta = {}
    if meta_lines:
        meta["titulo"] = meta_lines[0].replace("\ufeff", "").strip()
    for ln in meta_lines[1:15]:
        if "," in ln:
            k, v = ln.split(",", 1)
            meta[k.strip()] = v.strip()

    df = pd.read_csv(StringIO(text), sep=",", skiprows=header_row)
    df.columns = [c.strip() for c in df.columns]
    return df, meta


ADS_PERCENT_COLS = {
    "CTR",
    "Taxa de Conversão",
    "Taxa de Conversão Direta",
    "ROAS",
    "ROAS Direto",
    "ACOS",
    "ACOS Direto",
    "CTR do Produto",
}

ADS_NUMERIC_COLS = {
    "Impressões",
    "Cliques",
    "Conversões",
    "Conversões Diretas",
    "Itens Vendidos",
    "Itens Vendidos Diretos",
    "GMV",
    "Receita direta",
    "Despesas",
    "Custo",
    "Custo por Conversão",
    "Custo por Conversão Direta",
    "Impressões do Produto",
    "Cliques de Produtos",
}


def parse_ads_table(df: pd.DataFrame) -> pd.DataFrame:
    """Parse apenas colunas conhecidas; nao mexe em texto (nome, id etc)."""
    df2 = df.copy()
    for c in df2.columns:
        if c in ADS_PERCENT_COLS:
            df2[c] = df2[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df2[c] = df2[c].apply(parse_number_any_locale)
        # resto: mantem
    return df2


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_ads_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # PRIORIDADE: Impressões/Cliques (geral) -> se nao existir, usa do Produto
    col_imp = pick_first_existing(df, ["Impressões", "Impressões do Produto"])
    col_clk = pick_first_existing(df, ["Cliques", "Cliques de Produtos"])

    col_cost = pick_first_existing(df, ["Despesas", "Custo"])
    col_orders = pick_first_existing(df, ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"])

    # garante numeric
    for c in [col_imp, col_clk, col_cost, col_orders]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ctr_calc"] = np.where((col_imp and col_clk) and (df[col_imp] > 0), df[col_clk] / df[col_imp], 0.0)
    df["cvr_calc"] = np.where((col_clk and col_orders) and (df[col_clk] > 0), df[col_orders] / df[col_clk], 0.0)

    df["cpc"] = np.where((col_cost and col_clk) and (df[col_clk] > 0), df[col_cost] / df[col_clk], np.nan)
    df["cpa"] = np.where((col_cost and col_orders) and (df[col_orders] > 0), df[col_cost] / df[col_orders], np.nan)

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    # guarda quais colunas foram usadas
    df.attrs["imp_col"] = col_imp
    df.attrs["clk_col"] = col_clk
    df.attrs["cost_col"] = col_cost
    df.attrs["orders_col"] = col_orders

    # faturamento
    rev_col = pick_first_existing(df, ["GMV", "Receita direta"])
    df.attrs["rev_col"] = rev_col

    return df


# ============================
# 3) Formatação BR (para exibir)
# ============================

def fmt_brl(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    # formata 1,234,567.89 -> 1.234.567,89
    s = f"{v:,.2f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def fmt_int(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
    s = f"{v:,.0f}"
    s = s.replace(",", ".")
    return s


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x) * 100
    except Exception:
        return ""
    s = f"{v:.{digits}f}"
    s = s.replace(".", ",")
    return f"{s}%"


def make_display_df(df: pd.DataFrame, *, imp_col: str | None, clk_col: str | None, cost_col: str | None, orders_col: str | None, rev_col: str | None) -> pd.DataFrame:
    out = df.copy()

    # ints
    for c in [imp_col, clk_col, orders_col]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_int)

    # moeda
    for c in [cost_col, rev_col, "cpc", "cpa", "GMV", "Receita direta", "Despesas"]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_brl)

    # percent
    for c in ["ctr_calc", "cvr_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)

    for c in ["CTR", "Taxa de Conversão", "Taxa de Conversão Direta", "ROAS", "ROAS Direto", "ACOS", "ACOS Direto", "CTR do Produto"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    return out


# ============================
# 4) UI
# ============================

st.title("Shopee Ads – Auditoria (Estratégico)")

with st.sidebar:
    st.header("Parâmetros de alerta")
    min_impressions_ctr = st.number_input("Mín. impressões p/ avaliar CTR", value=1000, step=100)
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR / sem conversão", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões 'baixas' (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

st.markdown(
    """**Como usar**
- CSV **Dados gerais de anúncios**: aqui você tem o **resumo** (em muitos casos isso representa *campanhas*).
- CSV **Dados do grupo de anúncios**: aqui vem o **detalhe por produto/anúncio** (nome + ID do produto).

> Dica: para as recomendações (pausar/mover/escalar), o app prioriza o **nível de grupo** quando você subir esses arquivos.
"""
)

colA, colB = st.columns(2)
with colA:
    ads_general_file = st.file_uploader("CSV – Dados gerais de anúncios (Shopee)", type=["csv"], key="ads_general")
with colB:
    ads_group_files = st.file_uploader(
        "CSV – Dados por Grupo de Anúncios (1 ou mais)", type=["csv"], accept_multiple_files=True, key="ads_groups"
    )

st.divider()

# ============================
# 5) Carregamento Ads
# ============================

ads_general_df = None
ads_general_meta = {}

if ads_general_file is not None:
    df, meta = read_shopee_ads_csv(ads_general_file)
    df = parse_ads_table(df)
    ads_general_df = add_ads_metrics(df)
    ads_general_meta = meta

# grupos

group_dfs = []
if ads_group_files:
    for f in ads_group_files:
        dfg, metag = read_shopee_ads_csv(f)
        dfg = parse_ads_table(dfg)

        # o titulo do arquivo tem o nome do grupo/campanha
        group_name = metag.get("titulo", "")
        group_name = group_name.replace("\ufeff", "")
        group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
        if not group_name:
            group_name = Path(getattr(f, "name", "grupo")).stem

        dfg["Campanha/Grupo"] = group_name
        dfg = add_ads_metrics(dfg)

        # remove linha agregada do grupo (ID '-')
        if "ID do produto" in dfg.columns:
            dfg = dfg[dfg["ID do produto"].astype(str) != "-"]

        group_dfs.append(dfg)

ads_groups_df = pd.concat(group_dfs, ignore_index=True) if group_dfs else None


# ============================
# 6) Auditoria Ads
# ============================

st.header("1) Ads")

if ads_general_df is None and ads_groups_df is None:
    st.info("Suba pelo menos 1 arquivo (geral ou grupos) para iniciar.")
    st.stop()

# Fonte principal para alertas (prioriza grupos)
source_label = "Grupos (produto/anúncio)" if ads_groups_df is not None else "Geral (resumo)"

if ads_groups_df is not None and ads_general_df is not None:
    source_label = st.radio(
        "Fonte para alertas",
        ["Grupos (produto/anúncio)", "Geral (resumo)"] ,
        horizontal=True,
    )

alert_df = ads_groups_df if source_label.startswith("Grupos") else ads_general_df

# helper para pegar colunas usadas
imp_col = alert_df.attrs.get("imp_col")
clk_col = alert_df.attrs.get("clk_col")
cost_col = alert_df.attrs.get("cost_col")
orders_col = alert_df.attrs.get("orders_col")
rev_col = alert_df.attrs.get("rev_col")

# KPIs

def nsum(col):
    if not col or col not in alert_df.columns:
        return np.nan
    return float(np.nansum(pd.to_numeric(alert_df[col], errors="coerce")))

k1, k2, k3, k4, k5, k6 = st.columns(6)

k1.metric("Impressões", fmt_int(nsum(imp_col)) if imp_col else "n/a")
k2.metric("Cliques", fmt_int(nsum(clk_col)) if clk_col else "n/a")
ctr_total = (nsum(clk_col) / nsum(imp_col)) if imp_col and clk_col and nsum(imp_col) else np.nan
k3.metric("CTR (calc)", fmt_pct(ctr_total) if pd.notna(ctr_total) else "n/a")

k4.metric("Gasto", fmt_brl(nsum(cost_col)) if cost_col else "n/a")
k5.metric("Pedidos/Conv.", fmt_int(nsum(orders_col)) if orders_col else "n/a")

rev_total = nsum(rev_col) if rev_col else np.nan
k6.metric("GMV", fmt_brl(rev_total) if pd.notna(rev_total) else "n/a")

if ads_general_meta:
    with st.expander("Metadados do relatório (geral)", expanded=False):
        st.json(ads_general_meta)

# ============================
# 6.1 Tabelas base
# ============================

st.subheader("Base")

if source_label.startswith("Grupos"):
    # tabela por produto/anuncio
    base_df = alert_df.copy()
    # colunas preferidas
    name_col = "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in base_df.columns else None
    id_col = "ID do produto" if "ID do produto" in base_df.columns else None

    # filtro (status)
    if "Status do Anúncio" in base_df.columns:
        opts = ["(todos)"] + sorted([x for x in base_df["Status do Anúncio"].dropna().unique().tolist()])
        st_status = st.selectbox("Filtrar por Status do Anúncio", opts)
        if st_status != "(todos)":
            base_df = base_df[base_df["Status do Anúncio"] == st_status]

    show_cols = [c for c in [
        "Campanha/Grupo",
        name_col,
        id_col,
        imp_col,
        clk_col,
        "ctr_calc",
        "ctr_class",
        orders_col,
        "cvr_calc",
        "cvr_class",
        rev_col,
        cost_col,
        "cpc",
        "cpa",
    ] if c and c in base_df.columns]

    base_sorted = base_df.sort_values(by=cost_col, ascending=False) if cost_col in base_df.columns else base_df
    disp = make_display_df(base_sorted[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
    st.dataframe(disp, use_container_width=True, hide_index=True)

else:
    # tabela do geral (resumo)
    base_df = alert_df.copy()

    # filtro (Status)
    if "Status" in base_df.columns:
        opts = ["(todos)"] + sorted([x for x in base_df["Status"].dropna().unique().tolist()])
        st_status = st.selectbox("Filtrar por Status", opts)
        if st_status != "(todos)":
            base_df = base_df[base_df["Status"] == st_status]

    show_cols = [c for c in [
        "Nome do Anúncio",  # muitas contas usam isso como "campanha"
        "Status",
        "Tipos de Anúncios",
        imp_col,
        clk_col,
        "ctr_calc",
        "ctr_class",
        orders_col,
        "cvr_calc",
        "cvr_class",
        rev_col,
        cost_col,
        "cpc",
        "cpa",
    ] if c and c in base_df.columns]

    base_sorted = base_df.sort_values(by=cost_col, ascending=False) if cost_col in base_df.columns else base_df
    disp = make_display_df(base_sorted[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
    st.dataframe(disp, use_container_width=True, hide_index=True)


# ============================
# 6.2 Alertas
# ============================

st.subheader("Alertas e ações")

tabs = st.tabs([
    "Gastando sem converter",
    "CTR ruim",
    "CVR ruim",
    "Bons com pouca impressão",
    "Mover anúncio (competição no grupo)",
])

# Alertas operam sempre no dataframe "alert_df" (que pode ser geral ou grupos)

with tabs[0]:
    if cost_col and orders_col and clk_col:
        wasting = alert_df[
            (pd.to_numeric(alert_df[orders_col], errors="coerce").fillna(0) == 0)
            & (
                (pd.to_numeric(alert_df[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
                | (pd.to_numeric(alert_df[cost_col], errors="coerce").fillna(0) >= float(min_spend_no_conv))
            )
        ].copy()
        if wasting.empty:
            st.info("Nenhum item no critério.")
        else:
            wasting["ação"] = "Pausar/remover (gastando sem converter)"
            cols = [c for c in wasting.columns if c in {"Campanha/Grupo", "Anúncio / Nome do Produto", "ID do produto", "Nome do Anúncio", "Status", "Status do Anúncio"}]
            cols += [c for c in [imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "cpc", "cpa", "ação"] if c and c in wasting.columns]
            wasting_sorted = wasting.sort_values(by=cost_col, ascending=False)
            disp = make_display_df(wasting_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Nao encontrei colunas suficientes (gasto, cliques e pedidos/conversoes).")

with tabs[1]:
    if imp_col and clk_col:
        bad_ctr = alert_df[
            (pd.to_numeric(alert_df[imp_col], errors="coerce").fillna(0) >= min_impressions_ctr)
            & (pd.to_numeric(alert_df["ctr_calc"], errors="coerce").fillna(0) <= CTR_RUIM_MAX)
        ].copy()
        if bad_ctr.empty:
            st.info("Nenhum item no critério.")
        else:
            bad_ctr["ação"] = "Ajustar preço + cauda longa + imagem (CTR ruim)"
            cols = [c for c in bad_ctr.columns if c in {"Campanha/Grupo", "Anúncio / Nome do Produto", "ID do produto", "Nome do Anúncio"}]
            cols += [c for c in [imp_col, clk_col, "ctr_calc", "ctr_class", orders_col, "cvr_calc", rev_col, cost_col, "ação"] if c and c in bad_ctr.columns]
            bad_ctr_sorted = bad_ctr.sort_values(by="ctr_calc", ascending=True)
            disp = make_display_df(bad_ctr_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Nao encontrei colunas de impressoes/cliques.")

with tabs[2]:
    if clk_col and orders_col:
        bad_cvr = alert_df[
            (pd.to_numeric(alert_df[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
            & (pd.to_numeric(alert_df["cvr_calc"], errors="coerce").fillna(0) <= CVR_RUIM_MAX)
        ].copy()
        if bad_cvr.empty:
            st.info("Nenhum item no critério.")
        else:
            bad_cvr["ação"] = "Ajustar copy + gatilhos de conversao (CVR ruim)"
            cols = [c for c in bad_cvr.columns if c in {"Campanha/Grupo", "Anúncio / Nome do Produto", "ID do produto", "Nome do Anúncio"}]
            cols += [c for c in [imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", "cvr_class", rev_col, cost_col, "ação"] if c and c in bad_cvr.columns]
            bad_cvr_sorted = bad_cvr.sort_values(by="cvr_calc", ascending=True)
            disp = make_display_df(bad_cvr_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Nao encontrei colunas de cliques e pedidos/conversoes.")

with tabs[3]:
    if imp_col:
        good_low = alert_df[
            (pd.to_numeric(alert_df[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)
            & (
                (pd.to_numeric(alert_df["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                | (pd.to_numeric(alert_df["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
            )
        ].copy()
        if good_low.empty:
            st.info("Nenhum item no critério.")
        else:
            good_low["ação"] = "Escalar: aumentar entrega / revisar estrutura"
            cols = [c for c in good_low.columns if c in {"Campanha/Grupo", "Anúncio / Nome do Produto", "ID do produto", "Nome do Anúncio"}]
            cols += [c for c in [imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "ação"] if c and c in good_low.columns]
            good_low_sorted = good_low.sort_values(by=imp_col, ascending=True)
            disp = make_display_df(good_low_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Nao encontrei coluna de impressoes.")

with tabs[4]:
    if ads_groups_df is None:
        st.info("Esse alerta exige os CSVs de **Grupo de Anuncios**.")
    else:
        df_g = ads_groups_df.copy()
        g_imp = df_g.attrs.get("imp_col") or "Impressões"
        g_clk = df_g.attrs.get("clk_col") or "Cliques"
        g_cost = df_g.attrs.get("cost_col") or "Despesas"
        g_orders = df_g.attrs.get("orders_col") or "Conversões Diretas"

        # share de gasto no grupo
        df_g["spend_share"] = df_g.groupby("Campanha/Grupo")[g_cost].transform(lambda s: s / s.sum() if s.sum() else 0.0)

        # candidato promissor (bom CTR/CVR mas pouca entrega)
        prom = df_g[
            (pd.to_numeric(df_g[g_imp], errors="coerce").fillna(0) <= low_impressions_threshold)
            & (
                (pd.to_numeric(df_g["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                | (pd.to_numeric(df_g["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
            )
        ].copy()

        # precisa existir dominante no grupo
        dom = df_g[df_g["spend_share"] >= (dominance_spend_share / 100.0)][["Campanha/Grupo"]].drop_duplicates()
        prom = prom.merge(dom, on="Campanha/Grupo", how="inner")

        if prom.empty:
            st.info("Nenhum item no critério.")
        else:
            prom["ação"] = "Mover para outra campanha/grupo (competicao interna)"
            prom2 = prom.copy()
            prom2["spend_share_%"] = prom2["spend_share"] * 100

            cols = [
                "Campanha/Grupo",
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in prom2.columns else None,
                "ID do produto" if "ID do produto" in prom2.columns else None,
                g_imp,
                g_clk,
                "ctr_calc",
                g_orders,
                "cvr_calc",
                rev_col if rev_col in prom2.columns else None,
                g_cost,
                "spend_share_%",
                "ação",
            ]
            cols = [c for c in cols if c and c in prom2.columns]

            prom_sorted = prom2.sort_values(by="spend_share", ascending=False)
            disp = make_display_df(prom_sorted[cols], imp_col=g_imp, clk_col=g_clk, cost_col=g_cost, orders_col=g_orders, rev_col=rev_col)
            if "spend_share_%" in disp.columns:
                disp["spend_share_%"] = prom_sorted["spend_share_%"].apply(lambda v: fmt_pct(v/100.0, digits=1))
            st.dataframe(disp, use_container_width=True, hide_index=True)


# ============================
# 7) Vendas (vou manter como proximo passo)
# ============================

st.divider()
st.header("2) Vendas (mês vs mês)")
st.info("Nesta iteracao eu foquei em corrigir Ads (nome/ID/faturamento e formatacao BR). Se voce quiser, agora eu ajusto o bloco de Vendas com a mesma padronizacao de moeda/% e cruzamento com Ads.")
