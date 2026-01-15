import re
from pathlib import Path
from io import BytesIO, StringIO

import numpy as np
import pandas as pd
try:
    import streamlit as st
except ModuleNotFoundError:  # permite importar funcoes em ambiente sem streamlit
    st = None


if st is None:
    raise SystemExit("Streamlit nao esta instalado. Instale com: pip install streamlit pandas openpyxl")

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
# 2) Utilitários de parsing
# ============================

def _to_str(x):
    if x is None:
        return ""
    return str(x).strip()


def parse_percent(x) -> float:
    """Aceita '2.21%' ou '2,21%' -> 0.0221"""
    s = _to_str(x)
    if not s:
        return np.nan
    s = s.replace("%", "").strip()
    # Shopee às vezes usa '.' como decimal no CSV e em outros casos ','
    # Vamos tratar ambos.
    if "," in s and "." in s:
        # formato 1.234,56
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def parse_br_number(x) -> float:
    """Converte números brasileiros que podem vir como texto (ex.: '612.085', '1.234,56', '0')"""
    s = _to_str(x)
    if s == "" or s.lower() in {"nan", "-"}:
        return np.nan

    # remove espaços e símbolos
    s = s.replace("R$", "").replace(" ", "")

    # Se parece com percent, não parse aqui
    if s.endswith("%"):
        return np.nan

    # Se tem ambos '.' e ',', é 1.234,56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        # caso comum da Shopee nos XLSX: milhares com '.' (ex.: 612.085)
        # e também há casos de decimal com '.' vindo do CSV
        # Heurística:
        # - se tiver mais de um '.', provavelmente são milhares
        # - se tiver 1 '.' e exatamente 3 dígitos após, é milhares
        if s.count(".") >= 2:
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            if len(right) == 3 and left.isdigit() and right.isdigit():
                s = left + right
        # se tiver ',', é decimal
        if "," in s:
            s = s.replace(".", "").replace(",", ".")

    # remove qualquer coisa que não seja número/.-
    s = re.sub(r"[^0-9\.-]", "", s)
    try:
        return float(s)
    except Exception:
        return np.nan


def detect_csv_header_row(text: str) -> int:
    """Encontra a linha do cabeçalho real no CSV da Shopee (linha que começa com '#,')."""
    lines = text.splitlines()
    for i, line in enumerate(lines[:80]):
        if line.startswith("#,"):
            return i
    # fallback: procurar coluna Impressões e Cliques
    for i, line in enumerate(lines[:120]):
        if "Impress" in line and "Cliques" in line and "," in line:
            return i
    return 0


def read_shopee_ads_csv(uploaded_file) -> tuple[pd.DataFrame, dict]:
    """Lê CSVs de Ads da Shopee (geral ou por grupo). Retorna df e metadados."""
    raw = uploaded_file.read()
    # tenta decodificar com utf-8 (tem BOM)
    text = raw.decode("utf-8", errors="replace")

    header_row = detect_csv_header_row(text)

    # metadados: linhas antes do cabeçalho
    meta_lines = text.splitlines()[:header_row]
    meta = {}
    if meta_lines:
        meta["titulo"] = meta_lines[0].replace("\ufeff", "").strip()
    for ln in meta_lines[1:10]:
        if "," in ln:
            k, v = ln.split(",", 1)
            meta[k.strip()] = v.strip()

    df = pd.read_csv(StringIO(text), sep=",", skiprows=header_row)
    # normaliza colunas
    df.columns = [c.strip() for c in df.columns]

    return df, meta


def coerce_columns(
    df: pd.DataFrame,
    numeric_cols: set[str] | None = None,
    percent_cols: set[str] | None = None,
    currency_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Converte SOMENTE colunas conhecidas.

    Importante: os relatórios da Shopee têm colunas-texto com números (ex.: nome do anúncio com datas).
    Heurísticas quebram e transformam texto em NaN. Aqui evitamos isso.
    """

    df = df.copy()
    numeric_cols = numeric_cols or set()
    percent_cols = percent_cols or set()
    currency_cols = currency_cols or set()

    for c in df.columns:
        if c in percent_cols:
            df[c] = df[c].apply(parse_percent)
        elif c in numeric_cols or c in currency_cols:
            df[c] = df[c].apply(parse_br_number)
    return df


def fmt_percent(x) -> str:
    if pd.isna(x):
        return ""
    return f"{x*100:.2f}%"


def fmt_money(x) -> str:
    if pd.isna(x):
        return ""
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(x) -> str:
    if pd.isna(x):
        return ""
    return f"{int(round(float(x))):,}".replace(",", ".")


def add_ads_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona CTR e CVR recalculados + classes."""
    df = df.copy()

    # possíveis nomes
    col_imp = None
    col_clk = None
    col_cost = None
    col_orders = None

    # Preferir as colunas gerais (o relatório geral normalmente traz Impressões/Cliques;
    # as colunas "... do Produto" às vezes vêm como '-' e zeram tudo.)
    for cand in ["Impressões", "Impressões do Produto"]:
        if cand in df.columns:
            col_imp = cand
            break
    for cand in ["Cliques", "Cliques de Produtos"]:
        if cand in df.columns:
            col_clk = cand
            break
    for cand in ["Despesas", "Custo"]:
        if cand in df.columns:
            col_cost = cand
            break
    # pedidos (conversões diretas tende a ser mais próximo de compra)
    for cand in ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"]:
        if cand in df.columns:
            col_orders = cand
            break

    # recálculo
    if col_imp and col_clk:
        df["ctr_calc"] = np.where(df[col_imp] > 0, df[col_clk] / df[col_imp], 0.0)
    else:
        df["ctr_calc"] = np.nan

    if col_clk and col_orders:
        df["cvr_calc"] = np.where(df[col_clk] > 0, df[col_orders] / df[col_clk], 0.0)
    else:
        df["cvr_calc"] = np.nan

    if col_cost and col_clk:
        df["cpc"] = np.where(df[col_clk] > 0, df[col_cost] / df[col_clk], np.nan)
    else:
        df["cpc"] = np.nan

    if col_cost and col_orders:
        df["cpa"] = np.where(df[col_orders] > 0, df[col_cost] / df[col_orders], np.nan)
    else:
        df["cpa"] = np.nan

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    return df


def read_sales_xlsx(uploaded_file, sheet_name: str) -> pd.DataFrame:
    """Lê XLSX de tráfego do produto da Shopee mantendo tudo como string e parseando depois."""
    data = uploaded_file.read()
    bio = BytesIO(data)
    df = pd.read_excel(bio, sheet_name=sheet_name, dtype=str)
    df.columns = [c.strip() for c in df.columns]

    # colunas percent
    percent_cols = {
        "Taxa de Vendas",
        "CTR",
        "Taxa de Conversão de Pedidos",
    }

    # converter
    df2 = df.copy()
    for c in df2.columns:
        if c in percent_cols:
            df2[c] = df2[c].apply(parse_percent)
        else:
            df2[c] = df2[c].apply(parse_br_number)

    # manter também o nome do produto (texto)
    if "Produto" in df.columns:
        df2["Produto_nome"] = df["Produto"].astype(str)

    return df2


# ============================
# 3) UI
# ============================

st.title("Shopee Ads – Dashboard e Auditoria (Estratégico)")

with st.sidebar:
    st.header("Parâmetros de alerta")
    min_impressions_ctr = st.number_input("Mín. impressões p/ avaliar CTR", value=1000, step=100)
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR / sem conversão", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões 'baixas' (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

st.markdown(
    """**Como usar**
- Suba **1 CSV geral de anúncios** (obrigatório)
- Suba **1+ CSVs de grupo de anúncios** (opcional, melhora o alerta de *mover anúncio*)
- Suba **2 XLSX de tráfego do produto** (mês anterior e mês atual) para diagnóstico de queda e oportunidades
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

colC, colD = st.columns(2)
with colC:
    sales_prev_file = st.file_uploader("XLSX – Tráfego do Produto (mês anterior)", type=["xlsx"], key="sales_prev")
with colD:
    sales_curr_file = st.file_uploader("XLSX – Tráfego do Produto (mês atual)", type=["xlsx"], key="sales_curr")


# ============================
# 4) Carregamento + Normalização
# ============================

ads_general_df = None
ads_general_meta = {}

if ads_general_file is not None:
    df, meta = read_shopee_ads_csv(ads_general_file)

    ads_percent_cols = {
        "CTR",
        "CTR do Produto",
        "Taxa de Conversão",
        "Taxa de Conversão Direta",
        "ACOS",
        "ACOS Direto",
        "ROAS",
        "ROAS Direto",
    }
    ads_numeric_cols = {
        "Impressões",
        "Cliques",
        "Impressões do Produto",
        "Cliques de Produtos",
        "Conversões",
        "Conversões Diretas",
        "Itens Vendidos",
        "Itens Vendidos Diretos",
    }
    ads_currency_cols = {
        "Despesas",
        "Custo",
        "GMV",
        "Receita direta",
        "Custo por Conversão",
        "Custo por Conversão Direta",
        "Custo por Conversão",
        "Custo por Conversão Direta",
    }

    df = coerce_columns(df, numeric_cols=ads_numeric_cols, percent_cols=ads_percent_cols, currency_cols=ads_currency_cols)
    ads_general_df = add_ads_metrics(df)
    ads_general_meta = meta


group_dfs = []
if ads_group_files:
    for f in ads_group_files:
        dfg, metag = read_shopee_ads_csv(f)
        # Extrai nome do grupo do título
        group_name = metag.get("titulo", "").replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
        dfg["Grupo"] = group_name if group_name else Path(getattr(f, "name", "grupo")).stem

        dfg = coerce_columns(dfg, numeric_cols=ads_numeric_cols, percent_cols=ads_percent_cols, currency_cols=ads_currency_cols)
        dfg = add_ads_metrics(dfg)
        group_dfs.append(dfg)

ads_groups_df = pd.concat(group_dfs, ignore_index=True) if group_dfs else None


# ============================
# 5) ABA ADS
# ============================

st.header("1) Auditoria de Ads")

if ads_general_df is None:
    st.info("Suba o **CSV de dados gerais** para liberar a auditoria de Ads.")
else:
    # KPIs
    def _sum(col):
        return float(np.nansum(ads_general_df[col])) if col in ads_general_df.columns else np.nan

    # escolhe colunas principais (preferir gerais)
    imp_col = "Impressões" if "Impressões" in ads_general_df.columns else ("Impressões do Produto" if "Impressões do Produto" in ads_general_df.columns else None)
    clk_col = "Cliques" if "Cliques" in ads_general_df.columns else ("Cliques de Produtos" if "Cliques de Produtos" in ads_general_df.columns else None)
    cost_col = "Despesas" if "Despesas" in ads_general_df.columns else ("Custo" if "Custo" in ads_general_df.columns else None)
    orders_col = "Conversões Diretas" if "Conversões Diretas" in ads_general_df.columns else ("Conversões" if "Conversões" in ads_general_df.columns else None)

    faturamento_col = "Receita direta" if "Receita direta" in ads_general_df.columns else ("GMV" if "GMV" in ads_general_df.columns else None)

    kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
    kpi1.metric("Impressões", f"{_sum(imp_col):,.0f}" if imp_col else "n/a")
    kpi2.metric("Cliques", f"{_sum(clk_col):,.0f}" if clk_col else "n/a")
    kpi3.metric("CTR (calc)", f"{( _sum(clk_col) / _sum(imp_col) * 100):.2f}%" if imp_col and clk_col and _sum(imp_col) > 0 else "n/a")
    kpi4.metric("Gasto", f"R$ {_sum(cost_col):,.2f}" if cost_col else "n/a")
    kpi5.metric("Conversões/Pedidos", f"{_sum(orders_col):,.0f}" if orders_col else "n/a")
    kpi6.metric("Faturamento", fmt_money(_sum(faturamento_col)) if faturamento_col else "n/a")

    with st.expander("Metadados do relatório", expanded=False):
        st.json(ads_general_meta)

    # Tabela base (com filtros)
    st.subheader("Base de anúncios (geral)")

    # filtros
    f_status = None
    if "Status" in ads_general_df.columns:
        opts = ["(todos)"] + sorted([x for x in ads_general_df["Status"].dropna().unique().tolist()])
        f_status = st.selectbox("Filtrar por Status", opts)

    df_show = ads_general_df.copy()
    if f_status and f_status != "(todos)":
        df_show = df_show[df_show["Status"] == f_status]

    display_cols = [c for c in [
        "Nome do Anúncio",
        "Status",
        "Tipos de Anúncios",
        "ID do produto",
        faturamento_col,
        imp_col,
        clk_col,
        "ctr_calc",
        "ctr_class",
        orders_col,
        "cvr_calc",
        "cvr_class",
        cost_col,
        "cpc",
        "cpa",
    ] if c and c in df_show.columns]

    df_base = df_show[display_cols].copy()
    if cost_col and cost_col in df_base.columns:
        df_base = df_base.sort_values(by=cost_col, ascending=False)

    fmt_map = {
        "ctr_calc": fmt_percent,
        "cvr_calc": fmt_percent,
        "CTR": fmt_percent,
        "Taxa de Conversão": fmt_percent,
        "Taxa de Conversão Direta": fmt_percent,
        "cpc": fmt_money,
        "cpa": fmt_money,
    }
    if imp_col:
        fmt_map[imp_col] = fmt_int
    if clk_col:
        fmt_map[clk_col] = fmt_int
    if orders_col:
        fmt_map[orders_col] = fmt_int
    if cost_col:
        fmt_map[cost_col] = fmt_money
    if faturamento_col and faturamento_col in df_base.columns:
        fmt_map[faturamento_col] = fmt_money

    st.dataframe(df_base.style.format(fmt_map), use_container_width=True, hide_index=True)

    st.subheader("Alertas e ações")

    # --------- Alertas ---------
    alert_tabs = st.tabs([
        "Gastando sem converter",
        "CTR ruim",
        "CVR ruim",
        "Bons com pouca impressão",
        "Mover anúncio (grupo)",
    ])

    # 1) Gastando sem converter
    with alert_tabs[0]:
        if cost_col and orders_col and clk_col:
            wasting = ads_general_df[
                (ads_general_df[orders_col] == 0)
                & ((ads_general_df[clk_col] >= min_clicks_eval) | (ads_general_df[cost_col] >= min_spend_no_conv))
            ].copy()
            wasting["ação"] = "Pausar/remover (gastando sem converter)"
            cols = [c for c in display_cols if c in wasting.columns] + ["ação"]
            st.dataframe(wasting.sort_values(by=cost_col, ascending=False)[cols].style.format(fmt_map), use_container_width=True, hide_index=True)
        else:
            st.warning("Não encontrei colunas necessárias (gasto, cliques e conversões).")

    # 2) CTR ruim
    with alert_tabs[1]:
        if imp_col and clk_col:
            bad_ctr = ads_general_df[(ads_general_df[imp_col] >= min_impressions_ctr) & (ads_general_df["ctr_calc"] <= CTR_RUIM_MAX)].copy()
            bad_ctr["ação"] = "Ajustar preço + cauda longa + imagem (CTR ruim)"
            cols = [c for c in display_cols if c in bad_ctr.columns] + ["ação"]
            st.dataframe(bad_ctr.sort_values(by="ctr_calc", ascending=True)[cols].style.format(fmt_map), use_container_width=True, hide_index=True)
        else:
            st.warning("Não encontrei colunas de impressões/cliques para calcular CTR.")

    # 3) CVR ruim
    with alert_tabs[2]:
        if clk_col and orders_col:
            bad_cvr = ads_general_df[(ads_general_df[clk_col] >= min_clicks_eval) & (ads_general_df["cvr_calc"] <= CVR_RUIM_MAX)].copy()
            bad_cvr["ação"] = "Ajustar copy + gatilhos de conversão (CVR ruim)"
            cols = [c for c in display_cols if c in bad_cvr.columns] + ["ação"]
            st.dataframe(bad_cvr.sort_values(by="cvr_calc", ascending=True)[cols].style.format(fmt_map), use_container_width=True, hide_index=True)
        else:
            st.warning("Não encontrei colunas de cliques/conversões para calcular CVR.")

    # 4) Bons com pouca impressão
    with alert_tabs[3]:
        if imp_col:
            good_low = ads_general_df[
                (ads_general_df[imp_col] <= low_impressions_threshold)
                & ((ads_general_df["ctr_calc"] >= CTR_BOA_MIN) | (ads_general_df["cvr_calc"] >= CVR_BOA_MIN))
            ].copy()
            good_low["ação"] = "Aumentar entrega / revisar estrutura"
            cols = [c for c in display_cols if c in good_low.columns] + ["ação"]
            st.dataframe(good_low.sort_values(by=imp_col, ascending=True)[cols].style.format(fmt_map), use_container_width=True, hide_index=True)
        else:
            st.warning("Não encontrei coluna de impressões.")

    # 5) Mover anúncio (grupo)
    with alert_tabs[4]:
        if ads_groups_df is None:
            st.info("Suba 1+ CSVs de **Grupo de Anúncios** para liberar esta análise.")
        else:
            # identifica colunas do grupo
            g_imp = "Impressões" if "Impressões" in ads_groups_df.columns else None
            g_clk = "Cliques" if "Cliques" in ads_groups_df.columns else None
            g_cost = "Despesas" if "Despesas" in ads_groups_df.columns else None
            g_orders = "Conversões Diretas" if "Conversões Diretas" in ads_groups_df.columns else ("Conversões" if "Conversões" in ads_groups_df.columns else None)

            # remove a linha de total do grupo (normalmente ID do produto '-' e/ou é o próprio nome do grupo)
            df_g = ads_groups_df.copy()
            if "ID do produto" in df_g.columns:
                df_g = df_g[df_g["ID do produto"].notna()]
                df_g = df_g[df_g["ID do produto"].astype(str) != "-"]

            if not (g_cost and g_imp and g_clk):
                st.warning("Não encontrei colunas necessárias no relatório de grupo (impressões, cliques, despesas).")
            else:
                # share de gasto no grupo
                df_g["spend_share"] = df_g.groupby("Grupo")[g_cost].transform(lambda s: s / s.sum() if s.sum() else 0.0)

                # candidatos promissores com pouca entrega
                prom = df_g[
                    (df_g[g_imp] <= low_impressions_threshold)
                    & ((df_g["ctr_calc"] >= CTR_BOA_MIN) | (df_g["cvr_calc"] >= CVR_BOA_MIN))
                ].copy()

                # precisa existir alguém dominante no grupo
                dom = df_g[df_g["spend_share"] >= (dominance_spend_share / 100.0)][["Grupo"]].drop_duplicates()
                prom = prom.merge(dom, on="Grupo", how="inner")

                prom["ação"] = "Mover para outra campanha/grupo (competição interna)"

                show_cols = [c for c in [
                    "Grupo",
                    "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in prom.columns else None,
                    "ID do produto" if "ID do produto" in prom.columns else None,
                    g_imp,
                    g_clk,
                    "ctr_calc",
                    g_orders,
                    "cvr_calc",
                    g_cost,
                    "spend_share",
                    "ação",
                ] if c and c in prom.columns]

                fmt_map_g = fmt_map.copy()
                fmt_map_g["spend_share"] = fmt_percent
                if g_cost:
                    fmt_map_g[g_cost] = fmt_money
                if g_imp:
                    fmt_map_g[g_imp] = fmt_int
                if g_clk:
                    fmt_map_g[g_clk] = fmt_int
                if g_orders and g_orders in prom.columns:
                    fmt_map_g[g_orders] = fmt_int

                st.dataframe(prom.sort_values(by="spend_share", ascending=False)[show_cols].style.format(fmt_map_g), use_container_width=True, hide_index=True)


# ============================
# 6) ABA VENDAS (comparativo)
# ============================

st.header("2) Diagnóstico de Vendas (mês anterior vs mês atual)")

if sales_prev_file is None or sales_curr_file is None:
    st.info("Suba os **2 XLSX** de tráfego do produto para liberar a análise de vendas.")
else:
    # lista sheets
    def list_sheets(uploaded):
        # lê só nomes das abas sem carregar tudo
        data = uploaded.getvalue() if hasattr(uploaded, "getvalue") else uploaded.read()
        bio = BytesIO(data)
        xls = pd.ExcelFile(bio)
        return xls.sheet_names

    prev_sheets = list_sheets(sales_prev_file)
    curr_sheets = list_sheets(sales_curr_file)

    # escolhe sheet (tenta alinhar nomes)
    common = [s for s in prev_sheets if s in curr_sheets]
    default_sheet = common[0] if common else prev_sheets[0]

    sheet = st.selectbox("Escolha a aba (pedido pago/relaizado)", options=prev_sheets, index=prev_sheets.index(default_sheet))

    prev_df = read_sales_xlsx(sales_prev_file, sheet)
    curr_df = read_sales_xlsx(sales_curr_file, sheet)

    # colunas principais
    key = "ID do Item" if "ID do Item" in prev_df.columns and "ID do Item" in curr_df.columns else None
    if key is None:
        st.error("Não encontrei a coluna 'ID do Item' nas duas planilhas.")
    else:
        # padroniza
        prev = prev_df.copy()
        curr = curr_df.copy()
        prev = prev.rename(columns={key: "item_id"})
        curr = curr.rename(columns={key: "item_id"})

        # agregação por item
        def agg_sales(df):
            agg_map = {
                "Vendas (BRL)": "sum" if "Vendas (BRL)" in df.columns else "first",
                "Impressões de Produto": "sum" if "Impressões de Produto" in df.columns else "first",
                "Cliques Por Produto": "sum" if "Cliques Por Produto" in df.columns else "first",
                "Pedidos": "sum" if "Pedidos" in df.columns else "first",
            }
            if "Produto_nome" in df.columns:
                agg_map["Produto_nome"] = "first"
            out = df.groupby("item_id", as_index=False).agg(agg_map)
            # recomputa CTR e CVR
            if "Impressões de Produto" in out.columns and "Cliques Por Produto" in out.columns:
                out["ctr"] = np.where(out["Impressões de Produto"] > 0, out["Cliques Por Produto"] / out["Impressões de Produto"], 0.0)
            else:
                out["ctr"] = np.nan
            if "Cliques Por Produto" in out.columns and "Pedidos" in out.columns:
                out["cvr"] = np.where(out["Cliques Por Produto"] > 0, out["Pedidos"] / out["Cliques Por Produto"], 0.0)
            else:
                out["cvr"] = np.nan
            return out

        prev_a = agg_sales(prev).add_prefix("prev_")
        curr_a = agg_sales(curr).add_prefix("curr_")

        merged = prev_a.merge(curr_a, left_on="prev_item_id", right_on="curr_item_id", how="outer")
        merged["item_id"] = merged["prev_item_id"].fillna(merged["curr_item_id"])
        merged = merged.drop(columns=["prev_item_id", "curr_item_id"])

        # deltas
        for m in ["Vendas (BRL)", "Impressões de Produto", "Cliques Por Produto", "Pedidos", "ctr", "cvr"]:
            p = f"prev_{m}"
            c = f"curr_{m}"
            if p in merged.columns and c in merged.columns:
                merged[f"delta_{m}"] = merged[c] - merged[p]
                merged[f"delta_{m}_pct"] = np.where(merged[p] > 0, merged[f"delta_{m}"] / merged[p], np.nan)

        # diagnóstico
        def diagnose(row):
            # queda de vendas (receita) ou pedidos
            drop_sales = row.get("delta_Vendas (BRL)", 0)
            drop_orders = row.get("delta_Pedidos", 0)
            if (pd.notna(drop_sales) and drop_sales >= 0) and (pd.notna(drop_orders) and drop_orders >= 0):
                return ""

            imp_drop = row.get("delta_Impressões de Produto_pct", np.nan)
            ctr_drop = row.get("delta_ctr_pct", np.nan)
            cvr_drop = row.get("delta_cvr_pct", np.nan)

            if pd.notna(imp_drop) and imp_drop <= -0.20:
                return "Impressões caíram: colocar ADS (lei)"
            if pd.notna(ctr_drop) and ctr_drop <= -0.15:
                return "CTR caiu: ajustar preço + cauda longa + imagem (lei)"
            if pd.notna(cvr_drop) and cvr_drop <= -0.15:
                return "CVR caiu: ajustar copy + gatilhos de conversão (lei)"
            return "Queda sem sinal claro: checar preço/estoque/frete/concorrência"

        merged["acao_recomendada"] = merged.apply(diagnose, axis=1)

        # ligar com Ads para achar potenciais fora do Ads
        ads_item_ids = set()
        if ads_general_df is not None and "ID do produto" in ads_general_df.columns:
            ads_item_ids = set(ads_general_df["ID do produto"].dropna().astype(int).tolist())

        merged["esta_em_ads"] = merged["item_id"].astype(str).astype(float).astype("Int64").isin(ads_item_ids) if ads_item_ids else False

        # tabela de quedas
        st.subheader("Quedas (com diagnóstico)")
        if "delta_Vendas (BRL)" in merged.columns:
            drops = merged[merged["delta_Vendas (BRL)"] < 0].copy()
        elif "delta_Pedidos" in merged.columns:
            drops = merged[merged["delta_Pedidos"] < 0].copy()
        else:
            drops = merged.iloc[0:0].copy()

        show_cols = [
            "item_id",
            "curr_Produto_nome" if "curr_Produto_nome" in drops.columns else None,
            "prev_Vendas (BRL)",
            "curr_Vendas (BRL)",
            "delta_Vendas (BRL)",
            "prev_Impressões de Produto",
            "curr_Impressões de Produto",
            "delta_Impressões de Produto_pct",
            "prev_ctr",
            "curr_ctr",
            "delta_ctr_pct",
            "prev_cvr",
            "curr_cvr",
            "delta_cvr_pct",
            "acao_recomendada",
        ]
        show_cols = [c for c in show_cols if c and c in drops.columns]

        if len(drops) == 0:
            st.write("Nenhuma queda detectada pelos critérios de delta.")
        else:
            fmt_sales = {}
            for c in show_cols:
                if "Vendas (BRL)" in c or c in {"delta_Vendas (BRL)"}:
                    fmt_sales[c] = fmt_money
                elif c.endswith("_ctr") or c.endswith("_cvr") or c in {"prev_ctr", "curr_ctr", "prev_cvr", "curr_cvr"}:
                    fmt_sales[c] = fmt_percent
                elif c.endswith("_pct"):
                    fmt_sales[c] = fmt_percent
                elif "Impressões" in c or "Cliques" in c or "Pedidos" in c:
                    fmt_sales[c] = fmt_int

            st.dataframe(
                drops.sort_values(by="delta_Vendas (BRL)", ascending=True)[show_cols].head(200).style.format(fmt_sales),
                use_container_width=True,
                hide_index=True,
            )

        # oportunidades: fora do Ads
        st.subheader("Oportunidades: bons orgânicos fora do Ads")
        # regra: CTR boa (>=3%) e CVR boa (>=2%) com impressão baixa
        opp = merged.copy()
        opp = opp[(opp["esta_em_ads"] == False)]
        if "curr_ctr" in opp.columns and "curr_cvr" in opp.columns and "curr_Impressões de Produto" in opp.columns:
            opp = opp[(opp["curr_ctr"] >= CTR_BOA_MIN) & (opp["curr_cvr"] >= CVR_BOA_MIN) & (opp["curr_Impressões de Produto"] <= low_impressions_threshold)]
            opp["acao"] = "Produto com tração orgânica: incluir em Ads"

            opp_cols = [
                "item_id",
                "curr_Produto_nome" if "curr_Produto_nome" in opp.columns else None,
                "curr_Vendas (BRL)",
                "curr_Impressões de Produto",
                "curr_Cliques Por Produto" if "curr_Cliques Por Produto" in opp.columns else None,
                "curr_Pedidos" if "curr_Pedidos" in opp.columns else None,
                "curr_ctr",
                "curr_cvr",
                "acao",
            ]
            opp_cols = [c for c in opp_cols if c and c in opp.columns]

            fmt_opp = {}
            for c in opp_cols:
                if "Vendas (BRL)" in c:
                    fmt_opp[c] = fmt_money
                elif c in {"curr_ctr", "curr_cvr"}:
                    fmt_opp[c] = fmt_percent
                elif "Impressões" in c or "Cliques" in c or "Pedidos" in c:
                    fmt_opp[c] = fmt_int

            st.dataframe(opp.sort_values(by="curr_Vendas (BRL)", ascending=False)[opp_cols].head(200).style.format(fmt_opp), use_container_width=True, hide_index=True)
        else:
            st.warning("Não encontrei colunas necessárias nas planilhas de vendas para calcular oportunidades.")


st.caption(
    "Regras aplicadas (lei): Se impressões caem → colocar ADS. Se CTR cai → ajustar preço/cauda longa/imagem. Se CVR cai → ajustar copy/gatilhos."
)
