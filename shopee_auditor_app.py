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
# 2) Parsing robusto (CSV Shopee)
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
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def parse_number_br_aggressive(x) -> float:
    """
    Conversor AGRESSIVO para BR, porque o problema do GMV normalmente é:
    - 238.065,00  -> deve virar 238065.00
    - 238.065     -> deve virar 238065
    - 612.085     -> deve virar 612085 (milhar com ponto)
    - 7.129,40    -> deve virar 7129.40
    - pode vir com R$, espaços, etc.

    Regra:
    - Se tem vírgula: vírgula é decimal, ponto é milhar (remove ponto, troca vírgula por ponto)
    - Se não tem vírgula:
        - Se tem ponto e o bloco final tem 3 dígitos => ponto é milhar (remove)
        - Caso contrário => ponto é decimal
    """
    s = _to_str(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "")
    if s.endswith("%"):
        return np.nan

    # mantém só números, ponto, vírgula, sinal
    s = re.sub(r"[^0-9,\.\-]", "", s)
    if not s:
        return np.nan

    if "," in s:
        # BR: 1.234,56 ou 1234,56 ou 1.234.567,89
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # sem vírgula: pode ser milhar com ponto (612.085) ou decimal (32649.43)
        if s.count(".") >= 2:
            # muitos pontos => certamente milhar
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            # se a parte direita tem 3 dígitos => milhar
            if right.isdigit() and len(right) == 3 and left.replace("-", "").isdigit():
                s = left + right
            # senão, mantém decimal
        # senão, inteiro puro

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
        if ("Impress" in line and "Cliques" in line and "," in line) or ("GMV" in line and "," in line):
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
    for ln in meta_lines[1:30]:
        if "," in ln:
            k, v = ln.split(",", 1)
            meta[k.strip()] = v.strip()

    # CRÍTICO: lê TUDO como TEXTO para não estragar GMV
    df = pd.read_csv(
        StringIO(text),
        sep=",",
        skiprows=header_row,
        dtype=str,
        keep_default_na=False,
        na_values=[],
    )
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
    """Parse apenas colunas conhecidas; mantém texto (nome, id etc)."""
    df2 = df.copy()
    for c in df2.columns:
        if c in ADS_PERCENT_COLS:
            df2[c] = df2[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df2[c] = df2[c].apply(parse_number_br_aggressive)
    return df2


def pick_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def add_ads_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    col_imp = pick_first_existing(df, ["Impressões", "Impressões do Produto"])
    col_clk = pick_first_existing(df, ["Cliques", "Cliques de Produtos"])
    col_cost = pick_first_existing(df, ["Despesas", "Custo"])
    col_orders = pick_first_existing(df, ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"])

    # garante numeric
    for c in [col_imp, col_clk, col_cost, col_orders]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if col_imp and col_clk and col_imp in df.columns and col_clk in df.columns:
        df["ctr_calc"] = np.where(df[col_imp] > 0, df[col_clk] / df[col_imp], 0.0)
    else:
        df["ctr_calc"] = np.nan

    if col_clk and col_orders and col_clk in df.columns and col_orders in df.columns:
        df["cvr_calc"] = np.where(df[col_clk] > 0, df[col_orders] / df[col_clk], 0.0)
    else:
        df["cvr_calc"] = np.nan

    if col_cost and col_clk and col_cost in df.columns and col_clk in df.columns:
        df["cpc"] = np.where(df[col_clk] > 0, df[col_cost] / df[col_clk], np.nan)
    else:
        df["cpc"] = np.nan

    if col_cost and col_orders and col_cost in df.columns and col_orders in df.columns:
        df["cpa"] = np.where(df[col_orders] > 0, df[col_cost] / df[col_orders], np.nan)
    else:
        df["cpa"] = np.nan

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    df.attrs["imp_col"] = col_imp
    df.attrs["clk_col"] = col_clk
    df.attrs["cost_col"] = col_cost
    df.attrs["orders_col"] = col_orders

    rev_col = pick_first_existing(df, ["GMV", "Receita direta"])
    df.attrs["rev_col"] = rev_col

    # ACOS calc (spend / gmv)
    if col_cost and rev_col and col_cost in df.columns and rev_col in df.columns:
        df["acos_calc"] = np.where(pd.to_numeric(df[rev_col], errors="coerce").fillna(0) > 0,
                                   pd.to_numeric(df[col_cost], errors="coerce") / pd.to_numeric(df[rev_col], errors="coerce"),
                                   np.nan)
    else:
        df["acos_calc"] = np.nan

    return df


# ============================
# 3) Formatação BR (display)
# ============================

def fmt_brl(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x)
    except Exception:
        return ""
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
    s = f"{v:,.0f}".replace(",", ".")
    return s


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x) * 100
    except Exception:
        return ""
    s = f"{v:.{digits}f}".replace(".", ",")
    return f"{s}%"


def make_display_df(
    df: pd.DataFrame,
    *,
    imp_col: str | None,
    clk_col: str | None,
    cost_col: str | None,
    orders_col: str | None,
    rev_col: str | None,
) -> pd.DataFrame:
    out = df.copy()

    # ints
    for c in [imp_col, clk_col, orders_col]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_int)

    # moeda
    for c in [cost_col, rev_col, "cpc", "cpa", "GMV", "Receita direta", "Despesas", "Custo"]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_brl)

    # percent
    for c in ["ctr_calc", "cvr_calc", "acos_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    for c in ["CTR", "Taxa de Conversão", "Taxa de Conversão Direta", "ROAS", "ROAS Direto", "ACOS", "ACOS Direto", "CTR do Produto"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    return out


# ============================
# 4) UI
# ============================

st.title("Shopee Ads – Auditoria (Estratégico)")

with st.sidebar:
    st.header("Parâmetros (Ads)")
    min_impressions_ctr = st.number_input("Mín. impressões p/ avaliar CTR", value=1000, step=100)
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR / sem conversão", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões 'baixas' (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

    st.divider()
    st.header("Parâmetros (TACOS)")
    tacos_target_pct = st.number_input("TACOS bom (meta) %", value=8.0, step=0.5)  # você edita aqui
    tacos_warn_1_pp = 1.0
    tacos_warn_2_pp = 2.0

st.markdown(
    """**Como usar**
- CSV **Dados gerais de anúncios**: normalmente nível de **campanha/resumo**.
- CSV **Dados do grupo de anúncios**: nível de **produto/anúncio** (nome + ID do produto).
- (Opcional) **Vendas do mês atual**: usado para calcular **TACOS por campanha** usando faturamento total do mês.
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

        # nome do grupo/campanha a partir do título do relatório
        group_name = metag.get("titulo", "")
        group_name = group_name.replace("\ufeff", "")
        group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
        if not group_name:
            group_name = Path(getattr(f, "name", "grupo")).stem

        dfg["Campanha/Grupo"] = group_name
        dfg = add_ads_metrics(dfg)

        group_dfs.append(dfg)

ads_groups_df = pd.concat(group_dfs, ignore_index=True) if group_dfs else None


# ============================
# 6) Auditoria Ads
# ============================

st.header("1) Ads")

if ads_general_df is None and ads_groups_df is None:
    st.info("Suba pelo menos 1 arquivo (geral ou grupos) para iniciar.")
    st.stop()

source_label = "Grupos (produto/anúncio)" if ads_groups_df is not None else "Geral (campanha)"

if ads_groups_df is not None and ads_general_df is not None:
    source_label = st.radio(
        "Fonte para análises",
        ["Grupos (produto/anúncio)", "Geral (campanha)"],
        horizontal=True,
    )

alert_df = ads_groups_df if source_label.startswith("Grupos") else ads_general_df

imp_col = alert_df.attrs.get("imp_col")
clk_col = alert_df.attrs.get("clk_col")
cost_col = alert_df.attrs.get("cost_col")
orders_col = alert_df.attrs.get("orders_col")
rev_col = alert_df.attrs.get("rev_col")

def nsum(df, col):
    if not col or col not in df.columns:
        return np.nan
    return float(np.nansum(pd.to_numeric(df[col], errors="coerce")))

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Impressões", fmt_int(nsum(alert_df, imp_col)) if imp_col else "n/a")
k2.metric("Cliques", fmt_int(nsum(alert_df, clk_col)) if clk_col else "n/a")
ctr_total = (nsum(alert_df, clk_col) / nsum(alert_df, imp_col)) if imp_col and clk_col and nsum(alert_df, imp_col) else np.nan
k3.metric("CTR (calc)", fmt_pct(ctr_total) if pd.notna(ctr_total) else "n/a")
k4.metric("Gasto", fmt_brl(nsum(alert_df, cost_col)) if cost_col else "n/a")
k5.metric("Pedidos/Conv.", fmt_int(nsum(alert_df, orders_col)) if orders_col else "n/a")
rev_total = nsum(alert_df, rev_col) if rev_col else np.nan
k6.metric("GMV", fmt_brl(rev_total) if pd.notna(rev_total) else "n/a")

# ============================
# 6.1 Base (com filtro e soma)
# ============================

st.subheader("Base (com filtro + soma total)")

base_df = alert_df.copy()

if source_label.startswith("Grupos"):
    name_col = "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in base_df.columns else None
    id_col = "ID do produto" if "ID do produto" in base_df.columns else None

    # filtro por campanha/grupo
    if "Campanha/Grupo" in base_df.columns:
        opts = ["(todas)"] + sorted(base_df["Campanha/Grupo"].dropna().unique().tolist())
        sel_group = st.selectbox("Filtrar por Campanha/Grupo", opts, index=0)
        if sel_group != "(todas)":
            base_df = base_df[base_df["Campanha/Grupo"] == sel_group]

    show_cols = [c for c in [
        "Campanha/Grupo",
        name_col,
        id_col,
        imp_col,
        clk_col,
        "ctr_calc",
        orders_col,
        "cvr_calc",
        rev_col,
        cost_col,
        "acos_calc",
        "cpc",
        "cpa",
    ] if c and c in base_df.columns]

    # soma (do que está filtrado)
    sums = {
        "Impressões": nsum(base_df, imp_col) if imp_col else np.nan,
        "Cliques": nsum(base_df, clk_col) if clk_col else np.nan,
        "Gasto": nsum(base_df, cost_col) if cost_col else np.nan,
        "Pedidos": nsum(base_df, orders_col) if orders_col else np.nan,
        "GMV": nsum(base_df, rev_col) if rev_col else np.nan,
    }
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Σ Impressões (filtro)", fmt_int(sums["Impressões"]) if pd.notna(sums["Impressões"]) else "n/a")
    c2.metric("Σ Cliques (filtro)", fmt_int(sums["Cliques"]) if pd.notna(sums["Cliques"]) else "n/a")
    c3.metric("Σ Gasto (filtro)", fmt_brl(sums["Gasto"]) if pd.notna(sums["Gasto"]) else "n/a")
    c4.metric("Σ Pedidos (filtro)", fmt_int(sums["Pedidos"]) if pd.notna(sums["Pedidos"]) else "n/a")
    c5.metric("Σ GMV (filtro)", fmt_brl(sums["GMV"]) if pd.notna(sums["GMV"]) else "n/a")

    base_sorted = base_df.sort_values(by=cost_col, ascending=False) if cost_col in base_df.columns else base_df
    disp = make_display_df(base_sorted[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
    st.dataframe(disp, use_container_width=True, hide_index=True)

else:
    # campanha (geral)
    # tenta achar coluna de nome de campanha
    camp_name_col = pick_first_existing(base_df, ["Nome do Anúncio", "Campanha", "Nome da campanha", "Nome"])
    if camp_name_col is None:
        camp_name_col = base_df.columns[0]

    # filtro de campanha
    opts = ["(todas)"] + sorted(base_df[camp_name_col].dropna().unique().tolist())
    sel_camp = st.selectbox("Filtrar por Campanha", opts, index=0)
    if sel_camp != "(todas)":
        base_df = base_df[base_df[camp_name_col] == sel_camp]

    show_cols = [c for c in [
        camp_name_col,
        imp_col,
        clk_col,
        "ctr_calc",
        orders_col,
        "cvr_calc",
        rev_col,
        cost_col,
        "acos_calc",
        "cpc",
        "cpa",
    ] if c and c in base_df.columns]

    sums = {
        "Impressões": nsum(base_df, imp_col) if imp_col else np.nan,
        "Cliques": nsum(base_df, clk_col) if clk_col else np.nan,
        "Gasto": nsum(base_df, cost_col) if cost_col else np.nan,
        "Pedidos": nsum(base_df, orders_col) if orders_col else np.nan,
        "GMV": nsum(base_df, rev_col) if rev_col else np.nan,
    }
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Σ Impressões (filtro)", fmt_int(sums["Impressões"]) if pd.notna(sums["Impressões"]) else "n/a")
    c2.metric("Σ Cliques (filtro)", fmt_int(sums["Cliques"]) if pd.notna(sums["Cliques"]) else "n/a")
    c3.metric("Σ Gasto (filtro)", fmt_brl(sums["Gasto"]) if pd.notna(sums["Gasto"]) else "n/a")
    c4.metric("Σ Pedidos (filtro)", fmt_int(sums["Pedidos"]) if pd.notna(sums["Pedidos"]) else "n/a")
    c5.metric("Σ GMV (filtro)", fmt_brl(sums["GMV"]) if pd.notna(sums["GMV"]) else "n/a")

    base_sorted = base_df.sort_values(by=cost_col, ascending=False) if cost_col in base_df.columns else base_df
    disp = make_display_df(base_sorted[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
    st.dataframe(disp, use_container_width=True, hide_index=True)

# ============================
# 6.2 Campanhas: ACOS + TACOS + ROI
# ============================

st.divider()
st.subheader("Campanhas: ACOS + TACOS + ROI (com semáforo)")

# uploader de vendas (mês atual) para pegar faturamento total
sales_file = st.file_uploader("CSV/XLSX – Vendas (mês atual) para faturamento total (opcional)", type=["csv", "xlsx"], key="sales_month")

sales_total_revenue = None
sales_df = None

def read_any_table(uploaded):
    if uploaded is None:
        return None
    name = getattr(uploaded, "name", "").lower()
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded, dtype=str)
    # csv
    raw = uploaded.read()
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = raw.decode("latin-1", errors="replace")
    return pd.read_csv(StringIO(text), dtype=str, keep_default_na=False, na_values=[])

if sales_file is not None:
    sales_df = read_any_table(sales_file)
    # normaliza colunas
    sales_df.columns = [c.strip() for c in sales_df.columns]

    # tenta adivinhar coluna de faturamento
    cand = [c for c in sales_df.columns if any(k in c.lower() for k in ["fa]()_]()
