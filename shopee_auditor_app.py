import re
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Ads & Vendas Auditor", layout="wide")

# =========================================================
# 1) REGRAS (LEI)
# =========================================================
CTR_OTIMA_MIN = 0.04
CTR_OTIMA_MAX = 0.06
CTR_BOA_MIN = 0.03
CTR_RUIM_MAX = 0.015

CVR_OTIMA_MIN = 0.03
CVR_BOA_MIN = 0.02
CVR_RUIM_MAX = 0.015


def classify_ctr(ctr):
    if pd.isna(ctr):
        return "n/a"
    if CTR_OTIMA_MIN <= ctr <= CTR_OTIMA_MAX:
        return "ótima"
    if CTR_BOA_MIN <= ctr < CTR_OTIMA_MIN:
        return "boa"
    if ctr <= CTR_RUIM_MAX:
        return "ruim"
    return "média"


def classify_cvr(cvr):
    if pd.isna(cvr):
        return "n/a"
    if cvr >= CVR_OTIMA_MIN:
        return "ótima"
    if cvr >= CVR_BOA_MIN:
        return "boa"
    if cvr <= CVR_RUIM_MAX:
        return "ruim"
    return "média"


# =========================================================
# 2) PARSING ROBUSTO SHOPEE
# =========================================================
def _s(x):
    return "" if x is None else str(x).strip()


def parse_percent(x):
    s = _s(x).replace("%", "").replace(",", ".")
    try:
        return float(s) / 100
    except Exception:
        return np.nan


def parse_number_any_locale(x):
    s = _s(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "")
    s = re.sub(r"[^0-9,.\-]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        s = s.replace(".", "").replace(",", ".")
    elif s.count(".") >= 2:
        s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def detect_csv_header_row(text):
    for i, line in enumerate(text.splitlines()[:120]):
        if line.startswith("#,"):
            return i
    return 0


def read_shopee_ads_csv(uploaded_file):
    raw = uploaded_file.read()
    text = raw.decode("utf-8", errors="replace")
    header_row = detect_csv_header_row(text)

    meta_lines = text.splitlines()[:header_row]
    meta = {}
    if meta_lines:
        meta["titulo"] = meta_lines[0].replace("\ufeff", "").strip()

    df = pd.read_csv(StringIO(text), sep=",", skiprows=header_row)
    df.columns = [c.strip() for c in df.columns]
    return df, meta


ADS_PERCENT_COLS = {
    "CTR", "Taxa de Conversão", "Taxa de Conversão Direta",
    "ROAS", "ROAS Direto", "ACOS", "ACOS Direto", "CTR do Produto"
}

ADS_NUMERIC_COLS = {
    "Impressões", "Cliques",
    "Conversões", "Conversões Diretas",
    "Itens Vendidos", "Itens Vendidos Diretos",
    "GMV", "Receita direta",
    "Despesas", "Custo",
    "Impressões do Produto", "Cliques de Produtos",
}


def parse_ads_table(df):
    df = df.copy()
    for c in df.columns:
        if c in ADS_PERCENT_COLS:
            df[c] = df[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df[c] = df[c].apply(parse_number_any_locale)
    return df


def pick(df, cols):
    for c in cols:
        if c in df.columns:
            return c
    return None


# =========================================================
# 3) MÉTRICAS ADS
# =========================================================
def add_ads_metrics(df):
    df = df.copy()

    col_imp = pick(df, ["Impressões", "Impressões do Produto"])
    col_clk = pick(df, ["Cliques", "Cliques de Produtos"])
    col_cost = pick(df, ["Despesas", "Custo"])
    col_orders = pick(df, ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"])

    if col_imp and col_clk:
        df["ctr_calc"] = np.where(df[col_imp] > 0, df[col_clk] / df[col_imp], 0)
    else:
        df["ctr_calc"] = 0

    if col_clk and col_orders:
        df["cvr_calc"] = np.where(df[col_clk] > 0, df[col_orders] / df[col_clk], 0)
    else:
        df["cvr_calc"] = 0

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

    df.attrs["imp_col"] = col_imp
    df.attrs["clk_col"] = col_clk
    df.attrs["cost_col"] = col_cost
    df.attrs["orders_col"] = col_orders
    df.attrs["rev_col"] = "GMV" if "GMV" in df.columns else None

    return df


# =========================================================
# 4) FORMATAÇÃO BR
# =========================================================
def fmt_brl(x):
    if pd.isna(x):
        return ""
    s = f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def fmt_int(x):
    if pd.isna(x):
        return ""
    return f"{int(x):,}".replace(",", ".")


def fmt_pct(x):
    if pd.isna(x):
        return ""
    return f"{x*100:.2f}%".replace(".", ",")


# =========================================================
# 5) SIDEBAR — PARÂMETROS MANUAIS
# =========================================================
with st.sidebar:
    st.header("Parâmetros de análise")

    min_impressions_ctr = st.number_input(
        "Mín. impressões p/ avaliar CTR",
        value=1000,
        step=100
    )

    min_clicks_eval = st.number_input(
        "Mín. cliques p/ avaliar CVR",
        value=30,
        step=5
    )

    min_spend_no_conv = st.number_input(
        "Gasto mínimo p/ 'gastando sem converter' (R$)",
        value=50.0,
        step=10.0
    )

    low_impressions_threshold = st.number_input(
        "Impressões consideradas baixas",
        value=300,
        step=50
    )

    dominance_spend_share = st.slider(
        "Dominância de gasto no grupo (%)",
        min_value=50,
        max_value=95,
        value=70
    )

    st.divider()
    st.header("Uploads")
    ads_general_file = st.file_uploader("CSV – Dados Gerais de Anúncios", type="csv")
    ads_group_files = st.file_uploader(
        "CSV – Dados do Grupo de Anúncios",
        type="csv",
        accept_multiple_files=True
    )

# =========================================================
# 6) CARREGAMENTO ADS
# =========================================================
ads_general_df = None
ads_groups_df = None

if ads_general_file:
    df, meta = read_shopee_ads_csv(ads_general_file)
    ads_general_df = add_ads_metrics(parse_ads_table(df))

if ads_group_files:
    groups = []
    for f in ads_group_files:
        df, meta = read_shopee_ads_csv(f)
        df = parse_ads_table(df)
        df["Campanha/Grupo"] = meta.get("titulo", Path(f.name).stem)
        df = add_ads_metrics(df)
        if "ID do produto" in df.columns:
            df = df[df["ID do produto"].astype(str) != "-"]
        groups.append(df)
    ads_groups_df = pd.concat(groups, ignore_index=True)

# =========================================================
# 7) VISÃO PRINCIPAL
# =========================================================
st.title("Shopee Ads – Auditoria Profissional")

source = ads_groups_df if ads_groups_df is not None else ads_general_df

if source is None:
    st.info("Envie ao menos um CSV para iniciar.")
    st.stop()

imp = source.attrs["imp_col"]
clk = source.attrs["clk_col"]
cost = source.attrs["cost_col"]
orders = source.attrs["orders_col"]
rev = source.attrs["rev_col"]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Impressões", fmt_int(source[imp].sum()))
c2.metric("Cliques", fmt_int(source[clk].sum()))
c3.metric("CTR", fmt_pct(source[clk].sum() / source[imp].sum()))
c4.metric("Investimento", fmt_brl(source[cost].sum()))
c5.metric("Pedidos", fmt_int(source[orders].sum()))
c6.metric("GMV", fmt_brl(source[rev].sum()))

st.subheader("Base detalhada")
st.dataframe(source, use_container_width=True)
