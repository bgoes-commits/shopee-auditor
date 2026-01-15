import re
from io import StringIO
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
        s = s.replace(",", ".")
    try:
        return float(s) / 100.0
    except Exception:
        return np.nan


def parse_number_any_locale(x) -> float:
    """
    REGRA CORRETA PRA SHOPEE:
    - Se já for numérico (float/int): NÃO mexe.
    - Se vier como texto: converte BR (1.234,56) ou EUA (1234.56).
    """
    if isinstance(x, (int, float, np.integer, np.floating)) and not pd.isna(x):
        return float(x)

    s = _to_str(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "")
    if s.endswith("%"):
        return np.nan

    s = re.sub(r"[^0-9,\.\-]", "", s)

    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            # BR: 1.234,56
            s = s.replace(".", "").replace(",", ".")
        else:
            # EUA: 1,234.56
            s = s.replace(",", "")
    elif "," in s:
        # BR: 1234,56  (ou 1.234,56)
        s = s.replace(".", "").replace(",", ".")
    else:
        # EUA: 1234.56  OU inteiro 1234
        if s.count(".") >= 2:
            s = s.replace(".", "")

    try:
        return float(s)
    except Exception:
        return np.nan


def detect_csv_header_row(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines[:200]):
        if line.startswith("#,"):
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
    df2 = df.copy()
    for c in df2.columns:
        if c in ADS_PERCENT_COLS:
            df2[c] = df2[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df2[c] = df2[c].apply(parse_number_any_locale)
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

    # CTR/CVR com proteção
    if col_imp and col_clk:
        df["ctr_calc"] = np.where(df[col_imp] > 0, df[col_clk] / df[col_imp], 0.0)
    else:
        df["ctr_calc"] = 0.0

    if col_clk and col_orders:
        df["cvr_calc"] = np.where(df[col_clk] > 0, df[col_orders] / df[col_clk], 0.0)
    else:
        df["cvr_calc"] = 0.0

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
    df.attrs["rev_col"] = "GMV" if "GMV" in df.columns else None  # GMV fixo

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


def make_display_df(df: pd.DataFrame, *, imp_col, clk_col, cost_col, orders_col, rev_col) -> pd.DataFrame:
    out = df.copy()

    for c in [imp_col, clk_col, orders_col]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_int)

    for c in [cost_col, rev_col, "cpc", "cpa"]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_brl)

    for c in ["ctr_calc", "cvr_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(fmt_pct)

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
- CSV **Dados gerais de anúncios**: resumo (campanha/agregado, mas pode incluir vários tipos).
- CSV **Dados do grupo de anúncios**: detalhe por produto/anúncio (nome + ID)."""
)

colA, colB = st.columns(2)
with colA:
    ads_general_file = st.file_uploader("CSV – Dados gerais de anúncios (Shopee)", type=["csv"], key="ads_general")
with colB:
    ads_group_files = st.file_uploader("CSV – Dados por Grupo de Anúncios (1 ou mais)", type=["csv"], accept_multiple_files=True, key="ads_groups")

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

group_dfs = []
if ads_group_files:
    for f in ads_group_files:
        dfg, metag = read_shopee_ads_csv(f)
        dfg = parse_ads_table(dfg)

        group_name = metag.get("titulo", "")
        group_name = group_name.replace("\ufeff", "").replace("Ad Group -", "").strip()
        if not group_name:
            group_name = Path(getattr(f, "name", "grupo")).stem

        dfg["Campanha/Grupo"] = group_name
        dfg = add_ads_metrics(dfg)

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

source_label = "Grupos (produto/anúncio)" if ads_groups_df is not None else "Geral (resumo)"
if ads_groups_df is not None and ads_general_df is not None:
    source_label = st.radio("Fonte para alertas", ["Grupos (produto/anúncio)", "Geral (resumo)"], horizontal=True)

alert_df = ads_groups_df if source_label.startswith("Grupos") else ads_general_df

# ========== NOVO: RAW (TOTAL) vs FILTERED ==========
raw_df = alert_df.copy()  # SEMPRE total (para somar tudo)
filtered_df = raw_df      # vai receber filtros

# pega colunas usadas
imp_col = raw_df.attrs.get("imp_col")
clk_col = raw_df.attrs.get("clk_col")
cost_col = raw_df.attrs.get("cost_col")
orders_col = raw_df.attrs.get("orders_col")
rev_col = raw_df.attrs.get("rev_col")

# ====== Filtros opcionais (usuário pediu) ======
with st.sidebar:
    st.divider()
    st.header("Filtros (opcionais)")
    apply_filters_to_kpis = st.checkbox("Aplicar filtros também aos KPIs", value=False)

    # status varia por fonte
    status_col = None
    if source_label.startswith("Grupos"):
        if "Status do Anúncio" in raw_df.columns:
            status_col = "Status do Anúncio"
    else:
        if "Status" in raw_df.columns:
            status_col = "Status"

    selected_status = None
    if status_col:
        opts = sorted([x for x in raw_df[status_col].dropna().unique().tolist()])
        selected_status = st.multiselect("Status", opts, default=[])

    # tipos de anúncios (normalmente existe no geral)
    type_col = "Tipos de Anúncios" if "Tipos de Anúncios" in raw_df.columns else None
    selected_types = None
    if type_col:
        opts = sorted([x for x in raw_df[type_col].dropna().unique().tolist()])
        selected_types = st.multiselect("Tipos de Anúncios", opts, default=[])

    # método (no print aparece "Método de ..." — vamos capturar automaticamente)
    method_col = None
    for c in raw_df.columns:
        if c.lower().startswith("método"):
            method_col = c
            break
    selected_methods = None
    if method_col:
        opts = sorted([x for x in raw_df[method_col].dropna().unique().tolist()])
        selected_methods = st.multiselect(method_col, opts, default=[])

# aplica filtros (na tabela/alertas, e opcionalmente KPIs)
filtered_df = raw_df.copy()
if status_col and selected_status:
    filtered_df = filtered_df[filtered_df[status_col].isin(selected_status)]
if type_col and selected_types:
    filtered_df = filtered_df[filtered_df[type_col].isin(selected_types)]
if method_col and selected_methods:
    filtered_df = filtered_df[filtered_df[method_col].isin(selected_methods)]

kpi_df = filtered_df if apply_filters_to_kpis else raw_df

def nsum(df_, col):
    if not col or col not in df_.columns:
        return 0.0
    return float(np.nansum(pd.to_numeric(df_[col], errors="coerce")))

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Impressões", fmt_int(nsum(kpi_df, imp_col)) if imp_col else "n/a")
k2.metric("Cliques", fmt_int(nsum(kpi_df, clk_col)) if clk_col else "n/a")
ctr_total = (nsum(kpi_df, clk_col) / nsum(kpi_df, imp_col)) if imp_col and clk_col and nsum(kpi_df, imp_col) else np.nan
k3.metric("CTR (calc)", fmt_pct(ctr_total) if pd.notna(ctr_total) else "n/a")
k4.metric("Gasto", fmt_brl(nsum(kpi_df, cost_col)) if cost_col else "n/a")
k5.metric("Pedidos/Conv.", fmt_int(nsum(kpi_df, orders_col)) if orders_col else "n/a")
k6.metric("GMV", fmt_brl(nsum(kpi_df, rev_col)) if rev_col else "n/a")

if ads_general_meta:
    with st.expander("Metadados do relatório (geral)", expanded=False):
        st.json(ads_general_meta)

# ============================
# 6.1 Tabela base (usa FILTERED)
# ============================
st.subheader("Base (com filtros)")

base_df = filtered_df.copy()

if source_label.startswith("Grupos"):
    show_cols = [c for c in [
        "Campanha/Grupo",
        "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in base_df.columns else None,
        "ID do produto" if "ID do produto" in base_df.columns else None,
        imp_col, clk_col, "ctr_calc", "ctr_class",
        orders_col, "cvr_calc", "cvr_class",
        rev_col, cost_col, "cpc", "cpa",
    ] if c and c in base_df.columns]
else:
    show_cols = [c for c in [
        "Nome do Anúncio" if "Nome do Anúncio" in base_df.columns else None,
        "Status" if "Status" in base_df.columns else None,
        "Tipos de Anúncios" if "Tipos de Anúncios" in base_df.columns else None,
        imp_col, clk_col, "ctr_calc", "ctr_class",
        orders_col, "cvr_calc", "cvr_class",
        rev_col, cost_col, "cpc", "cpa",
    ] if c and c in base_df.columns]

sorted_df = base_df.sort_values(by=cost_col, ascending=False) if cost_col in base_df.columns else base_df
disp = make_display_df(sorted_df[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
st.dataframe(disp, use_container_width=True, hide_index=True)

# ============================
# 6.2 Alertas (usa FILTERED)
# ============================
st.subheader("Alertas e ações (com filtros)")

tabs = st.tabs([
    "Gastando sem converter",
    "CTR ruim",
    "CVR ruim",
    "Bons com pouca impressão",
    "Mover anúncio (competição no grupo)",
])

alert_work_df = filtered_df

with tabs[0]:
    if cost_col and orders_col and clk_col:
        wasting = alert_work_df[
            (pd.to_numeric(alert_work_df[orders_col], errors="coerce").fillna(0) == 0)
            & (
                (pd.to_numeric(alert_work_df[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
                | (pd.to_numeric(alert_work_df[cost_col], errors="coerce").fillna(0) >= float(min_spend_no_conv))
            )
        ].copy()
        if wasting.empty:
            st.info("Nenhum item no critério.")
        else:
            wasting["ação"] = "Pausar/remover (gastando sem converter)"
            cols = [c for c in [
                "Campanha/Grupo" if "Campanha/Grupo" in wasting.columns else None,
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in wasting.columns else None,
                "ID do produto" if "ID do produto" in wasting.columns else None,
                "Nome do Anúncio" if "Nome do Anúncio" in wasting.columns else None,
                imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "cpc", "cpa", "ação"
            ] if c and c in wasting.columns]
            wasting_sorted = wasting.sort_values(by=cost_col, ascending=False)
            disp = make_display_df(wasting_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas suficientes (gasto, cliques e pedidos/conversões).")

with tabs[1]:
    if imp_col and clk_col:
        bad_ctr = alert_work_df[
            (pd.to_numeric(alert_work_df[imp_col], errors="coerce").fillna(0) >= min_impressions_ctr)
            & (pd.to_numeric(alert_work_df["ctr_calc"], errors="coerce").fillna(0) <= CTR_RUIM_MAX)
        ].copy()
        if bad_ctr.empty:
            st.info("Nenhum item no critério.")
        else:
            bad_ctr["ação"] = "Ajustar preço + cauda longa + imagem (CTR ruim)"
            cols = [c for c in [
                "Campanha/Grupo" if "Campanha/Grupo" in bad_ctr.columns else None,
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in bad_ctr.columns else None,
                "ID do produto" if "ID do produto" in bad_ctr.columns else None,
                "Nome do Anúncio" if "Nome do Anúncio" in bad_ctr.columns else None,
                imp_col, clk_col, "ctr_calc", "ctr_class",
                orders_col, "cvr_calc", rev_col, cost_col, "ação"
            ] if c and c in bad_ctr.columns]
            bad_ctr_sorted = bad_ctr.sort_values(by="ctr_calc", ascending=True)
            disp = make_display_df(bad_ctr_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas de impressões/cliques.")

with tabs[2]:
    if clk_col and orders_col:
        bad_cvr = alert_work_df[
            (pd.to_numeric(alert_work_df[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
            & (pd.to_numeric(alert_work_df["cvr_calc"], errors="coerce").fillna(0) <= CVR_RUIM_MAX)
        ].copy()
        if bad_cvr.empty:
            st.info("Nenhum item no critério.")
        else:
            bad_cvr["ação"] = "Ajustar copy + gatilhos de conversão (CVR ruim)"
            cols = [c for c in [
                "Campanha/Grupo" if "Campanha/Grupo" in bad_cvr.columns else None,
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in bad_cvr.columns else None,
                "ID do produto" if "ID do produto" in bad_cvr.columns else None,
                "Nome do Anúncio" if "Nome do Anúncio" in bad_cvr.columns else None,
                imp_col, clk_col, "ctr_calc",
                orders_col, "cvr_calc", "cvr_class",
                rev_col, cost_col, "ação"
            ] if c and c in bad_cvr.columns]
            bad_cvr_sorted = bad_cvr.sort_values(by="cvr_calc", ascending=True)
            disp = make_display_df(bad_cvr_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas de cliques e pedidos/conversões.")

with tabs[3]:
    if imp_col:
        good_low = alert_work_df[
            (pd.to_numeric(alert_work_df[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)
            & (
                (pd.to_numeric(alert_work_df["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                | (pd.to_numeric(alert_work_df["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
            )
        ].copy()
        if good_low.empty:
            st.info("Nenhum item no critério.")
        else:
            good_low["ação"] = "Escalar: aumentar entrega / revisar estrutura"
            cols = [c for c in [
                "Campanha/Grupo" if "Campanha/Grupo" in good_low.columns else None,
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in good_low.columns else None,
                "ID do produto" if "ID do produto" in good_low.columns else None,
                "Nome do Anúncio" if "Nome do Anúncio" in good_low.columns else None,
                imp_col, clk_col, "ctr_calc",
                orders_col, "cvr_calc",
                rev_col, cost_col, "ação"
            ] if c and c in good_low.columns]
            good_low_sorted = good_low.sort_values(by=imp_col, ascending=True)
            disp = make_display_df(good_low_sorted[cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei coluna de impressões.")

with tabs[4]:
    if ads_groups_df is None or not source_label.startswith("Grupos"):
        st.info("Esse alerta exige os CSVs de **Grupo de Anúncios** e a fonte 'Grupos'.")
    else:
        df_g = alert_work_df.copy()
        g_imp = df_g.attrs.get("imp_col") or "Impressões"
        g_clk = df_g.attrs.get("clk_col") or "Cliques"
        g_cost = df_g.attrs.get("cost_col") or "Despesas"
        g_orders = df_g.attrs.get("orders_col") or "Conversões Diretas"

        df_g["spend_share"] = df_g.groupby("Campanha/Grupo")[g_cost].transform(lambda s: s / s.sum() if s.sum() else 0.0)

        prom = df_g[
            (pd.to_numeric(df_g[g_imp], errors="coerce").fillna(0) <= low_impressions_threshold)
            & (
                (pd.to_numeric(df_g["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                | (pd.to_numeric(df_g["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
            )
        ].copy()

        dom = df_g[df_g["spend_share"] >= (dominance_spend_share / 100.0)][["Campanha/Grupo"]].drop_duplicates()
        prom = prom.merge(dom, on="Campanha/Grupo", how="inner")

        if prom.empty:
            st.info("Nenhum item no critério.")
        else:
            prom["ação"] = "Mover para outra campanha/grupo (competição interna)"
            prom["spend_share_%"] = prom["spend_share"] * 100

            cols = [c for c in [
                "Campanha/Grupo",
                "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in prom.columns else None,
                "ID do produto" if "ID do produto" in prom.columns else None,
                g_imp, g_clk, "ctr_calc",
                g_orders, "cvr_calc",
                "GMV" if "GMV" in prom.columns else None,
                g_cost,
                "spend_share_%",
                "ação",
            ] if c and c in prom.columns]

            prom_sorted = prom.sort_values(by="spend_share", ascending=False)
            disp = make_display_df(prom_sorted[cols], imp_col=g_imp, clk_col=g_clk, cost_col=g_cost, orders_col=g_orders, rev_col="GMV")
            disp["spend_share_%"] = prom_sorted["spend_share_%"].apply(lambda v: f"{v:.1f}%".replace(".", ","))
            st.dataframe(disp, use_container_width=True, hide_index=True)

st.divider()
st.header("2) Vendas (mês vs mês)")
st.info("Quando você quiser, eu implemento aqui a comparação dos 2 XLSX (mês anterior vs mês atual) com as regras de ADS/CTR/CVR/copy.")
