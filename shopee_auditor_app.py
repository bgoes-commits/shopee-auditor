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
    for ln in meta_lines[1:25]:
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
    """Parse apenas colunas conhecidas; não mexe em texto (nome, id etc)."""
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

    # CTR / CVR com proteção de zero
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

    # ACOS calc (se GMV existir)
    if "GMV" in df.columns and col_cost:
        df["acos_calc"] = np.where(df["GMV"] > 0, df[col_cost] / df["GMV"], np.nan)
    else:
        df["acos_calc"] = np.nan

    # ROI calc: (GMV - gasto) / gasto
    if "GMV" in df.columns and col_cost:
        df["roi_calc"] = np.where(df[col_cost] > 0, (df["GMV"] - df[col_cost]) / df[col_cost], np.nan)
    else:
        df["roi_calc"] = np.nan

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    df.attrs["imp_col"] = col_imp
    df.attrs["clk_col"] = col_clk
    df.attrs["cost_col"] = col_cost
    df.attrs["orders_col"] = col_orders
    df.attrs["rev_col"] = "GMV" if "GMV" in df.columns else None

    return df


# ============================
# 3) Leitura vendas XLSX (faturamento total)
# ============================
def read_sales_xlsx(file) -> pd.DataFrame:
    # tenta primeira aba
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_revenue_col(df: pd.DataFrame) -> str | None:
    # candidatos comuns (Shopee/relatórios)
    candidates = [
        "GMV",
        "Faturamento",
        "Receita",
        "Receita Total",
        "Vendas",
        "Total de vendas",
        "Total",
        "Valor do Pedido",
        "Valor do pedido",
        "Pagamento do Comprador",
        "Pagamento do comprador",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: procura por colunas com palavras-chave
    for c in df.columns:
        cl = c.lower()
        if "fatur" in cl or "receit" in cl or ("venda" in cl and "qtd" not in cl) or "gmv" in cl:
            return c
    return None


def sum_revenue_from_sales(df: pd.DataFrame) -> float:
    col = find_revenue_col(df)
    if not col:
        return float("nan")
    vals = df[col].apply(parse_number_any_locale)
    return float(np.nansum(vals))


# ============================
# 4) Formatação BR (para exibir)
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

    # ints
    for c in [imp_col, clk_col, orders_col]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_int)

    # moeda
    for c in [cost_col, rev_col, "cpc", "cpa", "GMV"]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_brl)

    # percent
    for c in ["ctr_calc", "cvr_calc", "acos_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v, 2))

    # ROI (multiplicador)
    if "roi_calc" in out.columns:
        out["roi_calc"] = out["roi_calc"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}x".replace(".", ","))

    return out


# ============================
# 5) TACoS helpers
# ============================
def tacos_class(tacos: float, target: float) -> tuple[str, str]:
    """
    target em fração (ex.: 0.10 = 10%)
    - <= target: verde
    - target+1pp até target+2pp: amarelo
    - > target+2pp: vermelho
    """
    if pd.isna(tacos) or pd.isna(target):
        return ("", "n/a")
    if tacos <= target:
        return ("🟢", "ótimo")
    if tacos <= target + 0.02:  # +2 pontos percentuais
        if tacos <= target + 0.01:
            return ("🟡", "atenção (+1pp)")
        return ("🟡", "atenção (+2pp)")
    return ("🔴", "crítico")


# ============================
# 6) UI
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
- CSV **Dados gerais de anúncios**: visão de campanha (vários tipos e métodos).
- CSV **Dados do grupo de anúncios**: detalhe por produto/anúncio (nome + ID do produto).
- XLSX **Vendas (mês atual)**: usado para pegar o **faturamento total** e calcular **TACoS**.
"""
)

tabs_top = st.tabs(["Ads (Campanhas)", "Ads (Grupos)", "TACoS", "Vendas"])

# ============================
# Uploads
# ============================
with st.sidebar:
    st.divider()
    st.header("Uploads")

ads_general_file = st.sidebar.file_uploader("CSV – Dados gerais de anúncios (Shopee)", type=["csv"], key="ads_general")
ads_group_files = st.sidebar.file_uploader(
    "CSV – Dados por Grupo de Anúncios (1 ou mais)",
    type=["csv"],
    accept_multiple_files=True,
    key="ads_groups",
)

sales_month_current = st.sidebar.file_uploader(
    "XLSX – Vendas mês atual (para TACoS)",
    type=["xlsx"],
    key="sales_current",
)

# ============================
# Carregamento Ads
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
# Vendas (faturamento total)
# ============================
sales_total_revenue = np.nan
sales_df = None
if sales_month_current is not None:
    try:
        sales_df = read_sales_xlsx(sales_month_current)
        sales_total_revenue = sum_revenue_from_sales(sales_df)
    except Exception:
        sales_total_revenue = np.nan
        sales_df = None


# ======================================================
# TAB 1) Ads (Campanhas)
# ======================================================
with tabs_top[0]:
    st.header("Ads – Campanhas (Geral)")

    if ads_general_df is None:
        st.info("Suba o CSV **Dados gerais de anúncios** para ver campanhas.")
    else:
        imp_col = ads_general_df.attrs.get("imp_col")
        clk_col = ads_general_df.attrs.get("clk_col")
        cost_col = ads_general_df.attrs.get("cost_col")
        orders_col = ads_general_df.attrs.get("orders_col")
        rev_col = ads_general_df.attrs.get("rev_col")  # GMV

        def nsum(col):
            if not col or col not in ads_general_df.columns:
                return 0.0
            return float(np.nansum(pd.to_numeric(ads_general_df[col], errors="coerce")))

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Impressões", fmt_int(nsum(imp_col)) if imp_col else "n/a")
        k2.metric("Cliques", fmt_int(nsum(clk_col)) if clk_col else "n/a")
        ctr_total = (nsum(clk_col) / nsum(imp_col)) if imp_col and clk_col and nsum(imp_col) else np.nan
        k3.metric("CTR (calc)", fmt_pct(ctr_total) if pd.notna(ctr_total) else "n/a")
        k4.metric("Gasto", fmt_brl(nsum(cost_col)) if cost_col else "n/a")
        k5.metric("Pedidos/Conv.", fmt_int(nsum(orders_col)) if orders_col else "n/a")
        k6.metric("GMV", fmt_brl(nsum(rev_col)) if rev_col else "n/a")

        if ads_general_meta:
            with st.expander("Metadados do relatório (geral)", expanded=False):
                st.json(ads_general_meta)

        # Campanha = Nome do Anúncio (no geral)
        camp_col = "Nome do Anúncio" if "Nome do Anúncio" in ads_general_df.columns else None
        if not camp_col:
            st.warning("Não encontrei a coluna **Nome do Anúncio** para agrupar campanhas.")
        else:
            # agrega por campanha
            g = ads_general_df.groupby(camp_col, dropna=False).agg(
                impress=("Impressões", "sum") if "Impressões" in ads_general_df.columns else (imp_col, "sum"),
                cliques=("Cliques", "sum") if "Cliques" in ads_general_df.columns else (clk_col, "sum"),
                gasto=(cost_col, "sum") if cost_col else ("GMV", "sum"),
                gmv=("GMV", "sum") if "GMV" in ads_general_df.columns else (rev_col, "sum"),
                conv=(orders_col, "sum") if orders_col else ("GMV", "sum"),
            ).reset_index()

            g["ctr_calc"] = np.where(g["impress"] > 0, g["cliques"] / g["impress"], 0.0)
            g["cvr_calc"] = np.where(g["cliques"] > 0, g["conv"] / g["cliques"], 0.0)
            g["acos_calc"] = np.where(g["gmv"] > 0, g["gasto"] / g["gmv"], np.nan)
            g["roi_calc"] = np.where(g["gasto"] > 0, (g["gmv"] - g["gasto"]) / g["gasto"], np.nan)

            # tabela final (sem Status)
            show = g.rename(columns={camp_col: "Campanha"})
            show = show[["Campanha", "impress", "cliques", "ctr_calc", "conv", "cvr_calc", "gmv", "gasto", "acos_calc", "roi_calc"]]

            disp = show.copy()
            disp["impress"] = disp["impress"].apply(fmt_int)
            disp["cliques"] = disp["cliques"].apply(fmt_int)
            disp["conv"] = disp["conv"].apply(fmt_int)
            disp["ctr_calc"] = disp["ctr_calc"].apply(fmt_pct)
            disp["cvr_calc"] = disp["cvr_calc"].apply(fmt_pct)
            disp["gmv"] = disp["gmv"].apply(fmt_brl)
            disp["gasto"] = disp["gasto"].apply(fmt_brl)
            disp["acos_calc"] = disp["acos_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
            disp["roi_calc"] = disp["roi_calc"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}x".replace(".", ","))

            st.subheader("Campanhas (ACOS / ROI)")
            st.dataframe(disp, use_container_width=True, hide_index=True)


# ======================================================
# TAB 2) Ads (Grupos)
# ======================================================
with tabs_top[1]:
    st.header("Ads – Grupos (Produto/Anúncio)")

    if ads_groups_df is None:
        st.info("Suba os CSVs **Dados por Grupo de Anúncios** para ver os anúncios/produtos.")
    else:
        imp_col = ads_groups_df.attrs.get("imp_col")
        clk_col = ads_groups_df.attrs.get("clk_col")
        cost_col = ads_groups_df.attrs.get("cost_col")
        orders_col = ads_groups_df.attrs.get("orders_col")
        rev_col = ads_groups_df.attrs.get("rev_col")  # GMV

        def nsum(col):
            if not col or col not in ads_groups_df.columns:
                return 0.0
            return float(np.nansum(pd.to_numeric(ads_groups_df[col], errors="coerce")))

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Impressões", fmt_int(nsum(imp_col)) if imp_col else "n/a")
        k2.metric("Cliques", fmt_int(nsum(clk_col)) if clk_col else "n/a")
        ctr_total = (nsum(clk_col) / nsum(imp_col)) if imp_col and clk_col and nsum(imp_col) else np.nan
        k3.metric("CTR (calc)", fmt_pct(ctr_total) if pd.notna(ctr_total) else "n/a")
        k4.metric("Gasto", fmt_brl(nsum(cost_col)) if cost_col else "n/a")
        k5.metric("Pedidos/Conv.", fmt_int(nsum(orders_col)) if orders_col else "n/a")
        k6.metric("GMV", fmt_brl(nsum(rev_col)) if rev_col else "n/a")

        st.subheader("Base (grupos)")
        show_cols = [c for c in [
            "Campanha/Grupo",
            "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in ads_groups_df.columns else None,
            "ID do produto" if "ID do produto" in ads_groups_df.columns else None,
            imp_col, clk_col, "ctr_calc", "ctr_class",
            orders_col, "cvr_calc", "cvr_class",
            rev_col, cost_col, "cpc", "cpa",
            "acos_calc", "roi_calc"
        ] if c and c in ads_groups_df.columns]

        base_sorted = ads_groups_df.sort_values(by=cost_col, ascending=False) if cost_col in ads_groups_df.columns else ads_groups_df
        disp = make_display_df(base_sorted[show_cols], imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)
        st.dataframe(disp, use_container_width=True, hide_index=True)


# ======================================================
# TAB 3) TACoS
# ======================================================
with tabs_top[2]:
    st.header("TACoS – por campanha")

    if ads_general_df is None:
        st.info("Suba o CSV **Dados gerais de anúncios** para calcular TACoS por campanha.")
    elif pd.isna(sales_total_revenue):
        st.warning("Suba o **XLSX de vendas do mês atual** para obter o faturamento total e calcular TACoS.")
    else:
        st.success(f"Faturamento total (mês atual – XLSX): **{fmt_brl(sales_total_revenue)}**")

        # input do TACoS bom
        st.subheader("Regra TACoS (manual)")
        tacos_target_pct = st.number_input(
            "Digite o TACoS bom (%)",
            min_value=0.0,
            max_value=100.0,
            value=10.0,
            step=0.5
        )
        tacos_target = tacos_target_pct / 100.0

        # agrega gasto por campanha
        cost_col = ads_general_df.attrs.get("cost_col")
        camp_col = "Nome do Anúncio" if "Nome do Anúncio" in ads_general_df.columns else None
        if not camp_col or not cost_col:
            st.warning("Não encontrei colunas necessárias para TACoS (Nome do Anúncio / Despesas).")
        else:
            g = ads_general_df.groupby(camp_col, dropna=False).agg(
                gasto=(cost_col, "sum"),
                gmv=("GMV", "sum") if "GMV" in ads_general_df.columns else (ads_general_df.attrs.get("rev_col"), "sum"),
            ).reset_index()

            g["tacos"] = np.where(sales_total_revenue > 0, g["gasto"] / sales_total_revenue, np.nan)
            g["acos_calc"] = np.where(g["gmv"] > 0, g["gasto"] / g["gmv"], np.nan)
            g["roi_calc"] = np.where(g["gasto"] > 0, (g["gmv"] - g["gasto"]) / g["gasto"], np.nan)

            # classifica por bolinha
            icons, labels = [], []
            for v in g["tacos"].tolist():
                ic, lb = tacos_class(v, tacos_target)
                icons.append(ic)
                labels.append(lb)

            g["status_tacos"] = icons
            g["tacos_class"] = labels

            # display
            out = g.rename(columns={camp_col: "Campanha"})
            out = out[["status_tacos", "Campanha", "gasto", "gmv", "acos_calc", "tacos", "tacos_class", "roi_calc"]].copy()

            out["gasto"] = out["gasto"].apply(fmt_brl)
            out["gmv"] = out["gmv"].apply(fmt_brl)
            out["acos_calc"] = out["acos_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
            out["tacos"] = out["tacos"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
            out["roi_calc"] = out["roi_calc"].apply(lambda v: "" if pd.isna(v) else f"{v:.2f}x".replace(".", ","))

            st.subheader("TACoS por campanha (bolinhas)")
            st.dataframe(out.sort_values(by="gasto", ascending=False), use_container_width=True, hide_index=True)

            # KPI geral TACoS
            total_spend = float(np.nansum(pd.to_numeric(ads_general_df[cost_col], errors="coerce")))
            tacos_total = (total_spend / sales_total_revenue) if sales_total_revenue > 0 else np.nan
            st.caption(f"TACoS total (Ads / Faturamento): **{fmt_pct(tacos_total)}** | Gasto: **{fmt_brl(total_spend)}**")


# ======================================================
# TAB 4) Vendas
# ======================================================
with tabs_top[3]:
    st.header("Vendas (mês atual)")

    if sales_df is None:
        st.info("Suba o XLSX de vendas do mês atual (no menu lateral) para visualizar e validar o faturamento.")
    else:
        rev_col = find_revenue_col(sales_df)
        st.write(f"Coluna de faturamento identificada: **{rev_col if rev_col else 'não encontrada'}**")
        st.metric("Faturamento total (mês atual)", fmt_brl(sales_total_revenue) if pd.notna(sales_total_revenue) else "n/a")
        st.dataframe(sales_df.head(50), use_container_width=True)
