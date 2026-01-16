import re
from io import StringIO, BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Ads Auditor (Campanha + Loja)", layout="wide")

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
# 2) Parsing robusto Shopee CSV (Ads)
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
    BR agressivo:
    - 238.065,00 -> 238065.00
    - 612.085 -> 612085
    - 32649.43 -> 32649.43
    - 61223.00 -> 61223.00
    """
    s = _to_str(x)
    if not s or s.lower() in {"nan", "-"}:
        return np.nan

    s = s.replace("R$", "").replace(" ", "")
    if s.endswith("%"):
        return np.nan

    s = re.sub(r"[^0-9,\.\-]", "", s)
    if not s:
        return np.nan

    if "," in s:
        # decimal vírgula -> remove pontos (milhar) e troca vírgula por ponto
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # sem vírgula: se tiver múltiplos pontos, é milhar
        if s.count(".") >= 2:
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            # caso "612.085" (milhar) -> junta
            if right.isdigit() and len(right) == 3 and left.replace("-", "").isdigit():
                s = left + right

    try:
        return float(s)
    except Exception:
        return np.nan


def normalize_product_id(s: str) -> str:
    """
    IDs às vezes vêm como '2,3694E+10' no CSV.
    Aqui convertemos para '23694000000' (string).
    """
    t = _to_str(s)
    if not t or t in {"-", "nan", "None"}:
        return ""

    tt = t.replace(",", ".")
    try:
        if "e" in tt.lower():
            v = float(tt)
            if np.isfinite(v):
                return str(int(round(v)))
    except Exception:
        pass

    t_digits = re.sub(r"\D", "", t)
    return t_digits if t_digits else t


def detect_csv_header_row(text: str) -> int:
    lines = text.splitlines()
    for i, line in enumerate(lines[:500]):
        if line.startswith("#,") or line.startswith("#;"):
            return i
    for i, line in enumerate(lines[:800]):
        if "Anúncio / Nome do Produto" in line and "ID do produto" in line:
            return i
    return 0


def detect_delimiter(header_line: str) -> str:
    candidates = [",", ";", "\t"]
    counts = {sep: header_line.count(sep) for sep in candidates}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def read_shopee_csv(uploaded_file) -> tuple[pd.DataFrame, dict]:
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

    lines = text.splitlines()
    header_line = lines[header_row] if header_row < len(lines) else ""
    sep = detect_delimiter(header_line)
    meta["sep_detectado"] = sep

    df = pd.read_csv(
        StringIO(text),
        sep=sep,
        skiprows=header_row,
        dtype=str,
        keep_default_na=False,
        na_values=[],
        engine="python",
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
    df2 = df.copy()
    for c in df2.columns:
        if c in ADS_PERCENT_COLS:
            df2[c] = df2[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df2[c] = df2[c].apply(parse_number_br_aggressive)

    if "ID do produto" in df2.columns:
        df2["ID do produto"] = df2["ID do produto"].apply(normalize_product_id)

    return df2


def add_ads_metrics(df: pd.DataFrame, *, imp_col, clk_col, cost_col, orders_col, rev_col) -> pd.DataFrame:
    df = df.copy()

    for c in [imp_col, clk_col, cost_col, orders_col, rev_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ctr_calc"] = np.where((imp_col and clk_col and df[imp_col] > 0), df[clk_col] / df[imp_col], np.nan)
    df["cvr_calc"] = np.where((clk_col and orders_col and df[clk_col] > 0), df[orders_col] / df[clk_col], np.nan)

    df["cpc"] = np.where((cost_col and clk_col and df[clk_col] > 0), df[cost_col] / df[clk_col], np.nan)
    df["cpa"] = np.where((cost_col and orders_col and df[orders_col] > 0), df[cost_col] / df[orders_col], np.nan)

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    df["acos_calc"] = np.where((cost_col and rev_col and df[rev_col] > 0), df[cost_col] / df[rev_col], np.nan)

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
    return f"{v:,.0f}".replace(",", ".")


def fmt_pct(x, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        v = float(x) * 100
    except Exception:
        return ""
    return f"{v:.{digits}f}".replace(".", ",") + "%"


def make_display_df(df: pd.DataFrame, imp_col, clk_col, cost_col, orders_col, rev_col) -> pd.DataFrame:
    out = df.copy()

    for c in [imp_col, clk_col, orders_col]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_int)

    money_cols = [cost_col, rev_col, "cpc", "cpa"]
    money_cols = [c for c in money_cols if c and c in out.columns]
    seen = set()
    money_cols = [c for c in money_cols if not (c in seen or seen.add(c))]
    for c in money_cols:
        out[c] = out[c].apply(fmt_brl)

    for c in ["ctr_calc", "cvr_calc", "acos_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    if "CTR" in out.columns:
        out["CTR"] = out["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    return out


# ============================
# 4) Leitura Vendas (Loja) - XLSX com 2 abas
# ============================

def uploaded_bytes(uploaded_file) -> bytes:
    # Streamlit UploadedFile tem getvalue() (não consome)
    try:
        return uploaded_file.getvalue()
    except Exception:
        # fallback: lê (pode consumir)
        return uploaded_file.read()


def read_any_table(uploaded_file, *, sheet_name=None) -> pd.DataFrame:
    name = getattr(uploaded_file, "name", "").lower()
    data = uploaded_bytes(uploaded_file)

    if name.endswith(".xlsx") or name.endswith(".xls"):
        xls = pd.ExcelFile(BytesIO(data))
        if sheet_name is None:
            sheet_name = xls.sheet_names[0]
        df = xls.parse(sheet_name, dtype=str)
        df.attrs["sheet_names"] = xls.sheet_names
        df.attrs["sheet_used"] = sheet_name
    else:
        text = data.decode("utf-8", errors="replace")
        first = text.splitlines()[0] if text.splitlines() else ""
        sep = "," if first.count(",") >= first.count(";") else ";"
        df = pd.read_csv(StringIO(text), sep=sep, dtype=str, engine="python", keep_default_na=False)
        df.attrs["sheet_names"] = []
        df.attrs["sheet_used"] = None

    df.columns = [str(c).strip() for c in df.columns]
    return df


def list_excel_sheets(uploaded_file) -> list[str]:
    name = getattr(uploaded_file, "name", "").lower()
    if not (name.endswith(".xlsx") or name.endswith(".xls")):
        return []
    data = uploaded_bytes(uploaded_file)
    xls = pd.ExcelFile(BytesIO(data))
    return xls.sheet_names


def find_col(df: pd.DataFrame, exact: list[str], contains: list[str]) -> str | None:
    cols = [str(c).strip() for c in df.columns]
    for c in exact:
        if c in cols:
            return c
    low = {c: c.lower() for c in cols}
    for c, lc in low.items():
        if any(term in lc for term in contains):
            return c
    return None


def numeric_series_from(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].apply(parse_number_br_aggressive).astype(float)


# ============================
# 5) UI
# ============================

st.title("Shopee Ads – Campanhas + Loja (TACOS + Insights)")

with st.sidebar:
    st.header("Parâmetros (Ads)")
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões baixas (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

    st.divider()
    st.header("Sinalização (Campanhas)")
    acos_target_pct = st.number_input("ACOS bom (meta) %", value=12.0, step=0.5)
    acos_warn_pp = st.number_input("Tolerância ACOS p/ amarelo (+p.p.)", value=2.0, step=0.5)

    st.divider()
    st.header("Sinalização TACOS (Loja)")
    tacos_target_pct = st.number_input("TACOS bom (meta) %", value=10.0, step=0.5)
    tacos_warn_pp = st.number_input("Tolerância TACOS p/ amarelo (+p.p.)", value=2.0, step=0.5)

    st.divider()
    st.header("Insights (Loja)")
    min_revenue_candidate = st.number_input("Faturamento mínimo p/ sugerir Ads (R$)", value=1000.0, step=100.0)
    drop_alert_pct = st.number_input("Queda de faturamento p/ alerta (%)", value=20.0, step=5.0)

st.markdown(
    """
**Como funciona**
- Você sobe **CSV(s) de Dados do Grupo de Anúncios**.
- Em cada campanha, a **linha sem ID do produto** é o **TOTAL da campanha** (nível campanha).
- As linhas com **ID do produto** são os **anúncios/produtos** (nível anúncio).
- Depois, você sobe **Vendas do mês (Loja)** (Excel `producttraffic`) para calcular **TACOS** e gerar insights mês x mês.
"""
)

# ============================
# 6) Upload Ads (Grupos)
# ============================

ads_group_files = st.file_uploader(
    "CSV – Dados do Grupo de Anúncios (1 ou mais)",
    type=["csv"],
    accept_multiple_files=True,
    key="ads_groups",
)

if not ads_group_files:
    st.info("Suba 1 ou mais CSVs de **Dados do Grupo de Anúncios**.")
    st.stop()

frames = []
for f in ads_group_files:
    df_raw, meta = read_shopee_csv(f)
    df = parse_ads_table(df_raw)

    group_name = meta.get("titulo", "")
    group_name = group_name.replace("\ufeff", "")
    group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
    if not group_name:
        group_name = Path(getattr(f, "name", "grupo")).stem

    df["Campanha"] = group_name
    frames.append(df)

df_all = pd.concat(frames, ignore_index=True)

# colunas do Ads (exatas)
imp_col = "Impressões" if "Impressões" in df_all.columns else ("Impressões do Produto" if "Impressões do Produto" in df_all.columns else None)
clk_col = "Cliques" if "Cliques" in df_all.columns else ("Cliques de Produtos" if "Cliques de Produtos" in df_all.columns else None)
rev_col = "GMV" if "GMV" in df_all.columns else None
cost_col = "Despesas" if "Despesas" in df_all.columns else ("Custo" if "Custo" in df_all.columns else None)

orders_col = None
for cand in ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"]:
    if cand in df_all.columns:
        orders_col = cand
        break

df_all = add_ads_metrics(
    df_all,
    imp_col=imp_col,
    clk_col=clk_col,
    cost_col=cost_col,
    orders_col=orders_col,
    rev_col=rev_col,
)

id_col = "ID do produto" if "ID do produto" in df_all.columns else None
name_col = "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in df_all.columns else None

# ============================
# 7) Upload Vendas (Loja) - Excel producttraffic
# ============================

st.divider()
st.header("Vendas totais da loja (mês x mês) → KPIs + TACOS + Insights")

c1, c2 = st.columns(2)
with c1:
    sales_curr_file = st.file_uploader(
        "Vendas - mês ATUAL (Excel producttraffic)",
        type=["xlsx", "xls", "csv"],
        key="sales_curr",
    )
with c2:
    sales_prev_file = st.file_uploader(
        "Vendas - mês ANTERIOR (opcional)",
        type=["xlsx", "xls", "csv"],
        key="sales_prev",
    )

sales_curr_sheet = None
sales_prev_sheet = None

if sales_curr_file is not None:
    sheets = list_excel_sheets(sales_curr_file)
    if sheets:
        # geralmente tem 2 abas: (pedido realizado) e (pedido pago)
        default_idx = 1 if len(sheets) > 1 else 0
        sales_curr_sheet = st.selectbox("Aba (mês atual)", sheets, index=default_idx, key="sales_curr_sheet")

if sales_prev_file is not None:
    sheets2 = list_excel_sheets(sales_prev_file)
    if sheets2:
        default_idx2 = 1 if len(sheets2) > 1 else 0
        sales_prev_sheet = st.selectbox("Aba (mês anterior)", sheets2, index=default_idx2, key="sales_prev_sheet")

faturamento_atual = None
faturamento_anterior = None
df_sales_curr = None
df_sales_prev = None

# Lendo mês atual
if sales_curr_file is not None:
    df_sales_curr = read_any_table(sales_curr_file, sheet_name=sales_curr_sheet)

    col_rev_curr = find_col(
        df_sales_curr,
        exact=["Vendas (BRL)", "Faturamento", "GMV", "Receita", "Vendas", "Total", "Valor"],
        contains=["vendas (brl)", "fatur", "gmv", "receit", "vend", "total", "valor"],
    )
    if col_rev_curr is None:
        col_rev_curr = st.selectbox(
            "Escolha a coluna de faturamento (mês atual)",
            df_sales_curr.columns.tolist(),
            key="rev_curr_pick",
        )

    faturamento_atual = float(np.nansum(numeric_series_from(df_sales_curr, col_rev_curr)))
    st.caption(f"Aba usada (mês atual): **{df_sales_curr.attrs.get('sheet_used')}**")

# Lendo mês anterior
if sales_prev_file is not None:
    df_sales_prev = read_any_table(sales_prev_file, sheet_name=sales_prev_sheet)

    col_rev_prev = find_col(
        df_sales_prev,
        exact=["Vendas (BRL)", "Faturamento", "GMV", "Receita", "Vendas", "Total", "Valor"],
        contains=["vendas (brl)", "fatur", "gmv", "receit", "vend", "total", "valor"],
    )
    if col_rev_prev is None:
        col_rev_prev = st.selectbox(
            "Escolha a coluna de faturamento (mês anterior)",
            df_sales_prev.columns.tolist(),
            key="rev_prev_pick",
        )

    faturamento_anterior = float(np.nansum(numeric_series_from(df_sales_prev, col_rev_prev)))
    st.caption(f"Aba usada (mês anterior): **{df_sales_prev.attrs.get('sheet_used')}**")

# ============================
# 8) KPI Loja + KPI Ads + TACOS TOTAL
# ============================

# Spend total Ads (preferindo linhas TOTAL sem ID; fallback soma geral)
ads_spend_total = np.nan
if cost_col and cost_col in df_all.columns:
    if id_col and id_col in df_all.columns:
        id_clean = df_all[id_col].astype(str).str.strip()
        is_total = id_clean.isin(["", "-", "nan", "None"])
        if is_total.any():
            ads_spend_total = float(np.nansum(pd.to_numeric(df_all.loc[is_total, cost_col], errors="coerce")))
        else:
            ads_spend_total = float(np.nansum(pd.to_numeric(df_all[cost_col], errors="coerce")))
    else:
        ads_spend_total = float(np.nansum(pd.to_numeric(df_all[cost_col], errors="coerce")))

tacos_total = np.nan
if faturamento_atual and faturamento_atual > 0 and pd.notna(ads_spend_total):
    tacos_total = ads_spend_total / faturamento_atual

k1, k2, k3 = st.columns(3)
k1.metric("Faturamento total da loja (mês atual)", fmt_brl(faturamento_atual) if faturamento_atual is not None else "—")
k2.metric("Investimento Ads (Despesas)", fmt_brl(ads_spend_total) if pd.notna(ads_spend_total) else "—")
k3.metric("TACOS (total)", fmt_pct(tacos_total) if pd.notna(tacos_total) else "—")

if faturamento_anterior is not None and faturamento_anterior > 0 and faturamento_atual is not None:
    delta_loja = (faturamento_atual / faturamento_anterior) - 1
    st.metric("Variação do faturamento (mês x mês)", fmt_pct(delta_loja, digits=1))

# ============================
# 9) Visão geral de campanhas (TOTAL sem ID) + ACOS + TACOS + ROAS
# ============================

st.divider()
st.header("Visão Geral de Campanhas (totais)")

if id_col is None:
    st.warning("Não encontrei a coluna 'ID do produto'. Não consigo separar TOTAL vs anúncios.")
else:
    id_clean = df_all[id_col].astype(str).str.strip()
    is_total = id_clean.isin(["", "-", "nan", "None"])
    camp_total = df_all[is_total].copy()

    if camp_total.empty:
        st.warning("Não encontrei linhas TOTAL (sem ID).")
    else:
        grp = camp_total.groupby("Campanha", dropna=False).agg({
            imp_col: "sum" if imp_col else "sum",
            clk_col: "sum" if clk_col else "sum",
            orders_col: "sum" if orders_col else "sum",
            rev_col: "sum" if rev_col else "sum",
            cost_col: "sum" if cost_col else "sum",
        }).reset_index()

        grp = grp.rename(columns={
            imp_col: "Impressões",
            clk_col: "Cliques",
            orders_col: "Pedidos",
            rev_col: "GMV",
            cost_col: "Despesas",
        })

        grp["CTR"] = np.where(grp["Impressões"].fillna(0) > 0, grp["Cliques"].fillna(0) / grp["Impressões"].fillna(0), np.nan)
        grp["CVR"] = np.where(grp["Cliques"].fillna(0) > 0, grp["Pedidos"].fillna(0) / grp["Cliques"].fillna(0), np.nan)
        grp["ACOS"] = np.where(grp["GMV"].fillna(0) > 0, grp["Despesas"].fillna(0) / grp["GMV"].fillna(0), np.nan)
        grp["ROAS_calc"] = np.where(grp["Despesas"].fillna(0) > 0, grp["GMV"].fillna(0) / grp["Despesas"].fillna(0), np.nan)

        grp["CTR_status"] = grp["CTR"].apply(classify_ctr)
        grp["CVR_status"] = grp["CVR"].apply(classify_cvr)

        target_acos = acos_target_pct / 100.0
        warn_acos = acos_warn_pp / 100.0

        def acos_semaforo(x):
            if pd.isna(x):
                return "n/a"
            if x <= target_acos:
                return "🟢 ok"
            if x <= target_acos + warn_acos:
                return "🟡 atenção"
            return "🔴 crítico"

        grp["ACOS_sinal"] = grp["ACOS"].apply(acos_semaforo)

        # TACOS por campanha (alocado no faturamento total da loja)
        target_tacos = tacos_target_pct / 100.0
        warn_tacos = tacos_warn_pp / 100.0

        def tacos_semaforo(x):
            if pd.isna(x):
                return "n/a"
            if x <= target_tacos:
                return "🟢 ok"
            if x <= target_tacos + warn_tacos:
                return "🟡 atenção"
            return "🔴 crítico"

        if faturamento_atual and faturamento_atual > 0:
            grp["TACOS"] = grp["Despesas"] / faturamento_atual
            grp["TACOS_sinal"] = grp["TACOS"].apply(tacos_semaforo)
        else:
            grp["TACOS"] = np.nan
            grp["TACOS_sinal"] = "n/a"

        # filtro
        camp_opts = ["(todas)"] + sorted(grp["Campanha"].dropna().unique().tolist())
        sel_camp = st.selectbox("Filtrar campanha (visão geral)", camp_opts, index=0, key="camp_overview_filter")
        view_c = grp.copy()
        if sel_camp != "(todas)":
            view_c = view_c[view_c["Campanha"] == sel_camp]

        view_c = view_c.sort_values(by="Despesas", ascending=False)

        disp = view_c.copy()
        disp["Impressões"] = disp["Impressões"].apply(fmt_int)
        disp["Cliques"] = disp["Cliques"].apply(fmt_int)
        disp["Pedidos"] = disp["Pedidos"].apply(fmt_int)
        disp["GMV"] = disp["GMV"].apply(fmt_brl)
        disp["Despesas"] = disp["Despesas"].apply(fmt_brl)
        disp["CTR"] = disp["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS"] = disp["ACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["TACOS"] = disp["TACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ROAS"] = disp["ROAS_calc"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        cols = [
            "Campanha",
            "GMV",
            "Despesas",
            "ACOS",
            "ACOS_sinal",
            "TACOS",
            "TACOS_sinal",
            "ROAS",
            "Impressões",
            "Cliques",
            "CTR",
            "CTR_status",
            "Pedidos",
            "CVR",
            "CVR_status",
        ]
        st.dataframe(disp[cols], use_container_width=True, hide_index=True)

        st.caption("CTR/CVR seguem sua lei. ACOS/TACOS usam a meta e tolerância da sidebar (verde/amarelo/vermelho).")

# ============================
# 10) Insights (Loja + Ads)
# - usa planilha producttraffic: ID do Item / Vendas (BRL) / Impressões / Cliques / CTR / Taxa de Conversão / Pedidos
# ============================

st.divider()
st.header("Insights (Loja + Ads)")

if df_sales_curr is None:
    st.info("Suba as vendas do mês ATUAL (Excel producttraffic) para liberar insights da loja.")
else:
    # colunas padrão do producttraffic
    sales_id_col = find_col(
        df_sales_curr,
        exact=["ID do Item", "ID do produto", "Product ID", "ID Produto", "ID"],
        contains=["id do item", "id do produto", "product id", "id produto"],
    )
    if sales_id_col is None:
        sales_id_col = st.selectbox("Escolha a coluna de ID do produto (vendas mês atual)", df_sales_curr.columns.tolist(), key="sales_id_pick")

    sales_rev_col_curr = find_col(
        df_sales_curr,
        exact=["Vendas (BRL)", "Faturamento", "GMV", "Receita", "Vendas", "Total", "Valor"],
        contains=["vendas (brl)", "fatur", "gmv", "receit", "vend", "total", "valor"],
    )
    if sales_rev_col_curr is None:
        sales_rev_col_curr = st.selectbox("Escolha a coluna de faturamento (vendas mês atual)", df_sales_curr.columns.tolist(), key="sales_rev_pick2")

    # métricas loja para diagnóstico "lei"
    sales_imp_col_curr = find_col(
        df_sales_curr,
        exact=["Impressões de Produto", "Impressões do Produto", "Impressões"],
        contains=["impress"],
    )
    sales_clk_col_curr = find_col(
        df_sales_curr,
        exact=["Cliques Por Produto", "Cliques de Produtos", "Cliques"],
        contains=["clique"],
    )
    sales_ctr_col_curr = find_col(
        df_sales_curr,
        exact=["CTR"],
        contains=["ctr"],
    )
    sales_cvr_col_curr = find_col(
        df_sales_curr,
        exact=["Taxa de Conversão de Pedidos", "Taxa de Conversão", "Conversão"],
        contains=["convers"],
    )
    sales_orders_col_curr = find_col(
        df_sales_curr,
        exact=["Pedidos", "Pedidos Realizados", "Pedidos Pagos"],
        contains=["pedido"],
    )

    sales_curr = df_sales_curr.copy()
    sales_curr["prod_key"] = sales_curr[sales_id_col].astype(str).str.strip().apply(normalize_product_id)
    sales_curr["rev_curr"] = numeric_series_from(sales_curr, sales_rev_col_curr)

    if sales_imp_col_curr:
        sales_curr["imp_curr"] = numeric_series_from(sales_curr, sales_imp_col_curr)
    else:
        sales_curr["imp_curr"] = np.nan

    if sales_clk_col_curr:
        sales_curr["clk_curr"] = numeric_series_from(sales_curr, sales_clk_col_curr)
    else:
        sales_curr["clk_curr"] = np.nan

    if sales_ctr_col_curr:
        sales_curr["ctr_curr"] = sales_curr[sales_ctr_col_curr].apply(parse_percent)
    else:
        sales_curr["ctr_curr"] = np.where(
            sales_curr["imp_curr"].fillna(0) > 0, sales_curr["clk_curr"].fillna(0) / sales_curr["imp_curr"].fillna(0), np.nan
        )

    if sales_cvr_col_curr:
        sales_curr["cvr_curr"] = sales_curr[sales_cvr_col_curr].apply(parse_percent)
    else:
        if sales_orders_col_curr:
            sales_curr["ord_curr"] = numeric_series_from(sales_curr, sales_orders_col_curr)
            sales_curr["cvr_curr"] = np.where(
                sales_curr["clk_curr"].fillna(0) > 0, sales_curr["ord_curr"].fillna(0) / sales_curr["clk_curr"].fillna(0), np.nan
            )
        else:
            sales_curr["cvr_curr"] = np.nan

    # agregado mês atual
    sales_curr_agg = sales_curr.groupby("prod_key", dropna=False).agg({
        "rev_curr": "sum",
        "imp_curr": "sum",
        "clk_curr": "sum",
        "ctr_curr": "mean",
        "cvr_curr": "mean",
    }).reset_index()

    # mês anterior
    if df_sales_prev is not None:
        sales_id_col_prev = find_col(
            df_sales_prev,
            exact=[sales_id_col, "ID do Item", "ID do produto", "Product ID", "ID Produto", "ID"],
            contains=["id do item", "id do produto", "product id", "id produto"],
        )
        if sales_id_col_prev is None:
            sales_id_col_prev = st.selectbox("Escolha a coluna de ID do produto (vendas mês anterior)", df_sales_prev.columns.tolist(), key="sales_id_prev_pick")

        sales_rev_col_prev = find_col(
            df_sales_prev,
            exact=["Vendas (BRL)", "Faturamento", "GMV", "Receita", "Vendas", "Total", "Valor"],
            contains=["vendas (brl)", "fatur", "gmv", "receit", "vend", "total", "valor"],
        )
        if sales_rev_col_prev is None:
            sales_rev_col_prev = st.selectbox("Escolha a coluna de faturamento (vendas mês anterior)", df_sales_prev.columns.tolist(), key="sales_rev_prev_pick2")

        sales_imp_col_prev = find_col(
            df_sales_prev,
            exact=["Impressões de Produto", "Impressões do Produto", "Impressões"],
            contains=["impress"],
        )
        sales_clk_col_prev = find_col(
            df_sales_prev,
            exact=["Cliques Por Produto", "Cliques de Produtos", "Cliques"],
            contains=["clique"],
        )
        sales_ctr_col_prev = find_col(df_sales_prev, exact=["CTR"], contains=["ctr"])
        sales_cvr_col_prev = find_col(
            df_sales_prev,
            exact=["Taxa de Conversão de Pedidos", "Taxa de Conversão", "Conversão"],
            contains=["convers"],
        )

        sales_prev = df_sales_prev.copy()
        sales_prev["prod_key"] = sales_prev[sales_id_col_prev].astype(str).str.strip().apply(normalize_product_id)
        sales_prev["rev_prev"] = numeric_series_from(sales_prev, sales_rev_col_prev)

        if sales_imp_col_prev:
            sales_prev["imp_prev"] = numeric_series_from(sales_prev, sales_imp_col_prev)
        else:
            sales_prev["imp_prev"] = np.nan

        if sales_clk_col_prev:
            sales_prev["clk_prev"] = numeric_series_from(sales_prev, sales_clk_col_prev)
        else:
            sales_prev["clk_prev"] = np.nan

        if sales_ctr_col_prev:
            sales_prev["ctr_prev"] = sales_prev[sales_ctr_col_prev].apply(parse_percent)
        else:
            sales_prev["ctr_prev"] = np.where(
                sales_prev["imp_prev"].fillna(0) > 0, sales_prev["clk_prev"].fillna(0) / sales_prev["imp_prev"].fillna(0), np.nan
            )

        if sales_cvr_col_prev:
            sales_prev["cvr_prev"] = sales_prev[sales_cvr_col_prev].apply(parse_percent)
        else:
            sales_prev["cvr_prev"] = np.nan

        sales_prev_agg = sales_prev.groupby("prod_key", dropna=False).agg({
            "rev_prev": "sum",
            "imp_prev": "sum",
            "clk_prev": "sum",
            "ctr_prev": "mean",
            "cvr_prev": "mean",
        }).reset_index()

        sales_mom = sales_curr_agg.merge(sales_prev_agg, on="prod_key", how="left")
        sales_mom[["rev_prev", "imp_prev", "clk_prev", "ctr_prev", "cvr_prev"]] = sales_mom[
            ["rev_prev", "imp_prev", "clk_prev", "ctr_prev", "cvr_prev"]
        ].fillna(0.0)
        sales_mom["rev_delta_pct"] = np.where(sales_mom["rev_prev"] > 0, (sales_mom["rev_curr"] / sales_mom["rev_prev"]) - 1, np.nan)

        # deltas dos parâmetros
        sales_mom["imp_delta_pct"] = np.where(sales_mom["imp_prev"] > 0, (sales_mom["imp_curr"] / sales_mom["imp_prev"]) - 1, np.nan)
        sales_mom["ctr_delta_pp"] = (sales_mom["ctr_curr"] - sales_mom["ctr_prev"])  # em pontos (fração)
        sales_mom["cvr_delta_pp"] = (sales_mom["cvr_curr"] - sales_mom["cvr_prev"])
    else:
        sales_mom = sales_curr_agg.copy()
        sales_mom["rev_prev"] = np.nan
        sales_mom["rev_delta_pct"] = np.nan
        sales_mom["imp_delta_pct"] = np.nan
        sales_mom["ctr_delta_pp"] = np.nan
        sales_mom["cvr_delta_pp"] = np.nan

    # base ads por produto (apenas linhas com ID)
    if id_col is not None:
        ads_prod = df_all[~df_all[id_col].astype(str).str.strip().isin(["", "-", "nan", "None"])].copy()
        ads_prod["prod_key"] = ads_prod[id_col].astype(str).str.strip().apply(normalize_product_id)
    else:
        ads_prod = pd.DataFrame(columns=["prod_key"])

    ads_keys = set(ads_prod["prod_key"].dropna().astype(str).tolist())

    tabs_i = st.tabs([
        "Candidatos a Ads (fora do Ads)",
        "Queda mês x mês + diagnóstico (lei)",
        "Oportunidades (bom Ads + pouca entrega)",
    ])

    # 1) Candidatos a Ads
    with tabs_i[0]:
        cand = sales_mom.copy()
        cand = cand[~cand["prod_key"].astype(str).isin(ads_keys)]
        cand = cand[cand["rev_curr"] >= float(min_revenue_candidate)]

        if cand.empty:
            st.info("Nenhum candidato no critério (fora do Ads + faturamento mínimo).")
        else:
            cand = cand.sort_values("rev_curr", ascending=False)
            out = cand.copy()
            out["Faturamento (mês atual)"] = out["rev_curr"].apply(fmt_brl)
            if df_sales_prev is not None:
                out["Faturamento (mês anterior)"] = out["rev_prev"].apply(fmt_brl)
                out["Variação"] = out["rev_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")
                show = ["prod_key", "Faturamento (mês atual)", "Faturamento (mês anterior)", "Variação"]
            else:
                show = ["prod_key", "Faturamento (mês atual)"]

            st.dataframe(out[show], use_container_width=True, hide_index=True)
            st.caption("Ação: colocar Ads nesses produtos (já vendem bem e não estão anunciados).")

    # 2) Queda mês x mês + diagnóstico (lei)
    with tabs_i[1]:
        if df_sales_prev is None:
            st.info("Suba o mês anterior para liberar o diagnóstico de queda.")
        else:
            drops = sales_mom.copy()
            drops = drops[pd.notna(drops["rev_delta_pct"])]
            drops = drops[drops["rev_delta_pct"] <= -(float(drop_alert_pct) / 100.0)]

            if drops.empty:
                st.info("Nenhum produto com queda acima do limite.")
            else:
                def diagnose_lei(row):
                    # LEI (usando dados da loja):
                    # - Se IMPRESSÕES caíram: colocar ADS
                    # - Se CTR caiu: preço/cauda longa/imagem
                    # - Se CVR caiu: copy/gatilhos
                    imp_delta = row.get("imp_delta_pct")
                    ctr_pp = row.get("ctr_delta_pp")
                    cvr_pp = row.get("cvr_delta_pp")

                    # prioridade: impressão
                    if pd.notna(imp_delta) and imp_delta < 0:
                        return "Impressões caíram → colocar/fortalecer ADS"
                    # depois CTR
                    if pd.notna(ctr_pp) and ctr_pp < 0:
                        return "CTR caiu → ajustar preço + cauda longa + imagem"
                    # depois CVR
                    if pd.notna(cvr_pp) and cvr_pp < 0:
                        return "CVR caiu → ajustar copy + gatilhos de conversão"
                    return "Queda em vendas → checar estoque/preço concorrência/página"

                drops["Ação sugerida"] = drops.apply(diagnose_lei, axis=1)

                out = drops.copy()
                out["Fat atual"] = out["rev_curr"].apply(fmt_brl)
                out["Fat anterior"] = out["rev_prev"].apply(fmt_brl)
                out["Variação"] = out["rev_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")
                out["Imp Δ"] = out["imp_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")
                out["CTR Δ (p.p.)"] = out["ctr_delta_pp"].apply(lambda v: (f"{(v*100):.2f}".replace(".", ",") + " p.p.") if pd.notna(v) else "")
                out["CVR Δ (p.p.)"] = out["cvr_delta_pp"].apply(lambda v: (f"{(v*100):.2f}".replace(".", ",") + " p.p.") if pd.notna(v) else "")

                st.dataframe(
                    out[["prod_key", "Fat atual", "Fat anterior", "Variação", "Imp Δ", "CTR Δ (p.p.)", "CVR Δ (p.p.)", "Ação sugerida"]],
                    use_container_width=True,
                    hide_index=True,
                )

    # 3) Oportunidade: bom Ads + pouca impressão
    with tabs_i[2]:
        if ads_prod.empty or imp_col is None:
            st.info("Não tenho base de Ads por produto (ou não achei coluna de Impressões no Ads).")
        else:
            opp = ads_prod.copy()
            opp = opp.merge(sales_curr_agg[["prod_key", "rev_curr"]], on="prod_key", how="left")
            opp["rev_curr"] = opp["rev_curr"].fillna(0.0)

            opp = opp[
                (pd.to_numeric(opp[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)
                & (
                    (pd.to_numeric(opp["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                    | (pd.to_numeric(opp["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
                )
            ].copy()

            if opp.empty:
                st.info("Nenhuma oportunidade no critério.")
            else:
                opp = opp.sort_values(by=imp_col, ascending=True)
                out = opp.copy()
                out["CTR"] = out["ctr_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out["CVR"] = out["cvr_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out["GMV (ads)"] = out[rev_col].apply(fmt_brl) if rev_col in out.columns else ""
                out["Desp (ads)"] = out[cost_col].apply(fmt_brl) if cost_col in out.columns else ""
                out["Fat loja (mês atual)"] = out["rev_curr"].apply(fmt_brl)

                cols_show = ["Campanha"]
                if name_col in out.columns:
                    cols_show += [name_col]
                cols_show += ["prod_key", imp_col, clk_col, "CTR", "CVR", "GMV (ads)", "Desp (ads)", "Fat loja (mês atual)"]

                st.dataframe(out[cols_show], use_container_width=True, hide_index=True)
                st.caption("Ação: escalar/mover (bom desempenho com baixa entrega).")

# ============================
# 11) Visualização estruturada (Campanha → Total + Anúncios)
# ============================

st.divider()
st.header("Visualização Estruturada (Campanha → Total + Anúncios)")

campaigns = sorted(df_all["Campanha"].dropna().unique().tolist())
sel = st.selectbox("Filtrar campanha (detalhe)", ["(todas)"] + campaigns, index=0, key="camp_detail_filter")

view = df_all.copy()
if sel != "(todas)":
    view = view[view["Campanha"] == sel]

for camp, d in view.groupby("Campanha"):
    with st.expander(camp, expanded=(sel != "(todas)")):
        if id_col:
            id_clean = d[id_col].astype(str).str.strip()
            is_total = id_clean.isin(["", "-", "nan", "None"])
            df_total = d[is_total].copy()
            df_ads = d[~is_total].copy()
        else:
            df_total = pd.DataFrame()
            df_ads = d.copy()

        st.markdown("### Totais da campanha")
        if df_total.empty:
            st.info("Não encontrei a linha TOTAL (sem ID). Vou mostrar apenas os anúncios.")
        else:
            cols_total = [c for c in [
                "Campanha",
                name_col,
                imp_col,
                clk_col,
                "ctr_calc",
                orders_col,
                "cvr_calc",
                rev_col,
                cost_col,
                "acos_calc",
            ] if c and c in df_total.columns]
            disp_total = make_display_df(df_total[cols_total], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp_total, use_container_width=True, hide_index=True)

        st.markdown("### Anúncios (detalhado)")
        if df_ads.empty:
            st.warning("Não há anúncios com ID neste grupo.")
        else:
            cols_ads = [c for c in [
                "Campanha",
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
            ] if c and c in df_ads.columns]

            if cost_col and cost_col in df_ads.columns:
                df_ads = df_ads.sort_values(by=cost_col, ascending=False)

            disp_ads = make_display_df(df_ads[cols_ads], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp_ads, use_container_width=True, hide_index=True)

# ============================
# 12) Alertas Ads (produto/anúncio)
# ============================

st.divider()
st.header("Alertas (Ads – produto/anúncio)")

if id_col:
    only_ads = df_all[~df_all[id_col].astype(str).str.strip().isin(["", "-", "nan", "None"])].copy()
else:
    only_ads = df_all.copy()

tabs = st.tabs([
    "Gastando sem converter",
    "CTR ruim",
    "CVR ruim",
    "Bons com pouca impressão",
    "Mover anúncio (competição no grupo)",
])

with tabs[0]:
    if cost_col and orders_col and clk_col:
        wasting = only_ads[
            (pd.to_numeric(only_ads[orders_col], errors="coerce").fillna(0) == 0)
            & (
                (pd.to_numeric(only_ads[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
                | (pd.to_numeric(only_ads[cost_col], errors="coerce").fillna(0) >= float(min_spend_no_conv))
            )
        ].copy()
        if wasting.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            wasting["ação"] = "Pausar/remover (gastando sem converter)"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "acos_calc", "ação"] if c and c in wasting.columns]
            disp = make_display_df(wasting[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas suficientes (gasto, cliques e conversões/pedidos).")

with tabs[1]:
    if imp_col and clk_col:
        bad_ctr = only_ads[
            (pd.to_numeric(only_ads[imp_col], errors="coerce").fillna(0) > 0)
            & (pd.to_numeric(only_ads["ctr_calc"], errors="coerce").fillna(0) <= CTR_RUIM_MAX)
        ].copy()
        if bad_ctr.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            bad_ctr["ação"] = "Ajustar preço + cauda longa + imagem (CTR ruim)"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", "ctr_class", orders_col, "cvr_calc", rev_col, cost_col, "ação"] if c and c in bad_ctr.columns]
            disp = make_display_df(bad_ctr[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas de impressões/cliques.")

with tabs[2]:
    if clk_col and orders_col:
        bad_cvr = only_ads[
            (pd.to_numeric(only_ads[clk_col], errors="coerce").fillna(0) >= min_clicks_eval)
            & (pd.to_numeric(only_ads["cvr_calc"], errors="coerce").fillna(0) <= CVR_RUIM_MAX)
        ].copy()
        if bad_cvr.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            bad_cvr["ação"] = "Ajustar copy + gatilhos (CVR ruim)"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", "cvr_class", rev_col, cost_col, "ação"] if c and c in bad_cvr.columns]
            disp = make_display_df(bad_cvr[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei colunas de cliques e pedidos/conversões.")

with tabs[3]:
    if imp_col:
        good_low = only_ads[
            (pd.to_numeric(only_ads[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)
            & (
                (pd.to_numeric(only_ads["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                | (pd.to_numeric(only_ads["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
            )
        ].copy()
        if good_low.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            good_low["ação"] = "Escalar (bom com pouca impressão)"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "ação"] if c and c in good_low.columns]
            disp = make_display_df(good_low[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei coluna de impressões.")

with tabs[4]:
    if cost_col is None or id_col is None:
        st.info("Esse alerta exige colunas de custo e ID do produto.")
    else:
        df_g = only_ads.copy()
        g_cost = cost_col
        g_imp = imp_col
        g_clk = clk_col

        df_g["spend_share"] = df_g.groupby("Campanha")[g_cost].transform(lambda s: s / s.sum() if s.sum() else 0.0)

        prom = df_g.copy()
        if g_imp:
            prom = prom[(pd.to_numeric(prom[g_imp], errors="coerce").fillna(0) <= low_impressions_threshold)]
        prom = prom[
            (pd.to_numeric(prom["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
            | (pd.to_numeric(prom["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
        ]

        dom_camps = df_g[df_g["spend_share"] >= (dominance_spend_share / 100.0)][["Campanha"]].drop_duplicates()
        prom = prom.merge(dom_camps, on="Campanha", how="inner")

        if prom.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            prom["ação"] = "Mover pra outra campanha (competição interna)"
            prom["spend_share_%"] = prom["spend_share"] * 100

            cols = [c for c in [
                "Campanha", name_col, id_col,
                g_imp, g_clk, "ctr_calc",
                orders_col, "cvr_calc",
                rev_col, g_cost,
                "spend_share_%", "ação"
            ] if c and c in prom.columns]

            show = prom[cols].copy()
            show["spend_share_%"] = prom["spend_share_%"].apply(lambda v: fmt_pct(v / 100.0, digits=1))
            show = make_display_df(show, g_imp, g_clk, g_cost, orders_col, rev_col)
            st.dataframe(show, use_container_width=True, hide_index=True)
