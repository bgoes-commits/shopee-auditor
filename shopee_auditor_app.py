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
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        if s.count(".") >= 2:
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
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
    df["roas_calc"] = np.where((cost_col and df[cost_col] > 0), df[rev_col] / df[cost_col], np.nan) if (cost_col and rev_col) else np.nan

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


def fmt_pp(delta_fraction: float, digits: int = 2) -> str:
    if delta_fraction is None or (isinstance(delta_fraction, float) and np.isnan(delta_fraction)):
        return ""
    try:
        v = float(delta_fraction) * 100
    except Exception:
        return ""
    return f"{v:.{digits}f}".replace(".", ",") + " p.p."


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

    return out


# ============================
# 4) Leitura Vendas (Loja) - FIXO: aba "pedido pago" + coluna Vendas (BRL)
# ============================

def uploaded_bytes(uploaded_file) -> bytes:
    try:
        return uploaded_file.getvalue()
    except Exception:
        return uploaded_file.read()


def pick_paid_sheet(sheet_names: list[str]) -> str:
    # regra: sempre "pedido pago"
    # tenta por contains (sem acento / com acento)
    for s in sheet_names:
        if "pedido pago" in s.lower():
            return s
    # fallback: se não achar, usa a primeira
    return sheet_names[0] if sheet_names else ""


def read_sales_producttraffic_paid(uploaded_file) -> tuple[pd.DataFrame, str]:
    data = uploaded_bytes(uploaded_file)
    xls = pd.ExcelFile(BytesIO(data))
    sheet = pick_paid_sheet(xls.sheet_names)
    df = xls.parse(sheet, dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    return df, sheet


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

st.title("Shopee Ads – Campanhas + Loja (Pedidos pagos + TACOS + Insights)")

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
**Regras aplicadas**
- Vendas (loja): **sempre** usa a aba **(pedido pago)** e a coluna **Vendas (BRL)**.
- Ads: usa CSV(s) de **Dados do Grupo de Anúncios**.
- Linha **sem ID do produto** = **TOTAL da campanha**. Linhas com ID = **anúncios/produtos**.
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
# 7) Upload Vendas (Loja) - producttraffic (mês atual e anterior)
# ============================

st.divider()
st.header("Loja (Pedidos pagos) – mês anterior x mês analisado")

c1, c2 = st.columns(2)
with c1:
    sales_prev_file = st.file_uploader(
        "Vendas - mês ANTERIOR (Excel producttraffic)",
        type=["xlsx", "xls"],
        key="sales_prev",
    )
with c2:
    sales_curr_file = st.file_uploader(
        "Vendas - mês ANALISADO (Excel producttraffic)",
        type=["xlsx", "xls"],
        key="sales_curr",
    )

df_sales_prev = None
df_sales_curr = None
sheet_prev = ""
sheet_curr = ""

if sales_prev_file is not None:
    df_sales_prev, sheet_prev = read_sales_producttraffic_paid(sales_prev_file)

if sales_curr_file is not None:
    df_sales_curr, sheet_curr = read_sales_producttraffic_paid(sales_curr_file)

# ============================
# 8) KPI TOP: faturamento anterior | faturamento analisado | diferença
# ============================

faturamento_anterior = None
faturamento_atual = None

if df_sales_prev is not None:
    if "Vendas (BRL)" not in df_sales_prev.columns:
        st.error("Na planilha do mês anterior não encontrei a coluna EXATA: 'Vendas (BRL)'.")
    else:
        faturamento_anterior = float(np.nansum(numeric_series_from(df_sales_prev, "Vendas (BRL)")))

if df_sales_curr is not None:
    if "Vendas (BRL)" not in df_sales_curr.columns:
        st.error("Na planilha do mês analisado não encontrei a coluna EXATA: 'Vendas (BRL)'.")
    else:
        faturamento_atual = float(np.nansum(numeric_series_from(df_sales_curr, "Vendas (BRL)")))

kA, kB, kC = st.columns(3)
kA.metric("Faturamento (mês anterior) – pedidos pagos", fmt_brl(faturamento_anterior) if faturamento_anterior is not None else "—")
kB.metric("Faturamento (mês analisado) – pedidos pagos", fmt_brl(faturamento_atual) if faturamento_atual is not None else "—")

diff_txt = "—"
if faturamento_anterior is not None and faturamento_anterior > 0 and faturamento_atual is not None:
    delta = (faturamento_atual / faturamento_anterior) - 1
    diff_txt = f"{fmt_brl(faturamento_atual - faturamento_anterior)}  ({fmt_pct(delta, digits=1)})"
kC.metric("Diferença (R$ e %)", diff_txt)

if sheet_prev or sheet_curr:
    st.caption(
        f"Aba fixa usada (pedido pago) — anterior: **{sheet_prev or '—'}** | analisado: **{sheet_curr or '—'}**"
    )

# ============================
# 9) KPI ADS (embaixo): gasto | gmv | ACOS | ROAS | TACOS
# ============================

ads_spend_total = np.nan
ads_gmv_total = np.nan

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

if rev_col and rev_col in df_all.columns:
    if id_col and id_col in df_all.columns:
        id_clean = df_all[id_col].astype(str).str.strip()
        is_total = id_clean.isin(["", "-", "nan", "None"])
        if is_total.any():
            ads_gmv_total = float(np.nansum(pd.to_numeric(df_all.loc[is_total, rev_col], errors="coerce")))
        else:
            ads_gmv_total = float(np.nansum(pd.to_numeric(df_all[rev_col], errors="coerce")))
    else:
        ads_gmv_total = float(np.nansum(pd.to_numeric(df_all[rev_col], errors="coerce")))

acos_total = (ads_spend_total / ads_gmv_total) if (pd.notna(ads_spend_total) and pd.notna(ads_gmv_total) and ads_gmv_total > 0) else np.nan
roas_total = (ads_gmv_total / ads_spend_total) if (pd.notna(ads_spend_total) and ads_spend_total > 0 and pd.notna(ads_gmv_total)) else np.nan
tacos_total = (ads_spend_total / faturamento_atual) if (pd.notna(ads_spend_total) and faturamento_atual is not None and faturamento_atual > 0) else np.nan

st.subheader("Ads (mês analisado) – KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Despesas (Ads)", fmt_brl(ads_spend_total) if pd.notna(ads_spend_total) else "—")
k2.metric("GMV (Ads)", fmt_brl(ads_gmv_total) if pd.notna(ads_gmv_total) else "—")
k3.metric("ACOS", fmt_pct(acos_total) if pd.notna(acos_total) else "—")
k4.metric("ROAS", (str(round(roas_total, 2)).replace(".", ",")) if pd.notna(roas_total) else "—")
k5.metric("TACOS", fmt_pct(tacos_total) if pd.notna(tacos_total) else "—")

# ============================
# 10) Visão geral de campanhas (TOTAL sem ID) + ACOS + ROAS + TACOS
# ============================

st.divider()
st.header("Visão Geral de Campanhas (totais)")

def acos_semaforo(x, target, warn):
    if pd.isna(x):
        return "n/a"
    if x <= target:
        return "🟢 ok"
    if x <= target + warn:
        return "🟡 atenção"
    return "🔴 crítico"

def tacos_semaforo(x, target, warn):
    if pd.isna(x):
        return "n/a"
    if x <= target:
        return "🟢 ok"
    if x <= target + warn:
        return "🟡 atenção"
    return "🔴 crítico"

target_acos = acos_target_pct / 100.0
warn_acos = acos_warn_pp / 100.0
target_tacos = tacos_target_pct / 100.0
warn_tacos = tacos_warn_pp / 100.0

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
        grp["ROAS"] = np.where(grp["Despesas"].fillna(0) > 0, grp["GMV"].fillna(0) / grp["Despesas"].fillna(0), np.nan)

        grp["ACOS_sinal"] = grp["ACOS"].apply(lambda v: acos_semaforo(v, target_acos, warn_acos))
        grp["TACOS"] = (grp["Despesas"] / faturamento_atual) if (faturamento_atual is not None and faturamento_atual > 0) else np.nan
        grp["TACOS_sinal"] = grp["TACOS"].apply(lambda v: tacos_semaforo(v, target_tacos, warn_tacos))

        grp["CTR_status"] = grp["CTR"].apply(classify_ctr)
        grp["CVR_status"] = grp["CVR"].apply(classify_cvr)

        camp_opts = ["(todas)"] + sorted(grp["Campanha"].dropna().unique().tolist())
        sel_camp = st.selectbox("Filtrar campanha", camp_opts, index=0, key="camp_overview_filter")

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
        disp["ROAS"] = disp["ROAS"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        cols = [
            "Campanha",
            "GMV",
            "Despesas",
            "ACOS",
            "ACOS_sinal",
            "ROAS",
            "TACOS",
            "TACOS_sinal",
            "Impressões",
            "Cliques",
            "CTR",
            "CTR_status",
            "Pedidos",
            "CVR",
            "CVR_status",
        ]
        st.dataframe(disp[cols], use_container_width=True, hide_index=True)

# ============================
# 11) Insights (Loja + Ads) com números (validar)
# ============================

st.divider()
st.header("Insights (Loja + Ads) — com números para validar")

if df_sales_curr is None:
    st.info("Suba o Excel producttraffic do mês analisado para liberar os insights.")
else:
    # colunas fixas do producttraffic
    # (se alguma conta mudar o nome, deixei contains como fallback)
    sales_id_col = find_col(
        df_sales_curr,
        exact=["ID do Item"],
        contains=["id do item"],
    )
    prod_name_col = find_col(
        df_sales_curr,
        exact=["Produto"],
        contains=["produto"],
    )
    imp_store_col = find_col(
        df_sales_curr,
        exact=["Impressões de Produto"],
        contains=["impress"],
    )
    clk_store_col = find_col(
        df_sales_curr,
        exact=["Cliques Por Produto"],
        contains=["clique"],
    )
    ctr_store_col = find_col(
        df_sales_curr,
        exact=["CTR"],
        contains=["ctr"],
    )
    cvr_store_col = find_col(
        df_sales_curr,
        exact=["Taxa de Conversão de Pedidos"],
        contains=["taxa de convers", "convers"],
    )
    ord_store_col = find_col(
        df_sales_curr,
        exact=["Pedidos"],
        contains=["pedido"],
    )

    # monta base loja (mês analisado)
    s = df_sales_curr.copy()
    s["prod_key"] = s[sales_id_col].astype(str).str.strip().apply(normalize_product_id)
    s["Produto_nome"] = s[prod_name_col].astype(str).str.strip() if prod_name_col else ""
    s["rev_curr"] = numeric_series_from(s, "Vendas (BRL)")
    s["imp_curr"] = numeric_series_from(s, imp_store_col) if imp_store_col else np.nan
    s["clk_curr"] = numeric_series_from(s, clk_store_col) if clk_store_col else np.nan
    s["ctr_curr"] = s[ctr_store_col].apply(parse_percent) if ctr_store_col else np.where(s["imp_curr"].fillna(0) > 0, s["clk_curr"].fillna(0)/s["imp_curr"].fillna(0), np.nan)
    s["cvr_curr"] = s[cvr_store_col].apply(parse_percent) if cvr_store_col else np.nan
    s["ord_curr"] = numeric_series_from(s, ord_store_col) if ord_store_col else np.nan

    sales_curr_agg = s.groupby(["prod_key"], dropna=False).agg({
        "Produto_nome": "first",
        "rev_curr": "sum",
        "imp_curr": "sum",
        "clk_curr": "sum",
        "ctr_curr": "mean",
        "cvr_curr": "mean",
        "ord_curr": "sum",
    }).reset_index()

    # base loja mês anterior (se existir)
    if df_sales_prev is not None:
        sp = df_sales_prev.copy()
        sales_id_prev = find_col(sp, exact=["ID do Item"], contains=["id do item"])
        prod_name_prev = find_col(sp, exact=["Produto"], contains=["produto"])
        imp_prev_col = find_col(sp, exact=["Impressões de Produto"], contains=["impress"])
        clk_prev_col = find_col(sp, exact=["Cliques Por Produto"], contains=["clique"])
        ctr_prev_col = find_col(sp, exact=["CTR"], contains=["ctr"])
        cvr_prev_col = find_col(sp, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
        ord_prev_col = find_col(sp, exact=["Pedidos"], contains=["pedido"])

        sp["prod_key"] = sp[sales_id_prev].astype(str).str.strip().apply(normalize_product_id)
        sp["Produto_nome"] = sp[prod_name_prev].astype(str).str.strip() if prod_name_prev else ""
        sp["rev_prev"] = numeric_series_from(sp, "Vendas (BRL)")
        sp["imp_prev"] = numeric_series_from(sp, imp_prev_col) if imp_prev_col else np.nan
        sp["clk_prev"] = numeric_series_from(sp, clk_prev_col) if clk_prev_col else np.nan
        sp["ctr_prev"] = sp[ctr_prev_col].apply(parse_percent) if ctr_prev_col else np.where(sp["imp_prev"].fillna(0) > 0, sp["clk_prev"].fillna(0)/sp["imp_prev"].fillna(0), np.nan)
        sp["cvr_prev"] = sp[cvr_prev_col].apply(parse_percent) if cvr_prev_col else np.nan
        sp["ord_prev"] = numeric_series_from(sp, ord_prev_col) if ord_prev_col else np.nan

        sales_prev_agg = sp.groupby(["prod_key"], dropna=False).agg({
            "rev_prev": "sum",
            "imp_prev": "sum",
            "clk_prev": "sum",
            "ctr_prev": "mean",
            "cvr_prev": "mean",
            "ord_prev": "sum",
        }).reset_index()

        sales_mom = sales_curr_agg.merge(sales_prev_agg, on="prod_key", how="left")
        for c in ["rev_prev","imp_prev","clk_prev","ctr_prev","cvr_prev","ord_prev"]:
            sales_mom[c] = pd.to_numeric(sales_mom[c], errors="coerce").fillna(0.0)

        sales_mom["rev_delta_pct"] = np.where(sales_mom["rev_prev"] > 0, (sales_mom["rev_curr"]/sales_mom["rev_prev"]) - 1, np.nan)
        sales_mom["imp_delta_pct"] = np.where(sales_mom["imp_prev"] > 0, (sales_mom["imp_curr"]/sales_mom["imp_prev"]) - 1, np.nan)
        sales_mom["clk_delta_pct"] = np.where(sales_mom["clk_prev"] > 0, (sales_mom["clk_curr"]/sales_mom["clk_prev"]) - 1, np.nan)
        sales_mom["ctr_delta_pp"] = (sales_mom["ctr_curr"] - sales_mom["ctr_prev"])
        sales_mom["cvr_delta_pp"] = (sales_mom["cvr_curr"] - sales_mom["cvr_prev"])
    else:
        sales_mom = sales_curr_agg.copy()
        sales_mom["rev_prev"] = np.nan
        sales_mom["imp_prev"] = np.nan
        sales_mom["clk_prev"] = np.nan
        sales_mom["ctr_prev"] = np.nan
        sales_mom["cvr_prev"] = np.nan
        sales_mom["ord_prev"] = np.nan
        sales_mom["rev_delta_pct"] = np.nan
        sales_mom["imp_delta_pct"] = np.nan
        sales_mom["clk_delta_pct"] = np.nan
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
        "Candidatos a Ads (fora do Ads) — com motivos + números",
        "Queda mês x mês (loja) — com números + diagnóstico (lei)",
        "Oportunidades (Ads) — bom CTR/CVR + pouca entrega",
    ])

    # 1) candidatos a ADS
    with tabs_i[0]:
        cand = sales_curr_agg.copy()
        cand = cand[~cand["prod_key"].astype(str).isin(ads_keys)]
        cand = cand[cand["rev_curr"] >= float(min_revenue_candidate)]

        if cand.empty:
            st.info("Nenhum candidato no critério (fora do Ads + faturamento mínimo).")
        else:
            cand = cand.sort_values("rev_curr", ascending=False)

            def motivo_candidato(row):
                ctr = row.get("ctr_curr")
                cvr = row.get("cvr_curr")
                imp = row.get("imp_curr", np.nan)
                clk = row.get("clk_curr", np.nan)

                parts = ["Vende bem sem Ads"]
                if pd.notna(ctr):
                    parts.append(f"CTR {classify_ctr(ctr)}")
                if pd.notna(cvr):
                    parts.append(f"CVR {classify_cvr(cvr)}")
                if pd.notna(imp) and imp < low_impressions_threshold:
                    parts.append("baixa impressão (oportunidade de escala)")
                if pd.notna(clk) and clk < 10:
                    parts.append("poucos cliques (avaliar oferta/imagem)")
                return " | ".join(parts)

            out = cand.copy()
            out["Motivo"] = out.apply(motivo_candidato, axis=1)

            out_disp = out.copy()
            out_disp["Faturamento (mês)"] = out_disp["rev_curr"].apply(fmt_brl)
            out_disp["Impressões"] = out_disp["imp_curr"].apply(fmt_int)
            out_disp["Cliques"] = out_disp["clk_curr"].apply(fmt_int)
            out_disp["CTR"] = out_disp["ctr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
            out_disp["CVR"] = out_disp["cvr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
            out_disp["Pedidos"] = out_disp["ord_curr"].apply(fmt_int)

            cols = ["prod_key", "Produto_nome", "Faturamento (mês)", "Pedidos", "Impressões", "Cliques", "CTR", "CVR", "Motivo"]
            st.dataframe(out_disp[cols], use_container_width=True, hide_index=True)

    # 2) queda mes x mes com numeros
    with tabs_i[1]:
        if df_sales_prev is None:
            st.info("Suba o mês anterior para liberar queda mês x mês.")
        else:
            drops = sales_mom.copy()
            drops = drops[pd.notna(drops["rev_delta_pct"])]
            drops = drops[drops["rev_delta_pct"] <= -(float(drop_alert_pct) / 100.0)]

            if drops.empty:
                st.info("Nenhum produto com queda acima do limite.")
            else:
                def diagnose_lei(row):
                    # prioridade "lei"
                    if pd.notna(row.get("imp_delta_pct")) and row["imp_delta_pct"] < 0:
                        return "Impressões caíram → colocar/fortalecer ADS"
                    if pd.notna(row.get("ctr_delta_pp")) and row["ctr_delta_pp"] < 0:
                        return "CTR caiu → ajustar preço + cauda longa + imagem"
                    if pd.notna(row.get("cvr_delta_pp")) and row["cvr_delta_pp"] < 0:
                        return "CVR caiu → ajustar copy + gatilhos de conversão"
                    if pd.notna(row.get("clk_delta_pct")) and row["clk_delta_pct"] < 0:
                        return "Cliques caíram → revisar oferta/imagem/palavras"
                    return "Queda em vendas → checar estoque/preço concorrência/página"

                drops["Ação sugerida"] = drops.apply(diagnose_lei, axis=1)

                out = drops.copy()
                out_disp = out.copy()

                out_disp["Fat atual"] = out_disp["rev_curr"].apply(fmt_brl)
                out_disp["Fat anterior"] = out_disp["rev_prev"].apply(fmt_brl)
                out_disp["Δ Fat"] = out_disp["rev_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                out_disp["Imp atual"] = out_disp["imp_curr"].apply(fmt_int)
                out_disp["Imp anterior"] = out_disp["imp_prev"].apply(fmt_int)
                out_disp["Δ Imp"] = out_disp["imp_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                out_disp["Clk atual"] = out_disp["clk_curr"].apply(fmt_int)
                out_disp["Clk anterior"] = out_disp["clk_prev"].apply(fmt_int)
                out_disp["Δ Clk"] = out_disp["clk_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                out_disp["CTR atual"] = out_disp["ctr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out_disp["CTR anterior"] = out_disp["ctr_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out_disp["Δ CTR"] = out_disp["ctr_delta_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

                out_disp["CVR atual"] = out_disp["cvr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out_disp["CVR anterior"] = out_disp["cvr_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out_disp["Δ CVR"] = out_disp["cvr_delta_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

                out_disp["Pedidos atual"] = out_disp["ord_curr"].apply(fmt_int)
                out_disp["Pedidos anterior"] = out_disp["ord_prev"].apply(fmt_int)

                cols = [
                    "prod_key", "Produto_nome",
                    "Fat atual","Fat anterior","Δ Fat",
                    "Pedidos atual","Pedidos anterior",
                    "Imp atual","Imp anterior","Δ Imp",
                    "Clk atual","Clk anterior","Δ Clk",
                    "CTR atual","CTR anterior","Δ CTR",
                    "CVR atual","CVR anterior","Δ CVR",
                    "Ação sugerida"
                ]
                st.dataframe(out_disp[cols], use_container_width=True, hide_index=True)

    # 3) oportunidades no ads (bom CTR/CVR e pouca entrega)
    with tabs_i[2]:
        if ads_prod.empty or imp_col is None:
            st.info("Não tenho base de Ads por produto (ou não achei coluna de Impressões no Ads).")
        else:
            opp = ads_prod.copy()

            # pouco delivery + bom desempenho
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
                out["Impressões"] = out[imp_col].apply(fmt_int) if imp_col in out.columns else ""
                out["Cliques"] = out[clk_col].apply(fmt_int) if clk_col in out.columns else ""
                out["ACOS"] = out["acos_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out["Ação sugerida"] = "Criar/mover para campanha específica (orçamento) ou separar do grupo dominante"

                cols = ["Campanha"]
                if name_col in out.columns:
                    cols += [name_col]
                cols += ["prod_key", "Impressões", "Cliques", "CTR", "CVR", "GMV (ads)", "Desp (ads)", "ACOS", "Ação sugerida"]
                st.dataframe(out[cols], use_container_width=True, hide_index=True)

# ============================
# 12) Visualização estruturada (Campanha → Total + Anúncios) + INSIGHTS no detalhado
# ============================

st.divider()
st.header("Campanha → Totais + Anúncios (detalhado + recomendações)")

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
                imp_col,
                clk_col,
                "ctr_calc",
                orders_col,
                "cvr_calc",
                rev_col,
                cost_col,
                "acos_calc",
                "roas_calc",
            ] if c and c in df_total.columns]
            disp_total = make_display_df(df_total[cols_total], imp_col, clk_col, cost_col, orders_col, rev_col)
            if "roas_calc" in df_total.columns:
                disp_total["roas_calc"] = df_total["roas_calc"].apply(lambda v: "" if pd.isna(v) else str(round(float(v), 2)).replace(".", ","))
            st.dataframe(disp_total, use_container_width=True, hide_index=True)

        st.markdown("### Anúncios (detalhado)")
        if df_ads.empty:
            st.warning("Não há anúncios com ID neste grupo.")
        else:
            cols_ads = [c for c in [
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
                "roas_calc",
                "cpc",
                "cpa",
            ] if c and c in df_ads.columns]

            df_ads_sorted = df_ads.copy()
            if cost_col and cost_col in df_ads_sorted.columns:
                df_ads_sorted = df_ads_sorted.sort_values(by=cost_col, ascending=False)

            disp_ads = make_display_df(df_ads_sorted[cols_ads], imp_col, clk_col, cost_col, orders_col, rev_col)
            if "roas_calc" in df_ads_sorted.columns:
                disp_ads["roas_calc"] = df_ads_sorted["roas_calc"].apply(lambda v: "" if pd.isna(v) else str(round(float(v), 2)).replace(".", ","))
            st.dataframe(disp_ads, use_container_width=True, hide_index=True)

            st.markdown("### Recomendações rápidas (com base em CTR/CVR/Impressões)")
            t1, t2 = st.columns(2)

            # A) clique bom (CTR boa/ótima) mas conversão baixa -> ajustar copy/gatilhos (e/ou preço)
            with t1:
                st.markdown("**CTR bom, CVR ruim (ajustar copy/gatilhos/preço)**")
                if "ctr_calc" in df_ads.columns and "cvr_calc" in df_ads.columns and imp_col in df_ads.columns:
                    rec = df_ads.copy()
                    rec = rec[
                        (pd.to_numeric(rec[imp_col], errors="coerce").fillna(0) > 0)
                        & (pd.to_numeric(rec["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                        & (pd.to_numeric(rec["cvr_calc"], errors="coerce").fillna(0) <= CVR_RUIM_MAX)
                    ].copy()
                    if rec.empty:
                        st.caption("Nenhum anúncio no critério.")
                    else:
                        rec["ação"] = "CTR bom + CVR ruim → ajustar copy + gatilhos (e revisar preço/landing)"
                        show = [c for c in [name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "acos_calc", "ação"] if c and c in rec.columns]
                        disp = make_display_df(rec[show], imp_col, clk_col, cost_col, orders_col, rev_col)
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.caption("Não encontrei colunas suficientes para essa recomendação.")

            # B) CTR bom + CVR bom, mas pouca impressão -> separar campanha (orçamento)
            with t2:
                st.markdown("**CTR bom + CVR bom, mas pouca impressão (separar em campanha específica)**")
                if "ctr_calc" in df_ads.columns and "cvr_calc" in df_ads.columns and imp_col in df_ads.columns:
                    rec2 = df_ads.copy()
                    rec2 = rec2[
                        (pd.to_numeric(rec2[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)
                        & (pd.to_numeric(rec2["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
                        & (pd.to_numeric(rec2["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
                    ].copy()
                    if rec2.empty:
                        st.caption("Nenhum anúncio no critério.")
                    else:
                        rec2["ação"] = "Bom CTR/CVR + pouca entrega → mover/criar campanha dedicada (orçamento)"
                        show = [c for c in [name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "acos_calc", "ação"] if c and c in rec2.columns]
                        disp = make_display_df(rec2[show], imp_col, clk_col, cost_col, orders_col, rev_col)
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                else:
                    st.caption("Não encontrei colunas suficientes para essa recomendação.")

# ============================
# 13) Alertas Ads (produto/anúncio) + mover por dominância
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
            bad_ctr["ação"] = "CTR ruim → ajustar preço + cauda longa + imagem"
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
            bad_cvr["ação"] = "CVR ruim → ajustar copy + gatilhos de conversão"
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
            good_low["ação"] = "Bom com pouca impressão → escalar / ajustar estrutura"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "acos_calc", "ação"] if c and c in good_low.columns]
            disp = make_display_df(good_low[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei coluna de impressões.")

with tabs[4]:
    if cost_col is None or id_col is None:
        st.info("Esse alerta exige colunas de custo e ID do produto.")
    else:
        df_g = only_ads.copy()
        df_g["spend_share"] = df_g.groupby("Campanha")[cost_col].transform(lambda s: s / s.sum() if s.sum() else 0.0)

        prom = df_g.copy()
        if imp_col:
            prom = prom[(pd.to_numeric(prom[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold)]
        prom = prom[
            (pd.to_numeric(prom["ctr_calc"], errors="coerce").fillna(0) >= CTR_BOA_MIN)
            | (pd.to_numeric(prom["cvr_calc"], errors="coerce").fillna(0) >= CVR_BOA_MIN)
        ]

        dom_camps = df_g[df_g["spend_share"] >= (dominance_spend_share / 100.0)][["Campanha"]].drop_duplicates()
        prom = prom.merge(dom_camps, on="Campanha", how="inner")

        if prom.empty:
            st.info("Nenhum anúncio no critério.")
        else:
            prom["ação"] = "Mover para outra campanha/grupo (competição interna no orçamento)"
            prom["spend_share_%"] = prom["spend_share"] * 100

            cols = [c for c in [
                "Campanha", name_col, id_col,
                imp_col, clk_col, "ctr_calc",
                orders_col, "cvr_calc",
                rev_col, cost_col,
                "acos_calc",
                "spend_share_%", "ação"
            ] if c and c in prom.columns]

            show = prom[cols].copy()
            show["spend_share_%"] = prom["spend_share_%"].apply(lambda v: fmt_pct(v / 100.0, digits=1))
            disp = make_display_df(show, imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
