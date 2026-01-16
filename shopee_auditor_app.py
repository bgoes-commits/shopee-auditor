import re
from io import StringIO, BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Ads Auditor (Campanha + Loja)", layout="wide")


# ============================
# 1) Parsing / utils numéricos
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


# ============================
# 2) Formatação BR (display)
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


# ============================
# 3) CSV Shopee Ads (Grupo)
# ============================
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

    df["acos_calc"] = np.where((cost_col and rev_col and df[rev_col] > 0), df[cost_col] / df[rev_col], np.nan)
    df["roas_calc"] = np.where((cost_col and df[cost_col] > 0), df[rev_col] / df[cost_col], np.nan) if (cost_col and rev_col) else np.nan

    return df


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
# 4) Vendas (Loja) - producttraffic (pedido pago fixo)
# ============================
def uploaded_bytes(uploaded_file) -> bytes:
    try:
        return uploaded_file.getvalue()
    except Exception:
        return uploaded_file.read()


def pick_paid_sheet(sheet_names: list[str]) -> str:
    for s in sheet_names:
        if "pedido pago" in s.lower():
            return s
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
# 5) Excel export
# ============================
def make_excel_export(tables: dict[str, pd.DataFrame]) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, tdf in tables.items():
            if tdf is None:
                continue
            dfw = tdf.copy()
            safe_name = (sheet[:31]).strip() or "Sheet"
            dfw.to_excel(writer, index=False, sheet_name=safe_name)
    bio.seek(0)
    return bio.getvalue()


# ============================
# 6) UI
# ============================
st.title("Shopee Ads – Campanhas + Loja (Pedidos pagos + TACOS + Insights)")


with st.sidebar:
    st.header("Configuração das regras (geral)")

    st.subheader("CVR (taxa de conversão)")
    cvr_ruim_max_pct = st.number_input("CVR ruim (até %) → 🔴", value=1.5, step=0.1)
    cvr_bom_min_pct = st.number_input("CVR bom (a partir %) → 🟡", value=1.5, step=0.1)
    cvr_otimo_min_pct = st.number_input("CVR ótimo (a partir %) → 🟢", value=3.0, step=0.1)

    st.subheader("CTR (cliques / impressões)")
    ctr_ruim_max_pct = st.number_input("CTR ruim (até %) → 🔴", value=2.0, step=0.1)
    ctr_bom_min_pct = st.number_input("CTR bom (a partir %) → 🟡", value=3.0, step=0.1)
    ctr_otimo_min_pct = st.number_input("CTR ótimo (a partir %) → 🟢", value=4.0, step=0.1)

    st.subheader("ACOS / TACOS (ideal e limite)")
    acos_ideal_pct = st.number_input("ACOS ideal (%)", value=10.0, step=0.5)
    acos_limite_pct = st.number_input("ACOS limite (%)", value=12.0, step=0.5)
    tacos_ideal_pct = st.number_input("TACOS ideal (%)", value=8.0, step=0.5)
    tacos_limite_pct = st.number_input("TACOS limite (%)", value=10.0, step=0.5)

    st.divider()
    st.subheader("Outros filtros (Ads)")
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões baixas (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

    st.divider()
    st.subheader("Insights (Loja)")
    min_revenue_candidate = st.number_input("Faturamento mínimo p/ sugerir Ads (R$)", value=1000.0, step=100.0)
    drop_alert_pct = st.number_input("Queda de faturamento p/ alerta (%)", value=20.0, step=5.0)


# ============================
# 7) Classificações configuráveis (CTR/CVR)
# ============================
def classify_ctr(ctr: float) -> str:
    if pd.isna(ctr):
        return "n/a"

    red_max = ctr_ruim_max_pct / 100.0
    green_min = ctr_otimo_min_pct / 100.0
    yellow_min = ctr_bom_min_pct / 100.0

    if ctr <= red_max:
        return "vermelho"
    if ctr >= green_min:
        return "verde"
    if ctr >= yellow_min:
        return "amarelo"
    return "amarelo"


def classify_cvr(cvr: float) -> str:
    if pd.isna(cvr):
        return "n/a"

    red_max = cvr_ruim_max_pct / 100.0
    green_min = cvr_otimo_min_pct / 100.0
    yellow_min = cvr_bom_min_pct / 100.0

    if cvr < red_max:
        return "vermelho"
    if cvr >= green_min:
        return "verde"
    if cvr >= yellow_min:
        return "amarelo"
    return "vermelho"


def dot_from_class(cls: str) -> str:
    if cls == "verde":
        return "🟢"
    if cls == "amarelo":
        return "🟡"
    if cls == "vermelho":
        return "🔴"
    return "⚪"


def worst_class(a: str, b: str) -> str:
    order = {"verde": 0, "amarelo": 1, "vermelho": 2, "n/a": -1}
    return a if order.get(a, -1) >= order.get(b, -1) else b


def semaforo_ideal_limite(x: float, ideal: float, limite: float) -> tuple[str, str]:
    if pd.isna(x):
        return "⚪", "n/a"
    if x <= ideal:
        return "🟢", "ótimo"
    if x <= limite:
        return "🟡", "monitorar"
    return "🔴", "ajustar"


def action_for_row_new(ctr: float, cvr: float, imps: float, *, low_imp_threshold: int) -> tuple[str, str]:
    ctr_cls = classify_ctr(ctr) if pd.notna(ctr) else "n/a"
    cvr_cls = classify_cvr(cvr) if pd.notna(cvr) else "n/a"
    worst = worst_class(ctr_cls, cvr_cls)
    bolinha = dot_from_class(worst)

    reasons = []

    if ctr_cls == "vermelho":
        reasons.append("CTR 🔴 → ajustar preço + cauda longa + imagem (urgente)")
    elif ctr_cls == "amarelo":
        reasons.append("CTR 🟡 → otimizar preço + cauda longa + imagem")

    if cvr_cls == "vermelho":
        reasons.append("CVR 🔴 → ajustar copy + gatilhos + oferta/landing (urgente)")
    elif cvr_cls == "amarelo":
        reasons.append("CVR 🟡 → melhorar copy + gatilhos + oferta/landing")

    if pd.notna(imps) and imps <= low_imp_threshold:
        if ctr_cls == "verde" and cvr_cls in {"verde", "amarelo"}:
            reasons.append("Pouca impressão → criar/mover para campanha dedicada (orçamento/entrega)")
        else:
            reasons.append("Pouca impressão → pode estar travado por orçamento do grupo (considerar mover/separar)")

    if not reasons:
        reasons.append("Ok → manter e monitorar")

    return bolinha, " | ".join(reasons)


def action_for_store_row(ctr: float, cvr: float, imps: float, *,
                         low_imp_threshold: int,
                         not_in_ads: bool = False,
                         revenue: float | None = None) -> tuple[str, str]:
    bolinha, base = action_for_row_new(ctr, cvr, imps, low_imp_threshold=low_imp_threshold)
    extra = []
    if not_in_ads:
        extra.append("Fora do Ads → colocar Ads")
    if revenue is not None and pd.notna(revenue) and revenue > 0:
        extra.append("Já fatura → validar escala")
    if extra:
        return bolinha, base + " | " + " | ".join(extra)
    return bolinha, base


st.markdown(
    """
**Regras aplicadas**
- Vendas (loja): usa **(pedido pago)** e a coluna **Vendas (BRL)**.
- Ads: usa CSV(s) de **Dados do Grupo de Anúncios**.
- Linha **sem ID do produto** = **TOTAL da campanha**. Linhas com ID = **anúncios/produtos**.
- **TACOS** aparece **só** no **KPI geral do Ads** e no **nível de campanha**.
"""
)

# ============================
# 8) Upload Ads (Grupos)
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
# 9) Upload Vendas (Loja) - producttraffic (mês atual e anterior)
# ============================
st.divider()
st.header("Loja (Pedidos pagos) – mês anterior x mês analisado")

c1, c2 = st.columns(2)
with c1:
    sales_prev_file = st.file_uploader("Vendas - mês ANTERIOR (Excel producttraffic)", type=["xlsx", "xls"], key="sales_prev")
with c2:
    sales_curr_file = st.file_uploader("Vendas - mês ANALISADO (Excel producttraffic)", type=["xlsx", "xls"], key="sales_curr")

df_sales_prev = None
df_sales_curr = None
sheet_prev = ""
sheet_curr = ""

if sales_prev_file is not None:
    df_sales_prev, sheet_prev = read_sales_producttraffic_paid(sales_prev_file)

if sales_curr_file is not None:
    df_sales_curr, sheet_curr = read_sales_producttraffic_paid(sales_curr_file)

# ============================
# 10) KPIs TOP: faturamento anterior | faturamento analisado | diferença
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
    st.caption(f"Aba fixa usada (pedido pago) — anterior: **{sheet_prev or '—'}** | analisado: **{sheet_curr or '—'}**")


# ============================
# 11) KPI ADS (embaixo): gasto | gmv | ACOS | ROAS | TACOS
# ============================
def compute_total_from_campaign_rows(df: pd.DataFrame, col: str, id_col: str | None) -> float:
    if col is None or col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors="coerce")

    if id_col and id_col in df.columns:
        id_clean = df[id_col].astype(str).str.strip()
        is_total = id_clean.isin(["", "-", "nan", "None"])
        if is_total.any():
            return float(np.nansum(s[is_total]))
    return float(np.nansum(s))


ads_spend_total = compute_total_from_campaign_rows(df_all, cost_col, id_col)
ads_gmv_total = compute_total_from_campaign_rows(df_all, rev_col, id_col)

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
# 12) Visão geral de campanhas (TOTAL sem ID)
#     - Mostra: CVR, ACOS, TACOS + decisão combinada
# ============================
st.divider()
st.header("Visão Geral de Campanhas (decisão por CVR + ACOS + TACOS)")

camp_overview_df = None

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

        grp["CVR"] = np.where(grp["Cliques"].fillna(0) > 0, grp["Pedidos"].fillna(0) / grp["Cliques"].fillna(0), np.nan)
        grp["ACOS"] = np.where(grp["GMV"].fillna(0) > 0, grp["Despesas"].fillna(0) / grp["GMV"].fillna(0), np.nan)
        grp["ROAS"] = np.where(grp["Despesas"].fillna(0) > 0, grp["GMV"].fillna(0) / grp["Despesas"].fillna(0), np.nan)
        grp["TACOS"] = (grp["Despesas"] / faturamento_atual) if (faturamento_atual is not None and faturamento_atual > 0) else np.nan

        grp["CVR_cor"] = grp["CVR"].apply(classify_cvr)

        # Semáforo ACOS / TACOS (ideal / limite)
        acos_ideal = acos_ideal_pct / 100.0
        acos_limite = acos_limite_pct / 100.0
        tacos_ideal = tacos_ideal_pct / 100.0
        tacos_limite = tacos_limite_pct / 100.0

        grp[["ACOS_bolinha", "ACOS_status"]] = grp["ACOS"].apply(
            lambda v: pd.Series(semaforo_ideal_limite(v, acos_ideal, acos_limite))
        )
        grp[["TACOS_bolinha", "TACOS_status"]] = grp["TACOS"].apply(
            lambda v: pd.Series(semaforo_ideal_limite(v, tacos_ideal, tacos_limite))
        )

        def acao_cvr(cls: str) -> str:
            if cls == "verde":
                return "CVR ótimo → aumentar orçamento"
            if cls == "amarelo":
                return "CVR bom → manter"
            if cls == "vermelho":
                return "CVR ruim → ajustar"
            return "Monitorar"

        grp["Ação (CVR)"] = grp["CVR_cor"].apply(acao_cvr)

        def decisao_final(row) -> tuple[str, str]:
            cvr_cls = row["CVR_cor"]
            acos_st = row["ACOS_status"]      # ótimo / monitorar / ajustar
            tacos_st = row["TACOS_status"]    # ótimo / monitorar / ajustar

            # prioridade: ACOS/TACOS acima do limite -> ajustar
            if acos_st == "ajustar" or tacos_st == "ajustar":
                return "🔴", "AJUSTAR (ACOS/TACOS acima do limite)"

            # CVR ruim -> ajustar
            if cvr_cls == "vermelho":
                return "🔴", "AJUSTAR (CVR ruim)"

            # aumentar orçamento: CVR verde + ACOS ótimo + TACOS ótimo
            if cvr_cls == "verde" and acos_st == "ótimo" and tacos_st == "ótimo":
                return "🟢", "AUMENTAR ORÇAMENTO (CVR ótimo + ACOS/TACOS ótimos)"

            # CVR verde mas ACOS/TACOS monitorar
            if cvr_cls == "verde" and (acos_st == "monitorar" or tacos_st == "monitorar"):
                return "🟡", "MONITORAR (CVR ótimo, mas ACOS/TACOS pedem atenção)"

            # CVR amarelo: manter/monitorar
            if cvr_cls == "amarelo" and acos_st == "ótimo" and tacos_st == "ótimo":
                return "🟡", "MANTER (CVR bom)"

            return "🟡", "MONITORAR"

        grp[["Sinal", "Decisão"]] = grp.apply(lambda r: pd.Series(decisao_final(r)), axis=1)

        # display "limpo"
        disp = grp.copy()
        disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS"] = disp["ACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["TACOS"] = disp["TACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ROAS"] = disp["ROAS"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        cols = [
            "Sinal",
            "Campanha",
            "Decisão",
            "CVR",
            "CVR_cor",
            "Ação (CVR)",
            "ACOS_bolinha",
            "ACOS",
            "ACOS_status",
            "TACOS_bolinha",
            "TACOS",
            "TACOS_status",
            "ROAS",
        ]
        st.dataframe(disp[cols].sort_values(by="Campanha"), use_container_width=True, hide_index=True)

        # para export (numérico)
        camp_overview_df = grp.copy()


# ============================
# 13) Insights (Loja + Ads) — com bolinha + números
# ============================
st.divider()
st.header("Insights (Loja + Ads) — bolinhas + números")

candidates_df = None
drops_df = None
opportunities_df = None

if df_sales_curr is None:
    st.info("Suba o Excel producttraffic do mês analisado para liberar os insights.")
else:
    sales_id_col = find_col(df_sales_curr, exact=["ID do Item"], contains=["id do item"])
    prod_name_col = find_col(df_sales_curr, exact=["Produto"], contains=["produto"])
    imp_store_col = find_col(df_sales_curr, exact=["Impressões de Produto"], contains=["impress"])
    clk_store_col = find_col(df_sales_curr, exact=["Cliques Por Produto"], contains=["clique"])
    ctr_store_col = find_col(df_sales_curr, exact=["CTR"], contains=["ctr"])
    cvr_store_col = find_col(df_sales_curr, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
    ord_store_col = find_col(df_sales_curr, exact=["Pedidos"], contains=["pedido"])

    if sales_id_col is None or "Vendas (BRL)" not in df_sales_curr.columns:
        st.error("Não consegui localizar 'ID do Item' e/ou 'Vendas (BRL)' na planilha de pedidos pagos.")
    else:
        s = df_sales_curr.copy()
        s["prod_key"] = s[sales_id_col].astype(str).str.strip().apply(normalize_product_id)
        s["Produto_nome"] = s[prod_name_col].astype(str).str.strip() if prod_name_col else ""
        s["rev_curr"] = numeric_series_from(s, "Vendas (BRL)")
        s["imp_curr"] = numeric_series_from(s, imp_store_col) if imp_store_col else np.nan
        s["clk_curr"] = numeric_series_from(s, clk_store_col) if clk_store_col else np.nan
        s["ctr_curr"] = s[ctr_store_col].apply(parse_percent) if ctr_store_col else np.where(s["imp_curr"].fillna(0) > 0, s["clk_curr"].fillna(0) / s["imp_curr"].fillna(0), np.nan)
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

        # mês anterior
        if df_sales_prev is not None and "Vendas (BRL)" in df_sales_prev.columns:
            sp = df_sales_prev.copy()
            sales_id_prev = find_col(sp, exact=["ID do Item"], contains=["id do item"])
            prod_name_prev = find_col(sp, exact=["Produto"], contains=["produto"])
            imp_prev_col = find_col(sp, exact=["Impressões de Produto"], contains=["impress"])
            clk_prev_col = find_col(sp, exact=["Cliques Por Produto"], contains=["clique"])
            ctr_prev_col = find_col(sp, exact=["CTR"], contains=["ctr"])
            cvr_prev_col = find_col(sp, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
            ord_prev_col = find_col(sp, exact=["Pedidos"], contains=["pedido"])

            if sales_id_prev is not None:
                sp["prod_key"] = sp[sales_id_prev].astype(str).str.strip().apply(normalize_product_id)
                sp["rev_prev"] = numeric_series_from(sp, "Vendas (BRL)")
                sp["imp_prev"] = numeric_series_from(sp, imp_prev_col) if imp_prev_col else np.nan
                sp["clk_prev"] = numeric_series_from(sp, clk_prev_col) if clk_prev_col else np.nan
                sp["ctr_prev"] = sp[ctr_prev_col].apply(parse_percent) if ctr_prev_col else np.where(sp["imp_prev"].fillna(0) > 0, sp["clk_prev"].fillna(0) / sp["imp_prev"].fillna(0), np.nan)
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
                for c in ["rev_prev", "imp_prev", "clk_prev", "ctr_prev", "cvr_prev", "ord_prev"]:
                    sales_mom[c] = pd.to_numeric(sales_mom[c], errors="coerce").fillna(0.0)

                sales_mom["rev_delta_pct"] = np.where(sales_mom["rev_prev"] > 0, (sales_mom["rev_curr"] / sales_mom["rev_prev"]) - 1, np.nan)
                sales_mom["imp_delta_pct"] = np.where(sales_mom["imp_prev"] > 0, (sales_mom["imp_curr"] / sales_mom["imp_prev"]) - 1, np.nan)
                sales_mom["clk_delta_pct"] = np.where(sales_mom["clk_prev"] > 0, (sales_mom["clk_curr"] / sales_mom["clk_prev"]) - 1, np.nan)
                sales_mom["ctr_delta_pp"] = (sales_mom["ctr_curr"] - sales_mom["ctr_prev"])
                sales_mom["cvr_delta_pp"] = (sales_mom["cvr_curr"] - sales_mom["cvr_prev"])
            else:
                sales_mom = sales_curr_agg.copy()
                for c in ["rev_prev","imp_prev","clk_prev","ctr_prev","cvr_prev","ord_prev","rev_delta_pct","imp_delta_pct","clk_delta_pct","ctr_delta_pp","cvr_delta_pp"]:
                    sales_mom[c] = np.nan
        else:
            sales_mom = sales_curr_agg.copy()
            for c in ["rev_prev","imp_prev","clk_prev","ctr_prev","cvr_prev","ord_prev","rev_delta_pct","imp_delta_pct","clk_delta_pct","ctr_delta_pp","cvr_delta_pp"]:
                sales_mom[c] = np.nan

        # base ads por produto (linhas com ID)
        if id_col is not None:
            ads_prod = df_all[~df_all[id_col].astype(str).str.strip().isin(["", "-", "nan", "None"])].copy()
            ads_prod["prod_key"] = ads_prod[id_col].astype(str).str.strip().apply(normalize_product_id)
        else:
            ads_prod = pd.DataFrame(columns=["prod_key"])
        ads_keys = set(ads_prod["prod_key"].dropna().astype(str).tolist())

        tabs_i = st.tabs([
            "Candidatos a Ads (fora do Ads)",
            "Queda mês x mês (loja)",
            "Oportunidades (Ads) — bom + pouca entrega",
        ])

        # Candidatos
        with tabs_i[0]:
            cand = sales_curr_agg.copy()
            cand = cand[~cand["prod_key"].astype(str).isin(ads_keys)]
            cand = cand[cand["rev_curr"] >= float(min_revenue_candidate)]

            if cand.empty:
                st.info("Nenhum candidato no critério (fora do Ads + faturamento mínimo).")
            else:
                cand = cand.sort_values("rev_curr", ascending=False).copy()
                cand["Sinal"] = ""
                cand["Ajuste sugerido"] = ""
                for i, r in cand.iterrows():
                    sinal, ajuste = action_for_store_row(
                        r.get("ctr_curr", np.nan),
                        r.get("cvr_curr", np.nan),
                        r.get("imp_curr", np.nan),
                        low_imp_threshold=int(low_impressions_threshold),
                        not_in_ads=True,
                        revenue=float(r.get("rev_curr", 0.0)) if pd.notna(r.get("rev_curr", np.nan)) else None,
                    )
                    cand.at[i, "Sinal"] = sinal
                    cand.at[i, "Ajuste sugerido"] = ajuste

                out = cand.copy()
                out["Faturamento (mês)"] = out["rev_curr"].apply(fmt_brl)
                out["Pedidos"] = out["ord_curr"].apply(fmt_int)
                out["Impressões"] = out["imp_curr"].apply(fmt_int)
                out["Cliques"] = out["clk_curr"].apply(fmt_int)
                out["CTR"] = out["ctr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                out["CVR"] = out["cvr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

                cols = ["Sinal","prod_key","Produto_nome","Faturamento (mês)","Pedidos","Impressões","Cliques","CTR","CVR","Ajuste sugerido"]
                st.dataframe(out[cols], use_container_width=True, hide_index=True)

                candidates_df = cand.copy()

        # Queda MoM
        with tabs_i[1]:
            if df_sales_prev is None:
                st.info("Suba o mês anterior para liberar queda mês x mês.")
            else:
                drops = sales_mom.copy()
                drops = drops[pd.notna(drops["rev_delta_pct"])]
                drops = drops[drops["rev_delta_pct"] <= -(float(drop_alert_pct) / 100.0)].copy()

                if drops.empty:
                    st.info("Nenhum produto com queda acima do limite.")
                else:
                    drops["Sinal"] = ""
                    drops["Ajuste sugerido"] = ""
                    for i, r in drops.iterrows():
                        sinal, ajuste = action_for_store_row(
                            r.get("ctr_curr", np.nan),
                            r.get("cvr_curr", np.nan),
                            r.get("imp_curr", np.nan),
                            low_imp_threshold=int(low_impressions_threshold),
                            not_in_ads=(str(r.get("prod_key", "")) not in ads_keys),
                            revenue=float(r.get("rev_curr", 0.0)) if pd.notna(r.get("rev_curr", np.nan)) else None,
                        )

                        reasons = []
                        if pd.notna(r.get("imp_delta_pct")) and r["imp_delta_pct"] < 0:
                            reasons.append("Δ Impressões < 0 → colocar/fortalecer ADS")
                        if pd.notna(r.get("ctr_delta_pp")) and r["ctr_delta_pp"] < 0:
                            reasons.append("Δ CTR < 0 → preço/cauda longa/imagem")
                        if pd.notna(r.get("cvr_delta_pp")) and r["cvr_delta_pp"] < 0:
                            reasons.append("Δ CVR < 0 → copy/gatilhos")
                        if pd.notna(r.get("clk_delta_pct")) and r["clk_delta_pct"] < 0:
                            reasons.append("Δ Cliques < 0 → revisar oferta/imagem/palavras")

                        if reasons:
                            ajuste = ajuste + " | " + " | ".join(reasons)

                        drops.at[i, "Sinal"] = sinal
                        drops.at[i, "Ajuste sugerido"] = ajuste

                    out = drops.copy()
                    out["Fat atual"] = out["rev_curr"].apply(fmt_brl)
                    out["Fat anterior"] = out["rev_prev"].apply(fmt_brl)
                    out["Δ Fat"] = out["rev_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                    out["Pedidos atual"] = out["ord_curr"].apply(fmt_int)
                    out["Pedidos anterior"] = out["ord_prev"].apply(fmt_int)

                    out["Imp atual"] = out["imp_curr"].apply(fmt_int)
                    out["Imp anterior"] = out["imp_prev"].apply(fmt_int)
                    out["Δ Imp"] = out["imp_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                    out["Clk atual"] = out["clk_curr"].apply(fmt_int)
                    out["Clk anterior"] = out["clk_prev"].apply(fmt_int)
                    out["Δ Clk"] = out["clk_delta_pct"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

                    out["CTR atual"] = out["ctr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["CTR anterior"] = out["ctr_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["Δ CTR"] = out["ctr_delta_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

                    out["CVR atual"] = out["cvr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["CVR anterior"] = out["cvr_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["Δ CVR"] = out["cvr_delta_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

                    cols = [
                        "Sinal","prod_key","Produto_nome",
                        "Fat atual","Fat anterior","Δ Fat",
                        "Pedidos atual","Pedidos anterior",
                        "Imp atual","Imp anterior","Δ Imp",
                        "Clk atual","Clk anterior","Δ Clk",
                        "CTR atual","CTR anterior","Δ CTR",
                        "CVR atual","CVR anterior","Δ CVR",
                        "Ajuste sugerido",
                    ]
                    st.dataframe(out[cols], use_container_width=True, hide_index=True)

                    drops_df = drops.copy()

        # Oportunidades Ads
        with tabs_i[2]:
            if ads_prod.empty or imp_col is None:
                st.info("Não tenho base de Ads por produto (ou não achei coluna de Impressões no Ads).")
            else:
                opp = ads_prod.copy()
                if cost_col and cost_col in opp.columns:
                    opp[cost_col] = pd.to_numeric(opp[cost_col], errors="coerce")
                    opp = opp.sort_values(by=cost_col, ascending=False)

                opp["Sinal"] = ""
                opp["Ajuste sugerido"] = ""
                for i, r in opp.iterrows():
                    sinal, ajuste = action_for_row_new(
                        r.get("ctr_calc", np.nan),
                        r.get("cvr_calc", np.nan),
                        r.get(imp_col, np.nan) if imp_col else np.nan,
                        low_imp_threshold=int(low_impressions_threshold),
                    )
                    opp.at[i, "Sinal"] = sinal
                    opp.at[i, "Ajuste sugerido"] = ajuste

                # filtra os que têm pouca impressão e não são "ok"
                opp_f = opp[pd.to_numeric(opp[imp_col], errors="coerce").fillna(0) <= low_impressions_threshold].copy()
                if opp_f.empty:
                    st.info("Nenhuma oportunidade no critério (pouca impressão).")
                else:
                    out = opp_f.copy()
                    out["prod_key"] = out[id_col].astype(str).str.strip().apply(normalize_product_id) if id_col else ""
                    out["Impressões"] = out[imp_col].apply(fmt_int) if imp_col in out.columns else ""
                    out["Cliques"] = out[clk_col].apply(fmt_int) if clk_col in out.columns else ""
                    out["CTR"] = out["ctr_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["CVR"] = out["cvr_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    out["GMV (ads)"] = out[rev_col].apply(fmt_brl) if rev_col in out.columns else ""
                    out["Desp (ads)"] = out[cost_col].apply(fmt_brl) if cost_col in out.columns else ""
                    out["ACOS"] = out["acos_calc"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

                    cols = ["Sinal","Campanha"]
                    if name_col in out.columns:
                        cols += [name_col]
                    cols += ["prod_key","Impressões","Cliques","CTR","CVR","GMV (ads)","Desp (ads)","ACOS","Ajuste sugerido"]

                    st.dataframe(out[cols], use_container_width=True, hide_index=True)
                    opportunities_df = opp_f.copy()


# ============================
# 14) Campanha → Totais + Anúncios (detalhado + bolinha + ajuste)
# ============================
st.divider()
st.header("Campanha → Totais + Anúncios (detalhado + recomendações por linha)")

detailed_ads_all = []

campaigns = sorted(df_all["Campanha"].dropna().unique().tolist())
sel = st.selectbox("Filtrar campanha (detalhe)", ["(todas)"] + campaigns, index=0, key="camp_detail_filter")

view = df_all.copy()
if sel != "(todas)":
    view = view[view["Campanha"] == sel]

for camp, d in view.groupby("Campanha"):
    with st.expander(camp, expanded=(sel != "(todas)")):
        if id_col:
            id_clean2 = d[id_col].astype(str).str.strip()
            is_total2 = id_clean2.isin(["", "-", "nan", "None"])
            df_total = d[is_total2].copy()
            df_ads = d[~is_total2].copy()
        else:
            df_total = pd.DataFrame()
            df_ads = d.copy()

        st.markdown("### Totais da campanha (somente referência)")
        if df_total.empty:
            st.info("Não encontrei a linha TOTAL (sem ID). Vou mostrar apenas os anúncios.")
        else:
            cols_total = [c for c in [
                "Campanha",
                orders_col, "cvr_calc",
                cost_col, rev_col, "acos_calc", "roas_calc"
            ] if c and c in df_total.columns]
            disp_total = make_display_df(df_total[cols_total], imp_col, clk_col, cost_col, orders_col, rev_col)
            if "roas_calc" in df_total.columns and "roas_calc" in disp_total.columns:
                disp_total["roas_calc"] = df_total["roas_calc"].apply(lambda v: "" if pd.isna(v) else str(round(float(v), 2)).replace(".", ","))
            disp_total = disp_total.rename(columns={"cvr_calc": "CVR", "acos_calc": "ACOS", "roas_calc": "ROAS"})
            st.dataframe(disp_total, use_container_width=True, hide_index=True)

        st.markdown("### Anúncios (detalhado) — com bolinha + ajuste ao lado")
        if df_ads.empty:
            st.warning("Não há anúncios com ID neste grupo.")
        else:
            df_ads_sorted = df_ads.copy()
            for c in [imp_col, clk_col, orders_col, rev_col, cost_col]:
                if c and c in df_ads_sorted.columns:
                    df_ads_sorted[c] = pd.to_numeric(df_ads_sorted[c], errors="coerce")

            if cost_col and cost_col in df_ads_sorted.columns:
                df_ads_sorted = df_ads_sorted.sort_values(by=cost_col, ascending=False)

            df_ads_sorted["prod_key"] = df_ads_sorted[id_col].astype(str).str.strip().apply(normalize_product_id) if id_col in df_ads_sorted.columns else ""
            df_ads_sorted["Sinal"] = ""
            df_ads_sorted["Ajuste sugerido"] = ""

            for i, r in df_ads_sorted.iterrows():
                sinal, ajuste = action_for_row_new(
                    r.get("ctr_calc", np.nan),
                    r.get("cvr_calc", np.nan),
                    r.get(imp_col, np.nan) if imp_col else np.nan,
                    low_imp_threshold=int(low_impressions_threshold),
                )
                df_ads_sorted.at[i, "Sinal"] = sinal
                df_ads_sorted.at[i, "Ajuste sugerido"] = ajuste

            cols_ads = [c for c in [
                "Sinal",
                name_col, id_col,
                imp_col, clk_col, "ctr_calc",
                orders_col, "cvr_calc",
                rev_col, cost_col,
                "acos_calc", "roas_calc",
                "cpc", "cpa",
                "Ajuste sugerido",
            ] if c and c in df_ads_sorted.columns]

            disp_ads = make_display_df(df_ads_sorted[cols_ads], imp_col, clk_col, cost_col, orders_col, rev_col)
            if "roas_calc" in df_ads_sorted.columns and "roas_calc" in disp_ads.columns:
                disp_ads["roas_calc"] = df_ads_sorted["roas_calc"].apply(lambda v: "" if pd.isna(v) else str(round(float(v), 2)).replace(".", ","))

            disp_ads = disp_ads.rename(columns={
                "ctr_calc": "CTR",
                "cvr_calc": "CVR",
                "acos_calc": "ACOS",
                "roas_calc": "ROAS",
            })

            st.dataframe(disp_ads, use_container_width=True, hide_index=True)

            detailed_ads_all.append(df_ads_sorted.copy())


# ============================
# 15) Download Excel (todas as tabelas)
# ============================
st.divider()
st.header("Exportar (Excel)")

export_tables = {}

if camp_overview_df is not None:
    export_tables["Campanhas"] = camp_overview_df

if candidates_df is not None:
    export_tables["Candidatos_ADS"] = candidates_df

if drops_df is not None:
    export_tables["Queda_MesxMes"] = drops_df

if opportunities_df is not None:
    export_tables["Oportunidades_ADS"] = opportunities_df

if detailed_ads_all:
    export_tables["Anuncios_Detalhado"] = pd.concat(detailed_ads_all, ignore_index=True)

export_tables["ADS_Base_Completa"] = df_all.copy()

if export_tables:
    xlsx_bytes = make_excel_export(export_tables)
    st.download_button(
        "Baixar relatório Excel",
        data=xlsx_bytes,
        file_name="shopee_auditoria_relatorio.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
else:
    st.info("Nenhuma tabela pronta para exportar ainda.")
