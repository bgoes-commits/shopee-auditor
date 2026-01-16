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


def normalize_campaign_name(s: str) -> str:
    t = _to_str(s).lower()
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("grupo de anúncios", "grupo anuncios")
    t = t.replace("grupo de anuncios", "grupo anuncios")
    return t


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
# 6) Helpers MoM (NOVO): sinal pela queda de faturamento (GMV) e razões
# ============================
def mom_signal_and_reasons(
    gmv_prev: float, gmv_curr: float,
    imp_prev: float, imp_curr: float,
    clk_prev: float, clk_curr: float,
    ctr_prev: float, ctr_curr: float,
    cvr_prev: float, cvr_curr: float,
    pedidos_prev: float, pedidos_curr: float,
) -> tuple[str, str]:
    """
    REGRA PRINCIPAL (como você pediu):
    1) Primeiro sinaliza pelo GMV (faturamento Ads):
       - GMV_curr < GMV_prev -> 🔴 (queda)
       - GMV_curr > GMV_prev -> 🟢 (cresceu)
       - caso contrário -> 🟡 (estável / novo)
    2) Se caiu, lista o que caiu junto (com números).
    """
    # sinal principal
    if pd.notna(gmv_prev) and gmv_prev > 0 and pd.notna(gmv_curr):
        if gmv_curr < gmv_prev:
            sinal = "🔴"
        elif gmv_curr > gmv_prev:
            sinal = "🟢"
        else:
            sinal = "🟡"
    else:
        # sem base anterior
        sinal = "🟡"

    reasons = []

    # só lista "caiu junto" quando GMV caiu
    if sinal == "🔴":
        # impressões
        if pd.notna(imp_prev) and imp_prev > 0 and pd.notna(imp_curr) and imp_curr < imp_prev:
            reasons.append(f"Impressões caíram ({fmt_int(imp_prev)} → {fmt_int(imp_curr)}) → colocar/fortalecer ADS")
        # cliques
        if pd.notna(clk_prev) and clk_prev > 0 and pd.notna(clk_curr) and clk_curr < clk_prev:
            reasons.append(f"Cliques caíram ({fmt_int(clk_prev)} → {fmt_int(clk_curr)}) → revisar oferta/imagem/palavras")
        # CTR
        if pd.notna(ctr_prev) and pd.notna(ctr_curr) and ctr_curr < ctr_prev:
            reasons.append(f"CTR caiu ({fmt_pct(ctr_prev)} → {fmt_pct(ctr_curr)}) → preço/cauda longa/imagem")
        # CVR
        if pd.notna(cvr_prev) and pd.notna(cvr_curr) and cvr_curr < cvr_prev:
            reasons.append(f"CVR caiu ({fmt_pct(cvr_prev)} → {fmt_pct(cvr_curr)}) → copy/gatilhos")
        # pedidos (conversões)
        if pd.notna(pedidos_prev) and pedidos_prev > 0 and pd.notna(pedidos_curr) and pedidos_curr < pedidos_prev:
            reasons.append(f"Pedidos caíram ({fmt_int(pedidos_prev)} → {fmt_int(pedidos_curr)})")

    return sinal, (" | ".join(reasons) if reasons else ("Sem queda relevante nos drivers" if sinal == "🔴" else ""))


# ============================
# 7) UI
# ============================
st.title("Shopee Ads – Campanhas + Loja (Pedidos pagos + TACOS + Insights + Mês x Mês Ads)")

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
# 8) Classificações configuráveis (CTR/CVR) + semáforo meta
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

    if pd.notna(imps) and imps <= low_impressions_threshold:
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
- **Mês x mês (Ads)**: o **primeiro sinal** é pela **queda do GMV (faturamento Ads)**; se caiu, mostramos **o que caiu junto** (impressões/cliques/CTR/CVR/pedidos).
"""
)

# ============================
# 9) Upload Ads - mês analisado e mês anterior
# ============================
st.header("Ads – Upload (mês analisado e mês anterior)")

c_ads1, c_ads2 = st.columns(2)
with c_ads1:
    ads_group_files = st.file_uploader(
        "CSV – Grupo de anúncios (MÊS ANALISADO) (1 ou mais)",
        type=["csv"],
        accept_multiple_files=True,
        key="ads_groups_curr",
    )
with c_ads2:
    ads_group_prev_files = st.file_uploader(
        "CSV – Grupo de anúncios (MÊS ANTERIOR) (1 ou mais)",
        type=["csv"],
        accept_multiple_files=True,
        key="ads_groups_prev",
    )

if not ads_group_files:
    st.info("Suba 1 ou mais CSVs de **Dados do Grupo de Anúncios** do mês analisado.")
    st.stop()


def load_ads_files(files) -> pd.DataFrame:
    frames = []
    for f in files:
        df_raw, meta = read_shopee_csv(f)
        df = parse_ads_table(df_raw)

        group_name = meta.get("titulo", "")
        group_name = group_name.replace("\ufeff", "")
        group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
        if not group_name:
            group_name = Path(getattr(f, "name", "grupo")).stem

        df["Campanha"] = group_name
        df["Campanha_key"] = normalize_campaign_name(group_name)
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


df_all = load_ads_files(ads_group_files)
df_prev = load_ads_files(ads_group_prev_files) if ads_group_prev_files else pd.DataFrame()

imp_col = "Impressões" if "Impressões" in df_all.columns else ("Impressões do Produto" if "Impressões do Produto" in df_all.columns else None)
clk_col = "Cliques" if "Cliques" in df_all.columns else ("Cliques de Produtos" if "Cliques de Produtos" in df_all.columns else None)
rev_col = "GMV" if "GMV" in df_all.columns else None
cost_col = "Despesas" if "Despesas" in df_all.columns else ("Custo" if "Custo" in df_all.columns else None)

orders_col = None
for cand in ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"]:
    if cand in df_all.columns:
        orders_col = cand
        break

df_all = add_ads_metrics(df_all, imp_col=imp_col, clk_col=clk_col, cost_col=cost_col, orders_col=orders_col, rev_col=rev_col)

if not df_prev.empty:
    imp_col_prev = imp_col if imp_col in df_prev.columns else ("Impressões" if "Impressões" in df_prev.columns else ("Impressões do Produto" if "Impressões do Produto" in df_prev.columns else None))
    clk_col_prev = clk_col if clk_col in df_prev.columns else ("Cliques" if "Cliques" in df_prev.columns else ("Cliques de Produtos" if "Cliques de Produtos" in df_prev.columns else None))
    rev_col_prev = rev_col if rev_col in df_prev.columns else ("GMV" if "GMV" in df_prev.columns else None)
    cost_col_prev = cost_col if cost_col in df_prev.columns else ("Despesas" if "Despesas" in df_prev.columns else ("Custo" if "Custo" in df_prev.columns else None))

    orders_col_prev = None
    for cand in ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"]:
        if cand in df_prev.columns:
            orders_col_prev = cand
            break

    df_prev = add_ads_metrics(df_prev, imp_col=imp_col_prev, clk_col=clk_col_prev, cost_col=cost_col_prev, orders_col=orders_col_prev, rev_col=rev_col_prev)
else:
    imp_col_prev = clk_col_prev = rev_col_prev = cost_col_prev = orders_col_prev = None

id_col = "ID do produto" if "ID do produto" in df_all.columns else None
name_col = "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in df_all.columns else None

# ============================
# 10) Upload Vendas (Loja) - producttraffic (mês atual e anterior)
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
# 11) KPIs TOP: faturamento anterior | faturamento analisado | diferença
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
# 12) KPI ADS (mês analisado): gasto | gmv | ACOS | ROAS | TACOS
# ============================
def compute_total_from_campaign_rows(df: pd.DataFrame, col: str, id_col_: str | None) -> float:
    if col is None or col not in df.columns:
        return np.nan
    s = pd.to_numeric(df[col], errors="coerce")
    if id_col_ and id_col_ in df.columns:
        id_clean_ = df[id_col_].astype(str).str.strip()
        is_total_ = id_clean_.isin(["", "-", "nan", "None"])
        if is_total_.any():
            return float(np.nansum(s[is_total_]))
    return float(np.nansum(s))


ads_spend_total = compute_total_from_campaign_rows(df_all, cost_col, id_col)
ads_gmv_total = compute_total_from_campaign_rows(df_all, rev_col, id_col)

acos_total = (ads_spend_total / ads_gmv_total) if (pd.notna(ads_spend_total) and pd.notna(ads_gmv_total) and ads_gmv_total > 0) else np.nan
roas_total = (ads_gmv_total / ads_spend_total) if (pd.notna(ads_spend_total) and ads_spend_total > 0 and pd.notna(ads_gmv_total)) else np.nan
tacos_total = (ads_spend_total / faturamento_atual) if (pd.notna(ads_spend_total) and faturamento_atual is not None and faturamento_atual > 0) else np.nan

st.subheader("Ads (mês analisado) – KPIs")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Gasto Ads (mês analisado)", fmt_brl(ads_spend_total) if pd.notna(ads_spend_total) else "—")
k2.metric("GMV Ads (mês analisado)", fmt_brl(ads_gmv_total) if pd.notna(ads_gmv_total) else "—")
k3.metric("ACOS (mês analisado)", fmt_pct(acos_total) if pd.notna(acos_total) else "—")
k4.metric("ROAS (mês analisado)", (str(round(roas_total, 2)).replace(".", ",")) if pd.notna(roas_total) else "—")
k5.metric("TACOS (mês analisado)", fmt_pct(tacos_total) if pd.notna(tacos_total) else "—")

# ============================
# 13) Tabs principais
# ============================
st.divider()
tabs_top = st.tabs([
    "Campanhas (decisão)",
    "Ads mês x mês (Campanhas)",
    "Ads mês x mês (Anúncios)",
])

camp_overview_df = None
camp_mom_df = None
ads_mom_df = None

# ============================
# 13.1) Campanhas (decisão) - mês atual
# ============================
with tabs_top[0]:
    st.header("Visão Geral de Campanhas (decisão por CVR + ACOS + TACOS)")

    if id_col is None:
        st.warning("Não encontrei a coluna 'ID do produto'. Não consigo separar TOTAL vs anúncios.")
    else:
        id_clean = df_all[id_col].astype(str).str.strip()
        is_total = id_clean.isin(["", "-", "nan", "None"])
        camp_total = df_all[is_total].copy()

        if camp_total.empty:
            st.warning("Não encontrei linhas TOTAL (sem ID).")
        else:
            grp = camp_total.groupby(["Campanha_key", "Campanha"], dropna=False).agg({
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
                acos_st = row["ACOS_status"]
                tacos_st = row["TACOS_status"]

                if acos_st == "ajustar" or tacos_st == "ajustar":
                    return "🔴", "AJUSTAR (ACOS/TACOS acima do limite)"
                if cvr_cls == "vermelho":
                    return "🔴", "AJUSTAR (CVR ruim)"
                if cvr_cls == "verde" and acos_st == "ótimo" and tacos_st == "ótimo":
                    return "🟢", "AUMENTAR ORÇAMENTO (CVR ótimo + ACOS/TACOS ótimos)"
                if cvr_cls == "verde" and (acos_st == "monitorar" or tacos_st == "monitorar"):
                    return "🟡", "MONITORAR (CVR ótimo, mas ACOS/TACOS pedem atenção)"
                if cvr_cls == "amarelo" and acos_st == "ótimo" and tacos_st == "ótimo":
                    return "🟡", "MANTER (CVR bom)"
                return "🟡", "MONITORAR"

            grp[["Sinal", "Decisão"]] = grp.apply(lambda r: pd.Series(decisao_final(r)), axis=1)

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

            camp_overview_df = grp.copy()

# ============================
# 13.2) Ads mês x mês – Campanhas (ATUALIZADO: sinal principal GMV + "o que caiu junto")
# ============================
with tabs_top[1]:
    st.header("Ads mês x mês – Campanhas (comparação com mês anterior)")
    st.caption("O sinal principal é pela queda de GMV (faturamento Ads). Se GMV caiu, mostramos quais drivers caíram juntos (Impressões / Cliques / CTR / CVR / Pedidos).")

    if df_prev.empty:
        st.info("Suba os CSVs do **mês anterior** para liberar esta comparação.")
    elif id_col is None or (id_col not in df_prev.columns):
        st.warning("Não encontrei 'ID do produto' no mês anterior.")
    else:
        id_clean_c = df_all[id_col].astype(str).str.strip()
        is_total_c = id_clean_c.isin(["", "-", "nan", "None"])
        curr_tot = df_all[is_total_c].copy()

        id_clean_p = df_prev[id_col].astype(str).str.strip()
        is_total_p = id_clean_p.isin(["", "-", "nan", "None"])
        prev_tot = df_prev[is_total_p].copy()

        def campaign_totals(df_tot: pd.DataFrame, imp_, clk_, ord_, gmv_, spend_, faturamento_total: float | None) -> pd.DataFrame:
            g = df_tot.groupby(["Campanha_key", "Campanha"], dropna=False).agg({
                imp_: "sum" if imp_ and imp_ in df_tot.columns else "sum",
                clk_: "sum" if clk_ and clk_ in df_tot.columns else "sum",
                ord_: "sum" if ord_ and ord_ in df_tot.columns else "sum",
                gmv_: "sum" if gmv_ and gmv_ in df_tot.columns else "sum",
                spend_: "sum" if spend_ and spend_ in df_tot.columns else "sum",
            }).reset_index()

            g = g.rename(columns={
                imp_: "Impressões",
                clk_: "Cliques",
                ord_: "Pedidos",
                gmv_: "GMV",
                spend_: "Despesas",
            })

            g["CTR"] = np.where(g["Impressões"].fillna(0) > 0, g["Cliques"].fillna(0) / g["Impressões"].fillna(0), np.nan)
            g["CVR"] = np.where(g["Cliques"].fillna(0) > 0, g["Pedidos"].fillna(0) / g["Cliques"].fillna(0), np.nan)
            g["ACOS"] = np.where(g["GMV"].fillna(0) > 0, g["Despesas"].fillna(0) / g["GMV"].fillna(0), np.nan)
            g["ROAS"] = np.where(g["Despesas"].fillna(0) > 0, g["GMV"].fillna(0) / g["Despesas"].fillna(0), np.nan)
            g["TACOS"] = (g["Despesas"] / faturamento_total) if (faturamento_total is not None and faturamento_total > 0) else np.nan
            return g

        curr_camp = campaign_totals(curr_tot, imp_col, clk_col, orders_col, rev_col, cost_col, faturamento_atual)
        prev_camp = campaign_totals(prev_tot, imp_col_prev, clk_col_prev, orders_col_prev, rev_col_prev, cost_col_prev, faturamento_anterior)

        mom = curr_camp.merge(prev_camp, on="Campanha_key", how="outer", suffixes=("_curr", "_prev"))
        mom["Campanha"] = mom["Campanha_curr"].fillna(mom["Campanha_prev"])

        # deltas em %
        for base in ["GMV", "Despesas", "Impressões", "Cliques", "Pedidos"]:
            currv = pd.to_numeric(mom[f"{base}_curr"], errors="coerce")
            prevv = pd.to_numeric(mom[f"{base}_prev"], errors="coerce")
            mom[f"Δ_{base}_R$"] = currv - prevv
            mom[f"Δ_{base}_%"] = np.where(prevv > 0, (currv / prevv) - 1, np.nan)

        # deltas em p.p.
        for rate in ["CTR", "CVR", "ACOS", "TACOS"]:
            c = pd.to_numeric(mom[f"{rate}_curr"], errors="coerce")
            p = pd.to_numeric(mom[f"{rate}_prev"], errors="coerce")
            mom[f"Δ_{rate}_pp"] = c - p

        # NOVO: sinal principal por GMV + motivos (o que caiu junto)
        mom["Sinal (GMV)"] = ""
        mom["O que caiu junto"] = ""

        for i, r in mom.iterrows():
            sinal, reasons = mom_signal_and_reasons(
                r.get("GMV_prev", np.nan), r.get("GMV_curr", np.nan),
                r.get("Impressões_prev", np.nan), r.get("Impressões_curr", np.nan),
                r.get("Cliques_prev", np.nan), r.get("Cliques_curr", np.nan),
                r.get("CTR_prev", np.nan), r.get("CTR_curr", np.nan),
                r.get("CVR_prev", np.nan), r.get("CVR_curr", np.nan),
                r.get("Pedidos_prev", np.nan), r.get("Pedidos_curr", np.nan),
            )
            mom.at[i, "Sinal (GMV)"] = sinal
            mom.at[i, "O que caiu junto"] = reasons

        # decisão do mês atual (mantém sua regra CVR/ACOS/TACOS)
        acos_ideal = acos_ideal_pct / 100.0
        acos_limite = acos_limite_pct / 100.0
        tacos_ideal = tacos_ideal_pct / 100.0
        tacos_limite = tacos_limite_pct / 100.0

        mom["CVR_cor_curr"] = mom["CVR_curr"].apply(classify_cvr)
        mom[["ACOS_bol_curr", "ACOS_st_curr"]] = mom["ACOS_curr"].apply(lambda v: pd.Series(semaforo_ideal_limite(v, acos_ideal, acos_limite)))
        mom[["TACOS_bol_curr", "TACOS_st_curr"]] = mom["TACOS_curr"].apply(lambda v: pd.Series(semaforo_ideal_limite(v, tacos_ideal, tacos_limite)))

        def decisao_atual(row) -> str:
            if row["ACOS_st_curr"] == "ajustar" or row["TACOS_st_curr"] == "ajustar":
                return "AJUSTAR"
            if row["CVR_cor_curr"] == "vermelho":
                return "AJUSTAR"
            if row["CVR_cor_curr"] == "verde" and row["ACOS_st_curr"] == "ótimo" and row["TACOS_st_curr"] == "ótimo":
                return "AUMENTAR ORÇAMENTO"
            return "MONITORAR"

        mom["Ação recomendada (mês atual)"] = mom.apply(decisao_atual, axis=1)

        # DISPLAY (cabeçalhos claros)
        disp = mom.copy()

        # valores
        disp["GMV mês anterior"] = mom["GMV_prev"].apply(fmt_brl)
        disp["GMV mês analisado"] = mom["GMV_curr"].apply(fmt_brl)
        disp["Variação GMV (R$)"] = mom["Δ_GMV_R$"].apply(fmt_brl)
        disp["Variação GMV (%)"] = mom["Δ_GMV_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["Gasto Ads mês anterior"] = mom["Despesas_prev"].apply(fmt_brl)
        disp["Gasto Ads mês analisado"] = mom["Despesas_curr"].apply(fmt_brl)
        disp["Variação gasto Ads (R$)"] = mom["Δ_Despesas_R$"].apply(fmt_brl)
        disp["Variação gasto Ads (%)"] = mom["Δ_Despesas_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["CTR mês anterior"] = mom["CTR_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CTR mês analisado"] = mom["CTR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação CTR (p.p.)"] = mom["Δ_CTR_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        disp["CVR mês anterior"] = mom["CVR_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CVR mês analisado"] = mom["CVR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação CVR (p.p.)"] = mom["Δ_CVR_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        disp["ACOS mês anterior"] = mom["ACOS_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS mês analisado"] = mom["ACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação ACOS (p.p.)"] = mom["Δ_ACOS_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        disp["TACOS mês anterior"] = mom["TACOS_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["TACOS mês analisado"] = mom["TACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação TACOS (p.p.)"] = mom["Δ_TACOS_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        show_cols = [
            "Sinal (GMV)",
            "Campanha",
            "Ação recomendada (mês atual)",
            "GMV mês anterior", "GMV mês analisado", "Variação GMV (R$)", "Variação GMV (%)",
            "Gasto Ads mês anterior", "Gasto Ads mês analisado", "Variação gasto Ads (R$)", "Variação gasto Ads (%)",
            "CTR mês anterior", "CTR mês analisado", "Variação CTR (p.p.)",
            "CVR mês anterior", "CVR mês analisado", "Variação CVR (p.p.)",
            "ACOS mês anterior", "ACOS mês analisado", "Variação ACOS (p.p.)",
            "TACOS mês anterior", "TACOS mês analisado", "Variação TACOS (p.p.)",
            "O que caiu junto",
        ]
        st.dataframe(disp[show_cols].sort_values(by="Campanha"), use_container_width=True, hide_index=True)

        camp_mom_df = mom.copy()

# ============================
# 13.3) Ads mês x mês – Anúncios (mesma regra: sinal principal GMV e motivos)
# ============================
with tabs_top[2]:
    st.header("Ads mês x mês – Anúncios/Produtos (comparação com mês anterior)")
    st.caption("Sinal principal é pela queda do GMV do anúncio/produto. Se caiu, mostramos o que caiu junto (impressões/cliques/CTR/CVR/pedidos).")

    if df_prev.empty:
        st.info("Suba os CSVs do **mês anterior** para liberar esta comparação.")
    elif id_col is None or (id_col not in df_prev.columns):
        st.warning("Não encontrei 'ID do produto' no mês anterior.")
    else:
        def extract_ads(df: pd.DataFrame) -> pd.DataFrame:
            if id_col not in df.columns:
                return pd.DataFrame()
            id_clean = df[id_col].astype(str).str.strip()
            is_total = id_clean.isin(["", "-", "nan", "None"])
            out = df[~is_total].copy()
            out["prod_key"] = out[id_col].astype(str).str.strip().apply(normalize_product_id)
            out["Campanha_key"] = out["Campanha_key"].apply(_to_str)
            return out

        ads_curr = extract_ads(df_all)
        ads_prev = extract_ads(df_prev)

        def agg_ads(df: pd.DataFrame, imp_, clk_, ord_, gmv_, spend_) -> pd.DataFrame:
            g = df.groupby(["Campanha_key", "Campanha", "prod_key"], dropna=False).agg({
                imp_: "sum" if imp_ and imp_ in df.columns else "sum",
                clk_: "sum" if clk_ and clk_ in df.columns else "sum",
                ord_: "sum" if ord_ and ord_ in df.columns else "sum",
                gmv_: "sum" if gmv_ and gmv_ in df.columns else "sum",
                spend_: "sum" if spend_ and spend_ in df.columns else "sum",
            }).reset_index()

            g = g.rename(columns={
                imp_: "Impressões",
                clk_: "Cliques",
                ord_: "Pedidos",
                gmv_: "GMV",
                spend_: "Despesas",
            })

            if name_col and name_col in df.columns:
                name_map = df.dropna(subset=[name_col]).groupby(["Campanha_key", "prod_key"])[name_col].first().reset_index()
                g = g.merge(name_map, on=["Campanha_key", "prod_key"], how="left")

            g["CTR"] = np.where(g["Impressões"].fillna(0) > 0, g["Cliques"].fillna(0) / g["Impressões"].fillna(0), np.nan)
            g["CVR"] = np.where(g["Cliques"].fillna(0) > 0, g["Pedidos"].fillna(0) / g["Cliques"].fillna(0), np.nan)
            g["ACOS"] = np.where(g["GMV"].fillna(0) > 0, g["Despesas"].fillna(0) / g["GMV"].fillna(0), np.nan)
            g["ROAS"] = np.where(g["Despesas"].fillna(0) > 0, g["GMV"].fillna(0) / g["Despesas"].fillna(0), np.nan)
            return g

        curr_a = agg_ads(ads_curr, imp_col, clk_col, orders_col, rev_col, cost_col)
        prev_a = agg_ads(ads_prev, imp_col_prev, clk_col_prev, orders_col_prev, rev_col_prev, cost_col_prev)

        mom = curr_a.merge(prev_a, on=["Campanha_key", "prod_key"], how="outer", suffixes=("_curr", "_prev"))
        mom["Campanha"] = mom["Campanha_curr"].fillna(mom["Campanha_prev"])

        if name_col:
            mom["Produto/Anúncio"] = mom.get(f"{name_col}_curr").fillna(mom.get(f"{name_col}_prev"))
        else:
            mom["Produto/Anúncio"] = ""

        # deltas
        for base in ["GMV", "Despesas", "Impressões", "Cliques", "Pedidos"]:
            c = pd.to_numeric(mom[f"{base}_curr"], errors="coerce")
            p = pd.to_numeric(mom[f"{base}_prev"], errors="coerce")
            mom[f"Δ_{base}_abs"] = c - p
            mom[f"Δ_{base}_%"] = np.where(p > 0, (c / p) - 1, np.nan)

        for rate in ["CTR", "CVR", "ACOS"]:
            c = pd.to_numeric(mom[f"{rate}_curr"], errors="coerce")
            p = pd.to_numeric(mom[f"{rate}_prev"], errors="coerce")
            mom[f"Δ_{rate}_pp"] = c - p

        # NOVO: sinal por GMV + motivos (drivers)
        mom["Sinal (GMV)"] = ""
        mom["O que caiu junto"] = ""

        for i, r in mom.iterrows():
            sinal, reasons = mom_signal_and_reasons(
                r.get("GMV_prev", np.nan), r.get("GMV_curr", np.nan),
                r.get("Impressões_prev", np.nan), r.get("Impressões_curr", np.nan),
                r.get("Cliques_prev", np.nan), r.get("Cliques_curr", np.nan),
                r.get("CTR_prev", np.nan), r.get("CTR_curr", np.nan),
                r.get("CVR_prev", np.nan), r.get("CVR_curr", np.nan),
                r.get("Pedidos_prev", np.nan), r.get("Pedidos_curr", np.nan),
            )
            mom.at[i, "Sinal (GMV)"] = sinal
            mom.at[i, "O que caiu junto"] = reasons

        # Ajuste (mês atual) continua igual (pra ação por CTR/CVR + pouca impressão)
        mom["Sinal (qualidade mês atual)"] = ""
        mom["Ajuste recomendado (mês atual)"] = ""
        for i, r in mom.iterrows():
            sinal_q, ajuste = action_for_row_new(
                r.get("CTR_curr", np.nan),
                r.get("CVR_curr", np.nan),
                r.get("Impressões_curr", np.nan),
                low_imp_threshold=int(low_impressions_threshold),
            )
            mom.at[i, "Sinal (qualidade mês atual)"] = sinal_q
            mom.at[i, "Ajuste recomendado (mês atual)"] = ajuste

        # DISPLAY (cabeçalhos claros)
        disp = mom.copy()

        disp["ID do produto"] = mom["prod_key"].astype(str)

        disp["GMV mês anterior"] = mom["GMV_prev"].apply(fmt_brl)
        disp["GMV mês analisado"] = mom["GMV_curr"].apply(fmt_brl)
        disp["Variação GMV (R$)"] = mom["Δ_GMV_abs"].apply(fmt_brl)
        disp["Variação GMV (%)"] = mom["Δ_GMV_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["Gasto Ads mês anterior"] = mom["Despesas_prev"].apply(fmt_brl)
        disp["Gasto Ads mês analisado"] = mom["Despesas_curr"].apply(fmt_brl)
        disp["Variação gasto Ads (R$)"] = mom["Δ_Despesas_abs"].apply(fmt_brl)
        disp["Variação gasto Ads (%)"] = mom["Δ_Despesas_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["Impressões mês anterior"] = mom["Impressões_prev"].apply(fmt_int)
        disp["Impressões mês analisado"] = mom["Impressões_curr"].apply(fmt_int)

        disp["Cliques mês anterior"] = mom["Cliques_prev"].apply(fmt_int)
        disp["Cliques mês analisado"] = mom["Cliques_curr"].apply(fmt_int)

        disp["CTR mês anterior"] = mom["CTR_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CTR mês analisado"] = mom["CTR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação CTR (p.p.)"] = mom["Δ_CTR_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        disp["Pedidos mês anterior"] = mom["Pedidos_prev"].apply(fmt_int)
        disp["Pedidos mês analisado"] = mom["Pedidos_curr"].apply(fmt_int)

        disp["CVR mês anterior"] = mom["CVR_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CVR mês analisado"] = mom["CVR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação CVR (p.p.)"] = mom["Δ_CVR_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        disp["ACOS mês anterior"] = mom["ACOS_prev"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS mês analisado"] = mom["ACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["Variação ACOS (p.p.)"] = mom["Δ_ACOS_pp"].apply(lambda v: fmt_pp(v) if pd.notna(v) else "")

        show_cols = [
            "Campanha",
            "ID do produto",
            "Produto/Anúncio",
            "Sinal (GMV)",
            "GMV mês anterior", "GMV mês analisado", "Variação GMV (R$)", "Variação GMV (%)",
            "Gasto Ads mês anterior", "Gasto Ads mês analisado", "Variação gasto Ads (R$)", "Variação gasto Ads (%)",
            "Impressões mês anterior", "Impressões mês analisado",
            "Cliques mês anterior", "Cliques mês analisado",
            "CTR mês anterior", "CTR mês analisado", "Variação CTR (p.p.)",
            "Pedidos mês anterior", "Pedidos mês analisado",
            "CVR mês anterior", "CVR mês analisado", "Variação CVR (p.p.)",
            "ACOS mês anterior", "ACOS mês analisado", "Variação ACOS (p.p.)",
            "O que caiu junto",
            "Sinal (qualidade mês atual)",
            "Ajuste recomendado (mês atual)",
        ]
        st.dataframe(disp[show_cols].sort_values(by=["Campanha", "ID do produto"]), use_container_width=True, hide_index=True)

        ads_mom_df = mom.copy()

# ============================
# 14) Exportar Excel
# ============================
st.divider()
st.header("Exportar (Excel)")

export_tables = {}
export_tables["ADS_Base_Atual"] = df_all.copy()
if not df_prev.empty:
    export_tables["ADS_Base_Anterior"] = df_prev.copy()
if camp_overview_df is not None:
    export_tables["Campanhas_Atual"] = camp_overview_df
if camp_mom_df is not None:
    export_tables["Campanhas_MoM"] = camp_mom_df
if ads_mom_df is not None:
    export_tables["Anuncios_MoM"] = ads_mom_df

xlsx_bytes = make_excel_export(export_tables)
st.download_button(
    "Baixar relatório Excel",
    data=xlsx_bytes,
    file_name="shopee_auditoria_relatorio.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)
