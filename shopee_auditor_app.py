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
        # BR: remove milhares com ponto e troca vírgula decimal por ponto
        s = s.replace(".", "")
        s = s.replace(",", ".")
    else:
        # sem vírgula: pode ser 612.085 (milhar) ou 32649.43 (decimal)
        if s.count(".") >= 2:
            s = s.replace(".", "")
        elif s.count(".") == 1:
            left, right = s.split(".")
            # se 3 dígitos depois do ponto -> provavelmente milhar
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
    for i, line in enumerate(lines[:800]):
        if line.startswith("#,") or line.startswith("#;"):
            return i
    for i, line in enumerate(lines[:1200]):
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

    df["acos_calc"] = np.where((cost_col and rev_col and df[rev_col] > 0), df[cost_col] / df[rev_col], np.nan)
    df["roas_calc"] = np.where((cost_col and df[cost_col] > 0), df[rev_col] / df[cost_col], np.nan) if (cost_col and rev_col) else np.nan

    return df


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
            if tdf is None or not isinstance(tdf, pd.DataFrame) or tdf.empty:
                continue
            safe_name = (sheet[:31]).strip() or "Sheet"
            tdf.to_excel(writer, index=False, sheet_name=safe_name)
    bio.seek(0)
    return bio.getvalue()


# ============================
# 6) MoM (Ads): sinal primário GMV e ação "diminuir ROAS"
# ============================
def mom_signal_reasons_action_ads(
    gmv_prev: float, gmv_curr: float,
    imp_prev: float, imp_curr: float,
    clk_prev: float, clk_curr: float,
    ctr_prev: float, ctr_curr: float,
    cvr_prev: float, cvr_curr: float,
    roas_prev: float, roas_curr: float,
    roas_bom_threshold: float,
) -> tuple[str, str, str]:
    """
    - Sinal primário: GMV caiu vs mês anterior.
    - Motivos: o que caiu junto.
    - Ação: o que fazer (inclui "DIMINUIR ROAS" quando já performou antes e caiu agora).
    """
    # sinal primário
    if pd.notna(gmv_prev) and gmv_prev > 0 and pd.notna(gmv_curr):
        if gmv_curr < gmv_prev:
            sinal = "🔴"
        elif gmv_curr > gmv_prev:
            sinal = "🟢"
        else:
            sinal = "🟡"
    else:
        sinal = "🟡"

    motivos = []
    acao = "Monitorar"

    if sinal == "🔴":
        caiu_imp = pd.notna(imp_prev) and imp_prev > 0 and pd.notna(imp_curr) and imp_curr < imp_prev
        caiu_clk = pd.notna(clk_prev) and clk_prev > 0 and pd.notna(clk_curr) and clk_curr < clk_prev
        caiu_ctr = pd.notna(ctr_prev) and pd.notna(ctr_curr) and ctr_curr < ctr_prev
        caiu_cvr = pd.notna(cvr_prev) and pd.notna(cvr_curr) and cvr_curr < cvr_prev

        if caiu_imp:
            motivos.append("Impressões caíram")
        if caiu_clk:
            motivos.append("Cliques caíram")
        if caiu_ctr:
            motivos.append("CTR caiu")
        if caiu_cvr:
            motivos.append("CVR caiu")

        # ✅ regra Brenno: se no período anterior performou bem e caiu → diminuir ROAS
        ja_performou_bem = (
            pd.notna(gmv_prev) and gmv_prev > 0
            and pd.notna(roas_prev) and roas_prev >= roas_bom_threshold
        )
        if ja_performou_bem:
            acao = "DIMINUIR ROAS (afrouxar meta p/ recuperar entrega)"
            motivos.append("Antes performou bem (ROAS anterior bom)")

        # se não entrou no diminuir ROAS, decide pela “lei” de causa
        if acao == "Monitorar":
            if caiu_imp:
                acao = "Aumentar entrega (orçamento/lance/estrutura)"
            elif caiu_ctr:
                acao = "Preço + cauda longa + imagem"
            elif caiu_cvr:
                acao = "Copy + gatilhos + oferta/landing"
            elif caiu_clk:
                acao = "Revisar oferta/imagem/palavras"

    return sinal, (" | ".join(motivos) if motivos else ""), acao


# ============================
# 7) UI
# ============================
st.title("Shopee Ads – Campanhas + Loja (TACOS + Oportunidades + Mês x Mês)")

with st.sidebar:
    st.header("Configuração das regras")

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

    st.subheader("MoM: regra 'diminuir ROAS'")
    roas_bom = st.number_input("ROAS anterior considerado 'bom'", value=4.0, step=0.1)

    st.divider()
    st.subheader("Filtros")
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões baixas (oportunidade)", value=300, step=50)
    dominance_spend_share = st.slider("Dominância de gasto no grupo (%)", min_value=50, max_value=95, value=70)

    st.divider()
    st.subheader("Oportunidades (Loja)")
    min_revenue_candidate = st.number_input("Faturamento mínimo p/ sugerir Ads (R$)", value=1000.0, step=100.0)
    drop_alert_pct = st.number_input("Queda de faturamento p/ alerta (%)", value=20.0, step=5.0)


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
    """
    Retorna (bolinha, o_que_fazer) por anúncio/produto.
    """
    ctr_cls = classify_ctr(ctr) if pd.notna(ctr) else "n/a"
    cvr_cls = classify_cvr(cvr) if pd.notna(cvr) else "n/a"
    worst = worst_class(ctr_cls, cvr_cls)
    bolinha = dot_from_class(worst)

    actions = []
    if ctr_cls == "vermelho":
        actions.append("CTR ruim → preço + cauda longa + imagem")
    elif ctr_cls == "amarelo":
        actions.append("CTR ok → otimizar preço + cauda longa + imagem")

    if cvr_cls == "vermelho":
        actions.append("CVR ruim → copy + gatilhos + oferta/landing")
    elif cvr_cls == "amarelo":
        actions.append("CVR médio → melhorar copy + gatilhos")

    if pd.notna(imps) and imps <= low_imp_threshold:
        actions.append("Pouca impressão → mover/criar campanha dedicada (orçamento/entrega)")

    if not actions:
        actions.append("Ok → manter e monitorar")

    return bolinha, " | ".join(actions)


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
**Resumo**
- Loja: usa **Pedidos pagos** + coluna **Vendas (BRL)**.
- Ads: CSV(s) **Dados do grupo de anúncios**.
- Linha **sem ID do produto** = **TOTAL da campanha** (mesma visualização do seu print).
- Tabs incluem: **Campanhas**, **Oportunidades**, **Mês x mês** (Campanhas e Anúncios), e **Produtos da Loja mês x mês**.
"""
)

# ============================
# 8) Upload Ads - mês analisado e mês anterior
# ============================
st.header("Uploads – Ads (mês analisado + mês anterior)")

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
    imp_col_prev = "Impressões" if "Impressões" in df_prev.columns else ("Impressões do Produto" if "Impressões do Produto" in df_prev.columns else None)
    clk_col_prev = "Cliques" if "Cliques" in df_prev.columns else ("Cliques de Produtos" if "Cliques de Produtos" in df_prev.columns else None)
    rev_col_prev = "GMV" if "GMV" in df_prev.columns else None
    cost_col_prev = "Despesas" if "Despesas" in df_prev.columns else ("Custo" if "Custo" in df_prev.columns else None)

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

if id_col is None:
    st.error("Não encontrei a coluna 'ID do produto' no CSV. Sem isso não dá para separar TOTAL (campanha) de anúncios.")


# ============================
# 9) Upload Vendas (Loja) - producttraffic (mês atual e anterior)
# ============================
st.divider()
st.header("Uploads – Loja (Pedidos pagos) (mês anterior + mês analisado)")

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
# 10) KPI Loja (topo)
# ============================
faturamento_anterior = None
faturamento_atual = None

if df_sales_prev is not None and "Vendas (BRL)" in df_sales_prev.columns:
    faturamento_anterior = float(np.nansum(numeric_series_from(df_sales_prev, "Vendas (BRL)")))

if df_sales_curr is not None and "Vendas (BRL)" in df_sales_curr.columns:
    faturamento_atual = float(np.nansum(numeric_series_from(df_sales_curr, "Vendas (BRL)")))

kA, kB, kC = st.columns(3)
kA.metric("Loja – Faturamento mês anterior (pagos)", fmt_brl(faturamento_anterior) if faturamento_anterior is not None else "—")
kB.metric("Loja – Faturamento mês analisado (pagos)", fmt_brl(faturamento_atual) if faturamento_atual is not None else "—")

diff_txt = "—"
if faturamento_anterior is not None and faturamento_anterior > 0 and faturamento_atual is not None:
    delta = (faturamento_atual / faturamento_anterior) - 1
    diff_txt = f"{fmt_brl(faturamento_atual - faturamento_anterior)}  ({fmt_pct(delta, digits=1)})"
kC.metric("Diferença (R$ e %)", diff_txt)

if sheet_prev or sheet_curr:
    st.caption(f"ABA usada: anterior **{sheet_prev or '—'}** | analisado **{sheet_curr or '—'}** (sempre 'Pedido pago')")

# ============================
# 11) KPI ADS (mês analisado): gasto | gmv | ACOS | ROAS | TACOS
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

st.subheader("Ads – KPIs (mês analisado)")
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Gasto Ads", fmt_brl(ads_spend_total) if pd.notna(ads_spend_total) else "—")
k2.metric("GMV Ads", fmt_brl(ads_gmv_total) if pd.notna(ads_gmv_total) else "—")
k3.metric("ACOS", fmt_pct(acos_total) if pd.notna(acos_total) else "—")
k4.metric("ROAS", (str(round(roas_total, 2)).replace(".", ",")) if pd.notna(roas_total) else "—")
k5.metric("TACOS", fmt_pct(tacos_total) if pd.notna(tacos_total) else "—")


# ============================
# 12) Preparos: TOTAL campanha + anúncios
# ============================
def split_campaign_total_and_ads(df: pd.DataFrame, id_col_: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    id_clean = df[id_col_].astype(str).str.strip()
    is_total = id_clean.isin(["", "-", "nan", "None"])
    camp_total = df[is_total].copy()
    ads_rows = df[~is_total].copy()
    ads_rows["prod_key"] = ads_rows[id_col_].astype(str).str.strip().apply(normalize_product_id)
    return camp_total, ads_rows


camp_total_curr, ads_rows_curr = split_campaign_total_and_ads(df_all, id_col)
camp_total_prev, ads_rows_prev = (split_campaign_total_and_ads(df_prev, id_col) if not df_prev.empty else (pd.DataFrame(), pd.DataFrame()))

# chaves de produto em ads (mês atual)
ads_keys = set(ads_rows_curr["prod_key"].dropna().astype(str).tolist()) if not ads_rows_curr.empty else set()


# ============================
# 13) Tabs principais
# ============================
st.divider()
tabs = st.tabs([
    "Campanhas (decisão)",
    "Oportunidades",
    "Ads mês x mês (Campanhas)",
    "Ads mês x mês (Anúncios)",
    "Produtos (Loja) mês x mês",
])


# ============================
# Tab 1) Campanhas (decisão) - mês atual
# ============================
with tabs[0]:
    st.subheader("Campanhas – decisão (CVR + ACOS + TACOS)")

    if camp_total_curr.empty:
        st.warning("Não encontrei linhas TOTAL (sem ID) no mês analisado.")
    else:
        grp = camp_total_curr.groupby(["Campanha_key", "Campanha"], dropna=False).agg({
            imp_col: "sum",
            clk_col: "sum",
            orders_col: "sum",
            rev_col: "sum",
            cost_col: "sum",
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

        grp[["ACOS_bolinha", "ACOS_status"]] = grp["ACOS"].apply(lambda v: pd.Series(semaforo_ideal_limite(v, acos_ideal, acos_limite)))
        grp[["TACOS_bolinha", "TACOS_status"]] = grp["TACOS"].apply(lambda v: pd.Series(semaforo_ideal_limite(v, tacos_ideal, tacos_limite)))

        def decisao_final(row) -> tuple[str, str]:
            cvr_cls = row["CVR_cor"]
            acos_st = row["ACOS_status"]
            tacos_st = row["TACOS_status"]

            if acos_st == "ajustar" or tacos_st == "ajustar":
                return "🔴", "Ajustar (ACOS/TACOS acima do limite)"
            if cvr_cls == "vermelho":
                return "🔴", "Ajustar (CVR ruim)"
            if cvr_cls == "verde" and acos_st == "ótimo" and tacos_st == "ótimo":
                return "🟢", "Aumentar orçamento (CVR ótimo + ACOS/TACOS ótimos)"
            if cvr_cls == "verde" and (acos_st == "monitorar" or tacos_st == "monitorar"):
                return "🟡", "Monitorar (CVR ótimo, mas ACOS/TACOS pedem atenção)"
            if cvr_cls == "amarelo" and acos_st == "ótimo" and tacos_st == "ótimo":
                return "🟡", "Manter (CVR bom)"
            return "🟡", "Monitorar"

        grp[["Sinal", "O que fazer"]] = grp.apply(lambda r: pd.Series(decisao_final(r)), axis=1)

        disp = grp.copy()
        disp["GMV"] = disp["GMV"].apply(fmt_brl)
        disp["Despesas"] = disp["Despesas"].apply(fmt_brl)
        disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS"] = disp["ACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["TACOS"] = disp["TACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ROAS"] = disp["ROAS"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        # ✅ poucos cabeçalhos (só o necessário)
        cols = [
            "Sinal",
            "Campanha",
            "O que fazer",
            "GMV",
            "Despesas",
            "CVR",
            "ACOS_bolinha",
            "ACOS",
            "TACOS_bolinha",
            "TACOS",
            "ROAS",
        ]
        st.dataframe(disp[cols].sort_values(by="Campanha"), use_container_width=True, hide_index=True)


# ============================
# Tab 2) Oportunidades
# ============================
with tabs[1]:
    st.subheader("Oportunidades")

    t1, t2, t3 = st.tabs([
        "Loja → Candidatos a Ads (fora do Ads)",
        "Ads → Bons com pouca impressão",
        "Ads → Gastando sem converter",
    ])

    # 2.1) Loja candidatos a Ads
    with t1:
        if df_sales_curr is None or "Vendas (BRL)" not in df_sales_curr.columns:
            st.info("Suba o Excel do mês analisado (Pedidos pagos) para liberar candidatos.")
        else:
            sales_id_col = find_col(df_sales_curr, exact=["ID do Item"], contains=["id do item"])
            prod_name_col = find_col(df_sales_curr, exact=["Produto"], contains=["produto"])
            imp_store_col = find_col(df_sales_curr, exact=["Impressões de Produto"], contains=["impress"])
            clk_store_col = find_col(df_sales_curr, exact=["Cliques Por Produto"], contains=["clique"])
            ctr_store_col = find_col(df_sales_curr, exact=["CTR"], contains=["ctr"])
            cvr_store_col = find_col(df_sales_curr, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
            ord_store_col = find_col(df_sales_curr, exact=["Pedidos"], contains=["pedido"])

            if sales_id_col is None:
                st.warning("Não encontrei 'ID do Item' na aba de pedidos pagos.")
            else:
                s = df_sales_curr.copy()
                s["prod_key"] = s[sales_id_col].astype(str).str.strip().apply(normalize_product_id)
                s["Produto"] = s[prod_name_col].astype(str).str.strip() if prod_name_col else ""
                s["Faturamento"] = numeric_series_from(s, "Vendas (BRL)")
                s["Impressões"] = numeric_series_from(s, imp_store_col) if imp_store_col else np.nan
                s["Cliques"] = numeric_series_from(s, clk_store_col) if clk_store_col else np.nan
                s["CTR"] = s[ctr_store_col].apply(parse_percent) if ctr_store_col else np.where(s["Impressões"].fillna(0) > 0, s["Cliques"].fillna(0) / s["Impressões"].fillna(0), np.nan)
                s["CVR"] = s[cvr_store_col].apply(parse_percent) if cvr_store_col else np.nan
                s["Pedidos"] = numeric_series_from(s, ord_store_col) if ord_store_col else np.nan

                agg = s.groupby(["prod_key"], dropna=False).agg({
                    "Produto": "first",
                    "Faturamento": "sum",
                    "Impressões": "sum",
                    "Cliques": "sum",
                    "CTR": "mean",
                    "CVR": "mean",
                    "Pedidos": "sum",
                }).reset_index()

                # fora do Ads + com faturamento mínimo
                cand = agg[~agg["prod_key"].astype(str).isin(ads_keys)].copy()
                cand = cand[cand["Faturamento"] >= float(min_revenue_candidate)].copy()

                if cand.empty:
                    st.info("Nenhum candidato (fora do Ads + faturamento mínimo).")
                else:
                    cand["Sinal"] = ""
                    cand["O que fazer"] = ""
                    for i, r in cand.iterrows():
                        sinal, acao = action_for_store_row(
                            r.get("CTR", np.nan),
                            r.get("CVR", np.nan),
                            r.get("Impressões", np.nan),
                            low_imp_threshold=int(low_impressions_threshold),
                            not_in_ads=True,
                            revenue=float(r.get("Faturamento", 0.0)) if pd.notna(r.get("Faturamento", np.nan)) else None,
                        )
                        cand.at[i, "Sinal"] = sinal
                        cand.at[i, "O que fazer"] = acao

                    disp = cand.copy()
                    disp["Faturamento"] = disp["Faturamento"].apply(fmt_brl)
                    disp["Impressões"] = disp["Impressões"].apply(fmt_int)
                    disp["Cliques"] = disp["Cliques"].apply(fmt_int)
                    disp["Pedidos"] = disp["Pedidos"].apply(fmt_int)
                    disp["CTR"] = disp["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                    disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

                    cols = ["Sinal", "prod_key", "Produto", "Faturamento", "Pedidos", "Impressões", "Cliques", "CTR", "CVR", "O que fazer"]
                    st.dataframe(disp[cols].sort_values(by="Faturamento", ascending=False), use_container_width=True, hide_index=True)

    # 2.2) Ads bons com pouca impressão (mês atual)
    with t2:
        if ads_rows_curr.empty:
            st.info("Não há linhas de anúncios (somente TOTAL).")
        else:
            df = ads_rows_curr.copy()
            df["Impressões"] = pd.to_numeric(df[imp_col], errors="coerce") if imp_col else np.nan
            df["Cliques"] = pd.to_numeric(df[clk_col], errors="coerce") if clk_col else np.nan
            df["CTR"] = pd.to_numeric(df["ctr_calc"], errors="coerce")
            df["CVR"] = pd.to_numeric(df["cvr_calc"], errors="coerce")
            df["GMV"] = pd.to_numeric(df[rev_col], errors="coerce") if rev_col else np.nan
            df["Despesas"] = pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan
            df["ACOS"] = np.where(df["GMV"].fillna(0) > 0, df["Despesas"].fillna(0) / df["GMV"].fillna(0), np.nan)

            # oportunidade: pouca impressão mas CTR/CVR bons
            good = df[
                (df["Impressões"].fillna(0) <= int(low_impressions_threshold))
                & (
                    (df["CTR"].fillna(0) >= (ctr_bom_min_pct / 100.0))
                    | (df["CVR"].fillna(0) >= (cvr_bom_min_pct / 100.0))
                )
            ].copy()

            if good.empty:
                st.info("Nenhuma oportunidade no critério (boa performance + baixa impressão).")
            else:
                good["Sinal"] = ""
                good["O que fazer"] = ""
                for i, r in good.iterrows():
                    sinal, acao = action_for_row_new(r.get("CTR", np.nan), r.get("CVR", np.nan), r.get("Impressões", np.nan), low_imp_threshold=int(low_impressions_threshold))
                    good.at[i, "Sinal"] = sinal
                    good.at[i, "O que fazer"] = acao

                disp = good.copy()
                if name_col and name_col in disp.columns:
                    disp["Produto/Anúncio"] = disp[name_col].astype(str)
                else:
                    disp["Produto/Anúncio"] = ""
                disp["ID"] = disp["prod_key"].astype(str)

                disp["Impressões"] = disp["Impressões"].apply(fmt_int)
                disp["Cliques"] = disp["Cliques"].apply(fmt_int)
                disp["CTR"] = disp["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                disp["GMV"] = disp["GMV"].apply(fmt_brl)
                disp["Despesas"] = disp["Despesas"].apply(fmt_brl)
                disp["ACOS"] = disp["ACOS"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

                cols = ["Sinal", "Campanha", "Produto/Anúncio", "ID", "Impressões", "Cliques", "CTR", "CVR", "GMV", "Despesas", "ACOS", "O que fazer"]
                st.dataframe(disp[cols].sort_values(by="Campanha"), use_container_width=True, hide_index=True)

    # 2.3) Gastando sem converter (mês atual)
    with t3:
        if ads_rows_curr.empty:
            st.info("Não há linhas de anúncios (somente TOTAL).")
        else:
            df = ads_rows_curr.copy()
            df["Cliques"] = pd.to_numeric(df[clk_col], errors="coerce") if clk_col else np.nan
            df["Pedidos"] = pd.to_numeric(df[orders_col], errors="coerce") if orders_col else np.nan
            df["Despesas"] = pd.to_numeric(df[cost_col], errors="coerce") if cost_col else np.nan
            df["Impressões"] = pd.to_numeric(df[imp_col], errors="coerce") if imp_col else np.nan
            df["CTR"] = pd.to_numeric(df["ctr_calc"], errors="coerce")
            df["CVR"] = pd.to_numeric(df["cvr_calc"], errors="coerce")

            wasting = df[
                (df["Pedidos"].fillna(0) == 0)
                & (
                    (df["Cliques"].fillna(0) >= float(min_clicks_eval))
                    | (df["Despesas"].fillna(0) >= float(min_spend_no_conv))
                )
            ].copy()

            if wasting.empty:
                st.info("Nenhum item gastando sem converter no critério.")
            else:
                wasting["Sinal"] = "🔴"
                wasting["O que fazer"] = "Pausar/remover (gastando sem converter)"

                disp = wasting.copy()
                disp["Produto/Anúncio"] = disp[name_col].astype(str) if (name_col and name_col in disp.columns) else ""
                disp["ID"] = disp["prod_key"].astype(str)

                disp["Impressões"] = disp["Impressões"].apply(fmt_int)
                disp["Cliques"] = disp["Cliques"].apply(fmt_int)
                disp["CTR"] = disp["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                disp["Pedidos"] = disp["Pedidos"].apply(fmt_int)
                disp["CVR"] = disp["CVR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
                disp["Despesas"] = disp["Despesas"].apply(fmt_brl)

                cols = ["Sinal", "Campanha", "Produto/Anúncio", "ID", "Impressões", "Cliques", "CTR", "Pedidos", "CVR", "Despesas", "O que fazer"]
                st.dataframe(disp[cols].sort_values(by="Despesas", ascending=False), use_container_width=True, hide_index=True)


# ============================
# Tab 3) Ads mês x mês – Campanhas (poucos cabeçalhos + o que fazer)
# ============================
with tabs[2]:
    st.subheader("Ads mês x mês – Campanhas")

    if df_prev.empty or camp_total_prev.empty:
        st.info("Suba os CSVs do mês anterior para comparar campanhas.")
    else:
        def campaign_totals(df_tot: pd.DataFrame, imp_, clk_, ord_, gmv_, spend_, faturamento_total: float | None) -> pd.DataFrame:
            g = df_tot.groupby(["Campanha_key", "Campanha"], dropna=False).agg({
                imp_: "sum",
                clk_: "sum",
                ord_: "sum",
                gmv_: "sum",
                spend_: "sum",
            }).reset_index()

            g = g.rename(columns={imp_: "Impressões", clk_: "Cliques", ord_: "Pedidos", gmv_: "GMV", spend_: "Despesas"})
            g["CTR"] = np.where(g["Impressões"].fillna(0) > 0, g["Cliques"].fillna(0) / g["Impressões"].fillna(0), np.nan)
            g["CVR"] = np.where(g["Cliques"].fillna(0) > 0, g["Pedidos"].fillna(0) / g["Cliques"].fillna(0), np.nan)
            g["ACOS"] = np.where(g["GMV"].fillna(0) > 0, g["Despesas"].fillna(0) / g["GMV"].fillna(0), np.nan)
            g["ROAS"] = np.where(g["Despesas"].fillna(0) > 0, g["GMV"].fillna(0) / g["Despesas"].fillna(0), np.nan)
            g["TACOS"] = (g["Despesas"] / faturamento_total) if (faturamento_total is not None and faturamento_total > 0) else np.nan
            return g

        curr_camp = campaign_totals(camp_total_curr, imp_col, clk_col, orders_col, rev_col, cost_col, faturamento_atual)
        prev_camp = campaign_totals(camp_total_prev, imp_col_prev or imp_col, clk_col_prev or clk_col, orders_col_prev or orders_col, rev_col_prev or rev_col, cost_col_prev or cost_col, faturamento_anterior)

        mom = curr_camp.merge(prev_camp, on="Campanha_key", how="outer", suffixes=("_curr", "_prev"))
        mom["Campanha"] = mom["Campanha_curr"].fillna(mom["Campanha_prev"])

        mom["Δ_GMV_%"] = np.where(mom["GMV_prev"].fillna(0) > 0, (mom["GMV_curr"].fillna(0) / mom["GMV_prev"].fillna(0)) - 1, np.nan)

        mom["Sinal"] = ""
        mom["Motivos"] = ""
        mom["O que fazer"] = ""

        for i, r in mom.iterrows():
            sinal, motivos, acao = mom_signal_reasons_action_ads(
                r.get("GMV_prev", np.nan), r.get("GMV_curr", np.nan),
                r.get("Impressões_prev", np.nan), r.get("Impressões_curr", np.nan),
                r.get("Cliques_prev", np.nan), r.get("Cliques_curr", np.nan),
                r.get("CTR_prev", np.nan), r.get("CTR_curr", np.nan),
                r.get("CVR_prev", np.nan), r.get("CVR_curr", np.nan),
                r.get("ROAS_prev", np.nan), r.get("ROAS_curr", np.nan),
                float(roas_bom),
            )
            mom.at[i, "Sinal"] = sinal
            mom.at[i, "Motivos"] = motivos
            mom.at[i, "O que fazer"] = acao

        disp = mom.copy()
        disp["GMV ant"] = disp["GMV_prev"].apply(fmt_brl)
        disp["GMV atual"] = disp["GMV_curr"].apply(fmt_brl)
        disp["Δ GMV %"] = disp["Δ_GMV_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["Gasto atual"] = disp["Despesas_curr"].apply(fmt_brl)
        disp["CVR atual"] = disp["CVR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ACOS atual"] = disp["ACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["TACOS atual"] = disp["TACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ROAS atual"] = disp["ROAS_curr"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        cols = ["Sinal", "Campanha", "GMV ant", "GMV atual", "Δ GMV %", "Gasto atual", "CVR atual", "ACOS atual", "TACOS atual", "ROAS atual", "O que fazer", "Motivos"]
        st.dataframe(disp[cols].sort_values(by="Campanha"), use_container_width=True, hide_index=True)


# ============================
# Tab 4) Ads mês x mês – Anúncios (poucos cabeçalhos 'necessários' + o que fazer)
# ============================
with tabs[3]:
    st.subheader("Ads mês x mês – Anúncios/Produtos")

    if df_prev.empty or ads_rows_prev.empty:
        st.info("Suba os CSVs do mês anterior para comparar anúncios.")
    else:
        def agg_ads(df: pd.DataFrame, imp_, clk_, ord_, gmv_, spend_) -> pd.DataFrame:
            g = df.groupby(["Campanha_key", "Campanha", "prod_key"], dropna=False).agg({
                imp_: "sum",
                clk_: "sum",
                ord_: "sum",
                gmv_: "sum",
                spend_: "sum",
            }).reset_index()

            g = g.rename(columns={imp_: "Impressões", clk_: "Cliques", ord_: "Pedidos", gmv_: "GMV", spend_: "Despesas"})
            g["CTR"] = np.where(g["Impressões"].fillna(0) > 0, g["Cliques"].fillna(0) / g["Impressões"].fillna(0), np.nan)
            g["CVR"] = np.where(g["Cliques"].fillna(0) > 0, g["Pedidos"].fillna(0) / g["Cliques"].fillna(0), np.nan)
            g["ACOS"] = np.where(g["GMV"].fillna(0) > 0, g["Despesas"].fillna(0) / g["GMV"].fillna(0), np.nan)
            g["ROAS"] = np.where(g["Despesas"].fillna(0) > 0, g["GMV"].fillna(0) / g["Despesas"].fillna(0), np.nan)
            return g

        curr_a = agg_ads(ads_rows_curr, imp_col, clk_col, orders_col, rev_col, cost_col)
        prev_a = agg_ads(ads_rows_prev, imp_col_prev or imp_col, clk_col_prev or clk_col, orders_col_prev or orders_col, rev_col_prev or rev_col, cost_col_prev or cost_col)

        mom = curr_a.merge(prev_a, on=["Campanha_key", "prod_key"], how="outer", suffixes=("_curr", "_prev"))
        mom["Campanha"] = mom["Campanha_curr"].fillna(mom["Campanha_prev"])
        mom["ID"] = mom["prod_key"].astype(str)

        # nome (do mês atual se existir)
        if name_col and name_col in ads_rows_curr.columns:
            nm = ads_rows_curr.groupby(["Campanha_key", "prod_key"])[name_col].first().reset_index().rename(columns={name_col: "Produto/Anúncio"})
            mom = mom.merge(nm, on=["Campanha_key", "prod_key"], how="left")
        else:
            mom["Produto/Anúncio"] = ""

        mom["Δ_GMV_%"] = np.where(mom["GMV_prev"].fillna(0) > 0, (mom["GMV_curr"].fillna(0) / mom["GMV_prev"].fillna(0)) - 1, np.nan)

        mom["Sinal"] = ""
        mom["Motivos"] = ""
        mom["O que fazer"] = ""

        for i, r in mom.iterrows():
            sinal, motivos, acao = mom_signal_reasons_action_ads(
                r.get("GMV_prev", np.nan), r.get("GMV_curr", np.nan),
                r.get("Impressões_prev", np.nan), r.get("Impressões_curr", np.nan),
                r.get("Cliques_prev", np.nan), r.get("Cliques_curr", np.nan),
                r.get("CTR_prev", np.nan), r.get("CTR_curr", np.nan),
                r.get("CVR_prev", np.nan), r.get("CVR_curr", np.nan),
                r.get("ROAS_prev", np.nan), r.get("ROAS_curr", np.nan),
                float(roas_bom),
            )
            # complementa com qualidade mês atual (lei CTR/CVR + baixa impressão)
            q_sinal, q_acao = action_for_row_new(r.get("CTR_curr", np.nan), r.get("CVR_curr", np.nan), r.get("Impressões_curr", np.nan), low_imp_threshold=int(low_impressions_threshold))
            acao_final = acao
            if acao_final == "Monitorar" and q_acao:
                acao_final = q_acao

            mom.at[i, "Sinal"] = sinal
            mom.at[i, "Motivos"] = motivos
            mom.at[i, "O que fazer"] = acao_final

        disp = mom.copy()
        disp["GMV ant"] = disp["GMV_prev"].apply(fmt_brl)
        disp["GMV atual"] = disp["GMV_curr"].apply(fmt_brl)
        disp["Δ GMV %"] = disp["Δ_GMV_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")

        disp["Imp atual"] = disp["Impressões_curr"].apply(fmt_int)
        disp["CTR atual"] = disp["CTR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CVR atual"] = disp["CVR_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

        disp["ACOS atual"] = disp["ACOS_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["ROAS atual"] = disp["ROAS_curr"].apply(lambda v: "" if pd.isna(v) else str(round(v, 2)).replace(".", ","))

        cols = ["Sinal", "Campanha", "Produto/Anúncio", "ID", "GMV ant", "GMV atual", "Δ GMV %", "Imp atual", "CTR atual", "CVR atual", "ACOS atual", "ROAS atual", "O que fazer", "Motivos"]
        st.dataframe(disp[cols].sort_values(by=["Campanha", "Produto/Anúncio"]), use_container_width=True, hide_index=True)


# ============================
# Tab 5) Produtos (Loja) mês x mês (queda: sinal primário faturamento + o que fazer)
# ============================
with tabs[4]:
    st.subheader("Produtos (Loja) – mês x mês (sinal primário = faturamento caiu)")

    if df_sales_curr is None or "Vendas (BRL)" not in df_sales_curr.columns:
        st.info("Suba o Excel do mês analisado (Pedidos pagos).")
    else:
        sales_id_col = find_col(df_sales_curr, exact=["ID do Item"], contains=["id do item"])
        prod_name_col = find_col(df_sales_curr, exact=["Produto"], contains=["produto"])
        imp_store_col = find_col(df_sales_curr, exact=["Impressões de Produto"], contains=["impress"])
        clk_store_col = find_col(df_sales_curr, exact=["Cliques Por Produto"], contains=["clique"])
        ctr_store_col = find_col(df_sales_curr, exact=["CTR"], contains=["ctr"])
        cvr_store_col = find_col(df_sales_curr, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
        ord_store_col = find_col(df_sales_curr, exact=["Pedidos"], contains=["pedido"])

        s = df_sales_curr.copy()
        s["prod_key"] = s[sales_id_col].astype(str).str.strip().apply(normalize_product_id)
        s["Produto"] = s[prod_name_col].astype(str).str.strip() if prod_name_col else ""
        s["rev_curr"] = numeric_series_from(s, "Vendas (BRL)")
        s["imp_curr"] = numeric_series_from(s, imp_store_col) if imp_store_col else np.nan
        s["clk_curr"] = numeric_series_from(s, clk_store_col) if clk_store_col else np.nan
        s["ctr_curr"] = s[ctr_store_col].apply(parse_percent) if ctr_store_col else np.where(s["imp_curr"].fillna(0) > 0, s["clk_curr"].fillna(0) / s["imp_curr"].fillna(0), np.nan)
        s["cvr_curr"] = s[cvr_store_col].apply(parse_percent) if cvr_store_col else np.nan
        s["ord_curr"] = numeric_series_from(s, ord_store_col) if ord_store_col else np.nan

        curr_agg = s.groupby(["prod_key"], dropna=False).agg({
            "Produto": "first",
            "rev_curr": "sum",
            "imp_curr": "sum",
            "clk_curr": "sum",
            "ctr_curr": "mean",
            "cvr_curr": "mean",
            "ord_curr": "sum",
        }).reset_index()

        if df_sales_prev is not None and "Vendas (BRL)" in df_sales_prev.columns:
            sp = df_sales_prev.copy()
            sales_id_prev = find_col(sp, exact=["ID do Item"], contains=["id do item"])
            prod_name_prev = find_col(sp, exact=["Produto"], contains=["produto"])
            imp_prev_col = find_col(sp, exact=["Impressões de Produto"], contains=["impress"])
            clk_prev_col = find_col(sp, exact=["Cliques Por Produto"], contains=["clique"])
            ctr_prev_col = find_col(sp, exact=["CTR"], contains=["ctr"])
            cvr_prev_col = find_col(sp, exact=["Taxa de Conversão de Pedidos"], contains=["taxa de convers", "convers"])
            ord_prev_col = find_col(sp, exact=["Pedidos"], contains=["pedido"])

            sp["prod_key"] = sp[sales_id_prev].astype(str).str.strip().apply(normalize_product_id)
            sp["rev_prev"] = numeric_series_from(sp, "Vendas (BRL)")
            sp["imp_prev"] = numeric_series_from(sp, imp_prev_col) if imp_prev_col else np.nan
            sp["clk_prev"] = numeric_series_from(sp, clk_prev_col) if clk_prev_col else np.nan
            sp["ctr_prev"] = sp[ctr_prev_col].apply(parse_percent) if ctr_prev_col else np.where(sp["imp_prev"].fillna(0) > 0, sp["clk_prev"].fillna(0) / sp["imp_prev"].fillna(0), np.nan)
            sp["cvr_prev"] = sp[cvr_prev_col].apply(parse_percent) if cvr_prev_col else np.nan
            sp["ord_prev"] = numeric_series_from(sp, ord_prev_col) if ord_prev_col else np.nan

            prev_agg = sp.groupby(["prod_key"], dropna=False).agg({
                "rev_prev": "sum",
                "imp_prev": "sum",
                "clk_prev": "sum",
                "ctr_prev": "mean",
                "cvr_prev": "mean",
                "ord_prev": "sum",
            }).reset_index()

            mom = curr_agg.merge(prev_agg, on="prod_key", how="left")
            for c in ["rev_prev", "imp_prev", "clk_prev", "ctr_prev", "cvr_prev", "ord_prev"]:
                mom[c] = pd.to_numeric(mom[c], errors="coerce").fillna(0.0)
        else:
            mom = curr_agg.copy()
            mom["rev_prev"] = 0.0
            mom["imp_prev"] = 0.0
            mom["clk_prev"] = 0.0
            mom["ctr_prev"] = np.nan
            mom["cvr_prev"] = np.nan
            mom["ord_prev"] = 0.0

        mom["Δ_rev_%"] = np.where(mom["rev_prev"] > 0, (mom["rev_curr"] / mom["rev_prev"]) - 1, np.nan)

        mom["Sinal"] = ""
        mom["O que fazer"] = ""
        mom["Motivos"] = ""

        for i, r in mom.iterrows():
            prev_rev = r.get("rev_prev", np.nan)
            curr_rev = r.get("rev_curr", np.nan)

            if pd.notna(prev_rev) and prev_rev > 0 and pd.notna(curr_rev) and curr_rev < prev_rev:
                mom.at[i, "Sinal"] = "🔴"
                reasons = []
                if pd.notna(r.get("imp_prev", np.nan)) and r["imp_prev"] > 0 and pd.notna(r.get("imp_curr", np.nan)) and r["imp_curr"] < r["imp_prev"]:
                    reasons.append("Impressões caíram → colocar/fortalecer Ads")
                if pd.notna(r.get("clk_prev", np.nan)) and r["clk_prev"] > 0 and pd.notna(r.get("clk_curr", np.nan)) and r["clk_curr"] < r["clk_prev"]:
                    reasons.append("Cliques caíram → revisar oferta/imagem/palavras")
                if pd.notna(r.get("ctr_prev", np.nan)) and pd.notna(r.get("ctr_curr", np.nan)) and r["ctr_curr"] < r["ctr_prev"]:
                    reasons.append("CTR caiu → preço/cauda longa/imagem")
                if pd.notna(r.get("cvr_prev", np.nan)) and pd.notna(r.get("cvr_curr", np.nan)) and r["cvr_curr"] < r["cvr_prev"]:
                    reasons.append("CVR caiu → copy/gatilhos")
                mom.at[i, "Motivos"] = " | ".join(reasons) if reasons else "Queda sem driver claro"

                # o que fazer (prático)
                if any("Impressões caíram" in x for x in reasons):
                    mom.at[i, "O que fazer"] = "Colocar/fortalecer Ads (recuperar entrega)"
                elif any("CTR caiu" in x for x in reasons):
                    mom.at[i, "O que fazer"] = "Preço + cauda longa + imagem"
                elif any("CVR caiu" in x for x in reasons):
                    mom.at[i, "O que fazer"] = "Copy + gatilhos + oferta/landing"
                else:
                    mom.at[i, "O que fazer"] = "Investigar concorrência/estoque/preço"

            elif pd.notna(prev_rev) and prev_rev > 0 and pd.notna(curr_rev) and curr_rev > prev_rev:
                mom.at[i, "Sinal"] = "🟢"
                mom.at[i, "O que fazer"] = "Manter e escalar (se estiver sem Ads, considerar Ads)"
                mom.at[i, "Motivos"] = "Cresceu vs mês anterior"
            else:
                mom.at[i, "Sinal"] = "🟡"
                mom.at[i, "O que fazer"] = "Monitorar"
                mom.at[i, "Motivos"] = "Sem base anterior"

        disp = mom.copy()
        disp["Fat ant"] = disp["rev_prev"].apply(fmt_brl)
        disp["Fat atual"] = disp["rev_curr"].apply(fmt_brl)
        disp["Δ Fat %"] = disp["Δ_rev_%"].apply(lambda v: fmt_pct(v, digits=1) if pd.notna(v) else "")
        disp["Imp atual"] = disp["imp_curr"].apply(fmt_int)
        disp["CTR atual"] = disp["ctr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")
        disp["CVR atual"] = disp["cvr_curr"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

        cols = ["Sinal", "prod_key", "Produto", "Fat ant", "Fat atual", "Δ Fat %", "Imp atual", "CTR atual", "CVR atual", "O que fazer", "Motivos"]
        st.dataframe(disp[cols].sort_values(by=["Sinal", "Fat atual"], ascending=[True, False]), use_container_width=True, hide_index=True)


# ============================
# 15) BAIXAR TUDO (Excel + HTML + PDF)
# ============================
st.divider()
st.header("Baixar relatório completo")

# ✅ Se ainda não existir no topo do arquivo, crie:
# REPORT_TABLES: dict[str, pd.DataFrame] = {}
# E sempre que você fizer st.dataframe(...), registre a tabela:
# REPORT_TABLES["Nome da seção"] = df_final.copy()

# --------------------------------
# A) Excel COMPLETO (bases + todas as views exibidas)
# --------------------------------
excel_tables: dict[str, pd.DataFrame] = {}

# Bases brutas
excel_tables["BASE_ADS_ATUAL"] = df_all.copy()
if not df_prev.empty:
    excel_tables["BASE_ADS_ANTERIOR"] = df_prev.copy()

# Views (tabelas que aparecem na tela)
for name, df in REPORT_TABLES.items():
    if isinstance(df, pd.DataFrame) and not df.empty:
        sheet = f"VIEW_{name}"
        excel_tables[sheet] = df.copy()

xlsx_bytes = make_excel_export(excel_tables)

st.download_button(
    "⬇️ Baixar Excel COMPLETO (tudo)",
    data=xlsx_bytes,
    file_name="shopee_relatorio_completo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    use_container_width=True,
)

# --------------------------------
# B) HTML INTERATIVO (todas as views) – ordena / filtra / busca
# --------------------------------
def make_full_html_report(title: str, tables: dict[str, pd.DataFrame]) -> str:
    head = f"""
<!DOCTYPE html>
<html lang="pt-br">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">
<style>
  body {{ font-family: Arial, sans-serif; margin: 24px; }}
  h1 {{ margin-bottom: 8px; }}
  h2 {{ margin-top: 28px; }}
  .muted {{ color:#666; font-size: 12px; }}
  table.dataTable {{ width: 100% !important; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p class="muted">Relatório completo exportado do app Shopee Auditor.</p>
"""
    body = ""
    idx = 0
    for name, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
        idx += 1
        tid = f"tbl_{idx}"
        body += f"<h2>{name}</h2>\n"
        body += df.to_html(index=False, escape=True, table_id=tid, classes="display compact", border=0)
        body += "\n"

    tail = """
<script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>
<script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>
<script>
  $(document).ready(function(){
    $('table.display').DataTable({
      pageLength: 25,
      lengthMenu: [10, 25, 50, 100],
      order: [],
      language: {
        search: "Buscar:",
        lengthMenu: "Mostrar _MENU_ linhas",
        info: "Mostrando _START_ a _END_ de _TOTAL_",
        paginate: { previous: "Anterior", next: "Próximo" }
      }
    });
  });
</script>
</body>
</html>
"""
    return head + body + tail

if REPORT_TABLES:
    html_report = make_full_html_report("Relatório Shopee Auditor – Completo", REPORT_TABLES)
    st.download_button(
        "⬇️ Baixar HTML INTERATIVO (tudo)",
        data=html_report.encode("utf-8"),
        file_name="shopee_relatorio_completo.html",
        mime="text/html",
        use_container_width=True,
    )
    st.caption("Obs: HTML interativo usa internet (CDN). Se quiser 100% offline, posso embutir JS/CSS no arquivo.")
else:
    st.info("Ainda não há tabelas registradas em REPORT_TABLES. Registre as tabelas finais para habilitar HTML/PDF.")

# --------------------------------
# C) PDF EXECUTIVO (todas as views, com resumo por tabela)
# --------------------------------
def make_pdf_exec(title: str, tables: dict[str, pd.DataFrame]) -> bytes:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm

    def _safe_str(x) -> str:
        if x is None:
            return ""
        return str(x)

    bio = BytesIO()
    c = canvas.Canvas(bio, pagesize=landscape(A4))
    w, h = landscape(A4)

    c.setFont("Helvetica-Bold", 18)
    c.drawString(2*cm, h - 2*cm, title)

    y = h - 3.2*cm

    for name, df in tables.items():
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        c.setFont("Helvetica-Bold", 12)
        c.drawString(2*cm, y, name)
        y -= 0.7*cm

        preview = df.head(12).copy()
        cols = list(preview.columns)[:10]
        preview = preview[cols]

        c.setFont("Helvetica", 8)
        text = c.beginText(2*cm, y)

        text.textLine(" | ".join([_safe_str(x) for x in cols]))
        for _, row in preview.iterrows():
            text.textLine(" | ".join([_safe_str(row[col])[:60] for col in cols]))

        c.drawText(text)

        y -= (len(preview) + 3) * 0.35*cm

        if y < 3*cm:
            c.showPage()
            y = h - 2.5*cm

    c.save()
    bio.seek(0)
    return bio.getvalue()

if REPORT_TABLES:
    try:
        pdf_bytes = make_pdf_exec("Relatório Shopee Auditor – Executivo", REPORT_TABLES)
        st.download_button(
            "⬇️ Baixar PDF EXECUTIVO",
            data=pdf_bytes,
            file_name="shopee_relatorio_executivo.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception:
        st.warning("Para gerar PDF no Streamlit Cloud, adicione `reportlab` no requirements.txt.")
