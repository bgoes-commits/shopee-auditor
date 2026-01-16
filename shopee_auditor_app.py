import re
import unicodedata
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Shopee Ads Auditor (por Grupo)", layout="wide")

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
# 2) Parsing robusto Shopee CSV
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
    for i, line in enumerate(lines[:300]):
        if line.startswith("#,") or line.startswith("#;"):
            return i
    for i, line in enumerate(lines[:400]):
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
        # meta geralmente vem com vírgula, mesmo quando o csv é ';'
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

    # garante numeric nas colunas efetivamente usadas
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

    for c in [cost_col, rev_col, "cpc", "cpa", "GMV", "Despesas", "Receita direta", "Custo"]:
        if c and c in out.columns:
            out[c] = out[c].apply(fmt_brl)

    for c in ["ctr_calc", "cvr_calc", "acos_calc"]:
        if c in out.columns:
            out[c] = out[c].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    if "CTR" in out.columns:
        out["CTR"] = out["CTR"].apply(lambda v: fmt_pct(v) if pd.notna(v) else "")

    return out


# ============================
# 4) UI
# ============================

st.title("Shopee Ads – Estruturado por Campanha (via Grupo de Anúncios)")

with st.sidebar:
    st.header("Parâmetros")
    min_clicks_eval = st.number_input("Mín. cliques p/ avaliar CVR", value=30, step=5)
    min_spend_no_conv = st.number_input("Gasto mínimo p/ alerta sem conversão (R$)", value=50.0, step=10.0)
    low_impressions_threshold = st.number_input("Impressões baixas (oportunidade)", value=300, step=50)

st.markdown(
    """
**Visualização igual ao seu print**
- Em cada campanha, a **linha sem ID do produto** é o **TOTAL da campanha**.
- As linhas com **ID do produto** são os **anúncios**.
"""
)

ads_group_files = st.file_uploader(
    "CSV – Dados do Grupo de Anúncios (1 ou mais)",
    type=["csv"],
    accept_multiple_files=True,
    key="ads_groups",
)

if not ads_group_files:
    st.info("Suba 1 ou mais CSVs de **Dados do Grupo de Anúncios**.")
    st.stop()

# ============================
# 5) Carregar e unir tudo
# ============================

frames = []
metas = []

for f in ads_group_files:
    df_raw, meta = read_shopee_csv(f)
    df = parse_ads_table(df_raw)

    # campanha / grupo
    group_name = meta.get("titulo", "")
    group_name = group_name.replace("\ufeff", "")
    group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
    if not group_name:
        group_name = Path(getattr(f, "name", "grupo")).stem

    df["Campanha"] = group_name

    frames.append(df)
    metas.append((group_name, meta))

df_all = pd.concat(frames, ignore_index=True)

# ====== COLUNAS (fixo, pois nomes são exatos no seu excel) ======
imp_col = "Impressões" if "Impressões" in df_all.columns else ("Impressões do Produto" if "Impressões do Produto" in df_all.columns else None)
clk_col = "Cliques" if "Cliques" in df_all.columns else ("Cliques de Produtos" if "Cliques de Produtos" in df_all.columns else None)
rev_col = "GMV" if "GMV" in df_all.columns else None
cost_col = "Despesas" if "Despesas" in df_all.columns else ("Custo" if "Custo" in df_all.columns else None)

orders_col = None
for cand in ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"]:
    if cand in df_all.columns:
        orders_col = cand
        break

# ====== adiciona métricas depois do concat (attrs não se perdem assim) ======
df_all = add_ads_metrics(
    df_all,
    imp_col=imp_col,
    clk_col=clk_col,
    cost_col=cost_col,
    orders_col=orders_col,
    rev_col=rev_col,
)

# DEBUG opcional (deixe ligado até validar)
with st.expander("DEBUG (colunas e seleção)", expanded=False):
    st.write("Colunas existentes:", [c for c in ["GMV", "Despesas", "Custo", "Receita direta"] if c in df_all.columns])
    st.write("Selecionadas:", {"GMV": rev_col, "Despesas": cost_col, "Impressões": imp_col, "Cliques": clk_col, "Pedidos": orders_col})
    if metas:
        st.write("Separador detectado por arquivo (meta):")
        st.json({name: m.get("sep_detectado") for name, m in metas})

# ============================
# 6) Visualização estruturada
# ============================

st.header("Visualização Estruturada (Campanha → Total + Anúncios)")

campaigns = sorted(df_all["Campanha"].dropna().unique().tolist())
sel = st.selectbox("Filtrar campanha", ["(todas)"] + campaigns, index=0)

view = df_all.copy()
if sel != "(todas)":
    view = view[view["Campanha"] == sel]

id_col = "ID do produto" if "ID do produto" in view.columns else None
name_col = "Anúncio / Nome do Produto" if "Anúncio / Nome do Produto" in view.columns else None

for camp, d in view.groupby("Campanha"):
    with st.expander(camp, expanded=(sel != "(todas)")):
        # separação TOTAL vs ANÚNCIOS
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
# 7) Alertas (baseados nos anúncios)
# ============================

st.divider()
st.header("Alertas (apenas anúncios com ID)")

if id_col:
    only_ads = df_all[~df_all[id_col].astype(str).str.strip().isin(["", "-", "nan", "None"])].copy()
else:
    only_ads = df_all.copy()

tabs = st.tabs(["Gastando sem converter", "Bons com pouca impressão"])

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
            good_low["ação"] = "Oportunidade: mover/escalar (pouca impressão com bom desempenho)"
            cols = [c for c in ["Campanha", name_col, id_col, imp_col, clk_col, "ctr_calc", orders_col, "cvr_calc", rev_col, cost_col, "acos_calc", "ação"] if c and c in good_low.columns]
            disp = make_display_df(good_low[cols], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp, use_container_width=True, hide_index=True)
    else:
        st.warning("Não encontrei coluna de impressões.")
