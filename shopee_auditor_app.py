import re
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

    # troca vírgula por ponto p/ float científico
    tt = t.replace(",", ".")
    try:
        # se for notação científica
        if "e" in tt.lower():
            v = float(tt)
            if np.isfinite(v):
                return str(int(round(v)))
    except Exception:
        pass

    # se for número normal (só dígitos), mantém
    t_digits = re.sub(r"\D", "", t)
    return t_digits if t_digits else t


def detect_csv_header_row(text: str) -> int:
    lines = text.splitlines()
    # Shopee costuma ter '#,' no header real
    for i, line in enumerate(lines[:200]):
        if line.startswith("#,"):
            return i
    # fallback: detectar linha com colunas principais
    for i, line in enumerate(lines[:300]):
        if "Anúncio / Nome do Produto" in line and "ID do produto" in line:
            return i
    return 0


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

    # LER TUDO COMO TEXTO (CRÍTICO)
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
    df2 = df.copy()
    for c in df2.columns:
        if c in ADS_PERCENT_COLS:
            df2[c] = df2[c].apply(parse_percent)
        elif c in ADS_NUMERIC_COLS:
            df2[c] = df2[c].apply(parse_number_br_aggressive)

    # normaliza ID do produto
    if "ID do produto" in df2.columns:
        df2["ID do produto"] = df2["ID do produto"].apply(normalize_product_id)

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
    rev_col = pick_first_existing(df, ["GMV", "Receita direta"])

    # garante numeric
    for c in [col_imp, col_clk, col_cost, col_orders, rev_col]:
        if c and c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ctr_calc"] = np.where(df[col_imp] > 0, df[col_clk] / df[col_imp], np.nan) if col_imp and col_clk else np.nan
    df["cvr_calc"] = np.where(df[col_clk] > 0, df[col_orders] / df[col_clk], np.nan) if col_clk and col_orders else np.nan

    df["cpc"] = np.where(df[col_clk] > 0, df[col_cost] / df[col_clk], np.nan) if col_cost and col_clk else np.nan
    df["cpa"] = np.where(df[col_orders] > 0, df[col_cost] / df[col_orders], np.nan) if col_cost and col_orders else np.nan

    df["ctr_class"] = df["ctr_calc"].apply(classify_ctr)
    df["cvr_class"] = df["cvr_calc"].apply(classify_cvr)

    # ACOS calc
    if col_cost and rev_col:
        df["acos_calc"] = np.where(df[rev_col] > 0, df[col_cost] / df[rev_col], np.nan)
    else:
        df["acos_calc"] = np.nan

    df.attrs["imp_col"] = col_imp
    df.attrs["clk_col"] = col_clk
    df.attrs["cost_col"] = col_cost
    df.attrs["orders_col"] = col_orders
    df.attrs["rev_col"] = rev_col

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

    for c in [cost_col, rev_col, "cpc", "cpa", "GMV", "Receita direta", "Despesas", "Custo"]:
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
**O que este app faz (igual ao seu print):**
- Cada arquivo de **Grupo de Anúncios** vira uma **campanha**.
- Dentro da campanha:
  - **Linha sem ID** = **TOTAL da campanha**
  - **Linhas com ID** = **anúncios**
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

all_blocks = []
for f in ads_group_files:
    df_raw, meta = read_shopee_csv(f)
    df = parse_ads_table(df_raw)
    df = add_ads_metrics(df)

    # nome da campanha vem do título/meta ou do nome do arquivo
    group_name = meta.get("titulo", "")
    group_name = group_name.replace("\ufeff", "")
    group_name = group_name.replace("Ad Group -", "").replace("Report - Shopee Brasil", "").strip()
    if not group_name:
        group_name = Path(getattr(f, "name", "grupo")).stem

    df["Campanha"] = group_name
    all_blocks.append(df)

df_all = pd.concat(all_blocks, ignore_index=True)

imp_col = df_all.attrs.get("imp_col") or pick_first_existing(df_all, ["Impressões", "Impressões do Produto"])
clk_col = df_all.attrs.get("clk_col") or pick_first_existing(df_all, ["Cliques", "Cliques de Produtos"])
cost_col = df_all.attrs.get("cost_col") or pick_first_existing(df_all, ["Despesas", "Custo"])
orders_col = df_all.attrs.get("orders_col") or pick_first_existing(df_all, ["Conversões Diretas", "Conversões", "Itens Vendidos Diretos", "Itens Vendidos"])
rev_col = df_all.attrs.get("rev_col") or pick_first_existing(df_all, ["GMV", "Receita direta"])

# ============================
# 6) Visualização estruturada (campanha -> total + anúncios)
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
        # separação: TOTAL (sem ID) vs ANÚNCIOS (com ID)
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
                name_col,  # normalmente aparece o nome do grupo na linha total
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

            # ordena por gasto (mais relevante)
            if cost_col and cost_col in df_ads.columns:
                df_ads = df_ads.sort_values(by=cost_col, ascending=False)

            disp_ads = make_display_df(df_ads[cols_ads], imp_col, clk_col, cost_col, orders_col, rev_col)
            st.dataframe(disp_ads, use_container_width=True, hide_index=True)

# ============================
# 7) Alertas (opcional, no detalhe)
# ============================

st.divider()
st.header("Alertas (baseados nos anúncios)")

if id_col:
    only_ads = df_all[df_all[id_col].astype(str).str.strip().apply(lambda x: x not in ["", "-", "nan", "None"])].copy()
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
