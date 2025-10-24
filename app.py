# app.py — Leixões SC — Avaliação de Plantel
# (Sheets SoT + métricas dinâmicas + Funções antigo + médias ponderadas 60/40 + perfis dinâmicos)

import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ====================================
# CONFIGURAÇÕES GERAIS DE AMBIENTE
# ====================================

USE_SHEETS = True  # Fonte principal: Google Sheets (True) | CSV local (False)

# Cores
PRIMARY = "#d22222"  # vermelho Leixões
BLACK   = "#111111"
GREEN   = "#2e7d32"

# Página
st.set_page_config(
    page_title="Leixões SC — Avaliação de Plantel",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "⚽",
    layout="wide"
)

# --- Boot da sessão: garante que o session_state existe e tem defaults ---
def _boot_session():
    if "booted" in st.session_state:
        return
    st.session_state["booted"] = True
    today = datetime.today()
    st.session_state.setdefault("ano", today.year)
    st.session_state.setdefault("mes", today.month)
    st.session_state.setdefault("session_completed", set())

_boot_session()

# =========================
# CSS — Sidebar 315px + Branding + UI
# =========================
st.markdown("""
<style>
/* ====== LAYOUT GLOBAL ====== */
div.block-container {
  padding-top: 2.0rem !important;
  padding-bottom: 0.6rem;
}

/* ====== SIDEBAR ====== */
[data-testid="stSidebar"] {
  min-width: 315px !important;
  max-width: 315px !important;
  background: #f3f3f3;
  padding-left: .8rem;
  padding-right: .8rem;
}

/* Logo centrado e sem espaço morto */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{
  padding-top: 0 !important;
}
.sidebar-logo [data-testid="stImage"]{ margin: 0 !important; padding: 0 !important; }
.sidebar-logo img { display:block; margin: -18px auto 6px auto !important; width: 130px; height: auto; }

/* Título do projeto em vermelho Leixões, centrado e bold */
.sidebar-title {
  color: #d22222;
  font-weight: 800;
  font-size: 18px;
  line-height: 1.25;
  text-align: center;
  margin: 6px 0 12px 0;
}

/* ====== JOGADOR SELECIONADO ====== */
.player-hero-title { text-align: center; margin: 8px 0 10px 0; }
.player-hero-title .player-num,
.player-hero-title .player-name {
  font-weight: 800 !important;
  color: #111;
  font-size: 20px;
}
.player-hero-title .player-num { margin-right: 6px; }

/* ====== CABEÇALHOS ====== */
h2, h3 {
  font-weight: 800 !important;
  color: #b00000 !important;
  letter-spacing: 0.3px;
  margin-top: 0.3rem !important;
}
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.35rem !important; }

/* ====== BOTÕES / PROGRESS ====== */
.stButton > button{
  background:#d22222 !important;
  color:#fff !important;
  border:none !important;
  border-radius:8px !important;
  padding:.55rem .9rem !important;
  font-weight:700 !important;
}
.stButton > button:disabled{ opacity:.45 !important; }
[data-testid="stProgressBar"] > div > div{ background:#d22222 !important; }

/* ====== LISTA DE JOGADORES ====== */
.player-item { margin-bottom: 10px; }
.player-row-fixed { height: 60px; }
.player-row-fixed [data-testid="column"]{ display:flex; align-items:center; gap:10px; }
.player-row-fixed .img-wrap{ width:60px; height:60px; display:flex; align-items:center; justify-content:center; }
.player-row-fixed .img-wrap [data-testid="stImage"]{ margin:0 !important; padding:0 !important; }
.player-row-fixed .img-wrap img{ width:60px !important; height:60px !important; object-fit:cover; border-radius:10px; }
.player-row-fixed .btn-wrap .stButton{ width:100%; margin:0 !important; }
.player-row-fixed .btn-wrap .stButton > button{
  width:100% !important; height:60px !important;
  display:flex; align-items:center; justify-content:flex-start;
  white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
  padding:0 .70rem !important; font-size:1.00rem !important; margin:0 !important;
}
.status-dot{ width:12px; height:12px; border-radius:50%; display:inline-block; }
.status-done{ background:#2e7d32; }
.status-pending{ background:#cfcfcf; border:1px solid #bdbdbd; }
</style>
""", unsafe_allow_html=True)

# ==========================
# GOOGLE SHEETS CONFIG + HELPERS
# ==========================
def _get_sheet_id():
    sid = None
    try:
        sid = st.secrets["gcp_service_account"].get("SHEET_ID", None)
    except Exception:
        pass
    if not sid:
        sid = st.secrets.get("SHEET_ID", None)
    if not sid:
        raise ValueError("SHEET_ID não definido em secrets.")
    sid = str(sid).strip()
    if "docs.google.com/spreadsheets/d/" in sid:
        sid = sid.split("/d/")[1].split("/")[0]
    return sid

@st.cache_resource
def _get_gspread_client():
    import gspread
    from google.oauth2.service_account import Credentials
    sa = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(dict(sa), scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    return gspread.authorize(creds)

@st.cache_resource
def _open_sheet():
    gc = _get_gspread_client()
    sid = _get_sheet_id()
    return gc.open_by_key(sid)

@st.cache_data(ttl=120, show_spinner=False)
def read_sheet(tab: str) -> pd.DataFrame:
    if not USE_SHEETS:
        return pd.DataFrame()
    try:
        sh = _open_sheet()
        ws = sh.worksheet(tab)
        values = ws.get_all_values()
        if not values:
            return pd.DataFrame()
        header = [h.strip() for h in values[0]]
        data = values[1:]
        data = [row + [""]*(len(header)-len(row)) for row in data]
        df = pd.DataFrame(data, columns=header)
        return df
    except Exception:
        return pd.DataFrame()

def append_rows(tab: str, rows: list[list]) -> bool:
    try:
        sh = _open_sheet()
        try:
            ws = sh.worksheet(tab)
        except Exception:
            ws = sh.add_worksheet(title=tab, rows=1000, cols=50)
        ws.append_rows(rows, value_input_option="USER_ENTERED")
        read_sheet.clear()
        return True
    except Exception:
        return False

# ==========================
# PATHS FALLBACK CSV
# ==========================
DATA_DIR       = "data"
PLAYERS_CSV    = os.path.join(DATA_DIR, "jogadores.csv")
AVALIACOES_CSV = os.path.join(DATA_DIR, "avaliacoes.csv")
FUNCOES_CSV    = os.path.join(DATA_DIR, "funcoes.csv")
os.makedirs(DATA_DIR, exist_ok=True)

def _read_csv_flex(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns or [])
    for enc in ("utf-8","latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

# ==========================
# LOADERS
# ==========================
@st.cache_data(ttl=120)
def load_players() -> pd.DataFrame:
    df = read_sheet("players") if USE_SHEETS else _read_csv_flex(PLAYERS_CSV)
    if df.empty:
        st.error("Aba 'players' vazia/inexistente. Esperado: player_id|numero|nome|category")
        st.stop()
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"player_id","numero","nome","category"}
    if not need.issubset(df.columns):
        st.error("Aba 'players' inválida. Esperado: player_id|numero|nome|category")
        st.stop()
    df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    df["numero"]    = pd.to_numeric(df["numero"], errors="coerce").astype("Int64")
    df["nome"]      = df["nome"].astype(str).str.strip()
    df["category"]  = df["category"].astype(str).str.strip().str.upper()
    df = df.dropna(subset=["player_id","numero","nome","category"]).copy()
    df["player_id"] = df["player_id"].astype(int)
    df["numero"]    = df["numero"].astype(int)
    df = df.drop_duplicates(subset=["player_id"]).sort_values("numero")
    return df

@st.cache_data(ttl=120)
def load_metrics() -> pd.DataFrame:
    df = read_sheet("metrics")
    if df.empty:
        st.error("Aba 'metrics' vazia. Esperado: metric_id|label|scope|group|category|obrigatorio|ordem")
        st.stop()
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"metric_id","label","scope","group","category","obrigatorio","ordem"}
    if not need.issubset(df.columns):
        st.error("Colunas em falta na aba 'metrics'.")
        st.stop()
    df["metric_id"]   = df["metric_id"].astype(str).str.strip().str.upper()
    df["label"]       = df["label"].astype(str).str.strip()
    df["scope"]       = df["scope"].astype(str).str.strip().str.lower()
    df["group"]       = df["group"].astype(str).str.strip().str.lower()
    df["category"]    = df["category"].astype(str).str.strip().str.upper()
    df["obrigatorio"] = df["obrigatorio"].astype(str).str.strip().str.upper().isin(["TRUE","1","YES","SIM"])
    df["ordem"]       = pd.to_numeric(df["ordem"], errors="coerce").fillna(9999).astype(int)
    df = df.drop_duplicates(subset=["metric_id"]).sort_values(["scope","group","category","ordem","metric_id"])
    return df

@st.cache_data(ttl=120)
def load_avaliacoes() -> pd.DataFrame:
    df = read_sheet("avaliacoes") if USE_SHEETS else _read_csv_flex(AVALIACOES_CSV)
    if df.empty:
        cols = ["timestamp","ano","mes","avaliador","player_id","player_numero","player_nome","player_category","metric_id","score","observacoes"]
        return pd.DataFrame(columns=cols)
    df.columns = [c.strip().lower() for c in df.columns]
    # normaliza tipos
    for c in ("ano","mes","player_id","score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["avaliador"] = df["avaliador"].astype(str).str.strip()
    df["metric_id"] = df["metric_id"].astype(str).str.strip().str.upper()
    return df

@st.cache_data(ttl=120)
def load_funcoes_sheet() -> pd.DataFrame:
    df = read_sheet("funcoes") if USE_SHEETS else _read_csv_flex(FUNCOES_CSV)
    if df.empty:
        return pd.DataFrame(columns=["timestamp","ano","mes","avaliador","player_id","funcoes"])
    df.columns = [c.strip().lower() for c in df.columns]
    for c in ("ano","mes","player_id"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["avaliador"] = df["avaliador"].astype(str).str.strip()
    return df

# ---- Pesos / Perfis (aba `weights`) ----
DEFAULT_WEIGHTS = pd.DataFrame([
    # Equipa Técnica (ET): 60% total
    ["Treinador Principal", "ET", 0.60, 0.30],
    *[[f"Treinador Adjunto {i}", "ET", 0.60, 0.70/8] for i in range(1,9)],
    # Direção Desportiva (DD): 40% total
    ["Diretor Executivo", "DD", 0.40, 0.25],
    ["Lead Scout",        "DD", 0.40, 0.15],
], columns=["perfil","grupo","peso_grupo","peso_individual"])

@st.cache_data(ttl=120)
def load_weights_df() -> pd.DataFrame:
    """
    Lê a aba 'weights' e aceita vírgula decimal nos pesos.
    Se, após limpeza, ficar vazia, retorna DEFAULT_WEIGHTS.
    Esperado: perfil | grupo | peso_grupo | peso_individual
    grupos: 'ET' (Equipa Técnica) e 'DD' (Direção Desportiva)
    """
    try:
        if USE_SHEETS:
            df = read_sheet("weights")
            if not df.empty:
                df.columns = [c.strip().lower() for c in df.columns]
                need = {"perfil", "grupo", "peso_grupo", "peso_individual"}
                if need.issubset(df.columns):
                    # normaliza strings
                    df["perfil"] = df["perfil"].astype(str).str.strip()
                    df["grupo"]  = df["grupo"].astype(str).str.strip().str.upper()

                    # aceita vírgula decimal
                    for col in ("peso_grupo", "peso_individual"):
                        df[col] = (
                            df[col]
                            .astype(str)
                            .str.replace(",", ".", regex=False)
                        )
                        df[col] = pd.to_numeric(df[col], errors="coerce")

                    # descarta linhas inválidas
                    df = df.dropna(subset=["perfil", "grupo", "peso_grupo", "peso_individual"])
                    # se depois disto ainda tiver linhas, usa-as
                    if not df.empty:
                        return df
        # fallback se sheet ausente/ inválida/ vazia
        return DEFAULT_WEIGHTS.copy()
    except Exception:
        return DEFAULT_WEIGHTS.copy()

# ==========================
# SALVAR
# ==========================
def save_avaliacoes_bulk(rows_dicts: list[dict]):
    header = ["timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
              "player_category","metric_id","score","observacoes"]
    rows = [[rd.get(k,"") for k in header] for rd in rows_dicts]
    ok = False
    if USE_SHEETS:
        ok = append_rows("avaliacoes", rows)
    if not ok:
        df = _read_csv_flex(AVALIACOES_CSV, columns=header)
        df = pd.concat([df, pd.DataFrame(rows, columns=header)], ignore_index=True)
        df.to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")
    load_avaliacoes.clear()

def save_funcoes_tag(ano:int, mes:int, avaliador:str, player_id:int, funcoes_text:str):
    header = ["timestamp","ano","mes","avaliador","player_id","funcoes"]
    row = [[datetime.utcnow().isoformat(), ano, mes, avaliador, player_id, funcoes_text]]
    ok = False
    if USE_SHEETS:
        ok = append_rows("funcoes", row)
    if not ok:
        df = _read_csv_flex(FUNCOES_CSV, columns=header)
        df = pd.concat([df, pd.DataFrame(row, columns=header)], ignore_index=True)
        df.to_csv(FUNCOES_CSV, index=False, encoding="utf-8")
    load_funcoes_sheet.clear()

# ===== Catálogo de Funções (multiselect antigo) =====
FUNCOES_TAXONOMY_DEFAULT = [
    "Guarda Redes","Guarda Redes Construtor",
    "Lateral Profundo","Lateral Construtor","Lateral Defensivo",
    "Defesa Central Construtor","Defesa Central Agressivo",
    "Médio Defensivo Pivot","Segundo Médio","Médio Box-to-Box",
    "Médio Organizador","10 Organizador","Segundo Avançado",
    "Extremo Associativo","Extremo 1x1","Extremo de Profundidade",
    "Ponta de Lança Referência","Ponta de Lança Móvel",
]

@st.cache_data(ttl=600)
def load_funcoes_catalogo() -> list[str]:
    if USE_SHEETS:
        df = read_sheet("funcoes_catalogo")
        if not df.empty:
            col = df.columns[0]
            opts = [str(x).strip() for x in df[col] if str(x).strip()]
            if opts:
                return opts
    return FUNCOES_TAXONOMY_DEFAULT[:]

# =========================
# Helpers (formulário e admin)
# =========================
def foto_path_for(player_id: int, size: int = 60) -> str:
    base = f"assets/fotos/{player_id}"
    for ext in (".jpg",".jpeg",".png",".webp"):
        p = base + ext
        if os.path.exists(p): return p
    return f"https://placehold.co/{size}x{size}/cccccc/ffffff?text=%20"

def metrics_for_category(metrics: pd.DataFrame, category: str) -> dict[str, pd.DataFrame]:
    enc_pot = metrics[metrics["metric_id"].isin(["ENC_PERFIL","POT_FUT"])].copy()
    fis = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="fisicos")].copy()
    men = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="mentais")].copy()
    esp = metrics[(metrics["scope"]=="especifico") & (metrics["group"]=="categoria") & (metrics["category"]==category)].copy()
    return {"enc_pot": enc_pot, "fisicos": fis, "mentais": men, "especificos": esp}

def get_metric_ids_for_family(metrics: pd.DataFrame, category: str, family: str) -> list[str]:
    family = family.upper()
    if family == "FISICO":
        df = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="fisicos")]
    elif family == "MENTAL":
        df = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="mentais")]
    elif family == "ESPECIFICO":
        df = metrics[(metrics["scope"]=="especifico") & (metrics["group"]=="categoria") & (metrics["category"]==category)]
    else:
        return []
    return df.sort_values("ordem")["metric_id"].astype(str).str.upper().tolist()

def trimmed_weighted_mean(pairs: list[tuple[float,float]]) -> float | None:
    """
    pairs = [(score, weight_norm), ...]
    Trimming agregado: remove 1 menor e 1 maior score (se n>=3) e re-normaliza pesos.
    """
    vals = [(float(s), float(w)) for s,w in pairs if pd.notna(s) and pd.notna(w)]
    n = len(vals)
    if n == 0:
        return None
    vals.sort(key=lambda x: x[0])  # por score
    if n >= 3:
        vals = vals[1:-1]  # drop menor e maior
    if not vals:
        return None
    total_w = sum(w for _,w in vals)
    if total_w <= 0:
        return None
    return sum(s*w for s,w in vals) / total_w

def weighted_metric_mean(av_df: pd.DataFrame, weights_df: pd.DataFrame,
                         ano:int, mes:int, pid:int, metric_id:str) -> float | None:
    """
    Média ponderada 60/40 ET/DD com trimming agregado.
    Re-normaliza pesos quando um grupo não tem avaliações.
    """
    df = av_df[(av_df["ano"]==ano) & (av_df["mes"]==mes) &
               (av_df["player_id"]==pid) & (av_df["metric_id"]==metric_id)]
    if df.empty:
        return None

    df = df.copy()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    # junta grupos e pesos
    wd = weights_df.copy()
    wd["perfil"] = wd["perfil"].astype(str).str.strip()
    wd["grupo"]  = wd["grupo"].astype(str).str.upper()

    merged = df.merge(wd, left_on="avaliador", right_on="perfil", how="left")
    # se algum avaliador não estiver em weights, ignora (sem peso)
    merged = merged.dropna(subset=["grupo","peso_grupo","peso_individual","score"]).copy()

    if merged.empty:
        return None

    # re-normaliza por grupos presentes
    present_groups = merged["grupo"].unique().tolist()
    total_group_weight_present = wd[wd["grupo"].isin(present_groups)]["peso_grupo"].dropna().groupby(wd["grupo"]).first().sum()
    if total_group_weight_present <= 0:
        return None

    # peso do grupo normalizado (só grupos presentes)
    group_norm = {}
    for g in present_groups:
        g_total = wd[wd["grupo"]==g]["peso_grupo"].iloc[0]
        group_norm[g] = float(g_total) / float(total_group_weight_present)

    # dentro de cada grupo presente, normaliza por avaliadores que efetivamente avaliaram esta métrica
    pairs = []
    for g in present_groups:
        sub = merged[merged["grupo"]==g]
        if sub.empty:
            continue
        sum_ind = sub["peso_individual"].sum()
        if sum_ind <= 0:
            continue
        for _, r in sub.iterrows():
            w = group_norm[g] * (float(r["peso_individual"]) / float(sum_ind))
            pairs.append((float(r["score"]), w))

    return trimmed_weighted_mean(pairs)

def family_weighted_mean(av_df: pd.DataFrame, weights_df: pd.DataFrame,
                         metrics_df: pd.DataFrame, ano:int, mes:int, pid:int,
                         family:str, player_cat:str) -> float | None:
    ids = get_metric_ids_for_family(metrics_df, player_cat, family)
    vals = []
    for mid in ids:
        val = weighted_metric_mean(av_df, weights_df, ano, mes, pid, mid)
        if val is not None:
            vals.append(val)
    return float(np.mean(vals)) if vals else None

def standalone_weighted_mean(av_df: pd.DataFrame, weights_df: pd.DataFrame,
                             ano:int, mes:int, pid:int, metric_id:str) -> float | None:
    return weighted_metric_mean(av_df, weights_df, ano, mes, pid, metric_id)

# Etiquetas
def letter_grade(x):
    if x is None: return "-"
    if x > 3.5:  return "A"
    if x >= 3.0: return "B"
    if x >= 2.0: return "C"
    return "D"

def potential_suffix(x):
    if x is None: return ""
    if x > 3.5:  return "++"
    if x >= 3.0: return "+"
    if x >= 2.0: return ""
    return "-"

def prev_period(ano:int, mes:int):
    return (ano-1, 12) if mes == 1 else (ano, mes-1)

# Versatilidade
SETORES = {
    "Lateral Profundo":"DEFESA", "Lateral Construtor":"DEFESA", "Lateral Defensivo":"DEFESA",
    "Defesa Central Construtor":"DEFESA", "Defesa Central Agressivo":"DEFESA",
    "Guarda Redes":"GR", "Guarda Redes Construtor":"GR",
    "Médio Defensivo Pivot":"MEDIO", "Segundo Médio":"MEDIO", "Médio Box-to-Box":"MEDIO",
    "Médio Organizador":"MEDIO", "10 Organizador":"MEDIO",
    "Segundo Avançado":"ATAQUE", "Extremo Associativo":"ATAQUE", "Extremo 1x1":"ATAQUE",
    "Extremo de Profundidade":"ATAQUE", "Ponta de Lança Referência":"ATAQUE",
    "Ponta de Lança Móvel":"ATAQUE",
}

def consolidate_functions(funcoes_df, ano:int, mes:int, pid:int):
    df = funcoes_df[(funcoes_df["ano"]==ano)&(funcoes_df["mes"]==mes)&(funcoes_df["player_id"]==pid)]
    if df.empty: return set()
    # coluna 'funcoes' guarda string "A; B; C"
    funs = set()
    for s in df["funcoes"].astype(str):
        for f in [x.strip() for x in s.split(";") if x.strip()]:
            funs.add(f)
    return funs

def versatility_grade(funcoes:set[str]):
    if not funcoes: return "-"
    setores = {SETORES.get(f,"?") for f in funcoes}
    n_fun, n_set = len(funcoes), len(setores)
    if n_set >= 2 or n_fun >= 3:   return "A"
    if n_set == 1 and n_fun >= 3:  return "B"
    if n_set == 1 and n_fun >= 2:  return "C"
    return "D"

# =========================
# Carregar dados
# =========================
players = load_players()
metrics = load_metrics()
aval_all = load_avaliacoes()
funcoes_all = load_funcoes_sheet()
weights_df = load_weights_df()

if "session_completed" not in st.session_state:
    st.session_state["session_completed"]=set()

# =========================
# Sidebar — Branding + período + perfil + lista
# =========================
with st.sidebar:
    # Branding
    st.markdown("<div class='sidebar-logo'>", unsafe_allow_html=True)
    logo_path = "assets/logo.png"
    st.image(logo_path if os.path.exists(logo_path) else "https://placehold.co/140x140?text=Logo", clamp=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-title'>Leixões SC — Avaliação de Plantel</div>", unsafe_allow_html=True)

    # Período
    today = datetime.today()
    if "ano" not in st.session_state: st.session_state["ano"]=today.year
    if "mes" not in st.session_state: st.session_state["mes"]=today.month
    st.session_state["ano"] = st.number_input("Ano", min_value=2024, max_value=2100, value=st.session_state["ano"], step=1)
    st.session_state["mes"] = st.selectbox("Mês", list(range(1,13)), index=st.session_state["mes"]-1,
                                           format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize())

    st.markdown("---")
    st.write("**Utilizador**")

    # Perfis dinâmicos (weights) + opção vazia + Administrador
    perfis_weights = weights_df["perfil"].dropna().astype(str).tolist()
    PERFIS = ["— selecione —"] + perfis_weights + ["Administrador"]
    perfil = st.selectbox("Perfil", PERFIS, index=0)

    if perfil == "— selecione —":
        st.warning("Selecione o seu perfil na barra lateral para continuar.")
        st.stop()

    if perfil=="Administrador":
        code = st.text_input("Código de acesso", type="password", value="")
        if code != "leixoes2025":
            st.warning("Acesso restrito: introduza o código.")
            st.stop()

    st.markdown("---")
    st.write("🏃 **Jogadores**")

# 🔄 Atualizar dados (apenas limpa caches; mantém progresso local)
with st.sidebar:
    if st.button("🔄 Atualizar dados"):
        # limpa caches para forçar nova leitura ao Sheets
        try:
            load_avaliacoes.clear()
        except Exception:
            pass
        try:
            read_sheet.clear()
        except Exception:
            pass
        # NÃO limpar st.session_state["session_completed"] aqui
        st.success("Dados recarregados do Google Sheets.")
        st.rerun()

ano = int(st.session_state["ano"]); mes = int(st.session_state["mes"])

# progresso + seleção
def is_completed_for_player(av_df: pd.DataFrame, metrics: pd.DataFrame, avaliador: str,
                            ano:int, mes:int, player_id:int, player_cat:str) -> bool:
    sec = metrics_for_category(metrics, player_cat)
    req = pd.concat([
        sec["enc_pot"][sec["enc_pot"]["obrigatorio"]],
        sec["fisicos"][sec["fisicos"]["obrigatorio"]],
        sec["mentais"][sec["mentais"]["obrigatorio"]],
        sec["especificos"][sec["especificos"]["obrigatorio"]],
    ], ignore_index=True)
    needed = set(req["metric_id"].tolist())
    if av_df.empty or not needed:
        return False
    df = av_df
    try:
        m = ((df["avaliador"].astype(str)==avaliador) &
             (df["ano"].astype(int)==int(ano)) &
             (df["mes"].astype(int)==int(mes)) &
             (df["player_id"].astype(int)==int(player_id)))
        got = set(df.loc[m, "metric_id"].astype(str).str.upper().tolist())
        return needed.issubset(got)
    except Exception:
        return False

def has_funcoes_for(avaliador: str, ano:int, mes:int, player_id:int) -> bool:
    try:
        df = funcoes_all
        if df.empty:
            return False
        m = (
            (df.get("avaliador","").astype(str)==str(avaliador)) &
            (df.get("ano","")==ano) &
            (df.get("mes","")==mes) &
            (df.get("player_id","")==player_id)
        )
        return bool(m.any())
    except Exception:
        return False

# --- Leitura de 'funcoes' a partir do Sheets/CSV ---
@st.cache_data(ttl=60)
def load_funcoes_sheet() -> pd.DataFrame:
    df = read_sheet("funcoes") if USE_SHEETS else _read_csv_flex(FUNCOES_CSV)
    if df.empty:
        return pd.DataFrame(columns=["timestamp","ano","mes","avaliador","player_id","funcoes"])
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def required_metric_ids_for_category(metrics_df: pd.DataFrame, cat: str) -> set[str]:
    """Devolve o conjunto de metric_id obrigatórios para a categoria (posição) do jogador."""
    sec = metrics_for_category(metrics_df, cat)
    req = pd.concat(
        [
            sec["enc_pot"][sec["enc_pot"]["obrigatorio"]],
            sec["fisicos"][sec["fisicos"]["obrigatorio"]],
            sec["mentais"][sec["mentais"]["obrigatorio"]],
            sec["especificos"][sec["especificos"]["obrigatorio"]],
        ],
        ignore_index=True,
    )
    if req.empty:
        return set()
    return set(req["metric_id"].astype(str).str.upper().tolist())

def is_complete_in_sheet(av_df: pd.DataFrame, fn_df: pd.DataFrame,
                         perfil: str, ano: int, mes: int,
                         pid: int, cat: str) -> bool:
    """True se TODAS as métricas obrigatórias + pelo menos 1 registo em 'funcoes' existirem no Sheets."""
    req = required_metric_ids_for_category(metrics, cat)
    if not req:
        return False

    # linhas do avaliador/ano/mes/jogador
    m = (
        (av_df.get("avaliador", "").astype(str) == str(perfil)) &
        (av_df.get("ano", "").astype(str) == str(ano)) &
        (av_df.get("mes", "").astype(str) == str(mes)) &
        (av_df.get("player_id", "").astype(str) == str(pid))
    )
    got = set(av_df.loc[m, "metric_id"].astype(str).str.upper().tolist())
    metrics_ok = req.issubset(got)

    f = (
        (fn_df.get("avaliador", "").astype(str) == str(perfil)) &
        (fn_df.get("ano", "").astype(str) == str(ano)) &
        (fn_df.get("mes", "").astype(str) == str(mes)) &
        (fn_df.get("player_id", "").astype(str) == str(pid))
    )
    fun_ok = bool(fn_df.loc[f].shape[0] > 0)

    return metrics_ok and fun_ok


# 👉 lê 'funcoes' do Sheets/CSV
funcoes_sheet_df = load_funcoes_sheet()

# 👉 COMPLETOS apenas pelo que existe no Sheets (avaliacoes + funcoes)
completed_ids_sheet = []
for _, r in players.iterrows():
    pid  = int(r["player_id"])
    pcat = str(r["category"]).upper()
    if is_complete_in_sheet(aval_all, funcoes_sheet_df, perfil, ano, mes, pid, pcat):
        completed_ids_sheet.append(pid)

st.sidebar.progress(
    len(completed_ids_sheet)/len(players),
    text=f"Completos: {len(completed_ids_sheet)}/{len(players)}"
)

if "selecionado_id" not in st.session_state:
    st.session_state["selecionado_id"] = int(players.iloc[0]["player_id"])
selecionado_id = st.session_state["selecionado_id"]

for _, row in players.iterrows():
    pid   = int(row["player_id"])
    foto  = foto_path_for(pid, 60)
    label = f"#{int(row['numero']):02d} — {row['nome']}"
    with st.sidebar.container():
        st.markdown("<div class='player-item player-row-fixed'>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns([0.55,1.35,0.10], gap="small")
        with c1:
            st.markdown("<div class='img-wrap'>", unsafe_allow_html=True)
            st.image(foto, width=60, clamp=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
            if st.button(label, key=f"sel_{pid}"):
                selecionado_id = pid
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            done = (pid in completed_ids_sheet)
            st.markdown(f"<span class='status-dot {'status-done' if done else 'status-pending'}'></span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.session_state["selecionado_id"]=selecionado_id
sel = players[players["player_id"]==selecionado_id].iloc[0]
sel_cat = str(sel["category"]).upper()

# =========================
# Layout principal
# =========================
col1, col2 = st.columns([1.2, 2.2], gap="large")

# ---- COL1: Jogador + Formulário
with col1:
    st.markdown("#### Jogador Selecionado")
    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown(
            f"<div class='player-hero-title'>"
            f"<span class='player-num'>#{int(sel['numero'])}</span>"
            f"<span class='player-name'>{sel['nome']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
        st.image(foto_path_for(int(sel["player_id"]), 220), width=220, clamp=True)

    st.markdown("### Formulário de Avaliação")

    secs = metrics_for_category(metrics, sel_cat)

    def nota(label: str, key: str):
        opcoes = ["—", 1, 2, 3, 4]
        escolha = st.radio(label, opcoes, horizontal=True, index=0, key=key)
        return None if escolha == "—" else escolha

    respostas = {}

    if not secs["enc_pot"].empty:
        st.markdown("##### Encaixe & Potencial")
        for _, m in secs["enc_pot"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            respostas[mid] = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")

    if not secs["fisicos"].empty:
        st.markdown("##### Parâmetros Físicos")
        for _, m in secs["fisicos"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            respostas[mid] = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")

    if not secs["mentais"].empty:
        st.markdown("##### Parâmetros Mentais")
        for _, m in secs["mentais"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            respostas[mid] = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")

    if not secs["especificos"].empty:
        st.markdown(f"##### Específicos da Posição ({sel_cat})")
        for _, m in secs["especificos"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            respostas[mid] = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")

    # ===== Funções (obrigatório, multiselect antigo) =====
    funcoes_options = load_funcoes_catalogo()

    prev_funcoes = []
    try:
        fun_df = funcoes_all
        if not fun_df.empty:
            mf = (
                (fun_df.get("ano", 0)==ano) &
                (fun_df.get("mes", 0)==mes) &
                (fun_df.get("avaliador", "").astype(str) == str(perfil)) &
                (fun_df.get("player_id", 0)==int(sel["player_id"]))
            )
            if mf.any():
                last = fun_df.loc[mf].iloc[-1]
                prev_funcoes = [s.strip() for s in str(last.get("funcoes", "")).split(";") if s.strip()]
    except Exception:
        pass

st.markdown("### Posições em que apresenta domínio funcional")

# Catálogo local (CSV) apenas para rótulos de multiselect (mantido do teu código)
FUNCOES_CSV = os.path.join("data", "funcoes.csv")
@st.cache_data
def load_funcoes_catalogo_local():
    try:
        df = pd.read_csv(FUNCOES_CSV)
        if "funcao" not in df.columns:
            return []
        return df["funcao"].dropna().unique().tolist()
    except Exception:
        return []

funcoes = pd.DataFrame({"funcao": load_funcoes_catalogo_local()})

if funcoes.empty:
    st.warning("Nenhuma função encontrada em data/funcoes.csv.")
    funcoes_escolhidas = []
else:
    funcoes_disp = funcoes["funcao"].dropna().unique().tolist()
    funcoes_escolhidas = st.multiselect(
        "Escolha uma ou mais posições:",
        options=funcoes_disp,
        default=[],
    )

    obs = st.text_area("Observações")

    obrig = pd.concat([
        secs["enc_pot"][secs["enc_pot"]["obrigatorio"]],
        secs["fisicos"][secs["fisicos"]["obrigatorio"]],
        secs["mentais"][secs["mentais"]["obrigatorio"]],
        secs["especificos"][secs["especificos"]["obrigatorio"]],
    ], ignore_index=True)["metric_id"].tolist()

    faltam = [mid for mid in obrig if (respostas.get(mid) is None)]
    can_submit = (len(faltam)==0)

    if st.button("Submeter avaliação", type="primary", disabled=not can_submit):
        ts = datetime.utcnow().isoformat()
        base = dict(
            timestamp=ts, ano=ano, mes=mes, avaliador=perfil,
            player_id=int(sel["player_id"]), player_numero=int(sel["numero"]), player_nome=sel["nome"],
            player_category=sel_cat, observacoes=obs.replace("\n"," ").strip()
        )
        rows = []
        for mid, val in respostas.items():
            if val is None: continue
            rd = base.copy()
            rd["metric_id"] = str(mid).upper()
            rd["score"] = int(val)
            rows.append(rd)
        if rows:
            save_avaliacoes_bulk(rows)

        if len(funcoes_escolhidas) == 0:
            st.error("Selecione pelo menos uma Função antes de submeter.")
            st.stop()
        else:
            funcoes_str = "; ".join(funcoes_escolhidas)
            save_funcoes_tag(ano, mes, perfil, int(sel["player_id"]), funcoes_str)
        # Após save_avaliacoes_bulk(rows) e save_funcoes_tag(...)
        try:
            load_avaliacoes.clear()
            read_sheet.clear()
            load_funcoes_sheet.clear()
        except Exception:
            pass
    st.success("✅ Avaliação registada.")
    st.rerun()


    if not can_submit:
        st.info("⚠️ Responda todas as métricas obrigatórias (1–4) antes de submeter.")

    # Estado do mês
    aval_all = load_avaliacoes()
    completos = [int(r["player_id"]) for _, r in players.iterrows()
                 if completed_for_player(int(r["player_id"]), str(r["category"]).upper())]
    st.write(f"**Estado do mês:** {len(completos)}/{len(players)} jogadores avaliados.")

# ======================
# DASHBOARD DO ADMIN (médias ponderadas)
# ======================
if perfil == "Administrador":
    st.markdown("## Dashboard do Administrador")

    pid = int(selecionado_id)
    player_group = sel_cat

    ano_sel = int(ano); mes_sel = int(mes)
    ano_prev, mes_prev = prev_period(ano_sel, mes_sel)

    # Médias ponderadas por família (mês atual e anterior)
    fis_now  = family_weighted_mean(aval_all, weights_df, metrics, ano_sel, mes_sel, pid, "FISICO",      player_group)
    men_now  = family_weighted_mean(aval_all, weights_df, metrics, ano_sel, mes_sel, pid, "MENTAL",     player_group)
    esp_now  = family_weighted_mean(aval_all, weights_df, metrics, ano_sel, mes_sel, pid, "ESPECIFICO", player_group)

    fis_prv  = family_weighted_mean(aval_all, weights_df, metrics, ano_prev, mes_prev, pid, "FISICO",      player_group)
    men_prv  = family_weighted_mean(aval_all, weights_df, metrics, ano_prev, mes_prev, pid, "MENTAL",     player_group)
    esp_prv  = family_weighted_mean(aval_all, weights_df, metrics, ano_prev, mes_prev, pid, "ESPECIFICO", player_group)

    # Standalone ponderado
    lsc_now  = standalone_weighted_mean(aval_all, weights_df, ano_sel, mes_sel, pid, "ENC_PERFIL")
    pot_now  = standalone_weighted_mean(aval_all, weights_df, ano_sel, mes_sel, pid, "POT_FUT")

    # Média Global (3 famílias) + etiqueta final (letra + sufixo do potencial)
    medias = [x for x in [fis_now, men_now, esp_now] if x is not None]
    media_global = float(np.mean(medias)) if medias else None
    etiqueta = f"{letter_grade(media_global)}{potential_suffix(pot_now)}"

    # Versatilidade
    fun_set = consolidate_functions(funcoes_all, ano_sel, mes_sel, pid)
    vers = versatility_grade(fun_set)

    c1,c2,c3,c4,c5,c6,c7 = st.columns(7)
    c1.metric("Versatilidade", vers)
    c2.metric("Perfil LSC",    letter_grade(lsc_now))
    c3.metric("Físico",        letter_grade(fis_now))
    c4.metric("Mental",        letter_grade(men_now))
    c5.metric("Específico",    letter_grade(esp_now))
    c6.metric("Média Global",  letter_grade(media_global))
    c7.metric("Etiqueta Final", etiqueta)

    st.divider()

    # Radares (mês vs mês-1) com ponderação
    def labels_vals_for_family(family:str):
        ids = get_metric_ids_for_family(metrics, player_group, family)
        # labels (amigáveis)
        labels = []
        for mid in ids:
            lab = metrics.loc[metrics["metric_id"]==mid, "label"]
            labels.append(lab.iloc[0] if not lab.empty else mid)
        now = [standalone_weighted_mean(aval_all, weights_df, ano_sel, mes_sel, pid, mid) or 0 for mid in ids]
        prv = [standalone_weighted_mean(aval_all, weights_df, ano_prev, mes_prev, pid, mid) or 0 for mid in ids]
        return labels, now, prv

    def radar_two_traces(title, labels, vals_now, vals_prev):
        cat = labels + (labels[:1] if labels else [])
        now = vals_now + (vals_now[:1] if vals_now else [])
        prv = vals_prev + (vals_prev[:1] if vals_prev else [])
        fig = go.Figure()
        if any(vals_prev):
            fig.add_trace(go.Scatterpolar(r=prv, theta=cat, name="Mês anterior",
                                          line=dict(color="#c6c6c6",width=2)))
        if any(vals_now):
            fig.add_trace(go.Scatterpolar(r=now, theta=cat, name="Atual",
                                          line=dict(color="#d22222",width=3)))
        fig.update_layout(
            title=title, showlegend=True,
            polar=dict(radialaxis=dict(range=[0,4], tickvals=[1,2,3,4])),
            margin=dict(l=10,r=10,t=40,b=10), height=380
        )
        return fig

    colA,colB,colC = st.columns(3)
    lbl, vnow, vprv = labels_vals_for_family("FISICO")
    colA.plotly_chart(radar_two_traces("Radar Físico (ponderado)", lbl, vnow, vprv), use_container_width=True)

    lbl, vnow, vprv = labels_vals_for_family("MENTAL")
    colB.plotly_chart(radar_two_traces("Radar Mental (ponderado)", lbl, vnow, vprv), use_container_width=True)

    lbl, vnow, vprv = labels_vals_for_family("ESPECIFICO")
    pretty_group = player_group.title() if isinstance(player_group,str) else str(player_group)
    colC.plotly_chart(radar_two_traces(f"Radar Específico (ponderado) — {pretty_group}", lbl, vnow, vprv), use_container_width=True)

    st.caption("A linha cinza representa o mês anterior; a vermelha, o mês selecionado. Médias ponderadas 60/40 ET/DD (com trimming agregado).")

# ---- COL2: Instruções
with col2:
    st.markdown("#### Instruções")
    st.markdown("""
    <ol style="line-height:1.7; font-size:.95rem;">
      <li>Escolha o seu <strong>Nome de Utilizador</strong> na barra lateral.</li>
      <li>Escolha o <strong>jogador</strong> na barra lateral.</li>
      <li>Preencha todos os <strong>parâmetros obrigatórios</strong> (1–4).</li>
      <li>Selecione as <strong>Funções</strong> (pelo menos uma).</li>
      <li>Clique <strong>Submeter avaliação</strong>.</li>
    </ol>
    <p style="font-style: italic; font-size:.9rem;">
      As avaliações só são visíveis ao <strong>Administrador</strong>. O mês fecha quando os <strong>25/25</strong> estiverem completos.
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("© Leixões SC")
