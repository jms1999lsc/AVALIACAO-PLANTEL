# app.py — Leixões SC — Avaliação de Plantel
# Streamlit + Google Sheets (cache, batch-read, write-first) + UI afinada e alinhada

import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from gspread.exceptions import WorksheetNotFound, SpreadsheetNotFound, APIError

# =========================
# Configuração visual/tema
# =========================
PRIMARY = "#d22222"   # vermelho Leixões
BLACK   = "#111111"
GREEN   = "#2e7d32"

st.set_page_config(page_title="Leixões SC — Avaliação de Plantel", layout="wide")

# ---- CSS Global (sidebar estreita, botões vermelhos, progresso estilizado) ----
st.markdown(
    f"""
    <style>
    /* Sidebar mais estreita e com fundo leve */
    [data-testid="stSidebar"] {{
        min-width: 315px !important;
        max-width: 315px !important;
        background-color: #f3f3f3;
        padding-top: 1.0rem;
        padding-left: 0.6rem;
        padding-right: 0.6rem;
    }}
    /* Centraliza imagens (logo) na sidebar */
    [data-testid="stSidebar"] img {{
        display: block;
        margin-left: auto;
        margin-right: auto;
    }}
    /* Título da sidebar (vermelho Leixões) */
    .sidebar-title {{
        text-align: center;
        color: {PRIMARY};
        font-weight: 800;
        font-size: 15px;
        margin-top: 0.4rem;
        margin-bottom: 1.0rem;
    }}
    /* Rótulos menores e consistentes na sidebar */
    [data-testid="stSidebar"] label {{
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #333 !important;
    }}
    /* Buttons (toda a app) — vermelho Leixões + padding consistente */
    .stButton > button {{
        background-color: {PRIMARY} !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.9rem !important;
        font-weight: 700 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
    }}
    .stButton > button:disabled {{
        opacity: .45 !important;
    }}
    .stButton > button:hover:enabled {{
        filter: brightness(0.95);
    }}

    /* Progress bar – realça com vermelho Leixões */
    [data-testid="stProgressBar"] > div > div {{
        background-color: {PRIMARY} !important;
    }}
    [data-testid="stProgressBar"] {{
        height: 12px !important;
        border-radius: 99px !important;
        background-color: #e9e9e9 !important;
    }}

    /* Badges e cartões */
    .badge {{
        display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:.75rem;
    }}
    .player-card {{ padding:6px 6px; border-radius:10px; border:1px solid #eee; margin-bottom:6px; }}
    .player-card:hover {{ background:#fafafa; }}

    /* Títulos globais */
    h1, h2, h3, h4 {{ color: {BLACK}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# === Sidebar: logo centrado e maior ===
st.markdown(f"""
<style>
/* garante centragem de qualquer imagem na sidebar */
[data-testid="stSidebar"] img {{
  display:block; margin:0 auto;
}}
/* bloco do logo + título com alinhamento central */
.sidebar-brand {{
  display:flex; flex-direction:column; align-items:center; justify-content:center;
  text-align:center; margin-bottom:12px;
}}
.sidebar-brand .brand-title {{
  color:{PRIMARY}; font-weight:800; font-size:16px; margin-top:6px;
}}
</style>
""", unsafe_allow_html=True)

# === Sidebar: lista de jogadores — alinhamento perfeito a 60px ===
st.markdown("""
<style>
/* cada item tem 60px de altura */
.player-item { margin-bottom:10px; }
.player-row-fixed { height:60px; }

/* cada coluna do item centra verticalmente o conteúdo */
.player-row-fixed [data-testid="column"]{
  display:flex; align-items:center; gap:8px;
}

/* imagem 60×60 sem margens do Streamlit */
.player-row-fixed .img-wrap{
  width:60px; height:60px; display:flex; align-items:center; justify-content:center;
}
.player-row-fixed .img-wrap [data-testid="stImage"]{ margin:0 !important; padding:0 !important; }
.player-row-fixed .img-wrap img{
  width:60px !important; height:60px !important; object-fit:cover; border-radius:10px; display:block;
}

/* botão com a MESMA altura da imagem, texto centrado verticalmente */
.player-row-fixed .btn-wrap .stButton{ width:100%; margin:0 !important; }
.player-row-fixed .btn-wrap .stButton > button{
  width:100% !important;
  height:60px !important;
  display:flex; align-items:center; justify-content:flex-start;
  white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important;
  padding:0 0.60rem !important; font-size:0.96rem !important;
  margin:0 !important;
}

/* bolinha de estado */
.status-dot{ width:12px; height:12px; border-radius:50%; display:inline-block; }
.status-done{ background:#2e7d32; }
.status-pending{ background:#cfcfcf; border:1px solid #bdbdbd; }

/* bloco do jogador selecionado (título + foto) centrado */
.player-hero{ display:flex; flex-direction:column; align-items:center; }
.player-hero-title{ text-align:center; font-weight:700; margin:8px 0 10px 0; }
</style>
""", unsafe_allow_html=True)


# =========================
# Caminhos e ficheiros
# =========================
DATA_DIR       = "data"
PLAYERS_CSV    = os.path.join(DATA_DIR, "jogadores.csv")
FUNCOES_CSV    = os.path.join(DATA_DIR, "funcoes.csv")
AVALIACOES_CSV = os.path.join(DATA_DIR, "avaliacoes.csv")
FECHOS_CSV     = os.path.join(DATA_DIR, "fechos.csv")

# ============================================================
# 🔗 GOOGLE SHEETS (com fallback local) + anti-429
# ============================================================
USE_SHEETS = True  # Google Sheets ativo

REQUIRED_TABS = {
    "avaliacoes": [
        "timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
        "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial",
        "funcoes","observacoes",
    ],
    "fechos": ["timestamp","ano","mes","avaliador","completos","total","status"],
}

def _get_sheet_id():
    """Obtém SHEET_ID dos secrets; aceita ID puro ou URL completa."""
    sid = None
    try:
        sid = st.secrets["gcp_service_account"].get("SHEET_ID", None)
    except Exception:
        pass
    if not sid:
        sid = st.secrets.get("SHEET_ID", None)
    if not sid:
        raise ValueError("SHEET_ID não encontrado nos secrets. Defina-o em [gcp_service_account] ou na raiz.")
    sid = str(sid).strip()
    if "docs.google.com/spreadsheets/d/" in sid:
        try:
            sid = sid.split("/d/")[1].split("/")[0]
        except Exception:
            raise ValueError("Não foi possível extrair o SHEET_ID da URL fornecida.")
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
    try:
        return gc.open_by_key(sid)
    except SpreadsheetNotFound as e:
        raise RuntimeError(
            "SpreadsheetNotFound (404). Verifique o SHEET_ID e se a folha foi partilhada com a service account (Editor)."
        ) from e
    except APIError as e:
        raise RuntimeError(f"APIError ao abrir a Sheet (possível 404/quota). Detalhe: {e}") from e

@st.cache_data(ttl=120, show_spinner=False)
def gs_read_bulk():
    """
    Lê 'avaliacoes' e 'fechos' numa chamada via Spreadsheet.values_batch_get.
    Tolerante a quotas (429) e a abas em falta; devolve DataFrames vazios nesses casos.
    """
    sh = _open_sheet()
    try:
        res = sh.values_batch_get(ranges=["avaliacoes!A1:Z", "fechos!A1:Z"])
    except APIError as e:
        es = str(e)
        if "Quota exceeded" in es or "429" in es:
            return pd.DataFrame(), pd.DataFrame()
        raise

    vrs = res.get("valueRanges", [])

    def _to_df_safe(vr_dict):
        values = vr_dict.get("values", []) if isinstance(vr_dict, dict) else []
        if not values:
            return pd.DataFrame()
        raw_header = values[0]
        header = [str(h).strip() for h in raw_header if h is not None]
        if not header:
            return pd.DataFrame()
        rows = values[1:]
        fixed_rows = []
        ncols = len(header)
        for r in rows:
            r = list(r or [])
            if len(r) < ncols:
                r = r + [""] * (ncols - len(r))
            if len(r) > ncols:
                r = r[:ncols]
            fixed_rows.append(r)
        try:
            df = pd.DataFrame(fixed_rows, columns=header)
        except Exception:
            df = pd.DataFrame(fixed_rows)
            df.columns = [f"col_{i+1}" for i in range(df.shape[1])]
        return df

    df_av = _to_df_safe(vrs[0] if len(vrs) > 0 else {})
    df_f  = _to_df_safe(vrs[1] if len(vrs) > 1 else {})

    # coerção básica de tipos
    if not df_av.empty:
        for col in ["player_id","player_numero","ano","mes","encaixe","fisicas","mentais","impacto_of","impacto_def","potencial"]:
            if col in df_av.columns:
                df_av[col] = pd.to_numeric(df_av[col], errors="coerce")
    if not df_f.empty:
        for col in ["ano","mes","completos","total"]:
            if col in df_f.columns:
                df_f[col] = pd.to_numeric(df_f[col], errors="coerce")

    return df_av, df_f

def gs_append(sheet_name: str, row_dict: dict) -> bool:
    """
    Escreve diretamente sem chamadas de leitura:
    - Evita .worksheet() (faz GET); tenta usar cache local ou cria a aba direto.
    - Tenta até 3 vezes em caso de quota 429.
    """
    import time
    sh = _open_sheet()

    header = REQUIRED_TABS.get(sheet_name, list(row_dict.keys()))
    row = [row_dict.get(col, "") for col in header]

    # cache simples de worksheets por sessão
    if "_ws_cache" not in st.session_state:
        st.session_state["_ws_cache"] = {}
    ws_cache = st.session_state["_ws_cache"]

    ws = ws_cache.get(sheet_name, None)
    if ws is None:
        try:
            for w in sh.worksheets():
                if w.title == sheet_name:
                    ws = w
                    break
        except Exception:
            ws = None

    if ws is None:
        try:
            ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=max(len(header), 10))
            ws.update([header])
            ws_cache[sheet_name] = ws
        except Exception as e:
            st.error(f"Não consegui criar a aba '{sheet_name}': {e}")
            return False

    for attempt in range(3):
        try:
            ws.append_row(row, value_input_option="USER_ENTERED")
            return True
        except APIError as e:
            es = str(e)
            if "Quota exceeded" in es or "429" in es:
                time.sleep(2 * (attempt + 1))
                continue
            st.error(f"Erro API ao gravar na Sheet: {e}")
            return False
        except Exception as e:
            st.error(f"Erro inesperado ao gravar na Sheet: {e}")
            return False

    st.error("Falhou após várias tentativas (quota 429).")
    return False

def gs_replace_all(sheet_name: str, df: pd.DataFrame):
    """Substitui toda a aba por um DataFrame (usar com cuidado)."""
    try:
        sh = _open_sheet()
        try:
            if "_ws_cache" in st.session_state and sheet_name in st.session_state["_ws_cache"]:
                ws = st.session_state["_ws_cache"][sheet_name]
            else:
                ws = sh.worksheets()[0] if sh.worksheets() else sh.add_worksheet(title=sheet_name, rows=1000, cols=max(len(df.columns), 10))
                if ws.title != sheet_name:
                    ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=max(len(df.columns), 10))
        except Exception:
            ws = sh.add_worksheet(title=sheet_name, rows=1000, cols=max(len(df.columns), 10))
        ws.clear()
        if df.empty:
            ws.update([REQUIRED_TABS.get(sheet_name, [])])
        else:
            ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
    except Exception as e:
        st.error(f"Erro ao atualizar aba '{sheet_name}': {e}")

# ============================================================
# 📦 CSV locais (seeding + leitura robusta)
# ============================================================
def ensure_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(AVALIACOES_CSV):
        pd.DataFrame(columns=REQUIRED_TABS["avaliacoes"]).to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")
    if not os.path.exists(FECHOS_CSV):
        pd.DataFrame(columns=REQUIRED_TABS["fechos"]).to_csv(FECHOS_CSV, index=False, encoding="utf-8")

def _read_csv_flex(path: str) -> pd.DataFrame:
    """Lê CSV aceitando vírgula ou ponto e vírgula; tenta utf-8 depois latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

@st.cache_data(ttl=5)
def load_players() -> pd.DataFrame:
    ensure_files()
    df = _read_csv_flex(PLAYERS_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    colmap = {}
    for c in list(df.columns):
        if c in ["id","player_id","identificador","id_jogador"]:
            colmap[c] = "id"
        elif c in ["numero","número","num","nr"]:
            colmap[c] = "numero"
        elif c in ["nome","jogador","player","nome_jogador"]:
            colmap[c] = "nome"
    df = df.rename(columns=colmap)
    missing = [c for c in ["id","numero","nome"] if c not in df.columns]
    if missing:
        st.error(f"`data/jogadores.csv` inválido. Falta(m): {missing}. Esperado: id,numero,nome.")
        st.stop()
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    df["numero"] = pd.to_numeric(df["numero"], errors="coerce")
    df["nome"] = df["nome"].astype(str).str.strip()
    df = df.dropna(subset=["id","numero","nome"]).copy()
    df["id"] = df["id"].astype(int)
    df["numero"] = df["numero"].astype(int)
    df = df.drop_duplicates(subset=["id"]).sort_values("numero")
    return df

@st.cache_data(ttl=5)
def load_functions() -> pd.DataFrame:
    ensure_files()
    df = _read_csv_flex(FUNCOES_CSV)
    df.columns = [c.strip().lower() for c in df.columns]
    colmap = {}
    for c in list(df.columns):
        if c in ["codigo","código","cod","code"]:
            colmap[c] = "codigo"
        elif c in ["nome","funcao","função","role"]:
            colmap[c] = "nome"
        elif c in ["familia","família","grupo","family"]:
            colmap[c] = "familia"
    df = df.rename(columns=colmap)
    missing = [c for c in ["codigo","nome","familia"] if c not in df.columns]
    if missing:
        st.error(f"`data/funcoes.csv` inválido. Falta(m): {missing}. Esperado: codigo,nome,familia.")
        st.stop()
    df["codigo"]  = df["codigo"].astype(str).str.strip()
    df["nome"]    = df["nome"].astype(str).str.strip()
    df["familia"] = df["familia"].astype(str).str.strip()
    df = df.dropna(subset=["codigo","nome"]).copy()
    df = df.drop_duplicates(subset=["codigo"]).sort_values(["familia","nome"])
    return df

# ============================================================
# 📊 Reader/Writer (Sheets bulk + CSV fallback)
# ============================================================
def read_avaliacoes() -> pd.DataFrame:
    if USE_SHEETS:
        df_av, _ = gs_read_bulk()
        return df_av
    else:
        return pd.read_csv(AVALIACOES_CSV) if os.path.exists(AVALIACOES_CSV) else pd.DataFrame()

def read_fechos() -> pd.DataFrame:
    if USE_SHEETS:
        _, df_f = gs_read_bulk()
        return df_f
    else:
        return pd.read_csv(FECHOS_CSV) if os.path.exists(FECHOS_CSV) else pd.DataFrame()

def save_avaliacao(row: dict):
    if USE_SHEETS:
        ok = gs_append("avaliacoes", row)
        gs_read_bulk.clear()  # invalida cache p/ refletir mais depressa
        if not ok:
            df = pd.read_csv(AVALIACOES_CSV) if os.path.exists(AVALIACOES_CSV) else pd.DataFrame(columns=REQUIRED_TABS["avaliacoes"])
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(AVALIACOES_CSV) if os.path.exists(AVALIACOES_CSV) else pd.DataFrame(columns=REQUIRED_TABS["avaliacoes"])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")

def fechar_mes(avaliador: str, ano: int, mes: int, completos: int, total: int):
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "ano": int(ano), "mes": int(mes),
        "avaliador": avaliador, "completos": int(completos), "total": int(total),
        "status": "FECHADO" if completos == total else "INCOMPLETO",
    }
    if USE_SHEETS:
        ok = gs_append("fechos", row)
        gs_read_bulk.clear()
        if not ok:
            df = pd.read_csv(FECHOS_CSV) if os.path.exists(FECHOS_CSV) else pd.DataFrame(columns=REQUIRED_TABS["fechos"])
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(FECHOS_CSV, index=False, encoding="utf-8")
    else:
        df = pd.read_csv(FECHOS_CSV) if os.path.exists(FECHOS_CSV) else pd.DataFrame(columns=REQUIRED_TABS["fechos"])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(FECHOS_CSV, index=False, encoding="utf-8")

# =========================
# Regras de negócio úteis
# =========================
def trimmed_mean(vals):
    """Média aparada: exclui a maior e a menor quando n>=3."""
    vals = [float(v) for v in vals if pd.notna(v)]
    n = len(vals)
    if n == 0: return None
    if n >= 3: return (sum(vals) - min(vals) - max(vals)) / (n - 2)
    return sum(vals)/n

def foto_path_for(player_id: int, size: int = 44) -> str:
    """
    Devolve o caminho da foto do jogador (jpg/jpeg/png/webp).
    Se não existir, devolve um placeholder do tamanho pedido.
    """
    base = f"assets/fotos/{player_id}"
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = base + ext
        if os.path.exists(p):
            return p
    return f"https://placehold.co/{size}x{size}/cccccc/ffffff?text=%20"

def is_completed(df: pd.DataFrame, avaliador: str, ano: int, mes: int, player_id: int) -> bool:
    """Verifica se já existe avaliação para determinado jogador/avaliador/mês (tolerante a df vazio)."""
    if df is None or df.empty:
        return False
    needed = {"avaliador", "ano", "mes", "player_id"}
    if not needed.issubset(set(df.columns)):
        return False
    try:
        mask = (
            (df["avaliador"].astype(str) == str(avaliador))
            & (df["ano"].astype(int) == int(ano))
            & (df["mes"].astype(int) == int(mes))
            & (df["player_id"].astype(int) == int(player_id))
        )
        return not df.loc[mask].empty
    except Exception:
        return False

# =========================
# Carregamento base + Sidebar (branding + período)
# =========================
players = load_players()
funcs   = load_functions()

# memória local de avaliações feitas nesta sessão (perfil, ano, mes, player_id)
if "session_completed" not in st.session_state:
    st.session_state["session_completed"] = set()

# --- Sidebar: branding ajustado, sem espaço morto no topo ---
logo_path = "assets/logo.png"
with st.sidebar:
    # Remove padding superior com CSS customizado
    st.markdown(
        """
        <style>
            section[data-testid="stSidebar"] div[role="document"] {
                padding-top: 0rem !important;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Cria colunas para centralizar o logo
    cL, cC, cR = st.columns([1, 2, 1])
    with cC:
        if os.path.exists(logo_path):
            st.image(
                logo_path,
                width=130,           # ajusta aqui se quiser maior/menor
                use_column_width=False,
                clamp=True
            )
        else:
            st.image("https://placehold.co/130x130?text=Logo", width=130)

    # Título centrado logo abaixo do símbolo
    st.markdown(
        f"""
        <div style='text-align:center;
                    color:{PRIMARY};
                    font-weight:800;
                    font-size:15px;
                    margin-top:-6px;
                    margin-bottom:16px;'>
            Leixões SC — Avaliação de Plantel
        </div>
        """,
        unsafe_allow_html=True
    )


    today = datetime.today()
    if "ano" not in st.session_state:
        st.session_state["ano"] = today.year
    if "mes" not in st.session_state:
        st.session_state["mes"] = today.month

    st.session_state["ano"] = st.number_input("Ano", min_value=2024, max_value=2100,
                                              value=st.session_state["ano"], step=1)
    st.session_state["mes"] = st.selectbox(
        "Mês",
        list(range(1, 13)),
        index=st.session_state["mes"] - 1,
        format_func=lambda m: datetime(2000, m, 1).strftime("%B").capitalize(),
    )

ano = int(st.session_state["ano"])
mes = int(st.session_state["mes"])

st.sidebar.markdown("---")
st.sidebar.markdown('<div class="sidebar-title" style="color:#333;font-weight:700;">Utilizador</div>', unsafe_allow_html=True)
perfil = st.sidebar.selectbox(
    "Perfil",
    ["Utilizador 1","Utilizador 2","Utilizador 3","Utilizador 4","Utilizador 5","Utilizador 6","Utilizador 7","Administrador"]
)

if perfil == "Administrador":
    codigo = st.sidebar.text_input("Código de acesso", type="password", value="")
    if codigo != "leixoes2025":
        st.sidebar.warning("Acesso restrito: introduza o código.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.write("🏃 **Jogadores**")

# Leitura (bulk) 1x por rerun
df_all, df_fechos = read_avaliacoes(), read_fechos()

def completed_for_player(pid: int) -> bool:
    """Combina estado local e dados na Sheet, tolerante a erros."""
    try:
        in_session = (perfil, ano, mes, pid) in st.session_state["session_completed"]
    except Exception:
        in_session = False
    try:
        in_sheet = is_completed(df_all, perfil, ano, mes, pid)
    except Exception:
        in_sheet = False
    return in_session or in_sheet

completos_ids = [int(pid) for pid in players["id"].tolist() if completed_for_player(int(pid))]
st.sidebar.progress(len(completos_ids)/len(players), text=f"Completos: {len(completos_ids)}/{len(players)}")

# Seleção do jogador
if "selecionado_id" not in st.session_state:
    st.session_state["selecionado_id"] = int(players.iloc[0].id)
selecionado_id = st.session_state["selecionado_id"]

# ---- Lista de jogadores (img 60x60 / botão / dot) ----
# ---- Lista de jogadores (img 60x60 / botão / dot) ----
for _, row in players.iterrows():
    pid   = int(row["id"])
    foto  = foto_path_for(pid, 60)
    label = f"#{int(row['numero']):02d} — {row['nome']}"

    with st.sidebar.container():
        st.markdown("<div class='player-item player-row-fixed'>", unsafe_allow_html=True)

        # colunas: imagem / botão / dot (sem espaço morto)
        c1, c2, c3 = st.columns([0.55, 1.35, 0.10], gap="small")

        with c1:
            st.markdown("<div class='img-wrap'>", unsafe_allow_html=True)
            st.image(foto, width=60, clamp=True)  # força 60px
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='btn-wrap'>", unsafe_allow_html=True)
            if st.button(label, key=f"sel_{pid}"):
                selecionado_id = pid
            st.markdown("</div>", unsafe_allow_html=True)

        done = completed_for_player(pid)
        with c3:
            st.markdown(
                f"<span class='status-dot {'status-done' if done else 'status-pending'}'></span>",
                unsafe_allow_html=True
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.session_state["selecionado_id"] = selecionado_id
selecionado = players[players["id"]==selecionado_id].iloc[0]


# =========================
# Layout principal
# =========================
col1, col2 = st.columns([1.2, 2.2], gap="large")

with col1:
    st.markdown("#### Jogador selecionado")
    lsp, center, rsp = st.columns([1,2,1])
    with center:
        st.markdown(
            f"<div class='player-hero-title'><span class='badge'>#{int(selecionado['numero'])}</span> {selecionado['nome']}</div>",
            unsafe_allow_html=True
        )
        st.image(foto_path_for(int(selecionado['id']), 220), width=220, clamp=True)

    # Formulário
    st.markdown("### Formulário de Avaliação")

    def nota(label: str, key: str):
        """
        Compatível com todas as versões: mostra um '—' inicial (sem valor) e
        só passa a 1..4 depois do clique do utilizador.
        """
        opcoes = ["—", 1, 2, 3, 4]
        escolha = st.radio(label, opcoes, horizontal=True, index=0, key=key)
        return None if escolha == "—" else escolha

    encaixe   = nota("Encaixe no Perfil Leixões",       f"n_encaixe_{selecionado_id}_{ano}_{mes}_{perfil}")
    fisicas   = nota("Capacidades Físicas Exigidas",    f"n_fisicas_{selecionado_id}_{ano}_{mes}_{perfil}")
    mentais   = nota("Capacidades Mentais Exigidas",    f"n_mentais_{selecionado_id}_{ano}_{mes}_{perfil}")
    imp_of    = nota("Impacto Ofensivo na Equipa",      f"n_impof_{selecionado_id}_{ano}_{mes}_{perfil}")
    imp_def   = nota("Impacto Defensivo na Equipa",     f"n_impdef_{selecionado_id}_{ano}_{mes}_{perfil}")
    potencial = nota("Potencial Futuro",                 f"n_pot_{selecionado_id}_{ano}_{mes}_{perfil}")

    mult_opts = funcs["nome"].tolist()
    fun_sel = st.multiselect("Funções (obrigatório)", options=mult_opts, help="Pode escolher várias.")
    obs = st.text_area("Observações (visível apenas ao Administrador)")

    notas = [encaixe,fisicas,mentais,imp_of,imp_def,potencial]
    faltam_notas = any(v is None for v in notas)
    faltam_funcoes = len(fun_sel)==0
    can_submit = (not faltam_notas) and (not faltam_funcoes)

    if st.button("Submeter avaliação", type="primary", disabled=not can_submit):
        row = dict(
            timestamp=datetime.utcnow().isoformat(),
            ano=ano, mes=mes, avaliador=perfil,
            player_id=int(selecionado["id"]), player_numero=int(selecionado["numero"]), player_nome=selecionado["nome"],
            encaixe=int(encaixe), fisicas=int(fisicas), mentais=int(mentais),
            impacto_of=int(imp_of), impacto_def=int(imp_def), potencial=int(potencial),
            funcoes=";".join(fun_sel), observacoes=obs.replace("\n"," ").strip()
        )
        save_avaliacao(row)
        st.session_state["session_completed"].add((perfil,ano,mes,int(selecionado["id"])))
        st.success("✅ Avaliação registada.")
        st.rerun()

    if faltam_notas:
        st.info("⚠️ Selecione uma opção (1–4) em todas as dimensões.")
    elif faltam_funcoes:
        st.info("⚠️ Selecione pelo menos uma função.")

    # Estado do mês + Submeter mês
    df_all = read_avaliacoes()
    completos = [int(pid) for pid in players["id"].tolist() if is_completed(df_all, perfil, ano, mes, int(pid)) or (perfil,ano,mes,int(pid)) in st.session_state["session_completed"]]
    falta = len(players) - len(completos)
    st.write(f"**Estado do mês:** {len(completos)}/{len(players)} jogadores avaliados.")
    ja_fechado = False
    if not df_fechos.empty:
        mask = (df_fechos.get("avaliador","")==perfil) & (df_fechos.get("ano",0)==ano) & (df_fechos.get("mes",0)==mes)
        ja_fechado = not df_fechos[mask].empty
    if st.button("✅ Submeter mês (tudo preenchido)", type="secondary", disabled=(falta>0) or ja_fechado):
        fechar_mes(perfil, ano, mes, len(completos), len(players))
        st.success("📌 Mês marcado como submetido para este avaliador.")
        st.rerun()

# ======== COLUNA DIREITA — INSTRUÇÕES + BLOCO DO ADMIN ========
with col2:
    st.markdown("#### Instruções")

    # Bloco de instruções principais (igual para todos)
    st.markdown(
        """
        <ol style="line-height: 1.7; font-size: 0.95rem;">
            <li>Escolha o seu <strong>nome</strong> como Perfil em <strong>Utilizador</strong>.</li>
            <li>Escolha o <strong>jogador</strong> na barra lateral.</li>
            <li>Preencha as <strong>seis dimensões</strong> (1–4) e selecione as <strong>funções</strong> (pode escolher várias).</li>
            <li>Clique <strong>Submeter avaliação</strong> para registar o mês selecionado.</li>
        </ol>
        <p style="font-style: italic; font-size: 0.9rem;">
            As submissões ficam visíveis apenas ao <strong>Administrador</strong>.<br>
            O botão <strong>Submeter mês</strong> só fica ativo quando os <strong>25/25</strong> jogadores estiverem preenchidos.
        </p>
        """,
        unsafe_allow_html=True
    )

    # -----------------------------------------------------------------
    # BLOCO ADICIONAL VISÍVEL APENAS PARA O ADMINISTRADOR
    # -----------------------------------------------------------------
    if perfil == "Administrador":
        st.markdown("---")
        st.markdown("#### Painel do Administrador")

        # Dados agregados das avaliações
        total_avaliacoes = len(df_all) if 'df_all' in locals() else 0
        total_fechos = len(df_fechos) if 'df_fechos' in locals() else 0

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Avaliações registadas", total_avaliacoes)
        with col_b:
            st.metric("Fechos mensais", total_fechos)

        # Botão opcional de atualização
        if st.button("🔄 Atualizar dados", type="secondary"):
            st.cache_data.clear()
            st.rerun()

        # (opcional) — exportar dados
        st.download_button(
            "⬇️ Exportar avaliações (CSV)",
            df_all.to_csv(index=False).encode("utf-8") if 'df_all' in locals() else b"",
            "avaliacoes.csv",
            "text/csv",
        )

    # -----------------------------------------------------------------
    # Rodapé discreto
    st.markdown("---")
    st.caption("© Leixões SC")
