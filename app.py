# app.py — Leixões SC — Avaliação de Plantel (métricas dinâmicas + Sheets SoT)
import os
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ====================================
# CONFIGURAÇÕES GERAIS DE AMBIENTE
# ====================================

# Fonte principal de dados (True = Google Sheets, False = CSV local)
USE_SHEETS = True

# Cores de tema
PRIMARY = "#d22222"  # vermelho Leixões
BLACK   = "#111111"
GREEN   = "#2e7d32"

# Configurações da página
st.set_page_config(
    page_title="Leixões SC — Avaliação de Plantel",
    page_icon="assets/logo_mini.png" if os.path.exists("assets/logo_mini.png") else "⚽",
    layout="wide"
)

# =========================
# CSS — Sidebar (315px) + Branding Leixões SC + UI
# =========================
st.markdown(f"""
<style>
/* Sidebar com largura fixa 315px */
[data-testid="stSidebar"] {{
  min-width: 315px !important;
  max-width: 315px !important;
  background-color: #f3f3f3;
  padding-left: 0.8rem;
  padding-right: 0.8rem;
}}
/* Remove espaço morto no topo da sidebar */
section[data-testid="stSidebar"] div[role="document"] {{
  padding-top: 0 !important;
}}
/* Centralizar imagens da sidebar */
section[data-testid="stSidebar"] img {{
  display: block;
  margin: 0 auto;
}}
/* Branding centrado */
.sidebar-brand {{
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  margin-top: -14px;        /* puxa o bloco para cima (ajusta se quiseres) */
  margin-bottom: 14px;
}}
.sidebar-brand .brand-title {{
  color: {PRIMARY};         /* vermelho Leixões */
  font-weight: 800;         /* bold */
  font-size: 18px;
  line-height: 1.2;
  margin-top: 6px;
  text-align: center;
}}

/* Lista de jogadores (60px altura; alinhamento imagem/botão/estado) */
.player-item {{ margin-bottom: 10px; }}
.player-row-fixed {{ height: 60px; }}
.player-row-fixed [data-testid="column"] {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.player-row-fixed .img-wrap {{
  width: 60px; height: 60px;
  display: flex; align-items: center; justify-content: center;
}}
.player-row-fixed .img-wrap [data-testid="stImage"] {{
  margin: 0 !important; padding: 0 !important;
}}
.player-row-fixed .img-wrap img {{
  width: 60px !important; height: 60px !important; object-fit: cover;
  border-radius: 10px; display: block;
}}
.player-row-fixed .btn-wrap .stButton {{ width: 100%; margin: 0 !important; }}
.player-row-fixed .btn-wrap .stButton > button {{
  width: 100% !important; height: 60px !important;
  display: flex; align-items: center; justify-content: flex-start;
  white-space: nowrap !important; overflow: hidden !important; text-overflow: ellipsis !important;
  padding: 0 .70rem !important; font-size: 1.00rem !important; margin: 0 !important;
}}
.status-dot {{ width: 12px; height: 12px; border-radius: 50%; display: inline-block; }}
.status-done {{ background: {GREEN}; }}
.status-pending {{ background: #cfcfcf; border: 1px solid #bdbdbd; }}

/* Títulos / botões / progresso */
h1,h2,h3,h4 {{ color: {BLACK}; }}
.stButton > button {{
  background-color: {PRIMARY} !important; color: #fff !important; border: none !important;
  border-radius: 8px !important; padding: .55rem .9rem !important; font-weight: 700 !important;
}}
.stButton > button:disabled {{ opacity: .45 !important; }}
[data-testid="stProgressBar"] > div > div {{ background: {PRIMARY} !important; }}

/* Hero do jogador */
.player-hero-title {{ text-align: center; font-weight: 700; margin: 8px 0 10px 0; }}
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

# ==========================
# LOADERS (Sheets + fallback)
# ==========================
def _read_csv_flex(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns or [])
    for enc in ("utf-8","latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

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
    return df

@st.cache_data(ttl=120)
def load_fechos() -> pd.DataFrame:
    df = read_sheet("fechos") if USE_SHEETS else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=["timestamp","ano","mes","avaliador","completos","total","status"])
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def save_avaliacoes_bulk(rows_dicts: list[dict]):
    header = ["timestamp","ano","mes","avaliador","player_id","player_numero","player_nome","player_category","metric_id","score","observacoes"]
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

# =========================
# Helpers
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

def trimmed_mean(values: list[float]) -> float | None:
    vals = [float(v) for v in values if pd.notna(v)]
    n = len(vals)
    if n == 0: return None
    if n >= 3:
        return (sum(vals) - min(vals) - max(vals)) / (n - 2)
    return sum(vals)/n

def is_completed_for_player(av_df: pd.DataFrame, metrics: pd.DataFrame, avaliador: str, ano:int, mes:int, player_id:int, player_cat:str) -> bool:
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

# =========================
# Carregar dados
# =========================
players = load_players()
metrics = load_metrics()
aval_all = load_avaliacoes()
fechos   = load_fechos()

if "session_completed" not in st.session_state:
    st.session_state["session_completed"]=set()

# =========================
# Sidebar — Branding (centrado) + período + perfil + lista
# =========================
with st.sidebar:
    # Branding
    st.markdown("<div class='sidebar-brand'>", unsafe_allow_html=True)
    logo_path = "assets/logo.png"
    st.image(logo_path if os.path.exists(logo_path) else "https://placehold.co/160x160?text=Logo", width=160, clamp=True)
    st.markdown("<div class='brand-title'>Leixões SC — Avaliação de Plantel</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Período
    today = datetime.today()
    if "ano" not in st.session_state: st.session_state["ano"]=today.year
    if "mes" not in st.session_state: st.session_state["mes"]=today.month
    st.session_state["ano"] = st.number_input("Ano", min_value=2024, max_value=2100, value=st.session_state["ano"], step=1)
    st.session_state["mes"] = st.selectbox("Mês", list(range(1,13)), index=st.session_state["mes"]-1,
                                           format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize())

    st.markdown("---")
    st.write("**Utilizador**")
    PERFIS = [
        "Treinador Principal",
        "Treinador Adjunto 1",
        "Treinador Adjunto 2",
        "Treinador Adjunto 3",
        "Diretor Executivo",
        "Diretor Desportivo",
        "Lead Scout",
        "Administrador",
    ]
    perfil = st.selectbox("Perfil", PERFIS)
    if perfil=="Administrador":
        code = st.text_input("Código de acesso", type="password", value="")
        if code != "leixoes2025":
            st.warning("Acesso restrito: introduza o código.")
            st.stop()

    st.markdown("---")
    st.write("🏃 **Jogadores**")

ano = int(st.session_state["ano"]); mes = int(st.session_state["mes"])

# progresso
def completed_for_player(pid:int, pcat:str)->bool:
    in_session = (perfil,ano,mes,pid) in st.session_state["session_completed"]
    in_sheet   = is_completed_for_player(aval_all, metrics, perfil, ano, mes, pid, pcat)
    return in_session or in_sheet

completos_ids = []
for _, r in players.iterrows():
    pid, pcat = int(r["player_id"]), str(r["category"]).upper()
    if completed_for_player(pid, pcat): completos_ids.append(pid)
st.sidebar.progress(len(completos_ids)/len(players), text=f"Completos: {len(completos_ids)}/{len(players)}")

# seleção
if "selecionado_id" not in st.session_state:
    st.session_state["selecionado_id"] = int(players.iloc[0]["player_id"])
selecionado_id = st.session_state["selecionado_id"]

# lista
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
            if st.button(label, key=f"sel_{pid}"): selecionado_id = pid
            st.markdown("</div>", unsafe_allow_html=True)
        with c3:
            done = completed_for_player(pid, str(row["category"]).upper())
            st.markdown(f"<span class='status-dot {'status-done' if done else 'status-pending'}'></span>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

st.session_state["selecionado_id"]=selecionado_id
sel = players[players["player_id"]==selecionado_id].iloc[0]
sel_cat = str(sel["category"]).upper()

# =========================
# Layout principal
# =========================
col1, col2 = st.columns([1.2, 2.2], gap="large")

# ---- COL1: Jogador + Formulário Dinâmico
with col1:
    st.markdown("#### Jogador selecionado")
    _, mid, _ = st.columns([1,2,1])
    with mid:
        st.markdown(f"<div class='player-hero-title'><span class='badge'>#{int(sel['numero'])}</span> {sel['nome']}</div>", unsafe_allow_html=True)
        st.image(foto_path_for(int(sel['player_id']), 220), width=220, clamp=True)

    st.markdown("### Formulário de Avaliação")

    # Obter métricas aplicáveis
    def metrics_for_category(metrics: pd.DataFrame, category: str) -> dict[str, pd.DataFrame]:
        enc_pot = metrics[metrics["metric_id"].isin(["ENC_PERFIL","POT_FUT"])].copy()
        fis = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="fisicos")].copy()
        men = metrics[(metrics["scope"]=="transversal") & (metrics["group"]=="mentais")].copy()
        esp = metrics[(metrics["scope"]=="especifico") & (metrics["group"]=="categoria") & (metrics["category"]==category)].copy()
        return {"enc_pot": enc_pot, "fisicos": fis, "mentais": men, "especificos": esp}

    secs = metrics_for_category(metrics, sel_cat)

    # Radio sem pré-seleção (usa “—”)
    def nota(label: str, key: str):
        opcoes = ["—", 1, 2, 3, 4]
        escolha = st.radio(label, opcoes, horizontal=True, index=0, key=key)
        return None if escolha == "—" else escolha

    respostas = {}  # metric_id -> score

    # Seção Encaixe & Potencial
    if not secs["enc_pot"].empty:
        st.markdown("##### Encaixe & Potencial")
        for _, m in secs["enc_pot"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            val = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")
            respostas[mid] = val

    # Transversais Físicos
    if not secs["fisicos"].empty:
        st.markdown("##### Parâmetros Físicos (Transversais)")
        for _, m in secs["fisicos"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            val = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")
            respostas[mid] = val

    # Transversais Mentais
    if not secs["mentais"].empty:
        st.markdown("##### Parâmetros Mentais (Transversais)")
        for _, m in secs["mentais"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            val = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")
            respostas[mid] = val

    # Específicos da Categoria
    if not secs["especificos"].empty:
        st.markdown(f"##### Específicos da Posição ({sel_cat})")
        for _, m in secs["especificos"].iterrows():
            mid = m["metric_id"]; lab = m["label"]
            val = nota(lab, f"m_{mid}_{selecionado_id}_{ano}_{mes}_{perfil}")
            respostas[mid] = val

    # Tagging de Funções (Opção B)
    funcoes_text = st.text_input("Funções (tagging opcional — separa por ';')", value="")

    obs = st.text_area("Observações (visível apenas ao Administrador)")

    # Validação: todas obrigatórias respondidas
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
        if funcoes_text.strip():
            save_funcoes_tag(ano, mes, perfil, int(sel["player_id"]), funcoes_text.strip())
        st.session_state["session_completed"].add((perfil,ano,mes,int(sel["player_id"])))
        st.success("✅ Avaliação registada.")
        st.rerun()

    if not can_submit:
        st.info("⚠️ Responda todas as métricas obrigatórias (1–4) antes de submeter.")

    # Estado do mês
    aval_all = load_avaliacoes()
    completos = [int(r["player_id"]) for _, r in players.iterrows()
                 if is_completed_for_player(aval_all, metrics, perfil, ano, mes, int(r["player_id"]), str(r["category"]).upper())]
    st.write(f"**Estado do mês:** {len(completos)}/{len(players)} jogadores avaliados.")

# ---- COL2: Instruções + Painel Admin (Radares)
with col2:
    st.markdown("#### Instruções")
    st.markdown("""
    <ol style="line-height:1.7; font-size:.95rem;">
      <li>Escolha o <strong>jogador</strong> na barra lateral.</li>
      <li>Preencha todos os <strong>parâmetros obrigatórios</strong> (1–4).</li>
      <li>Clique <strong>Submeter avaliação</strong> — a gravação é 1 linha por métrica.</li>
    </ol>
    <p style="font-style: italic; font-size:.9rem;">
      As avaliações só são visíveis ao <strong>Administrador</strong>. O mês fecha quando os <strong>25/25</strong> estiverem completos.
    </p>
    """, unsafe_allow_html=True)

    if perfil == "Administrador":
        st.markdown("---")
        st.markdown("#### Painel do Administrador — Radar do Jogador Selecionado")

        df = load_avaliacoes()
        if not df.empty:
            try:
                m = ((df["ano"].astype(int)==ano) &
                     (df["mes"].astype(int)==mes) &
                     (df["player_id"].astype(int)==int(sel["player_id"])))
                d = df.loc[m].copy()
            except Exception:
                d = pd.DataFrame()
        else:
            d = pd.DataFrame()

        if d.empty:
            st.info("Ainda não há avaliações para este jogador neste mês.")
        else:
            d["score"] = pd.to_numeric(d["score"], errors="coerce")
            agg = (d.groupby("metric_id")["score"]
                     .apply(lambda s: (lambda vals: (sum(vals)-min(vals)-max(vals))/(len(vals)-2) if len(vals)>=3 else (sum(vals)/len(vals)))([float(x) for x in s.dropna().tolist()]))
                     .reset_index().rename(columns={"score":"tm"}))

            md = metrics[["metric_id","label","group","category"]].drop_duplicates()
            merged = agg.merge(md, on="metric_id", how="left")

            def radar_plot(df_sub: pd.DataFrame, title: str):
                if df_sub.empty:
                    st.info(f"Sem dados para o radar: {title}")
                    return
                df_sub = df_sub.dropna(subset=["tm"]).sort_values("label")
                if df_sub.empty:
                    st.info(f"Sem dados para o radar: {title}")
                    return
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=df_sub["tm"].tolist(),
                    theta=df_sub["label"].tolist(),
                    fill='toself',
                    name=title
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[1,4])),
                    showlegend=False,
                    margin=dict(l=20,r=20,t=30,b=20),
                    title=title
                )
                st.plotly_chart(fig, use_container_width=True)

            radar_plot(merged[merged["group"]=="fisicos"], "Radar Físico")
            radar_plot(merged[merged["group"]=="mentais"], "Radar Mental")
            radar_plot(merged[(merged["group"]=="categoria") & (merged["category"]==sel_cat)], f"Radar Específico ({sel_cat})")

    st.markdown("---")
    st.caption("© Leixões SC")
