# app.py — Leixões Plataforma de Avaliação (Streamlit + Google Sheets)

import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =========================
# Configuração visual/tema
# =========================
PRIMARY = "#d22222"   # vermelho Leixões
BLACK   = "#111111"
GREEN   = "#2e7d32"

st.set_page_config(page_title="Leixões — Avaliação", layout="wide")

st.markdown(
    f"""
<style>
.block-container {{ padding-top: .6rem; }}
h1, h2, h3, h4 {{ color: {BLACK}; }}
.sidebar-title {{ color: {PRIMARY}; font-weight: 700; font-size: 1.1rem; margin-bottom: .5rem; }}
.badge {{ display:inline-block; padding:2px 8px; border:1px solid #ddd; border-radius:999px; font-size:.75rem; }}
.player-card {{ padding:6px 6px; border-radius:10px; border:1px solid #eee; margin-bottom:6px; }}
.player-card:hover {{ background:#fafafa; }}
.small {{ font-size:.85rem; color:#666; }}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Caminhos e ficheiros
# =========================
DATA_DIR       = "data"
PLAYERS_CSV    = os.path.join(DATA_DIR, "jogadores.csv")
FUNCOES_CSV    = os.path.join(DATA_DIR, "funcoes.csv")
AVALIACOES_CSV = os.path.join(DATA_DIR, "avaliacoes.csv")
FECHOS_CSV     = os.path.join(DATA_DIR, "fechos.csv")

# ============================================================
# 🔗 INTEGRAÇÃO COM GOOGLE SHEETS (com fallback local em CSV)
# ============================================================
USE_SHEETS = True  # True = usa Google Sheets / False = CSV local

def _get_gspread_client():
    """Constrói cliente gspread com credenciais dos secrets."""
    import gspread
    from google.oauth2.service_account import Credentials
    sa = st.secrets["gcp_service_account"]  # precisa do bloco TOML nos Secrets
    creds = Credentials.from_service_account_info(dict(sa), scopes=[
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ])
    return gspread.authorize(creds)

def _get_sheet_id():
    """Obtém o SHEET_ID dos secrets. Aceita ID puro ou URL completa."""
    # 1) tentar dentro do bloco [gcp_service_account]
    sid = None
    try:
        sid = st.secrets["gcp_service_account"].get("SHEET_ID", None)
    except Exception:
        pass
    # 2) fallback para nível raiz
    if not sid:
        sid = st.secrets.get("SHEET_ID", None)
    if not sid:
        raise ValueError("SHEET_ID não encontrado nos secrets. Defina-o em [gcp_service_account].")

    sid = str(sid).strip()

    # aceitar URL completa e extrair o ID
    if "docs.google.com/spreadsheets/d/" in sid:
        # exemplo: https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
        try:
            sid = sid.split("/d/")[1].split("/")[0]
        except Exception:
            raise ValueError("Não foi possível extrair o SHEET_ID a partir da URL fornecida.")
    return sid

def _open_sheet():
    import gspread
    from gspread.exceptions import SpreadsheetNotFound, APIError

    gc = _get_gspread_client()
    sid = _get_sheet_id()
    try:
        return gc.open_by_key(sid)
    except SpreadsheetNotFound as e:
        # 404 específico — ID errado ou service account sem acesso (não partilhou)
        raise RuntimeError(
            "SpreadsheetNotFound (404). Verifique: "
            "1) o SHEET_ID está correto; "
            "2) a folha foi partilhada com a service account como Editor."
        ) from e
    except APIError as e:
        # outros 404 de API
        raise RuntimeError(f"APIError ao abrir a Sheet (possível 404). Detalhe: {e}") from e

REQUIRED_TABS = {
    "avaliacoes": [
        "timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
        "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial",
        "funcoes","observacoes",
    ],
    "fechos": ["timestamp","ano","mes","avaliador","completos","total","status"],
}
def ensure_gs_tabs():
    """Garante que as abas exigidas existem com os cabeçalhos corretos."""
    from gspread.exceptions import WorksheetNotFound
    sid = _get_sheet_id()
    sh = _open_sheet()  # se falhar aqui, já sai com mensagem clara
    existing = {ws.title for ws in sh.worksheets()}
    for tab, header in REQUIRED_TABS.items():
        if tab not in existing:
            ws = sh.add_worksheet(title=tab, rows=1000, cols=max(len(header), 10))
            ws.update([header])
        else:
            ws = sh.worksheet(tab)
            first_row = ws.row_values(1)
            if not first_row:
                ws.update([header])

def gs_read(sheet_name: str) -> pd.DataFrame:
    """Lê dados de uma aba; cria-a se ainda não existir."""
    try:
        sh = _open_sheet()
        try:
            ws = sh.worksheet(sheet_name)
        except WorksheetNotFound:
            ensure_gs_tabs()
            ws = sh.worksheet(sheet_name)
        data = ws.get_all_records()
        return pd.DataFrame(data) if data else pd.DataFrame()
    except Exception as e:
        st.warning(f"⚠️ Falha ao ler '{sheet_name}' da Google Sheet: {e}")
        return pd.DataFrame()

def gs_append(sheet_name: str, row_dict: dict):
    """Adiciona linha; cria aba e cabeçalho se precisarem."""
    try:
        sh = _open_sheet()
        try:
            ws = sh.worksheet(sheet_name)
        except WorksheetNotFound:
            ensure_gs_tabs()
            ws = sh.worksheet(sheet_name)
        header = ws.row_values(1)
        if not header:
            header = REQUIRED_TABS.get(sheet_name, list(row_dict.keys()))
            ws.update([header])
        row = [row_dict.get(col, "") for col in header]
        ws.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.error(f"Erro ao gravar no Google Sheets em '{sheet_name}': {e}")

def gs_replace_all(sheet_name: str, df: pd.DataFrame):
    """Substitui todo o conteúdo de uma aba pelo DataFrame atual."""
    try:
        sh = _open_sheet()
        ws = sh.worksheet(sheet_name)
        ws.clear()
        ws.update([df.columns.tolist()] + df.astype(str).values.tolist())
    except Exception as e:
        st.error(f"Erro ao atualizar aba '{sheet_name}': {e}")

# ============================================================
# 📦 Ficheiros locais (seeding + leitura robusta)
# ============================================================
def ensure_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    # avaliacoes.csv
    if not os.path.exists(AVALIACOES_CSV):
        pd.DataFrame(columns=[
            "timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
            "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial",
            "funcoes","observacoes",
        ]).to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")
    # fechos.csv
    if not os.path.exists(FECHOS_CSV):
        pd.DataFrame(columns=["timestamp","ano","mes","avaliador","completos","total","status"]).to_csv(
            FECHOS_CSV, index=False, encoding="utf-8"
        )

def _read_csv_flex(path: str) -> pd.DataFrame:
    """
    Lê CSV aceitando vírgula OU ponto e vírgula, e tenta UTF-8 → Latin-1.
    """
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, sep=None, engine="python", encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path)

@st.cache_data(ttl=5)
def load_players() -> pd.DataFrame:
    """Jogadores com tolerância a separadores/encoding e nomes de colunas."""
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
    """Funções com tolerância a separadores/encoding e nomes de colunas."""
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
# 📊 Leitura/Escrita de avaliações e fechos (Sheets ou CSV)
# ============================================================
def read_avaliacoes() -> pd.DataFrame:
    if USE_SHEETS:
        df = gs_read("avaliacoes")
        if df.empty:
            return df
        # Força coerência de tipos
        for col in ["player_id","ano","mes","encaixe","fisicas","mentais","impacto_of","impacto_def","potencial"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    else:
        return pd.read_csv(AVALIACOES_CSV) if os.path.exists(AVALIACOES_CSV) else pd.DataFrame()

def save_avaliacao(row: dict):
    """Adiciona uma avaliação individual."""
    if USE_SHEETS:
        gs_append("avaliacoes", row)
    else:
        df = pd.read_csv(AVALIACOES_CSV) if os.path.exists(AVALIACOES_CSV) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(AVALIACOES_CSV, index=False, encoding="utf-8")

def read_fechos() -> pd.DataFrame:
    if USE_SHEETS:
        return gs_read("fechos")
    else:
        return pd.read_csv(FECHOS_CSV) if os.path.exists(FECHOS_CSV) else pd.DataFrame()

def fechar_mes(avaliador: str, ano: int, mes: int, completos: int, total: int):
    """Regista fecho mensal do avaliador."""
    row = {
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "ano": int(ano),
        "mes": int(mes),
        "avaliador": avaliador,
        "completos": int(completos),
        "total": int(total),
        "status": "FECHADO" if completos == total else "INCOMPLETO",
    }
    if USE_SHEETS:
        gs_append("fechos", row)
    else:
        df = pd.read_csv(FECHOS_CSV) if os.path.exists(FECHOS_CSV) else pd.DataFrame()
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

def foto_path(player_id: int) -> str:
    """Foto do jogador (se existir) ou placeholder."""
    p = f"assets/fotos/{player_id}.jpg"
    return p if os.path.exists(p) else "https://placehold.co/60x60?text=%20"

def is_completed(df: pd.DataFrame, avaliador: str, ano: int, mes: int, player_id: int) -> bool:
    if df.empty:
        return False
    m = (df["avaliador"]==avaliador) & (df["ano"]==ano) & (df["mes"]==mes) & (df["player_id"]==player_id)
    return not df[m].empty

# =========================
# Carregamento base + Topbar
# =========================
players = load_players()
funcs   = load_functions()
df_all  = read_avaliacoes()

top_l, top_m, top_r = st.columns([0.22, 1, 0.55], vertical_alignment="center")
with top_l:
    if os.path.exists("assets/logo.png"):
        st.image("assets/logo.png", width=70)
with top_m:
    st.markdown("### Leixões — Plataforma de Avaliação")
with top_r:
    today = datetime.today()
    if "ano" not in st.session_state: st.session_state["ano"] = today.year
    if "mes" not in st.session_state: st.session_state["mes"] = today.month
    st.session_state["ano"] = st.number_input("Ano", min_value=2024, max_value=2100, value=st.session_state["ano"], step=1)
    st.session_state["mes"] = st.selectbox(
        "Mês",
        list(range(1,13)),
        index=st.session_state["mes"]-1,
        format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize()
    )
ano = int(st.session_state["ano"]); mes = int(st.session_state["mes"])
st.divider()
with st.expander("🔧 Diagnóstico Google Sheets", expanded=False):
    try:
        sid = _get_sheet_id()
        st.write("SHEET_ID efetivo:", sid)
        sh = _open_sheet()
        tabs = [ws.title for ws in sh.worksheets()]
        st.success("✅ Consegui abrir a Sheet.")
        st.write("Abas encontradas:", tabs)
        missing = [t for t in ["avaliacoes","fechos"] if t not in tabs]
        if missing:
            st.warning(f"Abas em falta: {missing}. Vou tentar criá-las automaticamente.")
            ensure_gs_tabs()
            st.info("Abas garantidas.")
    except Exception as e:
        import traceback
        st.error("❌ Erro ao abrir/validar a Sheet:")
        st.code("".join(traceback.format_exception_only(type(e), e)))

# Garante que as abas obrigatórias existem e têm cabeçalhos
if USE_SHEETS:
    try:
        ensure_gs_tabs()
    except Exception as _e:
        st.warning(f"Não consegui garantir as abas no Google Sheets: {_e}")

# =========================
# Sidebar — Perfil + Jogadores
# =========================
st.sidebar.markdown('<div class="sidebar-title">Utilizador</div>', unsafe_allow_html=True)
perfil = st.sidebar.selectbox(
    "Perfil",
    ["Avaliador 1","Avaliador 2","Avaliador 3","Avaliador 4","Avaliador 5","Avaliador 6","Avaliador 7","Administrador"]
)

if perfil == "Administrador":
    codigo = st.sidebar.text_input("Código de acesso", type="password", value="")
    if codigo != "leixoes2025":
        st.sidebar.warning("Acesso restrito: introduza o código.")
        st.stop()

st.sidebar.markdown("---")
st.sidebar.write("🏃 **Jogadores**")

# Progresso do avaliador no período
completos_ids = [int(pid) for pid in players["id"].tolist() if is_completed(df_all, perfil, ano, mes, int(pid))]
st.sidebar.progress(len(completos_ids)/len(players), text=f"Completos: {len(completos_ids)}/{len(players)}")

# Seleção do jogador
if "selecionado_id" not in st.session_state:
    st.session_state["selecionado_id"] = int(players.iloc[0].id)
selecionado_id = st.session_state["selecionado_id"]

for _, row in players.iterrows():
    pid = int(row["id"])
    foto = foto_path(pid)
    with st.sidebar.container():
        c1, c2, c3 = st.columns([0.35, 1.3, 0.6])
        c1.image(foto, width=36, clamp=True)
        label = f"#{int(row['numero']):02d} — {row['nome']}"
        if c2.button(label, key=f"sel_{pid}"):
            selecionado_id = pid
        done = pid in completos_ids
        c3.markdown(
            f"<span class='badge' style='border-color:{'#cfc' if done else '#eee'}; color:{GREEN if done else '#888'}'>{'🟢' if done else '—'}</span>",
            unsafe_allow_html=True
        )

st.session_state["selecionado_id"] = selecionado_id
selecionado = players[players["id"]==selecionado_id].iloc[0]

# =========================
# Layout principal
# =========================
col1, col2 = st.columns([1.2, 2.2], gap="large")

with col1:
    st.markdown("#### Jogador selecionado")
    st.markdown(
        f"<span class='badge'>#{int(selecionado['numero'])}</span> <b>{selecionado['nome']}</b>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    if perfil != "Administrador":
        st.subheader("Formulário de Avaliação")

        # controlos tipo “segmentos” 1-4 (com fallback)
        def seg(label, default=3):
            try:
                return st.segmented_control(label, options=[1,2,3,4], default=default)
            except Exception:
                return st.radio(label, [1,2,3,4], horizontal=True, index=default-1)

        encaixe   = seg("Encaixe no Perfil Leixões")
        fisicas   = seg("Capacidades Físicas Exigidas")
        mentais   = seg("Capacidades Mentais Exigidas")
        imp_of    = seg("Impacto Ofensivo na Equipa")
        imp_def   = seg("Impacto Defensivo na Equipa")
        potencial = seg("Potencial Futuro")

        mult_opts = funcs["nome"].tolist()
        fun_sel = st.multiselect(
            "Funções (obrigatório)",
            options=mult_opts,
            help="Pode escolher várias."
        )
        obs = st.text_area("Observações (visível apenas ao Administrador)")

        can_submit = len(fun_sel) > 0
        if st.button("Submeter avaliação", type="primary", disabled=not can_submit):
            row = dict(
                timestamp=datetime.utcnow().isoformat(),
                ano=ano, mes=mes, avaliador=perfil,
                player_id=int(selecionado["id"]),
                player_numero=int(selecionado["numero"]),
                player_nome=selecionado["nome"],
                encaixe=int(encaixe), fisicas=int(fisicas), mentais=int(mentais),
                impacto_of=int(imp_of), impacto_def=int(imp_def), potencial=int(potencial),
                funcoes=";".join(fun_sel), observacoes=obs.replace("\n"," ").strip()
            )
            save_avaliacao(row)
            st.success("✅ Avaliação registada.")
            st.rerun()

        # Submissão global do mês (informativo/registo)
        df_all = read_avaliacoes()  # recarrega
        completos_ids = [int(pid) for pid in players["id"].tolist() if is_completed(df_all, perfil, ano, mes, int(pid))]
        falta = len(players) - len(completos_ids)
        st.markdown("---")
        st.write(f"**Estado do mês:** {len(completos_ids)}/{len(players)} jogadores avaliados.")

        # verificar se já existe fecho para este avaliador/periodo
        df_fechos = read_fechos()
        ja_fechado = False
        if not df_fechos.empty:
            m = (df_fechos.get("avaliador","") == perfil) & (df_fechos.get("ano",0) == ano) & (df_fechos.get("mes",0) == mes)
            ja_fechado = not df_fechos[m].empty

        btn_disabled = (falta > 0) or ja_fechado
        if st.button("✅ Submeter mês (tudo preenchido)", type="secondary", disabled=btn_disabled,
                     help="Fica ativo quando os 25 estiverem avaliados. Regista o fecho deste período."):
            fechar_mes(perfil, ano, mes, len(completos_ids), len(players))
            st.success("📌 Mês marcado como submetido para este avaliador.")
            st.rerun()

with col2:
    if perfil == "Administrador":
        st.subheader("Dashboard do Administrador")

        df = read_avaliacoes()
        left, right = st.columns(2)
        anos_disp = sorted(df["ano"].dropna().unique().tolist() or [int(datetime.today().year)])
        with left:
            filt_ano = st.selectbox("Ano", anos_disp, index=len(anos_disp)-1)
        with right:
            meses_disp = sorted([int(x) for x in df[df["ano"]==filt_ano]["mes"].dropna().unique().tolist()] or [int(datetime.today().month)])
            filt_mes = st.selectbox("Mês", meses_disp, index=len(meses_disp)-1,
                                    format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize())

        df_m = df[(df["ano"]==filt_ano) & (df["mes"]==filt_mes)]
        st.markdown(f"**Período:** {datetime(2000,filt_mes,1).strftime('%B').capitalize()} {filt_ano}")

        if df_m.empty:
            st.info("Sem submissões para este período.")
        else:
            dims = ["encaixe","fisicas","mentais","impacto_of","impacto_def","potencial"]
            rows = []
            for pid, g in df_m.groupby("player_id"):
                rec = {
                    "player_id": int(pid),
                    "player_numero": int(g.iloc[0]["player_numero"]),
                    "player_nome": g.iloc[0]["player_nome"],
                }
                medias = []
                for d in dims:
                    val = trimmed_mean(g[d].astype(float).tolist())
                    rec[f"media_{d}"] = val
                    if val is not None:
                        medias.append(val)
                rec["media_global"] = float(np.mean(medias)) if medias else None
                rec["n_usadas"] = max(0, len(g)-2) if len(g)>=3 else len(g)
                rows.append(rec)
            agg = pd.DataFrame(rows).sort_values(["media_global","player_numero"], ascending=[False, True])

            st.markdown("#### Tabela de médias aparadas (por jogador)")
            st.dataframe(agg, use_container_width=True)

            csv_bytes = agg.to_csv(index=False).encode()
            st.download_button("📤 Exportar CSV do período", data=csv_bytes,
                               file_name=f"agregados_{filt_ano}_{filt_mes}.csv", mime="text/csv")

            st.markdown("---")
            st.markdown("#### Média global por jogador (barras)")
            chart = alt.Chart(agg).mark_bar(color=PRIMARY).encode(
                x=alt.X("player_nome:N", sort="-y", title="Jogador"),
                y=alt.Y("media_global:Q", title="Média Global (1–4)"),
                tooltip=["player_nome","media_global"]
            ).properties(height=320)
            st.altair_chart(chart, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Radar: comparação por jogador ao longo dos meses")
            pj = st.selectbox("Jogador", agg["player_nome"].tolist())
            meses_unicos = sorted(df[df["ano"]==filt_ano]["mes"].unique().tolist())
            meses_sel = st.multiselect("Meses a comparar", meses_unicos, default=[filt_mes])
            if meses_sel:
                categories = ["Encaixe","Cap. Físicas","Cap. Mentais","Imp. Ofensivo","Imp. Defensivo","Potencial"]
                theta = categories + categories[:1]
                fig = go.Figure()
                for msel in meses_sel:
                    dfn = df[(df["ano"]==filt_ano) & (df["mes"]==msel) & (df["player_nome"]==pj)]
                    if dfn.empty:
                        continue
                    vals = []
                    for d in dims:
                        vals.append(trimmed_mean(dfn[d].astype(float).tolist()) or 0)
                    vals.append(vals[0])  # fecha o polígono
                    fig.add_trace(
                        go.Scatterpolar(
                            r=vals, theta=theta, fill="none",
                            name=f"{datetime(2000,msel,1).strftime('%B').capitalize()}-{str(filt_ano)[-2:]}",
                            line=dict(color=PRIMARY)
                        )
                    )
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,4], tickvals=[1,2,3,4])),
                    showlegend=True, height=450
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Evolução mensal da média global (linha)")
            pj2 = st.selectbox("Jogador (evolução)", agg["player_nome"].tolist(), key="evol")
            evol_rows = []
            for msel in sorted(df[df["ano"]==filt_ano]["mes"].unique().tolist()):
                dfn = df[(df["ano"]==filt_ano) & (df["mes"]==msel) & (df["player_nome"]==pj2)]
                if dfn.empty:
                    continue
                medias = []
                for d in dims:
                    val = trimmed_mean(dfn[d].astype(float).tolist())
                    if val is not None:
                        medias.append(val)
                if medias:
                    evol_rows.append({"mes": int(msel), "media_global": float(np.mean(medias))})
            if evol_rows:
                e = pd.DataFrame(evol_rows).sort_values("mes")
                e["Mes"] = e["mes"].apply(lambda m: datetime(2000,m,1).strftime("%b").capitalize())
                line = alt.Chart(e).mark_line(point=True, color=PRIMARY).encode(
                    x=alt.X("Mes:N", sort=None),
                    y=alt.Y("media_global:Q", title="Média Global"),
                    tooltip=["Mes","media_global"]
                ).properties(height=260)
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("Sem dados suficientes para evolução.")
    else:
        st.subheader("Instruções")
        st.write(
            """
        1. Escolha o **jogador** na barra lateral.
        2. Preencha as **seis dimensões** (1–4) e selecione as **funções** (pode escolher várias).
        3. Clique **Submeter avaliação** para registar o mês selecionado.

        *As submissões ficam visíveis apenas ao **Administrador**.
        O botão **Submeter mês** só ativa quando os **25/25** estiverem preenchidos.*
        """
        )
