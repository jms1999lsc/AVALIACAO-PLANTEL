import os
from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# =====================
# Tema / Config
# =====================
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

DATA_DIR = "data"
AVALIACOES_CSV = os.path.join(DATA_DIR, "avaliacoes.csv")
FECHOS_CSV     = os.path.join(DATA_DIR, "fechos.csv")
PLAYERS_CSV    = os.path.join(DATA_DIR, "jogadores.csv")
FUNCOES_CSV    = os.path.join(DATA_DIR, "funcoes.csv")


# =====================
# Helpers de dados
# =====================
def ensure_files():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(AVALIACOES_CSV):
        pd.DataFrame(
            columns=[
                "timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
                "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial",
                "funcoes","observacoes",
            ]
        ).to_csv(AVALIACOES_CSV, index=False)
    if not os.path.exists(FECHOS_CSV):
        pd.DataFrame(
            columns=["timestamp","ano","mes","avaliador","completos","total","status"]
        ).to_csv(FECHOS_CSV, index=False)

@st.cache_data(ttl=5)
def load_players():
    return pd.read_csv(PLAYERS_CSV)

@st.cache_data(ttl=5)
def load_functions():
    return pd.read_csv(FUNCOES_CSV)

def read_avaliacoes():
    ensure_files()
    try:
        return pd.read_csv(AVALIACOES_CSV)
    except Exception:
        return pd.DataFrame()

def save_avaliacao(row: dict):
    df = read_avaliacoes()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(AVALIACOES_CSV, index=False)

def read_fechos():
    ensure_files()
    try:
        return pd.read_csv(FECHOS_CSV)
    except Exception:
        return pd.DataFrame(columns=["timestamp","ano","mes","avaliador","completos","total","status"])

def fechar_mes(avaliador: str, ano: int, mes: int, completos: int, total: int):
    df = read_fechos()
    # evita duplicados
    mask = (df["avaliador"]==avaliador) & (df["ano"]==ano) & (df["mes"]==mes)
    df = df[~mask]
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "ano": ano, "mes": mes, "avaliador": avaliador,
        "completos": completos, "total": total, "status": "SUBMETIDO"
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(FECHOS_CSV, index=False)

def trimmed_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    n = len(vals)
    if n == 0:
        return None
    if n >= 3:
        return (sum(vals) - min(vals) - max(vals)) / (n - 2)
    return sum(vals)/n

def foto_path(player_id: int):
    p = f"assets/fotos/{player_id}.jpg"
    # placeholder online; se preferires, troca por uma imagem local
    return p if os.path.exists(p) else "https://placehold.co/60x60?text=%20"

def is_completed(df: pd.DataFrame, avaliador: str, ano: int, mes: int, player_id: int) -> bool:
    return not df[
        (df["avaliador"]==avaliador) &
        (df["ano"]==ano) &
        (df["mes"]==mes) &
        (df["player_id"]==player_id)
    ].empty


# =====================
# Dados base + Topbar
# =====================
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


# =====================
# Sidebar — Perfil + Jogadores
# =====================
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
    # “card” do jogador
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


# =====================
# Layout principal
# =====================
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

        # controlos tipo “segmentos” 1-4
        def seg(label, default=3):
            try:
                return st.segmented_control(label, options=[1,2,3,4], default=default)
            except Exception:
                # fallback para versões mais antigas do streamlit
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

        # Submissão global do mês (apenas informativo/registo)
        df_all = read_avaliacoes()  # recarrega
        completos_ids = [int(pid) for pid in players["id"].tolist() if is_completed(df_all, perfil, ano, mes, int(pid))]
        falta = len(players) - len(completos_ids)
        st.markdown("---")
        st.write(f"**Estado do mês:** {len(completos_ids)}/{len(players)} jogadores avaliados.")
        # verificar se já foi fechado antes
        df_fechos = read_fechos()
        ja_fechado = not df_fechos[
            (df_fechos["avaliador"]==perfil) & (df_fechos["ano"]==ano) & (df_fechos["mes"]==mes)
        ].empty
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
