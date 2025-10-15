import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import altair as alt
from datetime import datetime

# =====================
# Instruções rápidas
# =====================
# Local:
#   pip install -r requirements.txt
#   streamlit run app.py
# Cloud:
#   Subir pastas/ficheiros para um repo GitHub e publicar no Streamlit Cloud.
# Admin code: leixoes2025

PRIMARY = "#d22222"  # vermelho
BLACK = "#111111"

st.set_page_config(page_title="Leixões — Avaliação", layout="wide")

st.markdown(f'''
<style>
.block-container {{ padding-top: 1.2rem; }}
h1, h2, h3, h4 {{ color: {BLACK}; }}
.sidebar-title {{ color: {PRIMARY}; font-weight: 700; font-size: 1.2rem; margin-bottom: .5rem; }}
.btn-red > button {{ background: {PRIMARY} !important; color: white !important; border: 0 !important; }}
.badge {{ display: inline-block; padding: 2px 8px; border:1px solid #ddd; border-radius: 999px; font-size:.8rem; }}
hr {{ border-top: 1px solid #eee; }}
</style>
''', unsafe_allow_html=True)

@st.cache_data(ttl=5)
def load_players():
    return pd.read_csv("data/jogadores.csv")

@st.cache_data(ttl=5)
def load_functions():
    return pd.read_csv("data/funcoes.csv")

def ensure_avaliacoes_file():
    try:
        pd.read_csv("data/avaliacoes.csv")
    except Exception:
        df = pd.DataFrame(columns=["timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
                                   "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial","funcoes","observacoes"])
        df.to_csv("data/avaliacoes.csv", index=False)

def read_avaliacoes():
    ensure_avaliacoes_file()
    try:
        return pd.read_csv("data/avaliacoes.csv")
    except Exception:
        return pd.DataFrame(columns=["timestamp","ano","mes","avaliador","player_id","player_numero","player_nome",
                                   "encaixe","fisicas","mentais","impacto_of","impacto_def","potencial","funcoes","observacoes"])

def append_avaliacao(row: dict):
    df = read_avaliacoes()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv("data/avaliacoes.csv", index=False)
    st.success("✅ Avaliação registada com sucesso.")

def trimmed_mean(vals):
    vals = [float(v) for v in vals if pd.notna(v)]
    n = len(vals)
    if n == 0:
        return None
    if n >= 3:
        return (sum(vals) - min(vals) - max(vals)) / (n - 2)
    return sum(vals)/n

def month_label(y:int,m:int):
    return datetime(y,m,1).strftime("%B-%y").capitalize()

players = load_players()
funcs = load_functions()

st.sidebar.markdown('<div class="sidebar-title">Leixões — Avaliação</div>', unsafe_allow_html=True)
perfil = st.sidebar.selectbox("Perfil de utilizador", ["Avaliador 1","Avaliador 2","Avaliador 3","Avaliador 4","Avaliador 5","Avaliador 6","Avaliador 7","Administrador"])

if perfil == "Administrador":
    codigo = st.sidebar.text_input("Código de acesso", type="password", value="")
    if codigo != "leixoes2025":
        st.sidebar.warning("Acesso restrito: introduza o código.")
        st.stop()

today = datetime.today()
ano = st.sidebar.number_input("Ano", min_value=2024, max_value=2100, value=today.year, step=1)
mes = st.sidebar.selectbox("Mês", list(range(1,13)), index=today.month-1, format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize())

st.sidebar.markdown("---")
st.sidebar.write("🏃 **Jogadores**")
selecionado_id = st.session_state.get("selecionado_id", int(players.iloc[0].id))
for _, row in players.iterrows():
    label = f"{int(row['numero']):02d} — {row['nome']}"
    if st.sidebar.button(label, key=f"btn_{row['id']}"):
        selecionado_id = int(row['id'])
st.session_state["selecionado_id"] = selecionado_id
selecionado = players[players["id"]==selecionado_id].iloc[0]

col1, col2 = st.columns([1.2, 2.2], gap="large")

with col1:
    st.markdown(f"### Jogador selecionado")
    st.markdown(f"<span class='badge'>#{int(selecionado['numero'])}</span> **{selecionado['nome']}**", unsafe_allow_html=True)
    st.markdown("---")

    if perfil != "Administrador":
        st.subheader("Formulário de Avaliação")
        def radio(label):
            return st.radio(label, [1,2,3,4], horizontal=True)
        encaixe = radio("Encaixe no Perfil Leixões")
        fisicas = radio("Capacidades Físicas Exigidas")
        mentais = radio("Capacidades Mentais Exigidas")
        imp_of = radio("Impacto Ofensivo na Equipa")
        imp_def = radio("Impacto Defensivo na Equipa")
        potencial = radio("Potencial Futuro")

        mult_opts = funcs["nome"].tolist()
        fun_sel = st.multiselect("Funções em que apresenta domínio funcional", options=mult_opts, help="Selecione uma ou mais funções (obrigatório).")
        obs = st.text_area("Observações (visível apenas ao Administrador)")

        can_submit = len(fun_sel) > 0
        if not can_submit:
            st.info("Selecione pelo menos uma função para poder submeter.")

        if st.button("Submeter", type="primary", disabled=not can_submit):
            row = dict(
                timestamp=datetime.utcnow().isoformat(),
                ano=int(ano), mes=int(mes),
                avaliador=perfil,
                player_id=int(selecionado['id']),
                player_numero=int(selecionado['numero']),
                player_nome=selecionado['nome'],
                encaixe=int(encaixe), fisicas=int(fisicas), mentais=int(mentais),
                impacto_of=int(imp_of), impacto_def=int(imp_def), potencial=int(potencial),
                funcoes=";".join(fun_sel),
                observacoes=obs.replace("\n"," ").strip()
            )
            append_avaliacao(row)
        st.caption("Notas submetidas não ficam visíveis ao avaliador; apenas o Administrador acede ao histórico.")

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
            filt_mes = st.selectbox("Mês", meses_disp, index=len(meses_disp)-1, format_func=lambda m: datetime(2000,m,1).strftime("%B").capitalize())

        df_m = df[(df["ano"]==filt_ano) & (df["mes"]==filt_mes)]
        st.markdown(f"**Período:** {datetime(2000,filt_mes,1).strftime('%B').capitalize()} {filt_ano}")

        if df_m.empty:
            st.info("Sem submissões para este período.")
        else:
            dims = ["encaixe","fisicas","mentais","impacto_of","impacto_def","potencial"]
            rows = []
            for pid, g in df_m.groupby("player_id"):
                rec = {"player_id": int(pid), "player_numero": int(g.iloc[0]['player_numero']), "player_nome": g.iloc[0]['player_nome']}
                medias = []
                for d in dims:
                    val = trimmed_mean(g[d].astype(float).tolist())
                    rec[f"media_{d}"] = val
                    if val is not None: medias.append(val)
                rec["media_global"] = float(np.mean(medias)) if medias else None
                rec["n_usadas"] = max(0, len(g)-2) if len(g)>=3 else len(g)
                rows.append(rec)
            agg = pd.DataFrame(rows).sort_values(["media_global","player_numero"], ascending=[False, True])

            st.markdown("#### Tabela de médias aparadas (por jogador)")
            st.dataframe(agg, use_container_width=True)

            csv_bytes = agg.to_csv(index=False).encode()
            st.download_button("📤 Exportar CSV do período", data=csv_bytes, file_name=f"agregados_{filt_ano}_{filt_mes}.csv", mime="text/csv")

            st.markdown("---")
            st.markdown("#### Média global por jogador (barras)")
            chart = alt.Chart(agg).mark_bar(color=PRIMARY).encode(
                x=alt.X('player_nome:N', sort='-y', title='Jogador'),
                y=alt.Y('media_global:Q', title='Média Global (1–4)'),
                tooltip=['player_nome','media_global']
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
                    if dfn.empty: continue
                    vals = []
                    for d in dims:
                        vals.append(trimmed_mean(dfn[d].astype(float).tolist()) or 0)
                    vals.append(vals[0])
                    fig.add_trace(go.Scatterpolar(r=vals, theta=theta, fill='none', name=f"{datetime(2000,msel,1).strftime('%B').capitalize()}-{str(filt_ano)[-2:]}", line=dict(color=PRIMARY)))
                fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,4], tickvals=[1,2,3,4])), showlegend=True, height=450)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("#### Evolução mensal da média global (linha)")
            pj2 = st.selectbox("Jogador (evolução)", agg["player_nome"].tolist(), key="evol")
            evol_rows = []
            for msel in sorted(df[df["ano"]==filt_ano]["mes"].unique().tolist()):
                dfn = df[(df["ano"]==filt_ano) & (df["mes"]==msel) & (df["player_nome"]==pj2)]
                if dfn.empty: continue
                medias = []
                for d in dims:
                    val = trimmed_mean(dfn[d].astype(float).tolist())
                    if val is not None: medias.append(val)
                if medias:
                    evol_rows.append({"mes": int(msel), "media_global": float(np.mean(medias))})
            if evol_rows:
                e = pd.DataFrame(evol_rows).sort_values("mes")
                e["Mes"] = e["mes"].apply(lambda m: datetime(2000,m,1).strftime('%b').capitalize())
                line = alt.Chart(e).mark_line(point=True, color=PRIMARY).encode(
                    x=alt.X('Mes:N', sort=None),
                    y=alt.Y('media_global:Q', title='Média Global'),
                    tooltip=['Mes','media_global']
                ).properties(height=260)
                st.altair_chart(line, use_container_width=True)
            else:
                st.info("Sem dados suficientes para evolução.")
    else:
        st.subheader("Instruções")
        st.write('''
        1. Escolha o **jogador** na barra lateral.
        2. Preencha as **seis dimensões** (1–4) e selecione as **funções** (pode escolher várias).
        3. Clique **Submeter** para registar a avaliação do mês selecionado.
        
        *As submissões ficam visíveis apenas ao **Administrador**.*
        ''')
