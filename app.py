# =========================
# Leixões SC — Avaliação de Plantel (Streamlit)
# =========================
import os
import json
import datetime as dt
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --- Google Sheets (opcional)
try:
    import gspread
    from google.oauth2.service_account import Credentials
except Exception:
    gspread = None
    Credentials = None

# ============= CONFIG ============
st.set_page_config(page_title="Leixões SC — Avaliação de Plantel", layout="wide", page_icon="⚽")

# Ativar des/ativar leitura via Google Sheets
USE_SHEETS = True

# ID do teu Spreadsheet (ajusta!)
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "<COLOCA_AQUI_O_ID_DA_TUA_SHEET>")

# Nome das abas no Google Sheets
TAB_AVALIACOES = "avaliacoes"   # ano, mes, player_id, metric_key, nota, avaliador, ...
TAB_PLAYERS     = "players"      # player_id, number, name, category, photo_url ...
TAB_METRICS     = "metrics"      # metric_key, family(FISICO/MENTAL/ESPECIFICO), ordem, group (ou group_norm)
TAB_WEIGHTS     = "weights"      # perfil, group(ET/DD), w_in_group, ativo

# Fallback local (CSV) em /data
DATA_DIR = "data"

# Pesos de grupo
W_ET = 0.60
W_DD = 0.40

# ============= CSS (logo centrado e título vermelho) ============
CSS = """
<style>
/* Sidebar compacta e colada ao topo */
section[data-testid="stSidebar"] div[data-testid="stSidebarContent"]{ padding-top: 6px !important; }

/* Largura fixa da sidebar */
[data-testid="stSidebar"][aria-expanded="true"]{ min-width: 315px; max-width: 315px; }

/* Emblema centrado e sem espaço extra */
.leixoes-logo-wrap { display:flex; justify-content:center; margin: -8px 0 6px 0; }
.leixoes-logo-wrap img { display:block; margin:0 auto; max-width: 220px; height:auto; }

/* Título vermelho (Leixões) centrado */
.leixoes-title { text-align:center; color:#D71920; font-weight:700; margin: 0 0 8px 0; }

/* Botões dos jogadores (quando usados) */
.btn-wrap button { width: 100%; text-align: left; }

/* Ajustes gerais */
div.block-container { padding-top: .6rem; padding-bottom: .6rem; }
.stTabs [role="tablist"] { margin-bottom: .4rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ============= HELPERS GSheets / CSV ============

def get_gs_client():
    if not USE_SHEETS:
        return None
    if gspread is None or Credentials is None:
        return None
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
        else:
            # como alternativa, lê de variável de ambiente GOOGLE_APPLICATION_CREDENTIALS (json)
            creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON", "")
            creds_dict = json.loads(creds_json) if creds_json else {}
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        return gspread.authorize(creds)
    except Exception as e:
        st.warning(f"Não consegui autenticar no Google Sheets: {e}")
        return None

gs_client = get_gs_client()

def gs_read_df(tab_name: str) -> pd.DataFrame:
    if gs_client and USE_SHEETS and SPREADSHEET_ID and SPREADSHEET_ID != "<COLOCA_AQUI_O_ID_DA_TUA_SHEET>":
        try:
            sh = gs_client.open_by_key(SPREADSHEET_ID)
            ws = sh.worksheet(tab_name)
            df = pd.DataFrame(ws.get_all_records())
            return df
        except Exception as e:
            st.warning(f"[GS] Falha ao ler '{tab_name}': {e}")
    # fallback CSV local
    csv_path = os.path.join(DATA_DIR, f"{tab_name}.csv")
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            st.warning(f"[CSV] Falha ao ler '{csv_path}': {e}")
    return pd.DataFrame()

@st.cache_data
def load_players() -> pd.DataFrame:
    df = gs_read_df(TAB_PLAYERS)
    # normaliza
    if not df.empty:
        cols = {c.lower(): c for c in df.columns}
        ren = {}
        for want in ["player_id","id","number","nome","name","category","group","photo_url","foto"]:
            if want in cols: ren[cols[want]] = want
        df = df.rename(columns=ren)
        if "player_id" not in df.columns:
            if "id" in df.columns:
                df["player_id"] = df["id"]
        # coerções simples
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce")
        df = df.dropna(subset=["player_id"])
        df["player_id"] = df["player_id"].astype(int)
        # nome
        if "name" not in df.columns and "nome" in df.columns:
            df["name"] = df["nome"]
        # group/category
        if "category" not in df.columns:
            if "group" in df.columns:
                df["category"] = df["group"]
        # photo_url
        if "photo_url" not in df.columns and "foto" in df.columns:
            df["photo_url"] = df["foto"]
    return df

@st.cache_data
def load_metrics() -> pd.DataFrame:
    df = gs_read_df(TAB_METRICS)
    if not df.empty:
        cols = {c.lower(): c for c in df.columns}
        ren = {}
        for want in ["metric_key","family","ordem","group","group_norm","label"]:
            if want in cols: ren[cols[want]] = want
        df = df.rename(columns=ren)
        # coerções
        df["ordem"] = pd.to_numeric(df.get("ordem"), errors="coerce")
        df["family"] = df.get("family","").astype(str).str.upper()
        # normaliza chave
        df["metric_key"] = df.get("metric_key", df.get("key","")).astype(str)
        # group_norm
        if "group_norm" not in df.columns:
            df["group_norm"] = df.get("group", "")
    return df

@st.cache_data
def load_weights_df() -> pd.DataFrame:
    df = gs_read_df(TAB_WEIGHTS)

    if df.empty:
        return pd.DataFrame(columns=["perfil","group","w_in_group","ativo"])

    # normalização agressiva
    ren = {}
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ["perfil","utilizador","user","avaliador","nome"]:
            ren[c] = "perfil"
        elif cl in ["group","grupo","grup"]:
            ren[c] = "group"
        elif cl in ["w_in_group","peso","peso_grupo","peso_interno","win_group"]:
            ren[c] = "w_in_group"
        elif cl in ["ativo","active","enabled","is_active"]:
            ren[c] = "ativo"
    df = df.rename(columns=ren)

    # cria colunas em falta
    for need in ["perfil","group","w_in_group","ativo"]:
        if need not in df.columns:
            df[need] = None

    # limpeza
    df["perfil"] = df["perfil"].astype(str).str.strip()
    df["group"] = df["group"].astype(str).str.strip().str.upper()
    df["ativo"] = df["ativo"].astype(str).str.strip().str.upper().isin(["TRUE","1","YES","SIM"])
    df["w_in_group"] = pd.to_numeric(df["w_in_group"], errors="coerce")

    df = df[(df["perfil"]!="") & (df["group"].isin(["ET","DD"])) & (df["ativo"]==True)]
    df = df.dropna(subset=["w_in_group"])

    # normaliza pesos para somarem 1.0 dentro de cada grupo
    for g in ["ET","DD"]:
        m = df["group"]==g
        sw = df.loc[m,"w_in_group"].sum()
        if sw and sw > 0:
            df.loc[m,"w_in_group"] = df.loc[m,"w_in_group"] / sw

    return df

weights_df = load_weights_df()

def weights_maps(df: pd.DataFrame) -> Tuple[Dict[str,str], Dict[str,float]]:
    map_group = {r["perfil"]: r["group"] for _,r in df.iterrows()}
    map_wrole = {r["perfil"]: float(r["w_in_group"]) for _,r in df.iterrows()}
    return map_group, map_wrole

MAP_GROUP_BY_PERFIL, MAP_WROLE_BY_PERFIL = weights_maps(weights_df)

# ============= FUNÇÕES DE CÁLCULO (ponderado + trimming agregado) ============

def weighted_metric_mean(aval_df: pd.DataFrame, ano:int, mes:int, pid:int, metric_key:str) -> Optional[float]:
    if aval_df is None or aval_df.empty:
        return None
    df = aval_df[(aval_df["ano"]==ano) & (aval_df["mes"]==mes) &
                 (aval_df["player_id"]==pid) & (aval_df["metric_key"]==metric_key)].copy()
    if df.empty:
        return None

    # nome do avaliador
    df["perfil_norm"] = df.apply(lambda r: str(r.get("avaliador") or r.get("perfil") or r.get("user") or r.get("utilizador") or ""), axis=1)
    df["group"] = df["perfil_norm"].map(MAP_GROUP_BY_PERFIL)
    df["w_in_group"] = df["perfil_norm"].map(MAP_WROLE_BY_PERFIL)
    df["nota"] = pd.to_numeric(df.get("nota", df.get("valor", None)), errors="coerce")

    # filtra válidos
    df = df.dropna(subset=["nota","group","w_in_group"])
    if df.empty:
        return None

    # trimming no agregado
    df_sorted = df.sort_values("nota").reset_index(drop=True)
    n = len(df_sorted)
    if n >= 7 or (n >= 5 and (n-2) >= 3):
        df_trim = df_sorted.iloc[1:-1].copy()
    else:
        df_trim = df_sorted
    if df_trim.empty:
        return None

    # média por grupo (re-normalizando pesos dos perfis presentes)
    result_by_group = {}
    for g, w_group in [("ET", W_ET), ("DD", W_DD)]:
        part = df_trim[df_trim["group"]==g]
        if part.empty:
            continue
        w = part["w_in_group"].astype(float).values
        r = part["nota"].astype(float).values
        sw = np.sum(w)
        if sw <= 0:
            continue
        r_group = float(np.sum(w*r)/sw)
        result_by_group[g] = r_group

    if not result_by_group:
        return None

    # combina 60/40 ou reescala para 100% se faltar grupo
    if "ET" in result_by_group and "DD" in result_by_group:
        return result_by_group["ET"]*W_ET + result_by_group["DD"]*W_DD
    elif "ET" in result_by_group:
        return result_by_group["ET"]
    else:
        return result_by_group["DD"]

def family_weighted_mean(aval_df, metrics_df, ano:int, mes:int, pid:int, family:str, player_group:str) -> Optional[float]:
    if family=="ESPECIFICO":
        col_group = "group_norm" if "group_norm" in metrics_df.columns else "group"
        mask = (metrics_df["family"]=="ESPECIFICO") & (metrics_df[col_group].astype(str).str.upper()==str(player_group).upper())
    else:
        mask = (metrics_df["family"]==family)
    keys = metrics_df.loc[mask].sort_values("ordem")["metric_key"].tolist()
    vals = []
    for k in keys:
        wm = weighted_metric_mean(aval_df, ano, mes, pid, k)
        if wm is not None:
            vals.append(wm)
    return float(np.mean(vals)) if vals else None

def standalone_weighted_mean(aval_df, ano:int, mes:int, pid:int, field:str) -> Optional[float]:
    return weighted_metric_mean(aval_df, ano, mes, pid, field)

def grade_letter(x: Optional[float]) -> str:
    if x is None: return "-"
    if x >= 3.5: return "A"
    if x >= 3.0: return "B"
    if x >= 2.0: return "C"
    return "D"

def potential_suffix(x: Optional[float]) -> str:
    if x is None: return ""
    if x >= 3.5: return "++"
    if x >= 3.0: return "+"
    if x >= 2.5: return ""
    return "-"

# ============= CARREGAR DADOS PRINCIPAIS ============
players  = load_players()
metrics  = load_metrics()

# Avaliações pode ser pesado; lê sob demanda quando precisar
@st.cache_data
def load_avaliacoes() -> pd.DataFrame:
    df = gs_read_df(TAB_AVALIACOES)
    if not df.empty:
        ren = {}
        for c in df.columns:
            cl = str(c).lower()
            if cl in ["ano","year"]: ren[c] = "ano"
            elif cl in ["mes","month"]: ren[c] = "mes"
            elif cl in ["player_id","id_jogador"]: ren[c] = "player_id"
            elif cl in ["metric_key","metric","pergunta","campo"]: ren[c] = "metric_key"
            elif cl in ["nota","score","valor"]: ren[c] = "nota"
            elif cl in ["avaliador","perfil","user","utilizador"]: ren[c] = "avaliador"
        df = df.rename(columns=ren)
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
        df["player_id"] = pd.to_numeric(df["player_id"], errors="coerce").astype("Int64")
    return df

# ============= UTIL / UI ============

def get_active_profiles_from_weights(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty: return []
    perfis = df["perfil"].dropna().astype(str).str.strip().tolist()
    seen=set(); out=[]
    for p in perfis:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def get_selected_player_id(players_df: pd.DataFrame) -> int:
    if "selected_player" in st.session_state:
        try: return int(st.session_state["selected_player"])
        except: pass
    if "selecionado_id" in st.session_state:
        try: return int(st.session_state["selecionado_id"])
        except: pass
    if not players_df.empty:
        return int(players_df.iloc[0]["player_id"])
    st.error("Sem jogadores disponíveis.")
    st.stop()

# ============= SIDEBAR ============

with st.sidebar:
    # logo
    logo_path_local = os.path.join(DATA_DIR, "logo.png")
    st.markdown('<div class="leixoes-logo-wrap">', unsafe_allow_html=True)
    if os.path.exists(logo_path_local):
        st.image(logo_path_local, use_column_width=False)
    else:
        st.write("")  # sem logo local
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div class="leixoes-title">Leixões SC — Avaliação de Plantel</div>', unsafe_allow_html=True)

    # Ano / Mês
    now = dt.date.today()
    anos = list(range(now.year-1, now.year+2))
    ano_sel = st.number_input("Ano", value=now.year, step=1)
    mes_map = {1:"January",2:"February",3:"March",4:"April",5:"May",6:"June",
               7:"July",8:"August",9:"September",10:"October",11:"November",12:"December"}
    mes_nome = st.selectbox("Mês", options=list(mes_map.values()), index=now.month-1)
    mes_sel = [k for k,v in mes_map.items() if v==mes_nome][0]

    # Utilizador (dinâmico de weights)
    st.subheader("Utilizador")
    PERFIS_ATIVOS = get_active_profiles_from_weights(weights_df)
    perfis_dropdown = [""] + PERFIS_ATIVOS + (["Administrador"] if "Administrador" not in PERFIS_ATIVOS else [])
    perfil = st.selectbox("Perfil", options=perfis_dropdown, index=0,
                          format_func=lambda v: "— selecione —" if v=="" else v, key="perfil_select")

if perfil == "":
    st.warning("⚠️ Selecione o seu perfil na barra lateral para continuar.")
    st.stop()

# ============= CONTEÚDO PRINCIPAL ============

colL, colR = st.columns([1.2, 1])

# Seleção de jogador (para simplificar: selectbox; se quiseres, troca pelos botões da tua sidebar original)
with colL:
    st.subheader("Jogador selecionado")

    if players.empty:
        st.error("Sem jogadores na tabela 'players'.")
        st.stop()

    # lista de display "#num — name"
    if "number" in players.columns:
        players["display"] = players.apply(lambda r: f"#{int(r['number'])} — {r['name']}", axis=1)
    else:
        players["display"] = players["name"]

    idx_default = 0
    selected_display = st.selectbox("Jogador", options=players["display"].tolist(), index=idx_default, label_visibility="collapsed")
    # guarda em sessão
    sel_row = players[players["display"]==selected_display].iloc[0]
    st.session_state["selected_player"] = int(sel_row["player_id"])
    pid = get_selected_player_id(players)

    # foto
    if "photo_url" in sel_row and pd.notna(sel_row["photo_url"]) and str(sel_row["photo_url"]).strip():
        st.image(sel_row["photo_url"], width=260)
    st.markdown(f"**#{int(sel_row.get('number',0))} {sel_row.get('name','')}**")

with colR:
    st.subheader("Instruções")
    st.markdown(
        """
1. Escolha o **jogador** na barra lateral (ou acima).
2. Preencha todos os **parâmetros obrigatórios** (1–4).
3. Selecione as **Funções** (pelo menos uma).
4. Clique **Submeter avaliação** — a gravação é 1 linha por métrica.

As submissões só são visíveis ao **Administrador**.
        """
    )

# Carregar avaliações quando precisarmos
aval_all = load_avaliacoes()

# Descobrir o grupo (categoria) do jogador
player_group_raw = str(sel_row.get("category") or sel_row.get("group") or "").upper().strip()
player_group = player_group_raw  # já normalizado

# ===================== PERFIL ADMINISTRADOR =====================
if perfil == "Administrador":
    st.markdown("---")
    st.subheader("Painel do Administrador")

    # KPIs do mês
    fis_now = family_weighted_mean(aval_all, metrics, ano_sel, mes_sel, pid, "FISICO", player_group)
    men_now = family_weighted_mean(aval_all, metrics, ano_sel, mes_sel, pid, "MENTAL", player_group)
    esp_now = family_weighted_mean(aval_all, metrics, ano_sel, mes_sel, pid, "ESPECIFICO", player_group)
    lsc_now = standalone_weighted_mean(aval_all, ano_sel, mes_sel, pid, "perfil_lsc")
    pot_now = standalone_weighted_mean(aval_all, ano_sel, mes_sel, pid, "potencial")

    # Média global
    fam_vals = [x for x in [fis_now, men_now, esp_now] if x is not None]
    media_global = float(np.mean(fam_vals)) if fam_vals else None

    gcol1, gcol2, gcol3, gcol4, gcol5 = st.columns(5)
    with gcol1: st.metric("Físico", f"{fis_now:.2f}" if fis_now is not None else "-", grade_letter(fis_now))
    with gcol2: st.metric("Mental", f"{men_now:.2f}" if men_now is not None else "-", grade_letter(men_now))
    with gcol3: st.metric("Específico", f"{esp_now:.2f}" if esp_now is not None else "-", grade_letter(esp_now))
    with gcol4: st.metric("Perfil LSC", f"{lsc_now:.2f}" if lsc_now is not None else "-", grade_letter(lsc_now))
    with gcol5: st.metric("Média Global", (f"{media_global:.2f}" if media_global is not None else "-"),
                           grade_letter(media_global) + potential_suffix(pot_now))

    # Radar simples (Físico / Mental / Específico)
    st.markdown("#### Radar (Mês atual)")
    cats = ["Físico","Mental","Específico"]
    vals = [v if v is not None else 0 for v in [fis_now, men_now, esp_now]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats+[cats[0]], fill='toself', name=f"{mes_nome} {ano_sel}"))
    fig.update_polars(radialaxis=dict(range=[0,4], dtick=1, showline=True))
    fig.update_layout(height=380, margin=dict(l=30,r=30,t=20,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.info("O painel usa **médias ponderadas** (60/40 ET/DD) com **trimming agregado**. A Média Global é a média simples das 3 famílias.")
    st.stop()

# ===================== FORMULÁRIO DE AVALIAÇÃO =====================
st.markdown("---")
st.subheader("Formulário de Avaliação")

# Perguntas stand-alone
st.markdown("**Encaixe no Perfil Leixões**")
perfil_lsc = st.radio("Encaixe no Perfil Leixões", options=[1,2,3,4], horizontal=True, label_visibility="collapsed", key="perfil_lsc_radio")

st.markdown("**Potencial Futuro**")
potencial = st.radio("Potencial Futuro", options=[1,2,3,4], horizontal=True, label_visibility="collapsed", key="potencial_radio")

# Parâmetros por família
def render_family_block(family_label, family_code):
    st.markdown(f"### {family_label}")
    # filtra métricas
    if family_code=="ESPECIFICO":
        col_group = "group_norm" if "group_norm" in metrics.columns else "group"
        m = (metrics["family"]=="ESPECIFICO") & (metrics[col_group].astype(str).str.upper()==player_group.upper())
        dfm = metrics[m].sort_values("ordem")
    else:
        dfm = metrics[metrics["family"]==family_code].sort_values("ordem")
    out = {}
    for _, r in dfm.iterrows():
        mkey = r["metric_key"]
        label = str(r.get("label") or r["metric_key"]).replace("_"," ").title()
        out[mkey] = st.radio(label, options=[1,2,3,4], horizontal=True, key=f"radio_{mkey}")
    return out

st.markdown("### Parâmetros Físicos")
ans_fis = render_family_block("Parâmetros Físicos","FISICO")

st.markdown("### Parâmetros Mentais")
ans_men = render_family_block("Parâmetros Mentais","MENTAL")

st.markdown("### Parâmetros Específicos à Função/Grupo")
ans_esp = render_family_block("Parâmetros Específicos","ESPECIFICO")

# Funções (dropdown multi) — de CSV local (podes trocar por Sheet se quiseres)
funcoes_path = os.path.join(DATA_DIR,"funcoes.csv")
funcoes_options = []
if os.path.exists(funcoes_path):
    try:
        funcoes_df = pd.read_csv(funcoes_path)
        col = [c for c in funcoes_df.columns if c.lower().strip() in ["funcao","função","nome","name","label"]]
        if col:
            funcoes_options = funcoes_df[col[0]].dropna().astype(str).tolist()
    except: pass

funcoes_sel = st.multiselect("Funções em que apresenta domínio funcional (obrigatório)", options=funcoes_options)

observ = st.text_area("Observações (visível apenas ao Administrador)", "")

def build_rows_to_append(ano,mes,pid,perfil,vals_dict) -> List[dict]:
    rows=[]
    for k,v in vals_dict.items():
        rows.append({
            "ano": int(ano),
            "mes": int(mes),
            "player_id": int(pid),
            "metric_key": str(k),
            "nota": int(v),
            "avaliador": str(perfil),
            "funcoes": ", ".join(funcoes_sel) if funcoes_sel else "",
            "obs": observ
        })
    return rows

# Botão de submissão
if st.button("Submeter avaliação", type="primary", use_container_width=False):
    if not funcoes_sel:
        st.error("Selecione pelo menos uma função.")
        st.stop()

    # juntar todas as respostas
    to_save = {}
    to_save["perfil_lsc"] = perfil_lsc
    to_save["potencial"] = potencial
    to_save.update(ans_fis)
    to_save.update(ans_men)
    to_save.update(ans_esp)

    rows = build_rows_to_append(ano_sel, mes_sel, pid, perfil, to_save)

    # gravar (GS ou CSV local append)
    if gs_client and USE_SHEETS and SPREADSHEET_ID and SPREADSHEET_ID != "<COLOCA_AQUI_O_ID_DA_TUA_SHEET>":
        try:
            sh = gs_client.open_by_key(SPREADSHEET_ID)
            ws = sh.worksheet(TAB_AVALIACOES)
            # garantir cabeçalhos
            if ws.row_count < 1:
                ws.append_row(list(rows[0].keys()))
            # append rows
            ws.append_rows([list(r.values()) for r in rows], value_input_option="RAW")
            st.success("Avaliação submetida com sucesso ao Google Sheets.")
        except Exception as e:
            st.error(f"Falha ao gravar no Google Sheets: {e}")
    else:
        # fallback local: cria/append CSV
        out_csv = os.path.join(DATA_DIR,f"{TAB_AVALIACOES}.csv")
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            df_old = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
            df_new = pd.DataFrame(rows)
            df_out = pd.concat([df_old, df_new], ignore_index=True)
            df_out.to_csv(out_csv, index=False)
            st.success(f"Avaliação gravada no CSV local: {out_csv}")
        except Exception as e:
            st.error(f"Falha ao gravar CSV local: {e}")
