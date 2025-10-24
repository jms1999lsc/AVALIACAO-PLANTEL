# =========================
# Layout principal
# =========================
col1, col2 = st.columns([1.2, 2.2], gap="large")

# ---- COL1: Jogador + (Formulário para não-admin)
with col1:
    st.markdown("#### Jogador selecionado")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        st.markdown(
            f"""
            <div class='player-hero-title'>
              <span class='player-num'>#{int(sel['numero'])}</span>
              <span class='player-name'>{sel['nome']}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.image(foto_path_for(int(sel["player_id"]), 220), width=220, clamp=True)

    # --- Só mostra o formulário para perfis que NÃO sejam Administrador ---
    if perfil != "Administrador":
        st.markdown("### Formulário de Avaliação")

        secs = metrics_for_category(metrics, sel_cat)

        def nota(label: str, key: str):
            opcoes = ["—", 1, 2, 3, 4]
            escolha = st.radio(label, opcoes, horizontal=True, index=0, key=key)
            return None if escolha == "—" else escolha

        respostas = {}

        # Encaixe & Potencial
        if not secs["enc_pot"].empty:
            st.markdown("##### Encaixe & Potencial")
            for _, m in secs["enc_pot"].iterrows():
                mid_m = m["metric_id"]; lab_m = m["label"]
                respostas[mid_m] = nota(lab_m, f"m_{mid_m}_{selecionado_id}_{ano}_{mes}_{perfil}")

        # Físicos
        if not secs["fisicos"].empty:
            st.markdown("##### Parâmetros Físicos")
            for _, m in secs["fisicos"].iterrows():
                mid_m = m["metric_id"]; lab_m = m["label"]
                respostas[mid_m] = nota(lab_m, f"m_{mid_m}_{selecionado_id}_{ano}_{mes}_{perfil}")

        # Mentais
        if not secs["mentais"].empty:
            st.markdown("##### Parâmetros Mentais")
            for _, m in secs["mentais"].iterrows():
                mid_m = m["metric_id"]; lab_m = m["label"]
                respostas[mid_m] = nota(lab_m, f"m_{mid_m}_{selecionado_id}_{ano}_{mes}_{perfil}")

        # Específicos
        if not secs["especificos"].empty:
            st.markdown(f"##### Específicos da Posição ({sel_cat})")
            for _, m in secs["especificos"].iterrows():
                mid_m = m["metric_id"]; lab_m = m["label"]
                respostas[mid_m] = nota(lab_m, f"m_{mid_m}_{selecionado_id}_{ano}_{mes}_{perfil}")

        # ===== Funções (obrigatório, multiselect) — DENTRO do col1 =====
        st.markdown("### Posições em que apresenta domínio funcional")

        # usa o catálogo global lido no arranque (evita NameError)
        funcoes_df = FUNCOES_CAT

        if funcoes_df.empty:
            st.warning("Nenhuma função encontrada em data/funcoes.csv.")
            funcoes_escolhidas = []
        else:
            funcoes_disp = (
                funcoes_df["funcao"].dropna().astype(str).str.strip().unique().tolist()
                if "funcao" in funcoes_df.columns else []
            )
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
            can_submit = (len(faltam) == 0)

            if st.button("Submeter avaliação", type="primary", disabled=not can_submit):
                ts = datetime.utcnow().isoformat()
                base = dict(
                    timestamp=ts, ano=ano, mes=mes, avaliador=perfil,
                    player_id=int(sel["player_id"]), player_numero=int(sel["numero"]), player_nome=sel["nome"],
                    player_category=sel_cat, observacoes=obs.replace("\n", " ").strip()
                )

                # grava 1 linha por métrica
                rows = []
                for mid_key, val in respostas.items():
                    if val is None:
                        continue
                    rd = base.copy()
                    rd["metric_id"] = str(mid_key).upper()
                    rd["score"] = int(val)
                    rows.append(rd)
                if rows:
                    save_avaliacoes_bulk(rows)

                # Funções obrigatórias
                if len(funcoes_escolhidas) == 0:
                    st.error("Selecione pelo menos uma Função antes de submeter.")
                    st.stop()
                else:
                    funcoes_str = "; ".join(funcoes_escolhidas)
                    save_funcoes_tag(ano, mes, perfil, int(sel["player_id"]), funcoes_str)

                st.session_state["session_completed"].add((perfil, ano, mes, int(sel["player_id"])))
                st.success("✅ Avaliação registada.")
                st.rerun()

            if not can_submit:
                st.info("⚠️ Responda todas as métricas obrigatórias (1–4) antes de submeter.")

            # Estado do mês (barra de progresso usa o helper completed_for_player)
            aval_all = load_avaliacoes()
            completos = [
                int(r["player_id"]) for _, r in players.iterrows()
                if completed_for_player(int(r["player_id"]), str(r["category"]).upper())
            ]
            st.write(f"**Estado do mês:** {len(completos)}/{len(players)} jogadores avaliados.")

# ---- COL2: Instruções + Painel Admin (Radares)
with col2:
    st.markdown("#### Instruções")
    st.markdown("""
    <ol style="line-height:1.7; font-size:.95rem;">
      <li>Escolha o seu <strong>Nome de Utilizador</strong> na barra lateral.</li>
      <li>Escolha o <strong>jogador</strong>, mais abaixo na barra lateral.</li>
      <li>Preencha todos os <strong>parâmetros obrigatórios</strong> (1–4).</li>
      <li>Selecione as <strong>Funções</strong> (pelo menos uma).</li>
      <li>Clique <strong>Submeter avaliação</strong></li>
    </ol>
    <p style="font-style: italic; font-size:.9rem;">
      As avaliações só são visíveis ao <strong>Administrador</strong>. O mês fecha quando os <strong>25/25</strong> estiverem completos.
    </p>
    """, unsafe_allow_html=True)

    # ====== DASHBOARD DO ADMIN (só para Administrador) ======
    if perfil == "Administrador":
        st.markdown("---")
        st.markdown("#### Painel do Administrador — Radar do Jogador Selecionado")

        # Seleção correta do jogador e grupo
        pid = int(selecionado_id)
        sel_admin = players.loc[players["player_id"] == pid].iloc[0]
        player_group = str(sel_admin.get("category")).upper()

        # Períodos
        ano_sel = int(ano); mes_sel = int(mes)
        ano_prev, mes_prev = prev_period(ano_sel, mes_sel)

        # Médias por família (mês atual e anterior)
        fis_now  = family_clean_mean(aval_all, metrics, ano_sel, mes_sel, pid, "FISICO",      player_group)
        men_now  = family_clean_mean(aval_all, metrics, ano_sel, mes_sel, pid, "MENTAL",     player_group)
        esp_now  = family_clean_mean(aval_all, metrics, ano_sel, mes_sel, pid, "ESPECIFICO", player_group)

        fis_prv  = family_clean_mean(aval_all, metrics, ano_prev, mes_prev, pid, "FISICO",      player_group)
        men_prv  = family_clean_mean(aval_all, metrics, ano_prev, mes_prev, pid, "MENTAL",     player_group)
        esp_prv  = family_clean_mean(aval_all, metrics, ano_prev, mes_prev, pid, "ESPECIFICO", player_group)

        # Standalone
        lsc_now  = standalone_clean_mean(aval_all, ano_sel, mes_sel, pid, "perfil_lsc")
        pot_now  = standalone_clean_mean(aval_all, ano_sel, mes_sel, pid, "potencial")

        # Média Global (3 famílias) + etiqueta final
        medias = [x for x in [fis_now, men_now, esp_now] if x is not None]
        media_global = float(np.mean(medias)) if medias else None

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

        etiqueta = f"{letter_grade(media_global)}{potential_suffix(pot_now)}"

        # Versatilidade (se quiseres no futuro, ler “funcoes” detalhado)
        vers = "-"  # placeholder (podes ligar à tua função consolidate_functions quando necessário)

        # ------ KPIs ------
        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        c1.metric("Versatilidade", vers)
        c2.metric("Perfil LSC",    letter_grade(lsc_now))
        c3.metric("Físico",        letter_grade(fis_now))
        c4.metric("Mental",        letter_grade(men_now))
        c5.metric("Específico",    letter_grade(esp_now))
        c6.metric("Média Global",  letter_grade(media_global))
        c7.metric("Etiqueta Final", etiqueta)

        st.divider()

        # ------ Radares (mês vs mês-1) ------
        def labels_vals_for_family(family:str):
            if family == "ESPECIFICO":
                mask = (metrics["group"].str.lower()=="categoria") & (metrics["category"]==player_group)
            elif family == "FISICO":
                mask = (metrics["group"].str.lower()=="fisicos") & (metrics["scope"].str.lower()=="transversal")
            else:
                mask = (metrics["group"].str.lower()=="mentais") & (metrics["scope"].str.lower()=="transversal")

            mlist = metrics.loc[mask].sort_values("ordem")["metric_id"].tolist()
            labels = metrics.set_index("metric_id").loc[mlist]["label"].tolist()

            def mcm(a, y, m, p, k):
                dfm = a[(a["ano"]==y) & (a["mes"]==m) & (a["player_id"]==p) & (a["metric_id"]==k)]
                vals = pd.to_numeric(dfm["score"], errors="coerce").dropna().tolist()
                if not vals: return 0.0
                vals = sorted(vals)
                n = len(vals)
                if n >= 7: vals = vals[1:-1]
                elif n >= 5 and (n-2) >= 3: vals = vals[1:-1]
                return float(np.mean(vals)) if vals else 0.0

            now = [mcm(aval_all, ano_sel, mes_sel, pid, k) for k in mlist]
            prv = [mcm(aval_all, ano_prev, mes_prev, pid, k) for k in mlist]
            return labels, now, prv

        colA, colB, colC = st.columns(3)
        lbl, vnow, vprv = labels_vals_for_family("FISICO")
        colA.plotly_chart(radar_two_traces("Radar Físico", lbl, vnow, vprv), use_container_width=True)

        lbl, vnow, vprv = labels_vals_for_family("MENTAL")
        colB.plotly_chart(radar_two_traces("Radar Mental", lbl, vnow, vprv), use_container_width=True)

        lbl, vnow, vprv = labels_vals_for_family("ESPECIFICO")
        pretty_group = player_group.title() if isinstance(player_group, str) else str(player_group)
        colC.plotly_chart(radar_two_traces(f"Radar Específico ({pretty_group})", lbl, vnow, vprv), use_container_width=True)

    st.markdown("---")
    st.caption("© Leixões SC")
