    # (Opcional) tabela de auditoria
with st.expander("Ver tabela detalhada (métricas e médias limpas)"):
    rows = []

    fams = ["FISICO", "MENTAL", "ESPECIFICO"]
    for fam in fams:
        if fam == "ESPECIFICO":
            # usa group_norm se existir; caso contrário, usa group
            if "group_norm" in metrics.columns:
                mask = (metrics["family"] == "ESPECIFICO") & (metrics["group_norm"] == player_group)
            else:
                mask = (metrics["family"] == "ESPECIFICO") & (metrics["group"] == player_group)
        else:
            mask = (metrics["family"] == fam)

        mlist = metrics.loc[mask].sort_values("ordem")["metric_key"].tolist()

        for k in mlist:
            # etiqueta amigável se existir 'label'
            label_k = (
                metrics.set_index("metric_key").loc[k]["label"]
                if ("label" in metrics.columns and k in metrics.set_index("metric_key").index)
                else k
            )
            rows.append({
                "Família": fam,
                "Métrica": label_k,
                "Média (mês)":   metric_clean_mean(aval_all, ano_sel, mes_sel, pid, k),
                "Média (mês-1)": metric_clean_mean(aval_all, ano_prev, mes_prev, pid, k),
            })

    if rows:
        df_rows = pd.DataFrame(rows)
        st.dataframe(df_rows, use_container_width=True)
    else:
        st.info("Sem métricas para apresentar neste jogador/família.")
