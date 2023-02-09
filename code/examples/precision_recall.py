list_real_topics = [12,22,23,29]
list_mix = [[12, 22],[23, 29]]
df_wtopic_testmix = pd.read_parquet('data/df_wtopic_testmix.parquet')
df_wtopic_testNOmix = pd.read_parquet('data/df_wtopic_testNOmix.parquet')
# You need also diz_vtopic_max_probability_based, diz_vtopic_max_topwords_based, diz_index_topicsAND, diz_index_topicsOR from baptism


list_mix_nomix_tot = ['mix','NOmix']
diz_df = {'mix': df_wtopic_testmix,'NOmix': df_wtopic_testNOmix}

diz_precision_results = {}
diz_recall_results = {}

for m in list_mix_nomix_tot:
    df = diz_df[m]

    list_topics_ = list_real_topics.copy()
    dizcountsORtop1, _ = compute_truepos_falsepos(df, diz_index_topics = diz_index_topicsOR, top = 1, columnname_topicdict = 'model_topic_dict',list_topics = list_topics_, mix = list_mix)
    recallORtop1, tportop1, _ = compute_recall_and_truepositive(df, diz_index_topicsOR, top = 1, columnname_topicdict = 'model_topic_dict',list_topics = list_real_topics)
    precisionORtop1 = {k: round(v/dizcountsORtop1[k],2) for k, v in tportop1.items() if v!= 0}
    precisionORtop1 = {k: precisionORtop1[k] if k in precisionORtop1 else 0 for k in dizcountsORtop1}
    precisionORtop1 = {k: v for k, v in sorted(precisionORtop1.items(), key=lambda item: item[0])}

    list_topics_ = list_real_topics.copy()
    dizcountsANDtop1, _ = compute_truepos_falsepos(df, diz_index_topics = diz_index_topicsAND, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_topics_, mix = list_mix)
    recallANDtop1, tpandtop1, _ = compute_recall_and_truepositive(df, diz_index_topicsAND, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    precisionANDtop1 = {k: round(v/dizcountsANDtop1[k],2) for k, v in tpandtop1.items() if v!= 0}
    precisionANDtop1 = {k: precisionANDtop1[k] if k in precisionANDtop1 else 0 for k in dizcountsANDtop1}
    precisionANDtop1 = {k: v for k, v in sorted(precisionANDtop1.items(), key=lambda item: item[0])}

    list_topics_ = list_real_topics.copy()
    dizcountsM1top1, _ = compute_truepos_falsepos(df, diz_index_topics = diz_vtopic_max_probability_based, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_topics_, mix = list_mix)
    recallM1top1, tpm1top1, _ = compute_recall_and_truepositive(df, diz_vtopic_max_probability_based, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    precisionM1top1 = {k: round(v/dizcountsM1top1[k],2) for k, v in tpm1top1.items() if v!= 0}
    precisionM1top1 = {k: precisionM1top1[k] if k in precisionM1top1 else 0 for k in dizcountsM1top1}
    precisionM1top1 = {k: v for k, v in sorted(precisionM1top1.items(), key=lambda item: item[0])}

    list_topics_ = list_real_topics.copy()
    dizcountsM2top1, _ = compute_truepos_falsepos(df, diz_index_topics = diz_vtopic_max_topwords_based, top = 1, columnname_topicdict = 'model_topic_dict',list_topics = list_topics_, mix = list_mix)
    recallM2top1, tpm2top1, _ = compute_recall_and_truepositive(df, diz_vtopic_max_topwords_based, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    precisionM2top1 = {k: round(v/dizcountsM2top1[k],2) for k, v in tpm2top1.items() if v!= 0}
    precisionM2top1 = {k: precisionM2top1[k] if k in precisionM2top1 else 0 for k in dizcountsM2top1}
    precisionM2top1 = {k: v for k, v in sorted(precisionM2top1.items(), key=lambda item: item[0])}

    precision_m_list = [precisionM1top1, precisionM2top1, precisionANDtop1, precisionORtop1]
    recall_m_list = [recallM1top1, recallM2top1, recallANDtop1, recallORtop1]
    
    
    diz_precision_results[m] = {'DistributionMethod': precisionM1top1, 'TopwordsMethod': precisionM2top1, 'AndMethod': precisionANDtop1, 'OrMethod': precisionORtop1}
    diz_recall_results[m] = {'DistributionMethod': recallM1top1, 'TopwordsMethod': recallM2top1, 'AndMethod': recallANDtop1, 'OrMethod': recallORtop1}

print('Precision: ', diz_precision_results)
print('Recall: ', diz_recall_results)