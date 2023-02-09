
df_wtopic_testmix = pd.read_parquet('data/df_wtopic_testmix.parquet')
df_wtopic_testNOmix = pd.read_parquet('data/df_wtopic_testNOmix.parquet')
list_real_topics = [12,22,23,29]
# You need also diz_vtopic_max_probability_based, diz_vtopic_max_topwords_based, diz_index_topicsAND, diz_index_topicsOR from baptism

list_mix_nomix_tot = ['mix','NOmix']
diz_df = {'mix': df_wtopic_testmix,'NOmix': df_wtopic_testNOmix}
diz_globalacc_results = {}
for m in list_mix_nomix_tot:
    df = diz_df[m]
    global_acc_distrib_m = compute_global_accuracy(df, diz_vtopic_max_probability_based, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    global_acc_topwords_m = compute_global_accuracy(df, diz_vtopic_max_topwords_based, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    global_acc_AND_m = compute_global_accuracy(df, diz_index_topicsAND, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    global_acc_OR_m = compute_global_accuracy(df, diz_index_topicsOR, top = 1, columnname_topicdict = 'model_topic_dict', list_topics = list_real_topics)
    
    diz_globalacc_results[m] = {'DistributionMethod': global_acc_distrib_m, 'TopwordsMethod': global_acc_topwords_m, 'AndMethod': global_acc_AND_m, 'OrMethod': global_acc_OR_m}
print(diz_globalacc_results)