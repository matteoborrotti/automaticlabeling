
with open(f"data/tmp/diz_5topwords.pickle", "rb") as handle:
    diz_topwords_5 = pickle.load(handle)


globalthresholdDistribution = 0.2
globalminDistanceWords = 1

described_topics10 = get_tp_topics(mdl, 10)

list_urls_mix= articles_proc[~articles_proc['topic2'].isnull()]['url']

ref_model_df, df_wtopic = compute_df_for_baptism(mdl, train_noCommonTrasversalWord, 
                                                            'paper_text_processed_nostopwords'
                                                            , list_urls_mix = list_urls_mix, num_topics = 16,  save = False)
trasversal_topics = get_trasversal_topic_ABS(ref_model_df, globalthresholdDistribution)
diz_vtopic_max_probability_based = create_max_diz_NOtrasversal(ref_model_df, trasversal_topics, columnname_reftopic = 'model_topic')
diz_vtopic_max_topwords_based = allocation_topwords_based(diz_topwords_5, described_topics10, globalminDistanceWords)

# final baptism for each method
diz_index_topicsOR = create_diz_index_topics_2Assign_(list_real_topics,diz_vtopic_max_probability_based, diz_vtopic_max_topwords_based)
diz_index_topicsAND = create_diz_index_topics_2Assign_AND(list_real_topics,diz_vtopic_max_probability_based, diz_vtopic_max_topwords_based)
diz_vtopic_max_probability_based = from_modeldiz_to_realdiz(diz_vtopic_max_probability_based)
diz_vtopic_max_topwords_based = from_modeldiz_to_realdiz(diz_vtopic_max_topwords_based)

list_diz_topics = [diz_vtopic_max_probability_based, diz_vtopic_max_topwords_based, diz_index_topicsOR, diz_index_topicsAND]
