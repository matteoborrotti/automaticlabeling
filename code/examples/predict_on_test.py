df_wtopic_test = create_df_with_topic_probability(mdl, test_noCommonTrasversalWord, 'paper_text_processed_nostopwords', train_or_test = 'test')

df_wtopic_test = edit_mix_topics(df_wtopic_test)

df_wtopic_test['model_topic_dict'] =  df_wtopic_test['model_topic'].apply(lambda x: dict(zip(list(range(16)),x)))

df_wtopic_testmix = df_wtopic_test[(df_wtopic_test['original_topic'].map(str)=='[12, 22]') |(df_wtopic_test['original_topic'].map(str)=='[23, 29]')].reset_index(drop=True)
df_wtopic_testNOmix = df_wtopic_test[~((df_wtopic_test['original_topic'].map(str)=='[12, 22]') | (df_wtopic_test['original_topic'].map(str)=='[23, 29]'))].reset_index(drop=True)

df_wtopic_testmix.to_parquet('data/df_wtopic_testmix.parquet')
df_wtopic_testNOmix.to_parquet('data/df_wtopic_testNOmix.parquet')