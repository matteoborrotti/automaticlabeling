
df_texts = pd.read_csv('data/scrapedTexts.csv')
df_texts.drop_duplicates(subset = 'txt', keep='first',inplace=True)
df_texts = df_texts[df_texts['txt']!=""].reset_index(drop=True)
df_texts.to_parquet('data/dfnodup.parquet')
pre_pre_df = pre_preprocess_df('data/dfnodup.parquet')
articles_proc = preprocess_df(pre_pre_df)
articles_proc.to_parquet(f'data/dfprep.parquet')