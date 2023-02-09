counts29 = pd.read_parquet(f'data/tmp/counts29.parquet')
counts23 = pd.read_parquet(f'data/tmp/counts23.parquet')
counts22 = pd.read_parquet(f'data/tmp/counts22.parquet')
counts12 = pd.read_parquet(f'data/tmp/counts12.parquet')

top_words_12 = counts12.sort_values(by='tot_count', ascending = False)[:100].index
top_words_22 = counts22.sort_values(by='tot_count', ascending = False)[:100].index
top_words_23 = counts23.sort_values(by='tot_count', ascending = False)[:100].index
top_words_29 = counts29.sort_values(by='tot_count', ascending = False)[:100].index
diz_topwords = {12: top_words_12, 22:top_words_22, 23:top_words_23, 29:top_words_29}

topics = [12,22,23,29]
diz_ = dict(zip(range(0,len(topics)), topics))


n_words = 5
words_removed, diz_topwords_edit = iteratively_remove_duplicate_words(diz_, diz_topwords,5)

with open(f"data/tmp/diz_{n_words}topwords.pickle", "wb") as handle:
    pickle.dump(diz_topwords_edit, handle)
