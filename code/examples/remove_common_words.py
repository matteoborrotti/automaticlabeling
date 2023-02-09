### Extract topwords per topic

vect = CountVectorizer()  
vects = vect.fit_transform(train.paper_text_processed_nostopwords)
document_term_matrix = pd.DataFrame(vects.todense())
document_term_matrix.columns = vect.get_feature_names_out()
document_term_matrix['____topic____'] = train['topic1'].map(str)
document_term_matrix['____topic____'] = document_term_matrix['____topic____'].astype('category')
del vects, vect

dt_12 = document_term_matrix[document_term_matrix['____topic____']=='12']
td_12 = dt_12.iloc[:,:-1].T
td_12['tot_count'] = td_12.sum(axis=1)
counts12 = td_12['tot_count']
del td_12, dt_12
counts12.to_frame().to_parquet(f'data/tmp/counts12.parquet')

dt_22 = document_term_matrix[document_term_matrix['____topic____']=='22']
td_22 = dt_22.iloc[:,:-1].T
td_22['tot_count'] = td_22.sum(axis=1)
counts22 = td_22['tot_count']
del td_22, dt_22
counts22.to_frame().to_parquet(f'data/tmp/counts22.parquet')

dt_23 = document_term_matrix[document_term_matrix['____topic____']=='23']
td_23 = dt_23.iloc[:,:-1].T
td_23['tot_count'] = td_23.sum(axis=1)
counts23 = td_23['tot_count']
del td_23, dt_23
counts23.to_frame().to_parquet(f'data/tmp/counts23.parquet')

dt_29 = document_term_matrix[document_term_matrix['____topic____']=='29']
td_29 = dt_29.iloc[:,:-1].T
td_29['tot_count'] = td_29.sum(axis=1)
counts29 = td_29['tot_count']
del td_29, dt_29
counts29.to_frame().to_parquet(f'data/tmp/counts29.parquet')

del document_term_matrix

counts12 = pd.read_parquet(f'data/tmp/counts12.parquet')
counts22 = pd.read_parquet(f'data/tmp/counts22.parquet')
counts23 = pd.read_parquet(f'data/tmp/counts23.parquet')
counts29 = pd.read_parquet(f'data/tmp/counts29.parquet')


top_words_12 = counts12.sort_values(by='tot_count', ascending = False)[:200].index
top_words_22 = counts22.sort_values(by='tot_count', ascending = False)[:200].index
top_words_23 = counts23.sort_values(by='tot_count', ascending = False)[:200].index
top_words_29 = counts29.sort_values(by='tot_count', ascending = False)[:200].index
list_top_words_topics = [top_words_12, top_words_22, top_words_23, top_words_29]

### Find words to remove (common to at least 3 out of 4 topics)

topics = ['Animali', 'Salute e benessere', 'Moda', 'Celebrità']
diz_ = dict(zip(range(0,len(topics)), topics))

comb = combinations(range(0,len(topics)), 3) # 3 = number of topics in which to look for common words

words_to_remove_total = set()
for triple in comb:
    words_to_remove = list_top_words_topics[triple[0]].intersection(list_top_words_topics[triple[1]]).intersection(list_top_words_topics[triple[2]]).values
    print(words_to_remove)
    print('-------------------------------------------------------------------------', diz_[triple[0]], '-', diz_[triple[1]], '-', diz_[triple[2]])
    words_to_remove_total.update(words_to_remove)

words_to_remove_total = list(words_to_remove_total)


### Remove from train and test

train = pd.read_parquet('data/train.parquet')
test = pd.read_parquet('data/test.parquet')

train_noCommonTrasversalWord  = train.copy()
train_noCommonTrasversalWord['paper_text_processed_nostopwords'] = train_noCommonTrasversalWord['paper_text_processed'].apply(lambda x: remove_stopwords(x, words_to_remove_total, []))
train_noCommonTrasversalWord.to_parquet(f'data/tmp/train_NOCOMMON.parquet')

test_noCommonTrasversalWord  = test.copy()
test_noCommonTrasversalWord['paper_text_processed_nostopwords'] = test_noCommonTrasversalWord['paper_text_processed'].apply(lambda x: remove_stopwords(x, words_to_remove_total, []))
test_noCommonTrasversalWord.to_parquet('data/tmp/test_NOCOMMON.parquet')

