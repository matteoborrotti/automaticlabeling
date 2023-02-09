articles_proc = pd.read_parquet('data/dfprep.parquet')
list_urls_mix= articles_proc[~articles_proc['topic2'].isnull()]['url']
list_real_topics = [12,22,23,29]
list_mix = [[12, 22],[23, 29]]

train_noCommonTrasversalWord = pd.read_parquet('data/tmp/train_NOCOMMON.parquet')
test_noCommonTrasversalWord = pd.read_parquet('data/tmp/test_NOCOMMON.parquet')

data_words_train = create_data_words(train_noCommonTrasversalWord, 'paper_text_processed_nostopwords')
datatrain=train_noCommonTrasversalWord.paper_text_processed_nostopwords.values.tolist()
disctionary_train, _, _ = create_corpora_voc(data_words_train)

data_words_test = create_data_words(test_noCommonTrasversalWord, 'paper_text_processed_nostopwords')
datatest=test_noCommonTrasversalWord.paper_text_processed_nostopwords.values.tolist()