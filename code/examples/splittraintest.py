articles_proc = pd.read_parquet(f'data/dfprep.parquet')

mixTexts = articles_proc[~articles_proc['topic2'].isnull()].reset_index(drop=True)
mixTexts['topic2'] = mixTexts['topic2'].apply(lambda x: int(x))
NOmixTexts = articles_proc[articles_proc['topic2'].isnull()].reset_index(drop=True)



XtrainMIX, XtestMIX, ytrainMIX, ytestMIX = train_test_split(mixTexts['url'], mixTexts['topic1'],
                                                    stratify=mixTexts['topic1'], 
                                                    test_size=0.20, shuffle=True)
XtrainNOMIX, XtestNOMIX, ytrainNOMIX, ytestNOMIX = train_test_split(NOmixTexts['url'], NOmixTexts['topic1'],
                                                    stratify=NOmixTexts['topic1'], 
                                                    test_size=0.20, shuffle=True)

X_trainURL = list(XtrainMIX.values) + list(XtrainNOMIX.values)
X_testURL = list(XtestMIX.values) + list(XtestNOMIX.values)

train = articles_proc[articles_proc['url'].isin(X_trainURL)].reset_index(drop=True)
test = articles_proc[articles_proc['url'].isin(X_testURL)].reset_index(drop=True)

train.to_parquet('data/train.parquet')
test.to_parquet('data/test.parquet')