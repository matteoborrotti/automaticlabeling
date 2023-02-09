mdl = tp.CTModel(tw=tp.TermWeight.PMI, k=16, rm_top = 0, min_df = 10)
for vec in data_words_train:
    mdl.add_doc(vec)
mdl.train(100)

mdl.save(f'data/tmp/ctm_model.bin')