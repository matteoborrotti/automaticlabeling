from collections import defaultdict
from itertools import combinations
from gensim import corpora


def from_modeldiz_to_realdiz(modeldiz):
    realdiz = defaultdict(list)
    for model in modeldiz:
        for real in modeldiz[model]:
            realdiz[real].append(model)
    return realdiz


def create_data_words(df, column_to_use):
    data_words = df[column_to_use].values.tolist()
    data_words = [[word for word in doc.split(' ') if word != ""] for doc in data_words]
    return data_words

def create_corpora_voc (data_words):
    # Create Dictionary
    dictionary = corpora.Dictionary(data_words)
    # Create Corpus
    texts = data_words
    # Term Document Frequency
    corpus = [dictionary.doc2bow(text) for text in texts]

    return dictionary, texts, corpus

def iteratively_remove_duplicate_words(diz_, diz_topwords_original, nwords=5):
    diz_topwords_edit = diz_topwords_original.copy()
    duplicate = 999
    counter = 0
    words_to_remove_total_final = set()
    while duplicate != 0 and counter < 15:
        list_top_words_topics = list(diz_topwords_edit.values())
        comb = combinations(list(diz_.keys()), 2)
        words_to_remove_total = set()
        duplicate = 0
        for couple in comb:
            words_to_remove = set(list_top_words_topics[couple[0]][:nwords]).intersection(
                set(list_top_words_topics[couple[1]][:nwords]))
            words_to_remove_total.update(words_to_remove)

            if len(words_to_remove) > 0:
                duplicate = 1

        words_to_remove_total_final.update(words_to_remove_total)
        for key in diz_topwords_original:
            diz_topwords_edit[key] = [topword for topword in diz_topwords_original[key] if
                                      topword not in words_to_remove_total_final]

        counter += 1

    for key in diz_topwords_original:
        diz_topwords_edit[key] = [topword for topword in diz_topwords_original[key] if
                                  topword not in words_to_remove_total_final][:nwords]

    return words_to_remove_total_final, diz_topwords_edit
