import numpy as np
import pandas as pd
from numpy import log as ln
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm


def extract_document_term_matrix(traintexts, used_vocabs, test=False, vect=None):
    if not test:
        vect = CountVectorizer()
        vect.fit(used_vocabs)

    vects = vect.transform(traintexts)

    document_term_matrix = pd.DataFrame(vects.todense())

    document_term_matrix.columns = vect.get_feature_names_out()
    if len(used_vocabs) != len(document_term_matrix.columns) and 0 < len(used_vocabs) - len(
            document_term_matrix.columns) <= 5:
        remaining = [w for w in used_vocabs if w not in document_term_matrix.columns]
        document_term_matrix[remaining] = 0
    document_term_matrix = document_term_matrix[used_vocabs]
    print("document_term created")

    return document_term_matrix, vect


def extract_topic_term_matrix_tp(model):
    data_topic_words = [model.get_topic_word_dist(topic_id) for topic_id in range(model.k)]
    topic_term_matrix = pd.DataFrame(data_topic_words)
    topic_term_matrix.columns = model.used_vocabs
    print("topic_term created")
    return topic_term_matrix


def extract_document_topic_matrix_tp(model, train=True, data_words_test=None):
    list_infers_train = []

    if train:
        for doc_idx in range(len(model.docs)):
            list_infers_train.append(model.infer(model.docs[doc_idx])[0])
    else:
        for doc in data_words_test:
            new_d = model.make_doc(doc)
            list_infers_train.append(model.infer(new_d)[0])

    document_topic_matrix = pd.DataFrame(list_infers_train)
    document_topic_matrix.columns = ["topic" + str(topic_id) for topic_id in range(model.k)]
    print("document_topic created")

    return document_topic_matrix


def perplexity(document_term_matrix, document_topic_matrix, topic_term_matrix):
    ll = 0
    for row_idx in tqdm(range(len(document_term_matrix)), leave=False):
        used_words_in_doc = [document_term_matrix.columns[i] for i in
                             np.where(document_term_matrix.iloc[row_idx] > 0)[0]]
        document_topic_doc = document_topic_matrix.to_numpy()[row_idx].reshape(1, -1)
        topic_term_doc = topic_term_matrix[used_words_in_doc]
        dotprod = np.matmul(document_topic_doc, topic_term_doc)
        logprod = np.matmul(ln(dotprod + 1e-16),
                            document_term_matrix.iloc[row_idx][used_words_in_doc].to_numpy().reshape(-1, 1))

        ll += logprod

    ll = ll.values[0][0]
    perplexity_value = np.exp(-ll / sum(sum(document_term_matrix.values)))

    return perplexity_value, - ll / sum(sum(document_term_matrix.values))
