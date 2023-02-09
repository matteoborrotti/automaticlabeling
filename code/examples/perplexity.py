_, vec = extract_document_term_matrix(datatrain, list(mdl.used_vocabs))

document_term_matrix_test, _ = extract_document_term_matrix(datatest, list(mdl.used_vocabs), test =True, vect = vec)
document_topic_matrix_test = extract_document_topic_matrix_tp(mdl, train =False, data_words_test= data_words_test)
topic_term_matrix_test = extract_topic_term_matrix_tp(mdl)
perplexity_value = perplexity(document_term_matrix_test, document_topic_matrix_test, topic_term_matrix_test)
