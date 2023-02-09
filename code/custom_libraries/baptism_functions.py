from collections import defaultdict
from itertools import combinations
import pandas as pd
from utilities import create_data_words


def get_tp_topics(mdl, top_n=10):
    sorted_topics = [k for k, v in enumerate(mdl.get_count_by_topics())]
    topics = dict()

    for k in sorted_topics:
        topic_wp = []
        for word, prob in mdl.get_topic_words(k, top_n=top_n):
            topic_wp.append((word, prob))

        topics[k] = topic_wp

    topics = sorted(topics.items(), key=lambda item: item[0], reverse=False)

    return topics


def create_df_with_topic_probability(model, original_df, colname_text_to_use, **kwargs):

    model_topics = []
    if 'train_or_test' in kwargs and kwargs['train_or_test'] == 'test':

        list_docs = create_data_words(original_df, colname_text_to_use)
        i = 0
        for doc in list_docs:
            if i % 1000 == 0:
                print('number of rows processed: ', i)
            new_d = model.make_doc(doc)
            list_topics = model.infer(new_d)[0]
            model_topics.append(list_topics)
            i += 1
    else:
        for doc in range(len(model.docs)):
            list_topics = model.infer(model.docs[doc])[0]
            list_topics = [round(t, 4) for t in list_topics]
            model_topics.append(list_topics)

    df = pd.DataFrame()
    df['url'] = original_df['url']
    df['original_topic'] = original_df['topic1']
    df['original_topic2'] = original_df['topic2']
    df['txt'] = original_df['txt']
    df[colname_text_to_use] = original_df[colname_text_to_use]
    df['model_topic'] = model_topics

    return df


def create_two_diz_w_lists(df_wtopic, num_topics, name_column_topic_dict):
    ## dictionary with list of values eg diz_orig = {12: {0:[0.2,0.06], 1: [0.8,0.0009]}, 23: {0:[0.7,0.06, 0.5], 1: [0.008,0.0012,0.032]}}
    ## it means that there are 2 documents with topic=12, one with 0.2 on topic 0 and one with 0.6 on topic 0

    diz_original_topic_doc = defaultdict(dict)
    diz_model_topic_doc = defaultdict(dict)
    for i in range(num_topics):
        diz_model_topic_doc[i] = defaultdict(list)

    for row_idx in range(len(df_wtopic)):
        # print(row_idx)
        topic_original = df_wtopic.loc[row_idx, 'original_topic']
        model_topics_list = df_wtopic.loc[row_idx, name_column_topic_dict]

        if topic_original not in diz_original_topic_doc:
            diz_original_topic_doc[topic_original] = defaultdict(list)

        for topic_model in range(num_topics):
            prob_topic = model_topics_list[topic_model]
            diz_original_topic_doc[topic_original][topic_model].append(prob_topic)

            prob_topic = model_topics_list[topic_model]
            diz_model_topic_doc[topic_model][topic_original].append(prob_topic)

    return diz_original_topic_doc, diz_model_topic_doc


def create_two_diz_w_sums(diz_original_topic_doc, diz_model_topic_doc, num_topics):
    ## dictionary with sum of values per list eg diz_orig =  {12: {0: 0.26, 1: 0.8009}, 23: {0: 1.26, 1: [0.0412]}}
    ## it means that the sum ov values on topic 0 for documents with topic=12 is 0.026

    diz_sum_original_topic = defaultdict(dict)
    for original_topic in diz_original_topic_doc:
        if original_topic not in diz_sum_original_topic:
            diz_sum_original_topic[original_topic] = defaultdict(list)
        for model_topic in diz_original_topic_doc[original_topic]:
            diz_sum_original_topic[original_topic][model_topic] = sum(diz_original_topic_doc[original_topic][model_topic])

    diz_sum_model_topic = defaultdict(dict)
    for i in range(num_topics):
        diz_sum_model_topic[i] = defaultdict(float)
    for model_topic in diz_model_topic_doc:
        for original_topic in diz_model_topic_doc[model_topic]:
            diz_sum_model_topic[model_topic][original_topic] = sum(diz_model_topic_doc[model_topic][original_topic])

    return diz_sum_original_topic, diz_sum_model_topic


def create_two_diz_w_perc(diz_sum_model_topic, diz_sum_original_topic):
    ## dictionary with percentage of values per list eg diz_orig =  {12: {0: 0.26/(0.26+0.8009), 1: 0.8009/(0.26+0.8009)},..}
    ## percentage is given by value over the sum of values

    for original_topic in diz_sum_model_topic:
        sum_ = sum(diz_sum_model_topic[original_topic].values())
        if sum_ == 0:
            sum_ = 1
        diz_sum_model_topic[original_topic] = {k: round(v / sum_, 2) for k, v in
                                               dict(diz_sum_model_topic[original_topic]).items()}

    for model_topic in diz_sum_original_topic:
        sum_ = sum(diz_sum_original_topic[model_topic].values())
        if sum_ == 0:
            sum_ = 1
        diz_sum_original_topic[model_topic] = {k: round(v / sum_, 2) for k, v in
                                               dict(diz_sum_original_topic[model_topic]).items()}

    return diz_sum_original_topic, diz_sum_model_topic


def from_diz_to_df(diz_perc, column_name_reftopic, **kwargs):
    df = pd.DataFrame()
    for idx, topic_ in enumerate(diz_perc):
        df.loc[idx, column_name_reftopic] = round(topic_, 0)
        diz_values = diz_perc[topic_]
        for topic in diz_values:
            df.loc[idx, topic] = diz_values[topic]
        if "tot_docs" in kwargs:
            df.loc[idx, 'tot_docs'] = kwargs['tot_docs'][topic_]
    return df


def create_max_diz(ref_model_df, topics_to_check):
    max_diz = {}
    for i in range(len(ref_model_df)):
        reftopic = ref_model_df.loc[i, 'model_topic']
        values = ref_model_df.iloc[i, 1:].to_dict()
        values_sorted = sorted(values.items(), key=lambda item: item[1], reverse=True)

        if reftopic in topics_to_check.keys():
            max_diz[reftopic] = {int(values_sorted[0][0]): values_sorted[0][1],
                                 int(values_sorted[1][0]): values_sorted[1][1]}
        else:
            max_diz[reftopic] = {int(values_sorted[0][0]): values_sorted[0][1]}
    return max_diz


def compute_df_for_baptism(model, original_df, colname_text_to_use,
                                                         **kwargs):

    df_wtopic = create_df_with_topic_probability(model, original_df, colname_text_to_use, **kwargs)
    df_wtopic['model_topic_dict'] = df_wtopic['model_topic'].apply(
        lambda x: dict(zip(list(range(kwargs['num_topics'])), x)))
    df_wtopic = df_wtopic[~df_wtopic['url'].isin(kwargs['list_urls_mix'])]
    df_wtopic.reset_index(drop=True, inplace=True)
    diz_original_topic_doc, diz_model_topic_doc = create_two_diz_w_lists(df_wtopic, kwargs['num_topics'],
                                                                         'model_topic_dict')

    diz_sum_original_topic, diz_sum_model_topic = create_two_diz_w_sums(diz_original_topic_doc, diz_model_topic_doc,
                                                                        kwargs['num_topics'])
    diz_perc_original_topic, diz_perc_model_topic = create_two_diz_w_perc(diz_sum_model_topic, diz_sum_original_topic)
    ref_model_df = from_diz_to_df(diz_perc_model_topic, 'model_topic')

    return ref_model_df, df_wtopic


def edit_mix_topics(df_wtopic_original):
    df_wtopic = df_wtopic_original.copy()
    df_wtopic['original_topic'] = df_wtopic['original_topic'].apply(lambda x: [x])
    list_topics = []
    for idx_row in range(len(df_wtopic)):
        first_topic_list = df_wtopic.loc[idx_row, 'original_topic']
        second_topic = df_wtopic.loc[idx_row, 'original_topic2']
        if str(second_topic) != 'nan':
            ltopic = first_topic_list.copy()
            ltopic.append(int(second_topic))
            ltopic.sort()
            list_topics.append(ltopic)
        else:
            list_topics.append(first_topic_list)
    df_wtopic['original_topic'] = list_topics
    return df_wtopic


def allocation_topwords_based(diz_topwords, described_topics, minDistanceWords):
    diz_vtopic = {}
    for topic_description in described_topics:
        Vtopic = topic_description[0]
        top_words_Vtopic = topic_description[1]
        diz_vtopic[Vtopic] = {}
        for rtopic in diz_topwords:
            diz_vtopic[Vtopic][rtopic] = 0
            for tuple_word in top_words_Vtopic:
                if tuple_word[0] in diz_topwords[rtopic]:
                    diz_vtopic[Vtopic][rtopic] += 1
    diz_vtopic_max = {}
    for vtopic in diz_vtopic:
        diz_vtopic_max[vtopic] = {}
        sorted_vtopic = sorted(diz_vtopic[vtopic].items(), key=lambda x: x[1], reverse=True)
        if sorted_vtopic[0][1] - sorted_vtopic[1][1] >= minDistanceWords:
            diz_vtopic_max[vtopic] = {sorted_vtopic[0][0]: sorted_vtopic[0][1]}

    return diz_vtopic_max


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


def get_trasversal_topic_ABS(df_refmodel, threshold):
    trasversal_topics = []
    for i in range(len(df_refmodel)):
        reftopic = df_refmodel.loc[i, 'model_topic']
        values = df_refmodel.iloc[i, 1:].to_dict()
        values_sorted = sorted(values.items(), key=lambda item: item[1], reverse=True)
        if (values_sorted[0][1] - values_sorted[1][1]) <= threshold:
            trasversal_topics.append(reftopic)
    return trasversal_topics


def create_max_diz_NOtrasversal(ref_model_df, trasversal_topics, columnname_reftopic):
    max_diz = {}
    for i in range(len(ref_model_df)):
        reftopic = ref_model_df.loc[i, columnname_reftopic]
        values = ref_model_df.iloc[i, 1:].to_dict()
        values_sorted = sorted(values.items(), key=lambda item: item[1], reverse=True)

        if reftopic not in trasversal_topics:
            max_diz[int(reftopic)] = {int(values_sorted[0][0]): values_sorted[0][1]}
        else:
            max_diz[int(reftopic)] = {}
    return max_diz


def create_diz_index_topics_2Assign_(list_rtopics, max_diz_prob_based, max_diz_topwords_based):
    diz_sum = defaultdict(list)
    for real_topic in list_rtopics:
        flag = 0
        for model_topic in max_diz_prob_based:
            if real_topic in max_diz_prob_based[model_topic].keys() or real_topic in max_diz_topwords_based[model_topic].keys():
                flag = 1
                diz_sum[real_topic].append(int(model_topic))
        if flag == 0:
            diz_sum[real_topic] = []
    values_to_remove = set()
    for key in diz_sum:
        values = diz_sum[key]
        for key2 in diz_sum:
            if key2 != key:
                values_intersection = list(set(values) & set(diz_sum[key2]))
                values_to_remove.update(values_intersection)
    for key in diz_sum:
        for value_to_remove in values_to_remove:
            if value_to_remove in diz_sum[key]:
                diz_sum[key].remove(value_to_remove)

    return diz_sum


def create_diz_index_topics_2Assign_AND(list_rtopics, max_diz_prob_based, max_diz_topwords_based):
    diz_sum = defaultdict(list)
    for real_topic in list_rtopics:
        flag = 0
        for model_topic in max_diz_prob_based:
            if real_topic in max_diz_prob_based[model_topic].keys() and real_topic in max_diz_topwords_based[model_topic].keys():
                flag = 1
                diz_sum[real_topic].append(int(model_topic))
        if flag == 0:
            diz_sum[real_topic] = []
    return diz_sum
