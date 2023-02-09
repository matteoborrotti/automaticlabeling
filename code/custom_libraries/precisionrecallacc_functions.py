import numpy as np
from baptism_functions import *


def compute_global_accuracy(df_w_topic, diz_index_topics, top, columnname_topicdict, list_topics, filter_topics=True):
    df_w_topic_str = df_w_topic.copy()
    df_w_topic_str['original_topic'] = df_w_topic_str['original_topic'].map(str)
    fpcount = 0
    tpcount = 0
    virtual_topics_assigned_total = defaultdict(list)

    for single_topic in list_topics:
        list_to_add = diz_index_topics[single_topic]
        virtual_topics_assigned_total[single_topic].append(list_to_add)

    for row_idx in range(len(df_w_topic)):
        real_topics_list = df_w_topic.loc[row_idx, 'original_topic']

        model_topic_dict_sorted = sorted(df_w_topic.loc[row_idx, columnname_topicdict].items(),
                                         key=lambda item: item[1], reverse=True)
        model_topic_list = []
        if filter_topics:
            model_topic_dict_sorted = [tuple_ for tuple_ in model_topic_dict_sorted if
                                       tuple_[0] in sum(list(diz_index_topics.values()),
                                                        [])]  # filters out topics not assigned
        model_topic_dict_sorted = dict(model_topic_dict_sorted)
        diz_model = {}
        for rtopic in virtual_topics_assigned_total:
            sumvalues = 0
            for vtopic in virtual_topics_assigned_total[rtopic][0]:
                if vtopic in model_topic_dict_sorted:
                    sumvalues += model_topic_dict_sorted[vtopic]
            diz_model[rtopic] = sumvalues
        model_topic_dict_sorted = sorted(diz_model.items(), key=lambda item: item[1], reverse=True)

        if len(real_topics_list) == 1:
            for tuple_ in model_topic_dict_sorted[:top]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])
            flag = 0
            for topic in real_topics_list:
                if topic in model_topic_list:
                    tpcount += 1
                    flag += 1
                    break
            if flag == 0:
                fpcount += 1

        else:
            # if the topic is a mixed topic,
            # the top1 means that in the first 2 positions there must be both the topics  
            # the top3 means that in the first 4 positions there must be both the topics  
            assert top == 1 or top == 3
            if top == 1:
                top_tmp = 2
            elif top == 3:
                top_tmp = 4

            model_topic_list = []
            for tuple_ in model_topic_dict_sorted[:top_tmp]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])
            flag = 0
            for topic in real_topics_list:
                if topic in model_topic_list:
                    flag += 1
            if flag == 2:
                tpcount += 1
            else:
                fpcount += 1

    global_acc = tpcount / (tpcount + fpcount)

    return global_acc


def compute_recall_and_truepositive(df_w_topic, diz_index_topics, top, columnname_topicdict, list_topics,
                                    filter_topics=True):
    df_w_topic_str = df_w_topic.copy()
    df_w_topic_str['original_topic'] = df_w_topic_str['original_topic'].map(str)
    tot_counts = df_w_topic_str.groupby(by='original_topic').size().to_dict()
    true_positive_dict = defaultdict(int)

    virtual_topics_assigned_total = defaultdict(list)
    for single_topic in list_topics:
        list_to_add = diz_index_topics[single_topic]
        virtual_topics_assigned_total[single_topic].append(list_to_add)

    for rtl in np.unique(df_w_topic['original_topic']):
        counts = 0
        for el in rtl:
            if el in virtual_topics_assigned_total:
                counts += 1
        if counts == len(rtl):
            true_positive_dict[str(rtl)] = 0

    misclassified_index = []
    for row_idx in range(len(df_w_topic)):
        real_topics_list = df_w_topic.loc[row_idx, 'original_topic']
        model_topic_dict_sorted = sorted(df_w_topic.loc[row_idx, columnname_topicdict].items(),
                                         key=lambda item: item[1], reverse=True)

        model_topic_list = []
        if filter_topics:
            model_topic_dict_sorted = [tuple_ for tuple_ in model_topic_dict_sorted if
                                       tuple_[0] in sum(list(diz_index_topics.values()),
                                                        [])]  # filters out topics not assigned
        model_topic_dict_sorted = dict(model_topic_dict_sorted)
        diz_model = {}
        for rtopic in virtual_topics_assigned_total:
            sumvalues = 0
            for vtopic in virtual_topics_assigned_total[rtopic][0]:
                if vtopic in model_topic_dict_sorted:
                    sumvalues += model_topic_dict_sorted[vtopic]
            diz_model[rtopic] = sumvalues

        model_topic_dict_sorted = sorted(diz_model.items(), key=lambda item: item[1], reverse=True)

        if len(real_topics_list) == 1:
            for tuple_ in model_topic_dict_sorted[:top]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])
            flag = 0
            for topic_r in model_topic_list:
                if topic_r in real_topics_list:
                    true_positive_dict[str(real_topics_list)] += 1
                    flag += 1
            if flag == 0:
                misclassified_index.append(row_idx)
        else:
            # if the topic is a mixed topic,
            # the top1 means that in the first 2 positions there must be both the topics  
            # the top3 means that in the first 4 positions there must be both the topics
            assert top == 1 or top == 3
            if top == 1:
                top_tmp = 2
            elif top == 3:
                top_tmp = 4

            model_topic_list = []
            for tuple_ in model_topic_dict_sorted[:top_tmp]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])
            flag = 0
            for topic_r in model_topic_list:
                if topic_r in real_topics_list:
                    flag += 1
            if flag == 2:
                true_positive_dict[str(real_topics_list)] += 1
            else:
                misclassified_index.append(row_idx)

    recall_ = {k: round(v / tot_counts[k], 2) for k, v in true_positive_dict.items()}

    return recall_, true_positive_dict, misclassified_index


def compute_truepos_falsepos(df_w_topic, diz_index_topics, top, columnname_topicdict, list_topics, mix,
                             filter_topics=True):
    df_w_topic_str = df_w_topic.copy()
    df_w_topic_str['original_topic'] = df_w_topic_str['original_topic'].map(str)
    truepos_falsepos = defaultdict(int)
    dizpositions = defaultdict(list)

    virtual_topics_assigned_total = defaultdict(list)

    for single_topic in list_topics:
        list_to_add = diz_index_topics[single_topic]
        virtual_topics_assigned_total[single_topic].append(list_to_add)

    list_topics.extend(mix)

    idx = 0
    for row_idx in range(len(df_w_topic)):
        real_topics_list = df_w_topic.loc[row_idx, 'original_topic']
        model_topic_dict_sorted = sorted(df_w_topic.loc[row_idx, columnname_topicdict].items(),
                                         key=lambda item: item[1], reverse=True)
        model_topic_list = []
        if filter_topics:
            model_topic_dict_sorted = [tuple_ for tuple_ in model_topic_dict_sorted if
                                       tuple_[0] in sum(list(diz_index_topics.values()),
                                                        [])]  # filters out topics not assigned
        model_topic_dict_sorted = dict(model_topic_dict_sorted)
        diz_model = {}
        for rtopic in virtual_topics_assigned_total:
            sumvalues = 0
            for vtopic in virtual_topics_assigned_total[rtopic][0]:
                if vtopic in model_topic_dict_sorted:
                    sumvalues += model_topic_dict_sorted[vtopic]
            diz_model[rtopic] = sumvalues

        model_topic_dict_sorted = sorted(diz_model.items(), key=lambda item: item[1], reverse=True)

        if len(real_topics_list) == 1:
            for tuple_ in model_topic_dict_sorted[:top]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])

            for topic in list_topics:
                if not isinstance(topic, list):
                    if topic in model_topic_list:
                        truepos_falsepos[str([topic])] += 1
                        dizpositions[str([topic])].append(idx)

        else:
            # if the topic is a mixed topic,
            # the top1 means that in the first 2 positions there must be both the topics  
            # the top3 means that in the first 4 positions there must be both the topics
            assert top == 1 or top == 3
            if top == 1:
                top_tmp = 2
            elif top == 3:
                top_tmp = 4

            model_topic_list = []
            for tuple_ in model_topic_dict_sorted[:top_tmp]:
                if tuple_[1] != 0:
                    model_topic_list.append(tuple_[0])

            for topic in list_topics:
                if isinstance(topic, list):
                    flag = 0
                    for single_topic in topic:
                        if single_topic in model_topic_list:
                            flag += 1
                    if flag == 2:
                        truepos_falsepos[str(topic)] += 1
                        dizpositions[str(topic)].append(idx)
        idx += 1
    return truepos_falsepos, dizpositions
