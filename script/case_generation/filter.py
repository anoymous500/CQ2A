import pandas as pd

from entity_extraction import acquire_dependence_tree
from model_access import *

import configparser

config = configparser.ConfigParser()

config.read(f'../../config.ini')


def check_from_ask_re(sent, answer, question, sim_model):

    prompt = f"According to the context '{sent}' and  question '{question}', the answer of the question should be relation of two entities, the form of answer is a word or phrase " \
                 f"please output the answer. Remember that the output should only be the answer without any other prompts."
    gen_a = connect_gpt_prompt_ask(prompt=prompt, client=client)
    sim = sim_model.similarity(answer, gen_a)
    sim_threshold = config['filter']['sim_threshold']
    if sim >= sim_threshold :
        return [gen_a, True]
    else:
        if answer in gen_a:
            return [gen_a, True]
        return [gen_a, False]


def check_answer(question,answer,q_type,type,sent,nlp=None):
    a_list = answer.split(' ')
    # 对于词组作为答案，词组中的实意词不应出现在问题中
    if len(question.split(' '))>len(sent.split(' '))+5:
        return False
    q_info = acquire_dependence_tree(question, nlp)
    q_lemma_list = [q_info[i][3] for i in range(len(q_info))]
    if q_type=='phrases':
        a_info = acquire_dependence_tree(answer, nlp)
        type_word = [a_info[i][3] for i in range(len(a_info)) if a_info[i][-4] in ['NOUN']]
        for word in type_word:
            if word in q_lemma_list:
                return False
        if 'there' in answer or 'here' in answer or 'his' in answer or 'her' in answer:
            return False
    elif q_type == 'words':
        if type=='NOUN':
            lemma_a = wnl.lemmatize(answer,'n')
            if lemma_a in q_lemma_list:
                return False
        elif type == 'VERB':
            lemma_a = wnl.lemmatize(answer, 'v')
            if lemma_a in q_lemma_list:
                return False

    return True
def rule_filter_entity(prompt_info,cond,ch_type):
    if prompt_info[2] in prompt_info[0]:
        return False
    if not check_answer(prompt_info[0], prompt_info[2], cond, ch_type, prompt_info[1], nlp):
        return False
    return True
def filter_entity(dataset):
    # [sent, context, answer, prompt, gpt_answer, str_question, str_reason]
    # [question,context,answer,explanation]
    for cond in ['phrases', 'words']:
        for ch_type in ['NOUN', 'VERB']:
            filter_q = []
            all_filter_q = []
            raw_q = pd.read_csv(f'../../data/result/case/{dataset}/entity_info_{dataset}_{cond}_{ch_type}_raw_q.csv',
                            sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
            flag = 0
            for prompt_info in raw_q:
                if rule_filter_entity(prompt_info,cond,ch_type):
                    filter_q.append(prompt_info)
                    flag = 1
                if flag:
                    ch_result = check_from_ask_re(prompt_info[1], prompt_info[2], prompt_info[0], sim_model)
                    raw_l = prompt_info.copy()
                    raw_l.append(ch_result[0])
                    all_filter_q.append(raw_l)
            q_dataframe = pd.DataFrame(filter_q)
            q_dataframe.to_csv(f'../../data/result/case/{dataset}/entity_info_{dataset}_{cond}_{ch_type}_rule_filter_q.csv',
                               mode='w',
                               index=False, header=False)
            q_dataframe_1 = pd.DataFrame(all_filter_q)
            q_dataframe_1.to_csv(
                f'../../data/result/case/{dataset}/entity_info_{dataset}_{cond}_{ch_type}_all_filter_q.csv',
                mode='w',
                index=False, header=False)

def rule_filter_rel(relation,str_reason,question,sub1,sub2):
    if relation not in str_reason or relation in question or relation in ['am', 'is', 'are', 'have', 'has']:
        return False
    if sub1 not in question or sub2 not in question:
        return False
    return True
def filter_rel(dataset):
    filter_q = []
    all_filter_q = []
    # [question,context,answer,explanation]
    raw_q = pd.read_csv(f'../../data/result/case/{dataset}/rel_info_{dataset}_raw_q.csv',
                        sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
    for prompt_info in raw_q:
        context = prompt_info[1]
        relation_triple = prompt_info[2]
        sub1 = relation_triple.split(',')[0]
        sub2 = relation_triple.split(',')[-1]
        relation = relation_triple.split(',')[1]
        while sub1[0] == ' ':
            sub1 = sub1[1:]
        while sub1[-1] == ' ':
            sub1 = sub1[:-1]
        while sub2[0] == ' ':
            sub2 = sub2[1:]
        while sub2[-1] == ' ':
            sub2 = sub2[:-1]
        while relation[0] == ' ':
            relation = relation[1:]
        while relation[-1] == ' ':
            relation = relation[:-1]
        question = prompt_info[0]
        str_reason = prompt_info[-1]
        flag = 0
        # relation triple中的信息应该出现在reason中
        if rule_filter_rel(relation,str_reason,question,sub1,sub2):
            filter_q.append(prompt_info)
            flag = 1
        if flag == 1:
            ch_result = check_from_ask_re(context, relation, question, sim_model)
            raw_l = prompt_info.copy()
            raw_l.append(ch_result[0])
            all_filter_q.append(raw_l)

    q_dataframe = pd.DataFrame(filter_q)
    q_dataframe.to_csv(f'../../data/result/case/{dataset}/rel_info_{dataset}_rule_filter_q.csv',
                       mode='w',
                       index=False, header=False)
    q_dataframe_1 = pd.DataFrame(all_filter_q)
    q_dataframe_1.to_csv(f'../../data/result/case/{dataset}/rel_info_{dataset}_all_filter_q.csv',
                       mode='w',
                       index=False, header=False)

if __name__ == '__main__':
    filter_entity('your_dataset')
    filter_rel('your_dataset')