import re
import numpy as np
from ..case_generation.model_access import *
import random
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from datetime import datetime
import configparser
import random

config = configparser.ConfigParser()

config.read(f'../../config.ini')


def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)

def select_ran(dataset,info_type):
    raw_file = f'../../data/result/test_result/bug_result/{info_type}_bug_result_{dataset}_all_filter.csv'
    ran_num = int(config['coverage']['rand_num'])
    sav_file = f'../../data/result/experiment_data/coverage/{dataset}/{info_type}_coverage_{dataset}_ran_{ran_num}.csv'
    data = pd.read_csv(raw_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
    ran_data = random.sample(data, ran_num)
    data = pd.DataFrame(ran_data)
    data.to_csv(sav_file, mode='w', index=False, header=False)

def choose_context(context_file,ran_num):
    with open(context_file, mode='r',encoding='utf-8',newline='') as f:
        raw = f.readlines()
        result_text = []
        for r in raw :
            result_text.append(r)
        f.close()
    random_text_list = random.sample(result_text,ran_num)
    return random_text_list


def find_q_context(context,dataset):
    # all_inf_q = pd.read_csv(f'experiment/final_result/{dataset}_final_inf_half_real.csv').values.tolist()
    # all_rel_q = pd.read_csv(f'experiment/final_result/{dataset}_final_rel_half_real.csv').values.tolist()
    # # q,c,a
    # inf_q = [[inf[2],inf[1],inf[3]] for inf in all_inf_q]
    # rel_q = [[inf[2],inf[1],inf[3]] for inf in all_rel_q]
    # all_q = inf_q+rel_q
    # q_data = pd.DataFrame(all_q)
    # q_data.to_csv(f'experiment/coverage/all_{dataset}_half_q.csv', sep=',',header=None,index=None)
    # all_test_case = pd.read_csv(f'experiment/QAQA/{dataset}_test_cases_all.csv').values.tolist()
    # qaqa_data = [[inf[0].split('\\n')[0],inf[0].split('\\n')[1],inf[1]] for inf in all_test_case]
    # qaqa = pd.DataFrame(qaqa_data)
    # qaqa.to_csv(f'experiment/coverage/QAQA_{dataset}_q.csv', sep=',',header=None,index=None)
    try:
        all_entity_q = pd.read_csv(f'../../data/result/test_result/bug_result/entity_bug_result_{dataset}_all_filter.csv').values.tolist()
        q_c_entity = [inf for inf in all_entity_q if jaccard_similarity(inf[1], context) > 0.85]
        all_entity_q = pd.read_csv(
            f'../../data/result/test_result/bug_result/rel_bug_result_{dataset}_all_filter.csv').values.tolist()
        q_c_rel = [inf for inf in all_entity_q if jaccard_similarity(inf[1], context) > 0.85]
        q_c = q_c_entity + q_c_rel
        q_list = [inf[0] for inf in q_c]
        q_list_str = ';'.join(q_list)
        prompt = f'''
                    I will provide you with a list of questions and the context from which these questions are generated. The list of questions includes all the questions generated based on this text. 
                    First, you need to deduplicate the list of questions and extract those that have different objectives or directions. Then, based on the context, you should determine which part of the context each question covers. If a question is unrelated to the context, output ' '.
                    Second, I need you to summarize the content of the first step, which involves merging the text covered by each sentence to identify which parts of the text are covered and which parts remain uncovered. You need to ensure that the content of these two parts is entirely the original text from the context.
                    The output should include: the parts of the text each question(after deduplicate) covers, the covered parts of the text, the uncovered parts of the text.
                    The format of output should be like(for a list of n questions):
                    q_question_1: ```question_1``` q_cover:```covered parts for question_1```
                    ...
                    q_question_n: ```question_n``` q_cover:```covered parts for question_n```
                    Uncovered parts: ```uncovered part```
                    Covered parts: ```covered part```
                    The question list is :[{q_list_str}]
                    The context is :{context}
                    The structure of the output should be simple, without excessive symbols or blank lines.
                                        '''
        answer = connect_gpt_prompt_cov(prompt, client)
        a = re.findall(r"```(.+)```", answer)
        uncover_parts_raw = answer.split('Uncovered parts')[-1].split('Covered parts')[0]
        cover_parts_raw = answer.split('Covered parts')[-1]
        # uncover_parts_raw = re.findall(r'Covered parts(.+)', answer)
        # cover_parts_raw = re.findall(r'Uncovered parts(.+)', answer)
        cover_parts = cover_parts_raw.replace('...', ' ')
        uncover_len = len(uncover_parts_raw.split(' '))
        cover_len = len(cover_parts.split(' '))
        all_len = len(context.split(' '))
        score = min(1 - uncover_len / all_len, cover_len / all_len)
        if score < 0:
            return None
        return score
    except:
        return None
    a = 1

if __name__ == '__main__':
    dataset = 'your_dataset'
    ran_c_num = int(config['coverage']['rand_c_num'])
    ran_context = choose_context(f'../../data/dataset/processed_data/{dataset}_dev_context.txt', ran_c_num)
    s_list = []
    for context in tqdm(ran_context):
        cover_score = find_q_context(context, dataset)
        if cover_score is not None:
            s_list.append(cover_score)
    with open('experiment/coverage/coverage_score.txt', 'a') as p:
        p.write(
            f'data_file:{dataset},average_score:{np.average(np.array(s_list))},all_data:{str(s_list)},time:{datetime.now()}. \n')

