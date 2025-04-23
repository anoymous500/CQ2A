from random import random

import pandas as pd
from tqdm import tqdm
import configparser

config = configparser.ConfigParser()

config.read(f'../../config.ini')
from entity_extraction import acquire_dependence_tree
from model_access import *

def entity_merge_data(dataset):
    df_list = []
    for file_tail in ['raw','rule_filter','all_filter']:
        for cond in ['phrases', 'words']:
            for ch_type in ['NOUN', 'VERB']:
                csv_file = f'../../data/result/case/{dataset}/entity_info_{dataset}_{cond}_{ch_type}_{file_tail}_q.csv'
                try:
                    f = pd.read_csv(csv_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
                    df_list.append(f)
                except:
                    continue
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df = combined_df.sample(len(combined_df))
        combined_df.to_csv(f'../../data/result/case/{dataset}/entity_info_{dataset}_{file_tail}_q.csv')

def data_change(dataset,info_type):
    if info_type == 'rel':
        in_file = f'../../data/result/case/{dataset}/rel_info_{dataset}_all_filter_q.csv'
        out_file = f'../../data/result/case/{dataset}/rel_sut_case_{dataset}_all_filter.tsv'
        raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
        a = []
        for r in raw:
            b = []
            # for i in range(len(r)):
            #     r[i].replace('\n','')
            # r[3]=r[3][:-1]
            b.append(r[0] + '\\n' + r[1])
            str_relation = r[2]
            # if b[0].remove():
            #     b[0] = b[0][:-1]
            relation = str_relation.strip('[]').split(',')
            gold_answer = relation[1]
            b.append(gold_answer)
            a.append(b)
        data = open(out_file, 'w', encoding='utf-8', newline=None)
        for c in a:
            print(c[0][:-2] + "\t" + c[1], file=data)
    elif info_type == 'entity':
        entity_merge_data(dataset)
        in_file = f'../../data/result/case/{dataset}/entity_info_{dataset}_all_filter_q.csv'
        out_file = f'../../data/result/case/{dataset}/entity_sut_case_{dataset}_all_filter.tsv'
        raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
        a = []
        for r in raw:
            b = []
            b.append(r[0] + '\\n' + r[1])
            gold_answer = r[2]
            b.append(gold_answer)
            a.append(b)
        data = open(out_file, 'w', encoding='utf-8', newline=None)
        for c in a:
            print(c[0][:-2] + "\t" + c[1], file=data)
    else:
        raise ValueError('info_type Wrong')

def sut_test(dataset,info_type):
    if info_type == 'rel':
        information_file = f'../../data/result/case/{dataset}/rel_info_{dataset}_all_filter_q.csv'
        input_file = f'../../data/result/case/{dataset}/entity_sut_case_{dataset}_all_filter.tsv'
        output_file = f'../../data/result/test_result/sut_result/entity_test_result_{dataset}_all_filter.csv'
    elif info_type == 'entity':
        information_file = f'../../data/result/case/{dataset}/entity_info_{dataset}_all_filter_q.csv'
        input_file = f'../../data/result/case/{dataset}/rel_sut_case_{dataset}_all_filter.tsv'
        output_file = f'../../data/result/test_result/sut_result/rel_test_result_{dataset}_all_filter.csv'
    else:
        raise ValueError('info_type Wrong')

    w_data = []
    raw = pd.read_csv(information_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    with open(input_file, "r", encoding='utf-8') as f:
        f = f.readlines()
        i = 0
        for line in tqdm(f):
            try:
                question, answer = line.split("\t")
                if answer[-1] == '\n':
                    answer = answer[:-1]
                sut_answer = run_model(question)[0]
                raw_q = raw[i][0]
                context = raw[i][1]
                w_data.append([raw_q, context, answer, sut_answer])
                i = i + 1
            except:
                i = i + 1
                continue
        data = pd.DataFrame(w_data)
        data.to_csv(output_file, sep=',', header=None, index=None)

def compare_through_llm(context,question,gold_answer,sut_answer, thread_id=None):
    prompt = f''' Based on the context "{context}" for the question "{question}" output the degree of consistency between the answer1 "{gold_answer}" and answer2 "{sut_answer}".
    The judge criteria of consistency includes
    1)High semantic similarity 
    2)whether the two answers are exactly pointed to the same thing based on the context
    3)If the content of answer1 includes in answer2 and all the content of answer2 directly appears in context
    Once any one of the above criterias is satisfied, they are considered to be consistent,and you should give a high score.
    The highest score can be 100 and the lowest score can be 0. you should make your reference based on the context. Output the score and explain why you give this score.
    The explanation should be specific enough to include the content of both answers
    the structure should be '[score] \\n [explain]'.
    '''
    raw_answer = connect_gpt_prompt(prompt, client, thread_id)
    try:
        result = raw_answer.split('\n')[0].split('[')[1].split(']')[0]
        explain = raw_answer.split('\n')[1]
        score_threshold = config['test']['gpt_score_threshold']
        if int(result) >= score_threshold:
            return 1, result, explain
        else:
            return 2, result, explain
    except:
        return 3, None, None

def compare_single(info_list):
    context = info_list[1]
    question = info_list[0]
    gold_answer = info_list[2]
    sut_answer = info_list[3]
    sim = sim_model.similarity(sut_answer, gold_answer)
    sim_threshold = config['test']['sim_threshold']
    if sim > sim_threshold:
        # 语义相似度极高的认为答案一致，否则用gpt协助判断
        info_list.extend([1,None,None])
    else:
        judge, result, explain = compare_through_llm(context,question,gold_answer,sut_answer)
        info_list.extend([judge, result, explain])
    return info_list

def bug_detect(dataset,info_type):
    if info_type == 'rel':
        r_file = f'../../data/result/test_result/sut_result/entity_test_result_{dataset}_all_filter.csv'
        w_file = f'../../data/result/test_result/bug_result/entity_bug_result_{dataset}_all_filter.csv'
    elif info_type == 'entity':
        r_file = f'../../data/result/test_result/sut_result/rel_test_result_{dataset}_all_filter.csv'
        w_file = f'../../data/result/test_result/bug_result/rel_bug_result_{dataset}_all_filter.csv'
    else:
        raise ValueError('info_type Wrong')

    raw = pd.read_csv(r_file, sep=',', encoding='ISO-8859-1')
    raw_list = raw.values.tolist()
    max_threads = 5
    results = []
    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(compare_single, prompt_info) for i, prompt_info in enumerate(raw_list)]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None:
                    results.append(result)
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")
    data = pd.DataFrame(results)
    data.to_csv(w_file, mode='w', index=False, header=False)

if __name__ == '__main__':
    for t in ['entity','rel']:
        sut_test('your_dataset',t)
        bug_detect('your_dataset',t)
