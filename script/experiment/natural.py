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

def check_gpt_score(dataset,info_type):
    file = f'../../data/result/test_result/bug_result/{info_type}_bug_result_{dataset}_all_filter.csv'
    record_file = f'../../data/result/experiment_data/natural/{dataset}/{info_type}_natural_{dataset}_all_filter.csv'
    data = pd.read_csv(file, header=None,encoding_errors='ignore').values.tolist()
    ran_num = int(config['natural']['rand_num'])
    ran_data = random.sample(data, ran_num)
    result_list = []
    aver_list = []
    for d in tqdm(ran_data):
        data_list = []
        question = d[0]
        context = d[1]
        data_list.extend([question,context])
        prompt = f'''
    I have a context here and a question generated based on this context. Please evaluate the naturalness of the question according to the following criteria: 
1、Whether the question is phrased in a way that a human would ask. 
2、Focus on the main semantic points of the question, ignoring contextual or supplementary information. Please summarize how many core semantic points the question has. A good question should have only one main semantic point. 
3、The entity and primary information in the question should all appear in the context
Each criteria has its own score, with scores ranging from 0 to 5. The rules for each score are as follows:
1. Whether the question is phrased in a way that a human would ask.
0 : Completely unnatural phrasing, sentence structure is incomplete or incorrect, making it unintelligible.
1 : Noticeable issues with sentence structure, unclear expression, and the phrasing doesn't sound natural.
2 : Partially natural but somewhat awkward or unclear in some places, which may confuse the reader.
3 : Mostly natural phrasing, but with minor details that don't quite sound like a human question.
4 : Clear and fluent phrasing, almost entirely natural, but with a few small issues.
5 : Perfectly natural phrasing, smooth sentence flow, sounds exactly like a human would ask.
2. Focus on the main semantic points of the question, ignoring contextual or supplementary information. Please summarize how many core semantic points the question has. A good question should have only one main semantic point.
0 : The question is off-topic, contains multiple unrelated or unnecessary semantic points, making it hard to extract the core information.
1 : The question contains two or more core semantic points, and the meaning is somewhat disordered, making it hard to identify the focus.
2 : The question contains multiple semantic points, but one is more prominent, with the others somewhat scattered, making the question a bit complex.
3 : The question contains several semantic points, but the main focus is clear, and the other points are supplementary.
4 : The question is very clear with a strong main point, and while there is some extra information, it doesn't detract from the main idea.
5 : The question contains only one core semantic point, completely focused on the topic, with almost no supplementary information.
3. The entity and primary information in the question should all appear in the context.
0 points: Entities or primary information in the question do not appear in the context at all, making the background of the question impossible to understand.
1 point: Some entities or information are mentioned in the context, but the relationships are unclear, making the context hard to understand.
2 points: Some entities or information are present in the context, but the connection is weak, making the background unclear.
3 points: Most of the entities or information in the question are mentioned in the context, but some inference or explanation is needed.
4 points: Almost all the entities or information in the question appear in the context, with only a small amount needing inference.
5 points: All entities and primary information in the question are fully supported by the context, clear and precise.
Based on the above standards, score the question in three aspects, provide an explanation for each score, and then calculate the average score.
 The output format should be as follows: 
Score1: [Score1] Explanation1: [Explanation1]  
Score2: [Score1] Explanation2: [Explanation1]  
Score3: [Score1] Explanation3: [Explanation1]  
The question and context to be evaluated are: Question: "{question}" Context: "{context}"
    '''
        try:
            result = connect_gpt_prompt_cov(prompt, client)
            score_pattern = r'Score.+:(.+)'
            explantaion_pattern = f'Explanation.+\:.+'
            score = re.findall(score_pattern, result)
            explanation = re.findall(explantaion_pattern, result)
            for i in range(len(score)):
                while score[i][0] not in '1234567890':
                    score[i] = score[i][1:]
                while score[i][-1] not in '1234567890':
                    score[i] = score[i][:-1]
            score_list = [int(s) for s in score]
            aver_score = np.average(np.array(score_list))
            explanation_list = [e.replace('Explanation', '') for e in explanation]
            assert len(score_list) == len(explanation_list) == 3
            for i in range(len(score_list)):
                data_list.append(score_list[i])
                data_list.append(explanation_list[i])

            data_list.append(aver_score)
            result_list.append(data_list)
            aver_list.append(aver_score)
            a = 1
        except:
            try:
                result = connect_gpt_prompt_cov(prompt, client)
                score_pattern = r'Score.+:(.+)'
                explantaion_pattern = f'Explanation.+\:.+'
                score = re.findall(score_pattern, result)
                explanation = re.findall(explantaion_pattern, result)
                for i in range(len(score)):
                    while score[i][0] not in '1234567890':
                        score[i] = score[i][1:]
                    while score[i][-1] not in '1234567890':
                        score[i] = score[i][:-1]
                score_list = [int(s) for s in score]
                aver_score = np.average(np.array(score_list))
                explanation_list = [e.replace('Explanation','') for e in explanation]
                assert len(score_list) == len(explanation_list) == 3
                for i in range(len(score_list)):
                    data_list.append(score_list[i])
                    data_list.append(explanation_list[i])
                data_list.append(aver_score)
                result_list.append(data_list)
                aver_list.append(aver_score)
            except:
                continue
    result_data = pd.DataFrame(result_list)
    result_data.to_csv(record_file, sep=',', header=None, index=None)
    return np.average(np.array(aver_list))

if __name__ == '__main__':
    dataset = 'your_dataset'
    for t in ['entity', 'rel']:
        avg_score = check_gpt_score(dataset,t)
        with open(f'../../data/result/experiment_data/natural/{dataset}/natural_score.txt', 'a') as p:
            p.write(f'dataset:{dataset},info_type:{t},average_score:{avg_score}. \n')