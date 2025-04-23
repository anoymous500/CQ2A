
import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor,as_completed
import re
from model_access import *
from entity_extraction import acquire_dependence_tree
import pandas as pd


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

def check_thread_process(prompt_info,thread_id,sim_model,cond,ch_type,nlp,raw_writer, check_type, pausewriter, writer):
    # [sent, context, g_answer, prompt, answer, str_question, str_reason]

    if prompt_info[2] in prompt_info[5]:
        pausewriter.writerow(prompt_info)
        return None
    if not check_answer(prompt_info[-2],prompt_info[2],cond,ch_type,prompt_info[0],nlp):
        pausewriter.writerow(prompt_info)
        return None
    prompt = f"According to the context '{prompt_info[1]}' and  question '{prompt_info[-2]}', the answer of the question is a word or phrase, " \
             f"please output the answer. Remember that the output should only be the answer without any other prompts."
    gen_a = connect_gpt_prompt_ask(prompt, client, thread_id)
    sim = sim_model.similarity(prompt_info[2], gen_a)
    pausewriter.writerow(prompt_info)
    if sim > 0.75:
        r_list = copy.deepcopy(prompt_info)
        r_list.append(gen_a)
        writer.writerow(r_list)
        return r_list
    else:
        return None

def entity_extraction(dataset):

    for cond in ['phrases', 'words']:
        for ch_type in ['NOUN', 'VERB']:
            info_data = pd.read_csv(
                f'../../data/result/relation_info/{dataset}/entity_info_{dataset}_{cond}_{ch_type}.csv',
                sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
            basic_info_list = []
            for info in info_data:
                sent = info[0]
                context = info[1]
                answer = info[2]
                prompt = f'''Design at least five questions for one word/phrase and its sentence in the context: '{context}'. 
                        You should make sure the answer of questions is the word/phrase in given sentence and then you should explain why you generate this question.
                                 Remember that your question should be exact enough to avoid the ambiguity according to the context.That is, the question should be specific enough that the answer to the question is unique.
                                 In addition, in order to increase the complexity of the question, you should try your best to add some additional information to the generated question according to the context above,the length of question should be longer than 25, but you need to ensure that the corresponding answer to the question should be the word or phrase I provide and you also need to make sure that the generated questions are natural, i.e. more human-like questions .
                                 .The output structure should be :"(1) Question:[]\\n   Reason:[] (2) ... (3) ... (4) ... (5) ...
                                 sentence:{sent}
                                 word:{answer}
                                 '''
                basic_info_list.append([sent, context, answer, prompt])
            results = get_threadpool_gpt(basic_info_list, 5)
            q_info_list = []
            for result in results:
                q_pattern = r'(Question\:.+)'
                r_pattern = r'(Reason\:.+)'
                num_pattern = r'\(\d+\) .+\n.+'
                answer = result[-1]
                try:
                    lists = re.findall(num_pattern, answer)
                except:
                    print(f'wrong answer:{answer}')
                    continue
                for qr_str in lists:
                    try:
                        str_question = re.findall(q_pattern, qr_str)[0]
                        str_question = str_question.split(':')[1][1:]
                        str_reason = re.findall(r_pattern, qr_str)[0]
                        str_reason = str_reason.split(':')[1][1:]
                        result_copy = copy.deepcopy(result)
                        result_copy.extend([str_question, str_reason])
                        q_info_list.append([result_copy[-2],result_copy[1],result_copy[2],result_copy[-1]])
                    #     [sent, context, answer, prompt, gpt_answer, str_question, str_reason]
                    except:
                        continue
            q_dataframe = pd.DataFrame(q_info_list)
            q_dataframe.to_csv(f'../../data/result/case/{dataset}/entity_info_{dataset}_{cond}_{ch_type}_raw_q.csv', mode='w',
                               index=False, header=False)

    # max_threads = 5
    # all_q = pd.read_csv(f'../../data/result/relation_info/{dataset}/entity_info_{dataset}_raw_q.csv', sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
    # with ThreadPoolExecutor(max_threads) as executor:
    #     # 提交任务到线程池中
    #     futures = [executor.submit(check_thread_process, prompt_info, i, sim_model, cond, ch_type, nlp, raw_writer,
    #                                check_type, pausewriter, writer) for i, prompt_info in tqdm(enumerate(remain_list))]
    #
    #     # 处理结果，as_completed 会在任务完成后返回结果
    #     for future in as_completed(futures):
    #         try:
    #             result = future.result()  # 获取每个线程的结果
    #             if result is not None:
    #                 results.append(result)
    #             # print(result)
    #         except Exception as e:
    #             print(f"Error occurred while processing result: {e}")


def rel_extraction(dataset):
    info_data = pd.read_csv(f'../../data/result/relation_info/{dataset}/rel_info_{dataset}.csv',
                            sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
    basic_info_list = []
    prompt = '''
                For relation triple like [entity1, relation, entity2],design at least ten different questions for it according to the context:{}.
                You'll need to meet the following requirements:
                 1) the answer of generated questions must be the relation in given triple [entity1, relation, entity2],
                 2) you should explain why your generated question satisfy the requirement, the explanation must strictly based on the generated question 
                 3) the entity1 and entity2 should appear in the generated question but the relation shouldn't appear in the generated question.
                 4) the question should be specific enough that the answer to the question is unique.
        Here are several examples of nice questions and their explanations:


        sentence : Alvin Dewey (Chris Cooper), the Kansas Bureau of Investigation's lead detective on the case, brushes him off, but Dewey's wife Marie (Amy Ryan) is a fan of Capote's writing and persuades her husband to invite Capote and Lee to their house for dinner
        relation triple:[Alvin Dewey,lead detective on the case,Kansas Bureau of Investigation]
        (1) Question: What is the relation of Alvin Dewey and Kansas Bureau of Investigation?
        Explanation:
        <1>The entity1 'Alvin Dewey' and entity2 'Kansas Bureau of Investigation' in the triple both appear in the generated question, and relation 'lead detective on the case' not appear in the question , satisfy the requirement.
        <2> The sentence mention "Alvin Dewey (Chris Cooper), the Kansas Bureau of Investigation's lead detective on the case", which suggests that Alvin Dewey is the lead detective on the case in Kansas Bureau of Investigation.So the relation of Alvin Dewey and Kansas Bureau of Investigation is "lead detective on the case" and the answer of generated question should be "lead detective on the case" ,satisfy the requirement.
        (2)Question: What role does Alvin Dewey have in the Kansas Bureau of Investigation?
        Explanation:
        <1>The entity1 'Alvin Dewey' and entity2 'Kansas Bureau of Investigation' in the triple both appear in the generated question,  and relation 'lead detective on the case' not appear in the question ,satisfy the requirement.
        <2>The sentence mention "Alvin Dewey (Chris Cooper), the Kansas Bureau of Investigation's lead detective on the case", which suggests that Alvin Dewey is the lead detective on the case in Kansas Bureau of Investigation. So Alvin Dewey have the role of "lead detective on the case" in the Kansas Bureau of Investigation and the answer of generated question should be "lead detective on the case" ,satisfy the requirement.
        (3)Question:How is Alvin Dewey connected to the Kansas Bureau of Investigation?
        Explanation:
        <1>The entity1 'Alvin Dewey' and entity2 'Kansas Bureau of Investigation' in the triple both appear in the generated question,  and relation 'lead detective on the case' not appear in the question ,satisfy the requirement.
        <2> The sentence mention "Alvin Dewey (Chris Cooper), the Kansas Bureau of Investigation's lead detective on the case", which suggests that Alvin Dewey is the lead detective on the case in Kansas Bureau of Investigation So Alvin Dewey is connected to the Kansas Bureau of Investigation by the role of "lead detective on the case" and the answer of generated question should be "lead detective on the case" ,satisfy the requirement.

        sentence:In the course of time, Lee's best-selling novel To Kill a Mockingbird is turned into a movie, but Capote is unable to share in the joy of his friend's success, too caught up in drinking through his own misery.
        relation triple:[Capotn, friend, Lee]
        (1)Question: What is the relation between Capotn and Lee?
        Explanation:
        <1>The entity1 'Capotn' and entity2 'Lee' in the triple both appear in the generated question, and relation 'friend' not appear in the question ,satisfy the requirement.
        <2> The sentence mention "Capote is unable to share in the joy of his friend's success", the success is that Lee's best-selling novel To Kill a Mockingbird is turned into a movie,which indicates Capotn is the friend of Lee.So the relation between Capotn and Lee is 'friend of' and the answer of generated question should be "friend" ,satisfy the requirement.
        (2)Question: What relationship exists between Capotn and Lee?
        Explanation:
        <1>The entity1 'Capotn' and entity2 'Lee' in the triple both appear in the generated question, and relation 'friend' not appear in the question , satisfy the requirement.
        <2>The sentence mention "Capote is unable to share in the joy of his friend's success", the success is that Lee's best-selling novel To Kill a Mockingbird is turned into a movie,which indicates Capotn is the friend of Lee.So the relationship 'friend of' exists between Capotn and Lee and the answer of generated question should be "friend" ,satisfy the requirement.
        (3)Question: To Lee, who is Capotn?
        Explanation:
        <1>The entity1 'Capotn' and entity2 'Lee' in the triple both appear in the generated question, and relation 'friend' not appear in the question , satisfy the requirement.
        <2>The sentence mention "Capote is unable to share in the joy of his friend's success", the success is that Lee's best-selling novel To Kill a Mockingbird is turned into a movie,which indicates Capotn is the friend of Lee.So to Lee , Capotn is his friend and the answer of generated question should be "friend" ,satisfy the requirement.
        The output structure should be :"(1) Question:[]\\n   Explanation:[] (2) ... (3) ... (4) ... (5) ...
        sentence: {}
        Relation triple: {}
                '''

    q_list = []
    q_raw_list = []
    total_raw_list = []
    prompt_list = []
    for i in range(len(info_data)):
        sent = info_data[i][0]
        context = info_data[i][1]
        relation_triple = info_data[i][3]
        prompt_list.append([sent, context, relation_triple, prompt.format(context, sent, relation_triple)])
    thread_num = 5
    results = get_threadpool_gpt(prompt_list, thread_num)
    num_pattern = r'\(\d+\) .+\n.+'
    q_info_list = []
    for prompt_info in results:
        q_pattern = r'(Question\:.+)'
        r_pattern = r'(Explanation\:.+)'
        num_pattern = r'\(\d+\) .+\n.+'
        answer = prompt_info[-1]
        try:
            lists = re.findall(num_pattern, answer)
        except:
            print(f'wrong answer:{answer}')
            continue
        for qr_str in lists:
            try:
                str_question = re.findall(q_pattern, qr_str)[0]
                str_question = str_question.split(':')[1][1:]
                str_reason = re.findall(r_pattern, qr_str)[0]
                str_reason = str_reason.split(':')[1][1:]
                result_copy = copy.deepcopy(prompt_info)
                result_copy.extend([str_question, str_reason])
                q_info_list.append([result_copy[-2],result_copy[1],result_copy[2],result_copy[-1]])
                #     [sent, context, rel_triple, prompt, gpt_answer, str_question, str_reason]
            except:
                continue
    q_dataframe_1 = pd.DataFrame(q_info_list)
    q_dataframe_1.to_csv(f'../../data/result/case/{dataset}/rel_info_{dataset}_raw_q.csv', mode='w', index=False, header=False)

if __name__ == '__main__':
    # entity_part
    entity_extraction('your_dataset')
    # rel_part
    rel_extraction('your_dataset')