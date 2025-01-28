import copy
import json
import time
import operator
from tqdm import tqdm
from model_access import *
from datetime import datetime
import csv
import multiprocessing
from functools import partial
import pandas as pd
from simcse import SimCSE
from multiprocessing import Pool
import argparse
import numpy as np
import random
from openai import OpenAI
from model_access import get_glm4,get_qwen,get_abab6
import logging
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import wordnet
import string
import csv
from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
import os
from nltk.tree import Tree
from transformers import pipeline
from queue import LifoQueue

from bs4 import BeautifulSoup
import requests
import stanza
# from spacy import displacy
import re
# import spacy
import openai
# proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
# stanza.download(lang='en', resources_url='stanford')
import requests
from collections import Counter
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler("log/consistency_rel_log.txt")
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
#从nltk库中导入需要的类，并且进行实例化
# from nltk.stem import WordNetLemmatizer
# wnl = WordNetLemmatizer()
# from simcse import SimCSE
# from sentence_transformers import util
# from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('sentence-transformers')
# import opennre
# opennre_model = opennre.get_model('wiki80_bert_softmax')

# nlp = stanza.Pipeline(lang='en')
# def indentify_entity(sent,nlp):
#     # model = opennre.get_model('wiki80_cnn_softmax')
#     # s_inf = acquire_dependence_tree(sent,nlp)
#     s_ent = nlp(sent).entities
#     ent_word_list = [(span.text,span.start_char,span.end_char) for span in s_ent]
#     sent_list = sent.split(' ')
#     ent_re_pair_list = []
#     for i in range(len(ent_word_list)):
#         for j in range(i+1,len(ent_word_list)):
#             ent_re_pair_list.append([ent_word_list[i],ent_word_list[j]])
#     result_re_pair_list = []
#     for ent_pair in ent_re_pair_list:
#         re_pair = opennre_model.infer({'text': sent, 'h': {'pos': (ent_pair[0][1], ent_pair[0][2])}, 't': {'pos': (ent_pair[1][1], ent_pair[1][2])}})
#         result_re_pair_list.append(re_pair)
#
# indentify_entity('Tom opens a company.',nlp)
nlp = stanza.Pipeline(lang='en')
OPENAI_KEY = 'your_key'
client = OpenAI(
        api_key=OPENAI_KEY,
    )

def acquire_dependence_tree(sentence,nlp):

    doc = nlp(sentence)
    print(*[
        f'id: {word.id}\tfeats: {word.feats if word.feats else "_"}\tword: {word.text}\tlemma: {word.lemma}\tupos: {word.upos}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
        for sent in doc.sentences for word in sent.words], sep='\n')
    return [[word.id, word.feats, word.text, word.lemma, word.upos, word.head,sent.words[word.head - 1].text if word.head > 0 else "root",word.deprel] for sent in doc.sentences for word in sent.words]


def relation_extraction_s(context):
    # 以sent为单位进行提问
    # context = '''The beautiful young Theodora Fitzgerald belongs to a family of noble lineage whose fortunes have waned and who have lived in near poverty for most of her life. The book begins with her arranged marriage to Josiah Brown, a nouveau-riche Australian in his fifties. The marriage was contracted for convenience: Josiah simply wants a pretty and aristocratic wife to improve his standing in society, and the Fitzgerald family are in need of Brown's financial resources. Theodora only agrees to the marriage for the sake of her father and sisters.Immediately after the wedding, Josiah falls ill. Theodora proves a dutiful and capable wife, and attends to her husband's every need, though she is secretly very unhappy. After a year of marriage, Josiah is well enough to visit Paris, where Theodora sees her father, Dominic, again for the first time since her wedding. She is thrilled to observe that at least he is receiving all the benefits she'd hoped to bring from her sacrifice: he now runs in aristocratic circles and is courting a wealthy American widow, Mrs. McBride. Theodora attends several social outings with her father, and at one dinner is introduced to Hector, Lord Bracondale. Theodora and Hector hit things off splendidly, and soon fall in love. Mrs. McBride is aware of Theodora's unhappy marriage, and seeing the situation she sympathetically arranges for Hector and Theodora to spend time together as often as possible. One day while Theodora and Hector are being chauffeured back to Paris after an outing at Versailles, the two indulge in a romantic encounter in the back of the car. Full of guilt thereafter, the two conclude they must behave themselves from now on and must no longer pursue each other romantically; they will, however, continue to be friendly to one another any time future social obligations might cause them to meet.Hector at this point is terribly in love with Theodora, and though he tries his best to live by his promise to her, he still goes out of his way to see her and to secure invitations to all the same gatherings that she attends. He fantasizes about marrying her and makes sure to introduce her to his mother and to his sister. However, Theodora's status as a newcomer into society, and the obvious favor that Hector pays her over other eligible women who desire his hand, causes ire and jealousy to be directed her way. Rumors begin to spread, and several people believe Hector and Theodora to be lovers. Morella Winmarleigh, a spurned candidate for Hector's hand, particularly sets out destroy Theodora. She maliciously switches a letter Theodora had written to Hector with another letter meant for Josiah. Meanwhile, without anyone else's knowledge, Theodora and Hector have concluded that they cannot attempt to remain friends any longer?￠????their love is too strong?￠????and so they must agree to never see each other again.The next day, Josiah receives Theodora's letter meant for Hector: the contents amount to Theodora asking Hector never to see her again, even though the two of them could be very happy together, because it is her duty to instead attend to the happiness of her husband Josiah. Josiah realizes for the first time how he has stood in the way of Theodora's happiness, and resolves to do his best to make her happy from now on. He forwards the letter to Hector and requests he never allow Theodora to learn of the mix-up. The next several months pass with Theodora and Josiah both trying their best to make the other happy, even while both are secretly miserable. Both begin to suffer from ill health. Ultimately, Josiah dies; eighteen months later, Mrs. McBride (now married to Dominic Fitzgerald) throws a picnic at Versailles to which both Theodora and Hector are invited. The book ends with the couple reunited, in a state of "passionate love and delirious happiness.'''
    # prompt = f'''Please output all the proper relations of noun entities in the following context with the structure of triple. For example,for context "Lucius Harney becomes Mr. Royall's boarder.",the output can be [Lucius Harney,becomes,Mr. Royall's boarder] and for context "Lucius Harney's cousin, Miss Hatchard, leaves the village.",the output can include [Miss Hatchard,cousin,Lucius Harney] and [Miss Hatchard,leaves,village]. Besides, to make the answer reasonably, you also need to explain why you choose this relation.So the structure of output should be shown as:"(1) [entity, relation1, entity],reason for relation1.     \\n  (2) [entity, relation2, entity],reason for relation2... " For clarity of referent, if a pronoun(like he,she,they,him and so on) is included in the relation triple, replace it with that it refers to. The context is :"{context}",based on the rules and examples, output all the relation triple of this context.'''
    # prompt = f'''Relation triple include three part: entity1, relation, entity2,please output all the proper relations of noun entities in the following context with the structure of triple. Here are some examples and rules.
    # For example,for context "Lucius Harney becomes Mr. Royall's boarder.",the output can be [Lucius Harney,becomes,Mr. Royall's boarder] and for context "Lucius Harney's cousin, Miss Hatchard, leaves the village.",the output can include [Miss Hatchard,cousin,Lucius Harney] and [Miss Hatchard,leaves,village].
    # The rules include (1)to make the answer reasonably, you need to explain why you choose this relation, the entities in the outputed triples should appear in your explanation .
    #  (2)the structure of output should be shown as:"(1) [entity, relation1, entity],[explanation for relation1].     \\n  (2) [entity, relation2, entity],[explanation for relation2]... ".
    #  (3)For clarity of referent, if a pronoun appears in the relation triple, replace it with the word it refers to.
    # The real context is :("{context}"),now based on above rules and examples, output all the relation triple of this context.'''
    prompt = '''Extract the relations between the NOUN entities in the given text and provide a reasonable explanation. 
    Here are some rules: 
    1.You should make sure that your extraction result without any ambiguity.If a pronoun(such as he,it,she,his,her and so on) appears in the relation triple, replace it with the word it refers to.
    2.You can make some inferences based on the text to extract the entity relationships implicit in the sentence and you should output extraction triples as much as possible, the more the better.                                                                                                                3.The extracted relation is a substantive verb or noun, not a linking verb(such as is, are, have and so on) and the output triple should include complete three parts splited by ',' and the two entities should not be the same.                             4. You need to extract as many relationships as possible,
    3.The extracted relation must be a substantive verb or noun, not a linking verb(such as 'is', 'are', 'have' and so on) and the outputed triple should include three parts splited by ',' and the two entities should not be the same. 
    Some examples are listed as follows.

        TEXT: "Lucius Harney becomes Mr. Royall's boarder."
        (1)Explanation: In the text Lucius Harney becomes Mr. Royall's boarder.
        Relations: [Lucius Harney, becomes, Mr. Royall's boarder.]

        TEXT: "Edward Marks, an official with the Montgomery County Democratic Party, argued that if Ms. Toth is not interested in the job, ‘she should get out..."
        (1)Explanation: Edward Marks is an official that works for the Montgomery County Democratic Party.
        Relations: [Edward Marks, Work For, Montgomery County Democratic Party]
        (2)Explanation: Edward Marks is a member of the Montgomery County Democratic Party.
        Relations: [Edward Marks, member, Montgomery County Democratic Party]
        
        TEXT: "Walter Hargrave, the brother of Helen's friend Milicent Hargrave, vies for Helen's affections"
        (1)Explanation: Walter Hargrave is the brother of Helen's friend Milicent Hargrave
        Relations: [Walter Hargrave, brother, Milicent Hargrave]
        (2)Explanation: Walter Hargrave vies for Helen's affections
        Relations: [Walter Hargrave, vies for Helen's affections, Helen]
        (3)Explanation: Milicent Hargrave is Helen's friend.
        Relations: [Helen, friend, Milicent Hargrave]

        TEXT: In 1910, the last year of Leo Tolstoy's life, his disciples, led by Vladimir Chertkov, manoeuvre against his wife, Sofya, for control over Tolstoy's works after his death.
        (1)Explanation: Sofya is Tolstoy's wife.
        Relations: [Sofya, wife, Tolstoy]
        (2)Explanation: his disciples, led by Vladimir Chertkov, manoeuvre against his wife, Sofya.
        Relations: [Vladimir Chertkov, manoeuvre against, Sofya]
        (3)Explanation: his disciples, led by Vladimir Chertkov.
        Relations: [disciples, led by, Vladimir Chertkov]

    TEXT:"{}" 
    Based on the TEXT and three rules, extract relation triples as many as you can.You should only output every explanation and relation as above without any other information'''
    sent_list = context.split('.')
    #
    prompt_list = []
    for sent in sent_list:
        if len(sent.split(' '))>8:
            prompt_list.append([sent, context, prompt.format(sent)])
    thread_num = 5

    prompt_list_1 = get_threadpool_gpt(prompt_list,thread_num)
#     answer = '''(1) [Indiana Jones, is kidnapped by, Soviet agents], Indiana Jones and his partner George "Mac" McHale are kidnapped in Nevada by Soviet agents under Colonel Dr. Irina Spalko.
#    (2) [Soviets, infiltrate, a warehouse labeled "Warehouse 51"], The Soviets infiltrate a warehouse labeled "Warehouse 51" and force Jones to locate an alien corpse with a crystal skull, recovered ten years earlier.
#    (3) [Mac, reveals, he is a double agent], Upon its discovery, Mac reveals he is a double agent working for the Soviets.
#    (4) [Jones, unsuccessfully attempts to retrieve, the skull], Jones escapes and unsuccessfully attempts to retrieve the crystal skull.
#
# and so on...'''
    pattern = r'\d+.+\n.+'
    prompt_info_list = []
    for p_info in prompt_list_1:
        answer = p_info[-1]
        fin_answer = []
        answer_list = re.findall(pattern, answer)
        a = 1
        for raw_answer in answer_list:
            try:
                num_pattern = r'\(\d+\) '
                raw_answer1 = re.sub(num_pattern, '', raw_answer)
                # raw_answer1 = raw_answer.replace(str_num,'')
                rel_pattern = r'(\[.+\])'
                str_relation = re.findall(rel_pattern, raw_answer1)[0]
                relation = str_relation.strip('[]').split(',')
                explain_pattern = r'(Explanation\:.+)'
                str_explain = re.findall(explain_pattern, raw_answer1)[0]
                str_explain = str_explain.split(':')[1][1:]
                a = 1
            except:
                logger = logging.getLogger()
                logger.info(f'relation wrong, wrong relation info is {raw_answer}')
                continue
            if len(relation) != 3:
                continue
            # 实体应该包含在解释中，实体应该为名词或名词词组
            try:
                ent_list = [relation[0], relation[2]]
            except:
                continue
            flag = 1
            # for entity in ent_list:
            #     e_inf = acquire_dependence_tree(entity, nlp)
            #     inf_head = [s[-2] for s in e_inf]
            #     try:
            #         index = inf_head.index('root')
            #         if entity not in str_explain or entity not in context or e_inf[index][-4] not in ['PROPN', 'NOUN']:
            #             flag = 0
            #             break
            #     except:
            #         flag = 0
            if flag:
                k = copy.deepcopy(p_info)
                prompt_info_list.append(k[:-1]+[str_relation,str_explain])

            a = 1
        a = 1
    # logger = logging.getLogger()
    # logger.info(f'extract {len(fin_answer)} relation triples from this context.')
    # ad_dataframe = pd.DataFrame(fin_answer)
    # ad_dataframe.to_csv(w_path, mode='a', index=False, header=False)
    return prompt_info_list

def check_from_ask_re(sent, answer, question, sim_model):

    prompt = f"According to the context '{sent}' and  question '{question}', the answer of the question should be relation of two entities, the form of answer is a word or phrase " \
                 f"please output the answer. Remember that the output should only be the answer without any other prompts."
    gen_a = connect_gpt_prompt_ask(prompt=prompt, client=client)
    sim = sim_model.similarity(answer, gen_a)
    if sim >= 0.7 :
        return [gen_a, True]
    else:
        if answer in gen_a:
            return [gen_a, True]
        return [gen_a, False]

def func_check_from_ask_re(raw_file,sim_model,w_path):
    # 从raw_file结果中进行重新提问筛选
    raw = pd.read_csv(raw_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    result_list = []
    for i in range(len(raw)):
        gena = raw[i][-1]
        answer = raw[i][1]
        sim = sim_model.similarity(answer, gena)
        gena_info = acquire_dependence_tree(gena, nlp)
        a_info = acquire_dependence_tree(answer, nlp)
        a_word = [a_info[i][3] for i in range(len(a_info)) if a_info[i][-4] in ['ADJ', 'NOUN', 'VERB', 'ADV']]
        gena_word = [gena_info[i][3] for i in range(len(gena_info)) if
                     gena_info[i][-4] in ['ADJ', 'NOUN', 'VERB', 'ADV']]
        if sim >= 0.9:
            result_list.append(raw[i])
        else:
            flag = 1
            for a in a_word:
                if a not in gena_word:
                    flag = 0
                    break
            if flag:
                result_list.append(raw[i])
    q_dataframe = pd.DataFrame(result_list)
    q_dataframe.to_csv(w_path, mode='w', index=False, header=False)

def check_rel_question(qr_str,prompt_info,totalraw_writer,raw_writer,writer,pause_writer):
    q_pattern = r'Question\:.+'
    total_raw_list = []
    q_raw_list = []
    q_list = []
    try:
        str_question = re.findall(q_pattern, qr_str)[0]
        str_question = str_question.split(':')[1][1:]
        str_reason = qr_str.split('Explanation:')[1]
        # rel_info_list[i].append(str_question)
        # rel_info_list[i].append(str_reason)
        # q_list.append(rel_info_list[i])
        tr_str = prompt_info[2][1:-1]
        sub1 = tr_str.split(',')[0]
        sub2 = tr_str.split(',')[-1]
        relation = tr_str.split(',')[1]
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

        context = prompt_info[1]
        sent = prompt_info[0]
        basic_info = prompt_info[:-2]
        flag = 1
        # relation triple中的信息应该出现在reason中
        if relation not in str_reason or relation in str_question or relation in ['am','is','are','have','has']:
            flag = 0
            pause_writer.writerow(prompt_info)
        if sub1 not in str_question or sub2 not in str_question:
            flag = 0
            pause_writer.writerow(prompt_info)
        raw_l = basic_info.copy()
        raw_l.append(str_question)
        raw_l.append(str_reason)
        # total_raw_list.append(raw_l)
        totalraw_writer.writerow(raw_l)
        ch_result = ['','']
        # 重新提问答案应该一致
        if flag != 0:
            ch_result = check_from_ask_re(sent, relation, str_question, sim_model)
            raw_l = basic_info.copy()
            raw_l.append(str_question)
            raw_l.append(str_reason)
            raw_l.append(ch_result[0])
            q_raw_list.append(raw_l)
            raw_writer.writerow(raw_l)
            pause_writer.writerow(prompt_info)
        # ch_result = check_from_ask_re(sent, rel_info_list[i][1], str_question, sim_model, rel_info_list[i][4])
        if flag and ch_result[1]:
            raw_l = basic_info.copy()
            raw_l.append(str_question)
            raw_l.append(str_reason)
            raw_l.append(ch_result[0])
            q_list.append(raw_l)
            writer.writerow(raw_l)
        return total_raw_list, q_raw_list, q_list
    except:
        return [], [], []

def rel_question_generate(rel_info_list,w_path,raw_w_path,totalraw_w_path=None,pause_file=None,dataset=None,data_interval=None):
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
    info_file_name_0 = f'rel_part/rel_{dataset}__{data_interval}_relresultinfo_{file_tail}.csv'
    info_file_name_1 = f'rel_part/rel_{dataset}__{data_interval}_questioninfo_{file_tail}.csv'
    if os.path.exists(info_file_name_1):
        load_data = pd.read_csv(info_file_name_1)
        all_info_list = load_data.values.tolist()
    else:
        for i in range(len(rel_info_list)):
            sent = rel_info_list[i][0]
            context = rel_info_list[i][1]
            relation_triple = rel_info_list[i][3]
            prompt_list.append([sent, context, relation_triple, prompt.format(context, sent, relation_triple)])
        thread_num = 10
        pause_file1 = f'experiment/pause_file/rel_{dataset}_{data_interval}_getallq_{file_tail}.csv'
        remain_list = []
        if not os.path.exists(pause_file1):
            empty = pd.DataFrame([])
            empty.to_csv(pause_file1, mode='w', index=False, header=False)
            pause_data1 = []
        else:
            pause_data1 = pd.read_csv(pause_file1)
            pause_data1 = pause_data1.values.tolist()
        for prompt_info in prompt_list:
            if prompt_info not in pause_data1:
                remain_list.append(prompt_info)
        with open(pause_file1, "a", newline='', encoding='utf-8') as csvfile1, \
            open(info_file_name_0, "a", newline='', encoding='utf-8') as csvfile2:
            pause_writer = csv.writer(csvfile1)
            writer = csv.writer(csvfile2)
            prompt_list_1 = get_threadpool_gpt(remain_list, thread_num, pause_writer, writer)
        prompt_list_1 = pd.read_csv(info_file_name_0,encoding_errors='ignore')
        prompt_list_1 = prompt_list_1.values.tolist()
        num_pattern = r'\(\d+\) .+\n.+'
        all_info_list = []
        for prompt_info in prompt_list_1:
            q_pattern = r'Question\:.+'
            # r_pattern = r'Explanation\:\n.+'
            answer = prompt_info[-1]
            try:
                lists = re.findall(num_pattern, answer)
            except:
                continue
            for qr_str in lists:
                all_info_list.append(prompt_info + [qr_str])
        q_dataframe_1 = pd.DataFrame(all_info_list)
        q_dataframe_1.to_csv(info_file_name_1, mode='w', index=False, header=False)
    max_threads = 10
    if not os.path.exists(pause_file):
        empty = pd.DataFrame([])
        empty.to_csv(pause_file, mode='w', index=False, header=False)
        pause_list = []
    else:
        pause_data = pd.read_csv(pause_file)
        pause_list = pause_data.values.tolist()
    remain_list = []
    for prompt_info in all_info_list:
        if prompt_info not in pause_list:
            remain_list.append(prompt_info)
    with open(totalraw_w_path, "a", newline='', encoding='utf-8') as csvfile1,\
        open(raw_w_path, "a", newline='', encoding='utf-8') as csvfile2, \
            open(w_path, "a", newline='', encoding='utf-8') as csvfile3, \
                open(pause_file, "a", newline='', encoding='utf-8') as csvfile4:
        totalraw_writer = csv.writer(csvfile1)
        raw_writer = csv.writer(csvfile2)
        writer = csv.writer(csvfile3)
        pause_writer = csv.writer(csvfile4)
        with ThreadPoolExecutor(max_threads) as executor:
            # 提交任务到线程池中
            futures = [executor.submit(check_rel_question, prompt_info[-1], prompt_info,totalraw_writer,raw_writer,writer,pause_writer)
                       for i, prompt_info in enumerate(remain_list)]

            # 处理结果，as_completed 会在任务完成后返回结果
            for future in as_completed(futures):
                try:
                    result = future.result()  # 获取每个线程的结果
                    total_raw_list.extend(result[0])
                    q_raw_list.extend(result[1])
                    q_list.extend(result[2])
                    # print(result)
                except Exception as e:
                    print(f"Error occurred while processing result: {e}")


    # q_dataframe = pd.DataFrame(total_raw_list)
    # q_dataframe.to_csv(totalraw_w_path, mode='a', index=False, header=False)
    # q_dataframe = pd.DataFrame(q_raw_list)
    # q_dataframe.to_csv(raw_w_path, mode='a', index=False, header=False)
    # q_dataframe = pd.DataFrame(q_list)
    # q_dataframe.to_csv(w_path, mode='a', index=False, header=False)

def func_check_sut_result(sut_file,output_file=None):
    # logger = logging.getLogger(__name__)
    # log_format = "%(levelname)s %(asctime)s - %(message)s"
    # logging.basicConfig(
    #     filename='log/consistency_log_random.txt',
    #     filemode='a',
    #     format=log_format,
    #     level=logging.INFO
    # )


    raw = pd.read_csv(sut_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(raw)):
        # golden_judge,auto_judge =  raw[i][3],raw[i][2],raw[i][0]
        # pred = check_sut(sut_answer,gold_answer,sent)
        # if len(raw[i]) == 5:
        #     raw[i].append(pred)
        # elif len(raw[i]) == 6:
        #     raw[i][-1] = pred
        if raw[i][5] == '1' or raw[i][5] == '2' or raw[i][5] == '3':
            raw[i][5] = eval(raw[i][5])
        if raw[i][6] == '1' or raw[i][6] == '2' or raw[i][6] == '3':
            raw[i][6] = eval(raw[i][6])
        if raw[i][5] == 2 and raw[i][6] == 2:
            tp += 1
        elif (raw[i][5] == 1 ) and raw[i][6] == 2:
            fp += 1
        # elif (raw[i][4] == 1 or raw[i][4] == 3) and raw[i][5] == 2:
        #     fp += 1
        elif (raw[i][5] == 2 )and raw[i][6] == 1:
            fn += 1
        # elif (raw[i][4] == 2 or raw[i][4] == 3 )and raw[i][5] == 1:
        #     fn += 1
        elif raw[i][5] == 1 and raw[i][6] == 1:
            tn += 1
    # data = pd.DataFrame(raw)
    # data.to_csv(output_file,sep=',',header=None,index=None)
    acc = (tp+tn)/(tp+tn+fp+fn)
    if fp!=0 or tp!=0:
        precision = tp/(fp+tp)
    else:
        precision = 1000000
    recall = tp/(tp+fn)

    logger.info(f'''{sut_file}
    tp:{tp},tn:{tn},fp:{fp},fn:{fn}
    acc:{acc};
    precision:{precision}
    recall:{recall}
    ''')
    print(f'tp:{tp},tn:{tn},fp:{fp},fn:{fn}')
    print(acc)
    print(precision)
    print(recall)

def rel_main(r_file,num1=None,num2=None,file_tail=None,data_interval=None):
    # 关系信息提取
    start_time = time.time()
    # # fee1 = get_account_balance(api_key)
    dataset = r_file.split('/')[-1].split('_')[0]
    with open(r_file, mode='r',encoding='utf-8',newline='') as f:
        raw = f.readlines()
        result_text = []
        for r in raw :
            result_text.append(r)
        f.close()
    if num1 is not None and num2 is not None:
        random_text_list = result_text[num1:num2]
    else:
        random_text_list = result_text
    max_threads = 10   # 同时运行的最大线程数
    results = []
    all_info = []
    info_file_name = f'rel_part/rel_{dataset}_{data_interval}_info_{file_tail}.csv'
    # contexts_to_prompts返回[[sent, context, phrase, prompt],...]
    if os.path.exists(info_file_name):
        load_data = pd.read_csv(info_file_name)
        all_info = load_data.values.tolist()
    else:
        with ThreadPoolExecutor(max_threads) as executor:
            # 提交任务到线程池中
            futures = [executor.submit(relation_extraction_s, context) for i, context in
                       enumerate(random_text_list)]

            # 处理结果，as_completed 会在任务完成后返回结果
            for future in as_completed(futures):
                try:
                    result = future.result()  # 获取每个线程的结果
                    if result is not None:
                        results.append(result)
                    # print(result)
                except Exception as e:
                    print(f"Error occurred while processing result: {e}")

        print('All contexts processed:')
        for result in results:
            all_info.extend(result)
        q_dataframe = pd.DataFrame(all_info)
        q_dataframe.to_csv(info_file_name, mode='w', index=False, header=False)

    a = 1
    # info_file_name = f'rel_part/rel_{dataset}_info_{file_tail}.csv'
    # q_dataframe = pd.DataFrame(all_info)
    # q_dataframe.to_csv(info_file_name, mode='a', index=False, header=False)
    # info_data = pd.read_csv(info_file_name)
    # all_info = info_data.values.tolist()
    file_name = f'rel_{dataset}_q_{file_tail}.csv'
    pause_file = f'experiment/pause_file/{dataset}_{data_interval}_gptresultinfo_{file_tail}.csv'
    rel_question_generate(all_info, f'rel_part/rel_{dataset}_q_{file_tail}.csv', f'rel_part/rel_{dataset}_q_raw_{file_tail}.csv',
                        f'rel_part/rel_{dataset}_q_totalraw_{file_tail}.csv',pause_file,dataset,data_interval)
    end_time = time.time()
    during_time = end_time - start_time
    # fee2 = get_account_balance(api_key)
    # fee = fee2 - fee1
    # with open('experiment/fact_data/fee.txt', 'a') as p:
    #     p.write(
    #         f'dataset:{dataset},info_type:relation_info,length:{len(random_text_list)},time:{datetime.now()},file_name:{file_name}'
    #         f'fee:{fee}' + '\n')
    with open('experiment/fact_data/real_time.txt', 'a') as p:
        p.write(f'dataset:{dataset},info_type:relation_info,context_length:{len(random_text_list)},q_length:{len(random_text_list)},time:{datetime.now()},file_name:{file_name}'
                f'during_time:{during_time}'+'\n')
    # rel_question_generate(raw_list,'rel_part/raw_rel_question_s_ten_lt_new_nc4.csv' ,'rel_part/sent_rel_question_s_ten_lt_new_nc4.csv',
    #                       'context_Dataset.txt','low_temperature','rel_part/totalraw_ten_lt_new_nc4.csv')
    # rel_question_generate(raw_list, 'rel_part/raw_rel_question_s_ten_pp_new_nc4.csv',
    #                       'rel_part/sent_rel_question_s_ten_pp_new_nc4.csv',
    #                       'context_Dataset.txt', 'presence_penalty', 'rel_part/totalraw_ten_pp_new_nc4.csv')
    # rel_question_generate(raw_list, 'rel_part/raw_rel_question_s_ht.csv', 'rel_part/sent_rel_question_s_ht.csv',
    #                       'context_Dataset.txt','high_temperature')
    # rel_question_generate(raw_list, 'rel_part/raw_rel_question_s_pp.csv', 'rel_part/sent_rel_question_s_pp.csv',
    #                       'context_Dataset.txt','presence_penalty')
    a = 1
model_name = "D:/huggingface_model/unifiedqa-v2-t5-large-1251000"  # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
unifiedqa_model = T5ForConditionalGeneration.from_pretrained(model_name)
device = torch.device('cuda:0')
unifiedqa_model.to(device)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = unifiedqa_model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
def compare_through_llm(context,question,gold_answer,sut_answer,mode, thread_id):
    prompt = ''
    if mode == 1:
        # question+分数
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
    elif mode == 2:
        # 分数
        prompt = f'''Based on the context "{context}" output the degree of consistency between the  expression1 "{gold_answer}" and expression2 "{sut_answer}".
                            The judge criteria of agreement  includes semantic similarity and whether the two answers are exactly the same and whether the content of both answers directly appears in context, if not, the score should be low. 
                            The highest score can be 100 and the lowest score can be 0. you should make your reference based on the context. Output the score and explain why you give this score the structure should be '[score] \\n [explain]'.
                '''

    elif mode == 3:
        # question + 直接判断
        prompt = f'''Based on the context "{context}" for the question "{question}" judge the consistency between the  answer1 "{gold_answer}" and answer2 "{sut_answer}".
                            The judge criteria of consistency includes
    1)High semantic similarity 
    2)The two answers are exactly pointed to the same thing based on the context
    3)If the content of answer1 includes in answer2 and all the content of answer2 directly appears in context
    Once any one of the above criteria is satisfied, they are considered to be consistent and you should output Yes,
                            otherwise output No. you should make your reference based on the context. Output your judgement and explain why you give this score.
                            The explanation should be specific enough to include the content of both answers
                            the structure should be '[Yes/No] \\n [explain]'.
                '''

    elif mode == 4:
        #  直接判断
        prompt = f'''Based on the context {context} judge the consistency between the expression {gold_answer} and expression {sut_answer}.
                            The judge criteria of agreement  includes semantic similarity and whether the two answers are exactly the same and whether the content of both answers directly appears in context.If the answers satisfy these criterias, you should output Yes,
                            otherwise output No. you should make your reference based on the context. Output your judgement and explain why you give this score the structure should be '[Yes/No] \\n [explain]'.
                '''
    raw_answer = connect_gpt_prompt(prompt, client, thread_id)
    try:
        result = raw_answer.split('\n')[0].split('[')[1].split(']')[0]
        explain = raw_answer.split('\n')[1]
        if mode == 1 or mode == 2:
            if int(result) >= 90:
                return 1, result, explain
            else:
                return 2, result, explain
        else:
            if result == 'Yes':
                return 1, result, explain
            else:
                return 2, result, explain
    except:
        return 3, None, None
def use_unifiedqa_inf(input_file,information_file,output_file):
    w_data = []
    raw = pd.read_csv(information_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    start_time = time.time()
    dataset = input_file.split('_')[1]
    with open(input_file, "r", encoding='utf-8') as f:
        f = f.readlines()
        i = 0
        for line in tqdm(f):
            try:
                question, answer = line.split("\t")
                if answer[-1] == '\n':
                    answer = answer[:-1]
                sut_answer = run_model(question)[0]
                sent = raw[i][0]
                raw_q = raw[i][5]
                context = raw[i][1]
                w_data.append([sent, context, raw_q, answer, sut_answer])
                i = i + 1
            except:
                i = i+1
                continue
        data = pd.DataFrame(w_data)
        data.to_csv(output_file,sep=',', header=None, index=None)
        end_time = time.time()
        during_time = end_time - start_time
        with open('experiment/fact_data/time.txt', 'a') as p:
            p.write(
                f'unified_time,dataset:{dataset},info_type:relation_info,length:{len(f)},time:{datetime.now()},file_name:{input_file}'
                f'during_time:{during_time}' + '\n')


def use_unifiedqa_rel(input_file,information_file,output_file):
    w_data = []
    raw = pd.read_csv(information_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    start_time = time.time()
    dataset = input_file.split('_')[1]
    with open(input_file, "r", encoding='utf-8') as f:
        f = f.readlines()
        i = 0
        for line in tqdm(f):
            try:
                question, answer = line.split("\t")
                if answer[-1] == '\n':
                    answer = answer[:-1]
                sut_answer = run_model(question)[0]
                sent = raw[i][0]
                raw_q = raw[i][4]
                context = raw[i][1]
                w_data.append([sent, context, raw_q, answer, sut_answer])
                i = i + 1
            except:
                i = i+1
                continue
        data = pd.DataFrame(w_data)
        data.to_csv(output_file,sep=',', header=None, index=None)
        end_time = time.time()
        during_time = end_time - start_time
        with open('experiment/fact_data/time.txt', 'a') as p:
            p.write(
                f'unified_time,dataset:{dataset},info_type:relation_info,length:{len(f)},time:{datetime.now()},file_name:{input_file}'
                f'during_time:{during_time}' + '\n')

def qa_to_sut_inf(in_file,out_file=None,rand_num=None):
    if rand_num is not None:
        raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
        raw = random.sample(raw.values.tolist(),rand_num)
        a = []
        for r in raw:
            b = []
            # for i in range(len(r)):
            #     r[i].replace('\n','')
            # r[3]=r[3][:-1]
            b.append(r[5] + '\\n' + r[1])
            # if b[0].remove():
            #     b[0] = b[0][:-1]
            b.append(r[2])
            a.append(b)
        data = open(out_file, 'w', encoding='utf-8', newline=None)
        for c in a:
            print(c[0][:-2] + "\t" + c[1], file=data)
    raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None,header=None)
    raw = raw.values.tolist()
    a = []
    for r in raw:
        b = []
        # for i in range(len(r)):
        #     r[i].replace('\n','')
        # r[3]=r[3][:-1]
        b.append(r[5] + '\\n' + r[1])
        # if b[0].remove():
        #     b[0] = b[0][:-1]
        b.append(r[2])
        a.append(b)
    data = open(out_file, 'w', encoding='utf-8',newline=None)
    for c in a:
        print(c[0][:-2] + "\t" + c[1], file=data)


def qa_to_sut_rel(in_file,out_file=None,rand_num=None):
    if rand_num is not None:
        raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
        raw = random.sample(raw.values.tolist(),rand_num)
        a = []
        for r in raw:
            b = []
            # for i in range(len(r)):
            #     r[i].replace('\n','')
            # r[3]=r[3][:-1]
            b.append(r[4] + '\\n' + r[1])
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
    else:
        raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
        raw = raw.values.tolist()
        a = []
        for r in raw:
            b = []
            # for i in range(len(r)):
            #     r[i].replace('\n','')
            # r[3]=r[3][:-1]
            b.append(r[4] + '\\n' + r[1])
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

def compare_single(info_list,mode, thread_id):
    context = info_list[1]
    question = info_list[2]
    gold_answer = info_list[3]
    sut_answer = info_list[4]
    sim = sim_model.similarity(sut_answer, gold_answer)
    if sim > 0.8:
        # 语义相似度极高的认为答案一致，否则用gpt协助判断
        info_list.extend([1,None,None])
    else:
        judge, result, explain = compare_through_llm(context,question,gold_answer,sut_answer,mode,thread_id)
        info_list.extend([judge, result, explain])
    return info_list
def merge_all_csv_for_con(w_file,type,file_tail):
    csv_files = [
                f'experiment/sut_output/boolq_all_output_{file_tail}.csv',
                 f'experiment/sut_output/nat_all_output_{file_tail}.csv',
                 f'experiment/sut_output/squad2_all_output_{file_tail}.csv',
                 f'experiment/sut_output/rel_squad2_all_output_{file_tail}.csv',
                 f'experiment/sut_output/rel_nat_all_output_{file_tail}.csv',
                 f'experiment/sut_output/rel_boolq_all_output_{file_tail}.csv'
                 ]
    csv_files = csv_files[0:3] if type =='inf' else csv_files[3:]
    df_list = [pd.read_csv(file, sep=',', encoding='ISO-8859-1', index_col=None, header=None) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.sample(len(combined_df))
    combined_df.to_csv(w_file, sep=',', header=None, index=None)
def judge_consistency(r_file, w_file, mode):
    start_time = time.time()
    # fee1 = get_account_balance(api_key)
    raw = pd.read_csv(r_file, sep=',', encoding='ISO-8859-1')
    raw_list = raw.values.tolist()
    raw_list = random.sample(raw_list, len(raw_list))
    max_threads = 5
    data_type = r_file.split('/')[-1].split('_')[0]
    results = []
    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(compare_single, prompt_info, mode, i) for i, prompt_info in enumerate(raw_list)]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in as_completed(futures):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None:
                    results.append(result)
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")
    data = pd.DataFrame(results)
    data.to_csv(w_file, mode='w', index=False, header=False)
    end_time = time.time()
    during_time = end_time - start_time
    # fee2 = get_account_balance(api_key)
    # fee = fee2 - fee1
    # with open('experiment/fact_data/fee.txt', 'a') as p:
    #     p.write(
    #         f'info_type:consistency,type:{data_type},length:{len(raw_list)},time:{datetime.now()},'
    #         f'fee:{fee}' + '\n')
    with open('experiment/fact_data/time.txt', 'a') as p:
        p.write(
            f'info_type:consistency,type:{data_type},length:{len(raw_list)},time:{datetime.now()},'
            f'during_time:{during_time}' + '\\n')
nlp = stanza.Pipeline(lang='en')
# relation_extraction(nlp)
sim_model = SimCSE("simcse")
# func_check_from_ask_re('rel_part/raw_rel_question_s_ten_lt_new_nc4.csv',sim_model,'rel_part/sent_rel_question_s_ten_lt_new_nc4.csv')
# rel_main('squad2_train_context.txt')
if __name__ == '__main__':
    file_tail = 'real'
    rel_main('Dataset/squad2_dev_context.txt', 0, 1203,file_tail,'1000_all')
    rel_main('Dataset/boolq_dev_context.txt', 0, 2938,file_tail,'2000_all')
    rel_main('Dataset/nat_dev_context.txt', 0, 114,file_tail,'70_all')
    # # # # # #
    # # # # # #
    qa_to_sut_rel(f'rel_part/rel_squad2_q_{file_tail}.csv', f'experiment/finetune/finetune_data/squad2/rel_squad2_finetune_{file_tail}.tsv',500)
    qa_to_sut_rel(f'rel_part/rel_nat_q_{file_tail}.csv', f'experiment/finetune/finetune_data/narrativeqa/rel_narrativeqa_finetune_{file_tail}.tsv',500)
    qa_to_sut_rel(f'rel_part/rel_boolq_q_{file_tail}.csv', f'experiment/finetune/finetune_data/boolq/rel_boolq_finetune_{file_tail}.tsv',100)
    # # #
    use_unifiedqa_rel(f'experiment/sut_test/rel_squad2_all_{file_tail}.tsv', f'rel_part/rel_squad2_q_{file_tail}.csv',
                      f'experiment/sut_output/rel_squad2_all_output_{file_tail}.csv')
    use_unifiedqa_rel(f'experiment/sut_test/rel_nat_all_{file_tail}.tsv', f'rel_part/rel_nat_q_{file_tail}.csv',
                      f'experiment/sut_output/rel_nat_all_output_{file_tail}.csv')
    use_unifiedqa_rel(f'experiment/sut_test/rel_boolq_all_{file_tail}.tsv', f'rel_part/rel_boolq_q_{file_tail}.csv',
                      f'experiment/sut_output/rel_boolq_all_output_{file_tail}.csv')
    # qa_to_sut(f'rel_part/sent_rel_question_s.csv', f'experiment/sut_test/rel_sut.tsv')
    # qa_to_sut_inf(f'experiment/fact_data/squad2_all_1.csv', f'experiment/sut_test/squad2_all_1.tsv')
    # qa_to_sut_inf(f'experiment/fact_data/nat_all_1.csv', f'experiment/sut_test/nat_all_1.tsv')
    # qa_to_sut_inf(f'experiment/fact_data/boolq_all_1.csv', f'experiment/sut_test/boolq_all_1.tsv')
    # use_unifiedqa_inf(f'experiment/sut_test/squad2_all_1.tsv',f'experiment/fact_data/squad2_all_1.csv','experiment/sut_output/squad2_all_output_1.csv')
    # use_unifiedqa_inf(f'experiment/sut_test/nat_all_1.tsv', f'experiment/fact_data/nat_all_1.csv',
    #                   'experiment/sut_output/nat_all_output_1.csv')
    # use_unifiedqa_inf(f'experiment/sut_test/boolq_all_1.tsv', f'experiment/fact_data/boolq_all_1.csv',
    #                   'experiment/sut_output/boolq_all_output_1.csv')
    # merge_all_csv_for_con(f'experiment/sut_output/rel_all_output_{file_tail}.csv','rel',file_tail)
    # judge_consistency(f'experiment/sut_output/rel_all_output_{file_tail}.csv', f'experiment/final_result/all_final_rel_{file_tail}.csv', 1)
    judge_consistency(f'experiment/sut_output/rel_boolq_all_output_{file_tail}.csv',
                      f'experiment/final_result/boolq_final_rel_{file_tail}.csv', 1)
    # judge_consistency(f'experiment/sut_output/squad2_all_output_{file_tail}.csv', f'experiment/final_result/squad2_final_inf_{file_tail}.csv', 1)

    # judge_consistency(f'experiment/sut_output/rel_nat_all_output_{file_tail}.csv',
    #                   f'experiment/final_result/nat_final_rel_{file_tail}.csv', 1)
    # judge_consistency(f'experiment/sut_output/rel_squad2_all_output_{file_tail}.csv',
    #                   f'experiment/final_result/squad2_final_rel_{file_tail}.csv', 1)
