import json
import time
import operator
import multiprocessing
import copy
from datetime import datetime
from functools import partial
import csv
import re
import pandas as pd
from openai import OpenAI
from multiprocessing import Pool
import argparse
import numpy as np
from tqdm import tqdm
from relation_extraction import use_unifiedqa_inf, qa_to_sut_inf, judge_consistency, merge_all_csv_for_con

from model_access import *
import random
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
# import spacy
import openai
# proxies = {'http': 'http://ip:port', 'https': 'http://ip:port'}
# stanza.download(lang='en', resources_url='stanford')
import requests
from collections import Counter

#从nltk库中导入需要的类，并且进行实例化
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from simcse import SimCSE
from sentence_transformers import util
from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('sentence-transformers')
import opennre
# opennre_model = opennre.get_model('wiki80_bert_softmax')
OPENAI_KEY = 'your_key'
client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )
nlp = stanza.Pipeline(lang='en')
NOUN_list = ['NOUN','PROPN']
punct = ['.','?','!']
a = 1
# word = 'bright'
# api_url = 'https://api.api-ninjas.com/v1/thesaurus?word={}'.format(word)
# response = requests.get(api_url, headers={'X-Api-Key': 'YOUR_API_KEY'})
# if response.status_code == requests.codes.ok:
#     print(response.text)
# else:
#     print("Error:", response.status_code, response.text)
# print(*[f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head-1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}' for sent in doc.sentences for word in sent.words], sep='\n')

def acquire_dependence_tree(sentence,nlp):

    doc = nlp(sentence)
    print(*[
        f'id: {word.id}\tfeats: {word.feats if word.feats else "_"}\tword: {word.text}\tlemma: {word.lemma}\tupos: {word.upos}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
        for sent in doc.sentences for word in sent.words], sep='\n')
    return [[word.id, word.feats, word.text, word.lemma, word.upos, word.head,sent.words[word.head - 1].text if word.head > 0 else "root",word.deprel] for sent in doc.sentences for word in sent.words]

def connect_gpt(prompt):
    OPENAI_KEY = 'your_key'
    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )


    response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
        temperature=0,
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    answer = response.choices[0].message.content.strip()
    return answer

# def draw_dependence_tree(sentence):
#     nlp = spacy.load('en_core_web_md')
#     doc = nlp(sentence)
#     displacy.serve(doc, style='dep')

def search_tree(child_str,type):
    np_list = []
    i = 0
    while i<len(child_str)-1:
        if child_str[i:i+2]==type:
            np = []
            q = LifoQueue()
            q.put('(')
            j = i+3
            while j<len(child_str):
                child_str = child_str+' '
                if child_str[j]=='(':
                    q.put(child_str[j])
                elif child_str[j]==')':
                    q.get()
                elif child_str[j-len(type)+1:j+1] == type:
                    q = LifoQueue()
                    q.put('(')
                    j = j+2
                    np = []
                    continue
                else:
                    np.append(child_str[j])
                if q.empty():
                    break
                j+=1
            np_list.append(np)
            i = j
        i = i+1
    return np_list

def output_phrase(nlp, sent, type):
    doc = nlp(sent)
    # a = doc.sentences[0].constituency
    Tree.fromstring(str(doc.sentences[0].constituency)).draw()
    for sentence in doc.sentences:
        str_c = str(sentence.constituency)
        phrase_list = search_tree(str_c,type)
        phrase_list = [''.join(m) for m in phrase_list]
        result = []
        for raw in phrase_list:
            raw_list = raw.split(' ')
            phrase = []
            for r in raw_list:
                if r.upper()!=r:
                    phrase.append(r.lower())
                elif len(r)==1:
                    phrase.append(r.lower())
                else:
                    continue
            if len(phrase)>=2:
                result.append(phrase)
        return result
    f = 1

def extract_phrase(tree_str, label):
    phrases = []
    trees = Tree.fromstring(tree_str)
    # trees.draw()
    for tree in trees:
        for subtree in tree.subtrees():
            if subtree.label() == label:
                t = subtree
                t = ' '.join(t.leaves())
                phrases.append(t)

    return phrases

def compare_phrase_list(origin,new):
    origin_phrase_list = [output_phrase(nlp, origin, type) for type in ['PP', 'NP']]
    # [[['of', 'freshly', 'brewed', 'coffee'], ['in', 'the', 'morning']],[['The', 'smell'], ['freshly', 'brewed', 'coffee'], ['the', 'morning'], ['the', 'best', 'way'], ['the', 'day']]]
    origin_phrase_list = [[phrase for phrase in i if len(i) > 1] for i in origin_phrase_list]
    new_phrase_list = [output_phrase(nlp, new, type) for type in ['PP', 'NP']]
    new_phrase_list = [[phrase for phrase in i if len(i) > 1] for i in new_phrase_list]
    for i in range(len(origin_phrase_list)):
        for p_phrase in origin_phrase_list[i]:
            if p_phrase not in new_phrase_list[i]:
                return False
    return True

# all_list---[[q,a],...]
def choose_sim_answer(model,all_list,sel_a,ch_num,method_type):
    # all_list 为所有q，a对
    if method_type == 1:
        results = model.search(sel_a,top_k=ch_num)
    elif method_type == 2:
        results = model.search(sel_a, top_k=5)
        if len(results)>ch_num:
            results = random.sample(results,ch_num)
    if len(results)==0:
        return None
    re_list = []
    a_list = [r[1] for r in all_list]
    for r in results:
        q = r[0]
        index = a_list.index(q)
        re_list.append(all_list[index])

    # re_list = [[all_list[all_list[1].index(r[0])][0],r[0]] for r in results]
    return re_list
    # similarities = list(model.similarity(sel_a, a_list))
    # return select_method(similarities,rand_list,ch_num,method_type)


def change_through_llm(sent,type,verb=None):
    new_sent = ''
    prompt = ''
    if type=='SVC':
        prompt = f'For sentence "{sent}", swap the place of subject and the whole subject complement of it. If it has pronoun，change the pronoun to its right case, remembering that the output contains only the modified sentence and no other information.'
    elif type == 'SVO':
        prompt = f'For the sentence "{sent}", only change the voice of the sentence where the verb "{verb}" is located to the passive voice with "by", and the other verbs will not change. Remember that the output only contains the modified sentence, not the other information.'
    elif type == 'passive':
        prompt = f'For the sentence "{sent}", only change the passive voice of the part where the verb "{verb}" and "by" is located to the active voice in the right grammar, and the other parts should keep the same. Remember that the output only contains the modified sentence, not the other information.'
    elif type == 'OBL':
        prompt = f'Please modify the form of the indirect object in sentence "{sent}" and keep the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    elif type == 'VO':
        prompt = f'Please change the position of the vocative in sentence "{sent}", and keep the correctness of the sentence grammar and the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    elif type == 'ADVCL':
        prompt = f'Please change the position of the adverbial and the adverbial clause in sentence "{sent}", and keep the correctness of the sentence grammar and the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    elif type == 'NMOD':
        prompt = f'Please change the form of the possessive case in sentence "{sent}", and keep the correctness of the sentence grammar and the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    elif type == 'CONJ_N':
        prompt = f'For the parallel NOUNs and PROPNs associated with "and" or "or" in sentence "{sent}" , only exchange the positions of the nouns and propns.Remember that the other part of the sentence should be in the former place and the output result only contains the modified sentence without any prompts.'
    elif type == 'CONJ_ADJ':
        prompt = f'For the parallel adjectives associated with "and" or "or" in sentence "{sent}" , only exchange the positions of the adjectives.Remember that the other part of the sentence should be in the former place and the output result only contains the modified sentence without any prompts.'
        # prompt = f'Please exchange the positions between the coordinating adjectives in sentence "{sent}" and keep the correctness of the sentence grammar and the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    elif type == 'CONJ_ADV':
        prompt = f'For the parallel adverbs associated with "and" or "or" in sentence "{sent}" , only exchange the positions of the adverbs.Remember that the other part of the sentence should be in the former place and the output result only contains the modified sentence without any prompts.'
        # prompt = f'Please exchange the positions between the coordinating adverbs in sentence "{sent}" and keep the correctness of the sentence grammar and the meaning of the sentence unchanged. Remember that the output result only contains the modified sentence.'
    answer_dict = dict()
    answer = []
    for i in range(5):
        try:
            answer = func_timeout(30, connect_gpt, args=(prompt,))
        except:
            try:
                answer = func_timeout(30, connect_gpt, args=(prompt,))
            except:
                answer = func_timeout(30, connect_gpt, args=(prompt,))
        # answer = connect_gpt(prompt)
        if answer[0] == '"' and answer[-1] == '"':
            answer = answer[1:-1]
        if answer_dict.get(answer):
            answer_dict[answer] = answer_dict[answer]+1
        else:
            answer_dict[answer] = 1
    max_key = max(answer_dict.items(), key=operator.itemgetter(1))[0]
    if answer_dict[max_key] == 1:
        return [sent, '']
    return [sent, max_key]

def ask_through_llm(prompt=None):
    try:
        answer = func_timeout(30, connect_gpt, args=(prompt,))
    except:
        try:
            answer = func_timeout(30, connect_gpt, args=(prompt,))
        except:
            answer = func_timeout(30, connect_gpt, args=(prompt,))
    return answer

def generate_question_llm(context=None,g_answer=None,sent=None, prompt_type=None):
    re_list = []
    prompt = ''
    if prompt_type == 0:
        prompt = f'''Design at least five questions for one sentence,you should make sure the answer of questions is the word/phrase in given sentence and then you should explain why you generate this question,
             Here are two examples:
             sentence : The Witch took a coffin and 
             
             threw it with contempt into a ditch
             word : contempt 
             (1) Question: What does the witch threw a coffin into a ditch with?
             
             (2) Question: What is the witch's attitude when she threw a coffin into a ditch?
             
             
             sentence :  He wonders why Dietrichson did not file a claim for his broken leg, and deduces he did not know about the policy
             word : break 
             (1) Question: What is the state of Dietrichson's leg?
             
             (2) Question: Based on the policy ,in which condition of his leg can Dietrichson be helped?
             
             (3) Question: According to Dietrichson, what happens to his leg?
             
             sentence : He reads a comic strip where Buck Rogers, stranded, calls for help by building a makeshift communication device and is inspired to try it himself
             word : help
             (1) Question: What does Buck Rogers call for by building a makeshift communication device and is inspired to try it himself ? 
             
             (2) Question: When soldiers turned their swords into ploughshares on red anvils, what would be stopped ?' 
             
             remember that your question should be exact enough to avoid the ambiguity according to the context '{context}'.
             .The output structure should be :"(1) Question:[]\\n   Reason:[] (2) ... (3) ... (4) ... (5) ...
             sentence:{sent}
             word:{g_answer}
             '''
    elif prompt_type == 1:
        prompt = f'''Design at least five questions for one word/phrase and its sentence in the context: '{context}'. You should make sure the answer of questions is the word/phrase in given sentence and then you should explain why you generate this question.
                 Remember that your question should be exact enough to avoid the ambiguity according to the context.That is, the question should be specific enough that the answer to the question is unique.
                 In addition, in order to increase the complexity of the question, you should try your best to add some additional information to the generated question according to the context above,the length of question should be longer than 25, but you need to ensure that the corresponding answer to the question should be the word or phrase I provide and you also need to make sure that the generated questions are natural, i.e. more human-like questions .
                 .The output structure should be :"(1) Question:[]\\n   Reason:[] (2) ... (3) ... (4) ... (5) ...
                 sentence:{sent}
                 word:{g_answer}
                 '''
    return prompt


def judge_sent(sent, d_nlp, writer):

    s_inf = acquire_dependence_tree(sent, d_nlp)
    pos_list = [w_inf[-4] for w_inf in s_inf]
    text_list = [w_inf[-6] for w_inf in s_inf]
    deprel_list = [w_inf[-1] for w_inf in s_inf]
    head_list = [w_inf[-2] for w_inf in s_inf]
    hid_list = [w_inf[-3] for w_inf in s_inf]
    result_list = []
    flag = 1
    sent_type = ''
    # if 'nsubj:pass' in deprel_list and ('by' in text_list or 'By' in text_list):
    #     dep_arr = np.array(deprel_list)
    #     obj_index_list = np.where(dep_arr == 'nsubj:pass')[0]
    #     for i in obj_index_list:
    #         #     sent_dep = []
    #         #     for j in range(i+1):
    #         #         if head_list[j] == head_list[i] and (deprel_list[j] in ['root', 'ccomp', 'advcl'] or deprel_list[j] == 'nsubj'):
    #         #             sent_dep.append(deprel_list[j])
    #         nsubj_plc = 0
    #         sent_dep = [deprel_list[j] for j in range(len(deprel_list)) if
    #                     head_list[j] == head_list[i] and deprel_list[j]!='root' and 'advcl' not in deprel_list[j]]
    #         sent_text = [text_list[j] for j in range(len(text_list)) if
    #                      (head_list[j] == head_list[i] and deprel_list[j]!='root' and 'advcl' not in deprel_list[j])
    #                      or (deprel_list[j]=='case' and deprel_list[hid_list[j]-1] in ['obl','obl:agent'])]
    #         if 'by' not in sent_text:
    #             continue
    #         sent_dep.append(deprel_list[hid_list[i] - 1])
    #         if ('obl' in sent_dep or 'obl:agent' in sent_dep) and 'aux:pass' in sent_dep:
    #             for k in range(len(sent_dep)):
    #                 if sent_dep[k] == 'root' or 'advcl' in sent_dep[k]:
    #                     sent_type = 'passive'
    #                     flag = 0
    #                     verb = text_list[hid_list[i] - 1]
    #                     result = change_through_llm(sent, sent_type, verb)
    #                     new_re = [result[0], sent_type, result[1]]
    #                     if result[1] != None:
    #                         s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                         if check_answer(result, sent_type, s_inf, s_inf_new, verb=verb):
    #                             new_re.append(result[1])
    #                             result_list.append(new_re)
    #                         else:
    #                             new_re.append('')
    #                             result_list.append(new_re)
    #                     else:
    #                         new_re.append('')
    #                         result_list.append(new_re)
    # if 'obj' in deprel_list and deprel_list[hid_list[deprel_list.index('obj')]-1] in ['root','ccomp','advcl']:
    #     dep_arr = np.array(deprel_list)
    #     obj_index_list = np.where(dep_arr == 'obj')[0]
    #     for i in obj_index_list:
    #     #     sent_dep = []
    #     #     for j in range(i+1):
    #     #         if head_list[j] == head_list[i] and (deprel_list[j] in ['root', 'ccomp', 'advcl'] or deprel_list[j] == 'nsubj'):
    #     #             sent_dep.append(deprel_list[j])
    #         nsubj_plc = 0
    #         sent_dep = [deprel_list[j] for j in range(len(deprel_list)) if head_list[j]==head_list[i] and deprel_list[j] == 'nsubj' ]
    #
    #         sent_dep.append(deprel_list[hid_list[i]-1])
    #         for k in range(len(sent_dep)):
    #             if sent_dep[k] in ['root','ccomp','advcl']:
    #                 if 'nsubj' in sent_dep:
    #                     sent_type = 'SVO'
    #                     flag = 0
    #                     verb = text_list[hid_list[i]-1]
    #                     # result = change_through_llm(sent, sent_type, verb)
    #                     # s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                     # if check_answer(result, sent_type, s_inf, s_inf_new, verb=verb):
    #                     #     result_list.append(result)
    #                     result = change_through_llm(sent, sent_type, verb)
    #                     new_re = [result[0],sent_type,result[1]]
    #                     if result[1] != None:
    #                         s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                         if check_answer(result, sent_type, s_inf, s_inf_new, verb=verb):
    #                             new_re.append(result[1])
    #                             result_list.append(new_re)
    #                         else:
    #                             new_re.append('')
    #                             result_list.append(new_re)
    #                     else:
    #                         new_re.append('')
    #                         result_list.append(new_re)
    #
    #
    # 专有名词或者有the 修饰的名词
    # if 'cop' in deprel_list:
    #     if pos_list[hid_list[deprel_list.index('cop')]-1]=='PROPN' :
    #         sent_type = 'SVC'
    #         flag = 0
    #         result = change_through_llm(sent, sent_type)
    #         new_re = [result[0], sent_type, result[1]]
    #         s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #         if check_answer(result, sent_type, s_inf, s_inf_new):
    #             new_re.append(result[1])
    #             result_list.append(new_re)
    #         else:
    #             new_re.append('')
    #             result_list.append(new_re)
    #     elif pos_list[hid_list[deprel_list.index('cop')]-1]=='PRON' :
    #         sent_type = 'SVC'
    #         result = change_through_llm(sent, sent_type)
    #         new_re = [result[0], sent_type, result[1]]
    #         flag = 0
    #         s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #         if check_answer(result, sent_type, s_inf, s_inf_new):
    #             new_re.append(result[1])
    #             result_list.append(new_re)
    #         else:
    #             new_re.append('')
    #             result_list.append(new_re)
    #     elif pos_list[hid_list[deprel_list.index('cop')]-1]=='NOUN' :
    #         pp_phrase_list = output_phrase(d_nlp,sent,'PP')
    #         for phrase in pp_phrase_list:
    #             if text_list[hid_list[deprel_list.index('cop')] - 1] in phrase:
    #                 break
    #         for i in range(len(deprel_list)):
    #             if deprel_list[i]=='det' and head_list[hid_list[i]-1]=='root' and text_list[i]=='the':
    #                 sent_type = 'SVC'
    #                 flag = 0
    #                 result = change_through_llm(sent, sent_type)
    #                 new_re = [result[0], sent_type, result[1]]
    #                 s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                 if check_answer(result, sent_type, s_inf, s_inf_new):
    #                     new_re.append(result[1])
    #                     result_list.append(new_re)
    #                 else:
    #                     new_re.append('')
    #                     result_list.append(new_re)
    #             if deprel_list[i] == 'nmod:poss' and head_list[hid_list[i]-1]=='root':
    #                 sent_type = 'SVC'
    #                 flag = 0
    #                 result = change_through_llm(sent, sent_type)
    #                 new_re = [result[0], sent_type, result[1]]
    #                 s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                 if check_answer(result, sent_type, s_inf, s_inf_new):
    #                     new_re.append(result[1])
    #                     result_list.append(new_re)
    #                 else:
    #                     new_re.append('')
    #                     result_list.append(new_re)
    #             if deprel_list[i] == 'compound' and head_list[hid_list[i] - 1] == 'root' :
    #                 sent_type = 'SVC'
    #                 flag = 0
    #                 result = change_through_llm(sent, sent_type)
    #                 new_re = [result[0], sent_type, result[1]]
    #                 s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #                 if check_answer(result, sent_type, s_inf, s_inf_new):
    #                     new_re.append(result[1])
    #                     result_list.append(new_re)
    #                 else:
    #                     new_re.append('')
    #                     result_list.append(new_re)
    # if 'iobj' in deprel_list :
    #     # 间接宾语
    #     sent_type = 'OBL'
    #     flag = 0
    #     result = change_through_llm(sent, sent_type)
    #     new_re = [result[0], sent_type, result[1]]
    #     s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #     if check_answer(result, sent_type, s_inf, s_inf_new):
    #         new_re.append(result[1])
    #     else:
    #         new_re.append('')
    #         result_list.append(new_re)
    # if 'vocative' in deprel_list:
    #     # 呼格词
    #     sent_type = 'VO'
    #     result = change_through_llm(sent, sent_type)
    #     s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #     if check_answer(result, sent_type, d_nlp, s_inf, s_inf_new):
    #         result_list.append(result)
    # if 'advcl' in deprel_list:
    #     # 状语短语以及从句
    #     sent_type = 'ADVCL'
    #     flag = 0
    #     result = change_through_llm(sent, sent_type)
    #     new_re = [result[0], sent_type, result[1]]
    #     s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #     if check_answer(result, sent_type, s_inf, s_inf_new):
    #         new_re.append(result[1])
    #         result_list.append(new_re)
    #     else:
    #         new_re.append('')
    #         result_list.append(new_re)
    # if 'nmod:poss' in deprel_list and pos_list[deprel_list.index('nmod:poss')]=='PROPN':
    #     # 名词修饰语
    #     sent_type = 'NMOD'
    #     flag = 0
    #     result = change_through_llm(sent, sent_type)
    #     new_re = [result[0], sent_type, result[1]]
    #     s_inf_new = acquire_dependence_tree(result[1], d_nlp)
    #     if check_answer(result, sent_type, s_inf, s_inf_new):
    #         new_re.append(result[1])
    #         result_list.append(new_re)
    #     else:
    #         new_re.append('')
    #         result_list.append(new_re)
    if 'conj' in deprel_list:
        # 名词修饰语
        if pos_list[deprel_list.index('cc')-1]=='ADJ':
            sent_type = 'CONJ_ADJ'
        elif pos_list[deprel_list.index('cc')-1] in ['NOUN','PROPN']:
            sent_type = 'CONJ_N'
        elif pos_list[deprel_list.index('cc')-1]=='ADV':
            sent_type = 'CONJ_ADV'
        else:
            pass
        if sent_type != '':
            flag = 0
            result = change_through_llm(sent, sent_type)
            new_re = [result[0], sent_type, result[1]]
            s_inf_new = acquire_dependence_tree(result[1], d_nlp)
            if check_answer(result, sent_type, s_inf, s_inf_new):
                new_re.append(result[1])
                result_list.append(new_re)
            else:
                new_re.append('')
                result_list.append(new_re)

        else:
            flag = 1
    if flag == 1:
        result_list.append([sent,sent_type,'',''])
    for r in result_list:
        writer.writerow(r)
    return result_list

def new_gen_phrases(nlp,sent,type):
    # 返回提取的词组，以及词组的首个词（对于动词,非原型）
    def a(l):
        return len(l)
    if type == 'VP':
        w_type = 'v'
    elif type == 'NP':
        w_type = 'n'
    fin_phrases = []
    fin_words = []
    doc = nlp(sent)
    if type == 'VP':
        for i in range(len(doc.sentences)):
            phrase_verb = []
            verb_phrases = extract_phrase(str(doc.sentences[i].constituency), type)
            for ph in verb_phrases:
                word = ph.split(' ')[0]
                if len(ph.split(' '))>=2:
                    phrase_verb.append(word)
            # phrase_verb = [wnl.lemmatize(ph.split('')[0], w_type) for ph in verb_phrases]
            SBAR_phrases = extract_phrase(str(doc.sentences[i].constituency), 'SBAR') \
                           + extract_phrase(str(doc.sentences[i].constituency), 'SBARQ') + extract_phrase(
                str(doc.sentences[i].constituency), 'SQ')
            SBAR_phrases.sort(key=a, reverse=True)
            for i in range(len(verb_phrases)):
                for j in range(len(SBAR_phrases)):
                    if SBAR_phrases[j] in verb_phrases[i] and SBAR_phrases[j] != verb_phrases[i]:
                        verb_phrases[i] = verb_phrases[i].replace(SBAR_phrases[j], '')
            verb_phrases = list(set(verb_phrases))
            verb_phrases.sort(key=a, reverse=True)
            for i in range(len(verb_phrases) - 1):
                flag = 1
                for j in range(i + 1, len(verb_phrases)):
                    if verb_phrases[j] in verb_phrases[i]:
                        flag = 0
                        break
                verb_words = [word for word in verb_phrases[i].split(' ') if len(word)>0]
                if flag == 1 and (len(verb_words) > 1):
                    fin_phrases.append(verb_phrases[i])

            if len(verb_phrases) > 0:
                verb_last_words = [word for word in verb_phrases[-1].split(' ') if len(word) > 0]
                if len(verb_last_words)>1:
                    fin_phrases.append(verb_phrases[-1])
            fin_words.extend(phrase_verb)
    elif type == 'NP':
        for i in range(len(doc.sentences)):
            phrase_noun = []
            noun_phrases = extract_phrase(str(doc.sentences[i].constituency), type)
            # for ph in verb_phrases:
            #     word = ph.split(' ')[0]
            #     phrase_verb.append(word)
            # phrase_verb = [wnl.lemmatize(ph.split('')[0], w_type) for ph in verb_phrases]
            SBAR_phrases = extract_phrase(str(doc.sentences[i].constituency), 'SBAR') \
                           + extract_phrase(str(doc.sentences[i].constituency), 'SBARQ') + extract_phrase(
                str(doc.sentences[i].constituency), 'SQ')
            SBAR_phrases.sort(key=a, reverse=True)
            for i in range(len(noun_phrases)):
                for j in range(len(SBAR_phrases)):
                    if SBAR_phrases[j] in noun_phrases[i] and SBAR_phrases[j] != noun_phrases[i]:
                        noun_phrases[i] = noun_phrases[i].replace(SBAR_phrases[j], '')
            noun_phrases = list(set(noun_phrases))
            noun_phrases.sort(key=a, reverse=True)
            for i in range(len(noun_phrases) - 1):
                if noun_phrases[i] == ' ':
                    continue
                flag = 1
                if len(noun_phrases[i].split(' ')) <= 4:
                    # 如果np长度小于4，则直接选择它，如果里面还包含np则放弃里面的np(' ')(为了保证所有格词组不被分开，如'the brothers' influence')
                    for j in range(i + 1, len(noun_phrases)):
                        if noun_phrases[j] in noun_phrases[i]:
                            noun_phrases[j] = ' '
                else:
                    for j in range(i + 1, len(noun_phrases)):
                        if noun_phrases[j] in noun_phrases[i]:
                            flag = 0
                            break
                noun_words = [word for word in noun_phrases[i].split(' ') if len(word)>0]
                if flag == 1 and (len(noun_words) > 1):
                    fin_phrases.append(noun_phrases[i])

            if len(noun_phrases) > 0 and noun_phrases[-1] != ' ':
                noun_last_words = [word for word in noun_phrases[-1].split(' ') if len(word) > 0]
                if len(noun_last_words)>1:
                    fin_phrases.append(noun_phrases[-1])
            # fin_words.extend(phrase_verb)
    return [fin_phrases, fin_words]



not_word = ['generate','continue','become','remain','deem','let','take','express','conclude','give','consider','demand','resolve',
            'cause','do','turn','discover','realize','start','convince','be','involve','begin','choose','ensue','get','endeavour','assume','insinuate','have']

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

def check_thread_process(prompt_info,thread_id,sim_model,cond,ch_type,nlp,raw_writer, check_type):
    # [sent, context, g_answer, prompt, answer, str_question, str_reason]
    if prompt_info[2] in prompt_info[5]:
        return None
    # if not check_answer(prompt_info[-2],prompt_info[2],cond,ch_type,prompt_info[0],nlp):
    #     return None
    else:
        raw_writer.writerow(prompt_info)
    prompt = f"According to the context '{prompt_info[1]}' and  question '{prompt_info[-2]}', the answer of the question is a word or phrase, " \
             f"please output the answer. Remember that the output should only be the answer without any other prompts."
    if check_type == 'gpt':
        gen_a = connect_gpt_prompt(prompt, client, thread_id)
    elif check_type == 'llama':
        gen_a = connect_llama_prompt(prompt)
    sim = sim_model.similarity(prompt_info[2], gen_a)
    if sim > 0.75:
        r_list = copy.deepcopy(prompt_info)
        r_list.append(gen_a)
        return r_list
    else:
        return None


def check_from_ask_1(new_q_info_list, sim_model, cond, ch_type, nlp, raw_writer):
    max_threads = 10
    results = []
    check_type = 'gpt'
    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(check_thread_process, prompt_info, i, sim_model, cond, ch_type, nlp, raw_writer, check_type) for i, prompt_info in tqdm(enumerate(new_q_info_list))]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in as_completed(futures):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None:
                    results.append(result)
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")

    return results


def judge_verb(s_inf):
    # 判断非主要动词 如try to play中的try
    inf_pos = [s[4] for s in s_inf]
    inf_text = [s[2] for s in s_inf]
    inf_dep = [s[-1] for s in s_inf]
    inf_head = [s[-2] for s in s_inf]
    aim_verb_1 = [s_inf[i][3] for i in range(len(s_inf)-2) if inf_pos[i]=='VERB' and inf_dep[i+1]=='mark' and inf_pos[i+2]=='VERB']
    aim_verb_2 = [s_inf[i][3] for i in range(len(s_inf) - 2) if s_inf[i][1] == 'Tense=-Past|VerbForm=Part' and
                  inf_pos[i] == 'VERB' and inf_dep[i] == 'amod' and inf_pos[inf_text.index(inf_head[i])]=='NOUN']
    aim_verb = list(set(aim_verb_1+aim_verb_2))
    return aim_verb
def search_once_word1(context,nlp,model,ran_list,ch_type=None,sav_pro_file=None):
    # 提取context信息，构建prompt保存在文件中

    phrase_type = ''
    if ch_type == 'NOUN':
        phrase_type = 'NP'
    elif ch_type == 'VERB':
        phrase_type = 'VP'
    # c_inf = acquire_dependence_tree(context, nlp)

    sent_list = context.split('.')
    ask_dataset_1 = []
    ask_dataset_2 = []
    ask_dataset_3 = []
    re_list1 = []
    re_list2 = []
    re_list3 = []
    for sent in sent_list:
        phrase_list = []
        check_word_list = []
        all_phrase_list = new_gen_phrases(nlp, sent, phrase_type)
        for i in all_phrase_list[0]:
            phrase_list.append(i)
        for i in all_phrase_list[1]:
            check_word_list.append(i)
        # try:
        #
        # except:
        #     continue
        s_inf = acquire_dependence_tree(sent, nlp)
        c_lemma_list = [s_inf[i][3] for i in range(len(s_inf))]
        c_text_list = [s_inf[i][2] for i in range(len(s_inf))]
        context = sent.lower()
    # sent_list = context.split('.')
        c_list = sent.split(' ')
        if ch_type == 'NOUN':
            ch_type_list = [s_inf[i][3] for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][-1] in ['root','obl','obj','iobj','nsubj','nsubj:pass']
                            ]
            ch_type_list1 = []
            for ch in ch_type_list:
                flag = 1
                for j in phrase_list:
                    if ch in j:
                        flag = 0
                        break
                if flag == 1:
                    ch_type_list1.append(ch)
            ch_type_list = ch_type_list1
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][-1] in ['root','obl','obj','iobj','nsubj','nsubj:pass']]
        elif ch_type == 'VERB':
            # ch_type_list = [s_inf[i][3] for i in range(len(s_inf)-1) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            ch_type_list = [s_inf[i][3] for i in range(len(s_inf)-1) if
                            s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
                                s_inf) and s_inf[i][2] not in check_word_list]
            ch_type_index = [i for i in range(len(s_inf)-1) if
                            s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
                                s_inf) and s_inf[i][2] not in check_word_list]
            try:
                if s_inf[-1][-4] == ch_type :
                    ch_type_list.append(s_inf[-1][3])
                    # ch_type_index.append(len(s_inf))
            except:
                pass
            # ch_type_list1 = []
            # for i in ch_type_index:
            #     flag = 1
            #     for j in phrase_list:
            #         if s_inf[i][2] in j:
            #             flag = 0
            #             break
            #     if flag == 1:
            #         ch_type_list1.append(s_inf[i][3])
            # ch_type_list = ch_type_list1

        word_dict = Counter(ch_type_list)
        word_1_dict = dict(filter(lambda x: x[1]==1, word_dict.items()))
        ch_words = word_1_dict.keys()
        b = 1
        for word in ch_words:
            ch_sent = sent
            prompt1 = f"Next I will give you a sentence and a word in this sentence, " \
                                                "I need you to generate a question and its answer must be the word ." \
                                                "I will give you two examples: the first one ,for sentence " \
                                                "'The Witch took a coffin and threw it with contempt into a ditch' and word 'contempt', " \
                                                "the generated question can be 'What does the witch threw a coffin into a ditch with?'; " \
                                                "the second one,for sentence " \
                                                "'War would be practised no more as soldiers turned their swords into ploughshares on red anvils' " \
                                                "and word 'war', the generated question can be " \
                                                "'What would be practised no more as soldiers turned their swords into ploughshares on red anvils' " \
                                                f"Like the above example, generate the question for sentence '{ch_sent}' and word '{word}', " \
                                                f"you can just output the question without any prompts."\
                if ch_type == 'NOUN' else f"Next I will give you a sentence and a word in this sentence, " \
                                                "I need you to generate a question and its answer must be the word ." \
                                                "I will give you two examples: the first one ,for sentence " \
                                                "'Max and Kate are met by the police moments after' and word 'met', " \
                                                "the generated question can be 'What are Max and Kate done by the police?'; " \
                                                "the second one,for sentence " \
                                                "'They manage to open the door , but a dangerous security system prevents them from going any farther' " \
                                                "and word 'prevents', the generated question can be " \
                                                "'What does the dangerous security system do to them from going any farther?' " \
                                                f"Like the above example, generate the question for sentence '{ch_sent}' and word '{word}', " \
                                                f"you can just output the question without any prompts."
            ask_dataset_1.append([context, word,ch_sent, prompt1])
            sample_list1 = choose_sim_answer(model, ran_list, word, ch_num=2,method_type=1)
            print(sample_list1)
            try:
            	prompt2 = f"Next I will give you a sentence and a word in this sentence,I need you to generate a question and its answer must be the word .Here are two examples of the answer and the generated question." \
                      f"(1) answer:'{sample_list1[0][1]}',question:'{sample_list1[0][0]}';(2) answer:'{sample_list1[1][1]}',question:'{sample_list1[1][0]}'.Like the above question in the example, generate new question for sentence '{ch_sent}' and word '{word}'," \
                      f"you should just output the question without any prompts or the answer."
            	ask_dataset_2.append([context, word, ch_sent, prompt2])
            except:
            	pass
            sample_list2 = choose_sim_answer(model, ran_list, word, ch_num=2,method_type=2)
            print(sample_list2)
            try:
            	prompt3 = f"Next I will give you a sentence and a word in this sentence,I need you to generate a question and its answer must be the word .Here are two examples of the answer and the generated question." \
                      f"(1) answer:'{sample_list2[0][1]}',question:'{sample_list2[0][0]}';(2) answer:'{sample_list2[1][1]}',question:'{sample_list2[1][0]}'.Like the above question in the example, generate new question for sentence '{ch_sent}' and word '{word}'," \
                      f"you should just output the question without any prompts or the answer."
            	ask_dataset_3.append([context, word, ch_sent, prompt3])
            except:
            	pass
    ad_dataframe_1 = pd.DataFrame(ask_dataset_1)
    ad_dataframe_2 = pd.DataFrame(ask_dataset_2)
    ad_dataframe_3 = pd.DataFrame(ask_dataset_3)
    ad_dataframe_1.to_csv(sav_pro_file + ch_type + '_1n.csv',mode='a',index=False, header=False)
    ad_dataframe_2.to_csv(sav_pro_file + ch_type + '_2n.csv',mode='a', index=False, header=False)
    ad_dataframe_3.to_csv(sav_pro_file + ch_type + '_3n.csv',mode='a', index=False, header=False)
    print(f'本context信息提取完成，共{len(ask_dataset_1)}条')

    # for i in range(len(ask_dataset_1)):
    #     try:
    #         q1 = ask_through_llm(ask_dataset_1[i][0], ask_dataset_1[i][1], ask_dataset_1[i][2], ask_dataset_1[i][3])
    #         q2 = ask_through_llm(ask_dataset_2[i][0], ask_dataset_2[i][1], ask_dataset_2[i][2], ask_dataset_2[i][3])
    #         q3 = ask_through_llm(ask_dataset_3[i][0], ask_dataset_3[i][1], ask_dataset_3[i][2], ask_dataset_3[i][3])
    #         flag = (check_answer(q1, ask_dataset_1[i][1], 'words', ch_type) and check_from_ask(ask_dataset_1[i][2], ask_dataset_1[i][1], q1, 'words', ch_type))
    #         if flag:
    #             re_list1.append([ask_dataset_1[i][2], ask_dataset_1[i][1], q1, ask_dataset_1[i][0]])
    #         flag = (check_answer(q2, ask_dataset_2[i][1], 'words', ch_type) and check_from_ask(ask_dataset_2[i][2], ask_dataset_2[i][1], q2, 'words', ch_type))
    #         if flag:
    #             re_list2.append([ask_dataset_2[i][2], ask_dataset_2[i][1], q1, ask_dataset_2[i][0]])
    #         flag = (check_answer(q3, ask_dataset_3[i][1], 'words', ch_type) and check_from_ask(ask_dataset_3[i][2], ask_dataset_3[i][1], q3, 'words', ch_type))
    #         if flag:
    #             re_list3.append([ask_dataset_3[i][2], ask_dataset_3[i][1], q1, ask_dataset_3[i][0]])
    #     except:
    #         continue
    # for r in re_list1:
    #     writer1.writerow(r)
    # for r in re_list2:
    #     writer2.writerow(r)
    # for r in re_list3:
    #     writer3.writerow(r)
    a = 1

def new_search_word(context,nlp, ch_type=None, prompt_type=None,thread_id=None):
    # 提取context信息，构建prompt保存在文件中

    phrase_type = ''
    if ch_type == 'NOUN':
        phrase_type = 'NP'
    elif ch_type == 'VERB':
        phrase_type = 'VP'
    # c_inf = acquire_dependence_tree(context, nlp)

    sent_list = context.split('.')
    total_info = []
    return_list = []
    for sent in sent_list:
        if len(sent.split(' '))<10:
            continue
        phrase_list = []
        check_word_list = []
        all_phrase_list = new_gen_phrases(nlp, sent, phrase_type)
        for i in all_phrase_list[0]:
            phrase_list.append(i)
        for i in all_phrase_list[1]:
            check_word_list.append(i)
        # try:
        #
        # except:
        #     continue
        s_inf = acquire_dependence_tree(sent, nlp)
        c_lemma_list = [s_inf[i][3] for i in range(len(s_inf))]
        c_text_list = [s_inf[i][2] for i in range(len(s_inf))]
        # sent_list = context.split('.')
        c_list = sent.split(' ')
        ch_type_list = []
        if ch_type == 'NOUN':
            ch_type_list = [s_inf[i][3] for i in range(len(s_inf)) if s_inf[i][-4] == ch_type \
                            # and s_inf[i][-1] in ['root', 'obl', 'obj', 'iobj', 'nsubj',
                            #                                              'nsubj:pass']
                            ]
            ch_type_list1 = []
            for ch in ch_type_list:
                flag = 1
                for j in phrase_list:
                    if ch in j:
                        flag = 0
                        break
                if flag == 1:
                    ch_type_list1.append(ch)
            ch_type_list = ch_type_list1
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][-1] in ['root','obl','obj','iobj','nsubj','nsubj:pass']]
        elif ch_type == 'VERB':
            # ch_type_list = [s_inf[i][3] for i in range(len(s_inf)-1) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            ch_type_list = [s_inf[i][3] for i in range(len(s_inf) - 1) if
                            s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
                                s_inf) and s_inf[i][2] not in check_word_list]
            # ch_type_index = [i for i in range(len(s_inf) - 1) if
            #                  s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
            #                      s_inf) and s_inf[i][2] not in check_word_list]
            try:
                if s_inf[-1][-4] == ch_type:
                    ch_type_list.append(s_inf[-1][3])
                    # ch_type_index.append(len(s_inf))
            except:
                pass
        word_dict = Counter(ch_type_list)
        word_1_dict = dict(filter(lambda x: x[1] == 1, word_dict.items()))
        ch_words = word_1_dict.keys()
        for word in ch_words:
            prompt = generate_question_llm(context, word, sent, prompt_type)
            return_list.append([sent, context, word, prompt])
            # q_info_list = generate_question_llm(context,word,sent)
            # for q_info in q_info_list:
            #     # [sent, context, g_answer, str_question, str_reason]
            #     if q_info[2] in q_info[4] and check_answer(q_info[3], q_info[2], 'words', ch_type, q_info[0], nlp):
            #         raw_writer.writerow(q_info)
            #     if q_info[2] in q_info[4] and check_answer(q_info[3],q_info[2],'words',ch_type,q_info[0],nlp) \
            #         and check_from_ask_1(q_info[1],q_info[2],q_info[3],'words', ch_type, sim_model):
            #         writer.writerow(q_info)
                    # total_info.append(q_info)
    return return_list





def search_once_word1_1(context,nlp,model,ran_list,ch_type=None,sav_pro_file=None):
    # 提取context信息，构建prompt保存在文件中

    phrase_type = ''
    if ch_type == 'NOUN':
        phrase_type = 'NP'
    elif ch_type == 'VERB':
        phrase_type = 'VP'
    # c_inf = acquire_dependence_tree(context, nlp)

    sent_list = context.split('.')
    ask_dataset_1 = []
    ask_dataset_2 = []
    ask_dataset_3 = []
    re_list1 = []
    re_list2 = []
    re_list3 = []
    for sent in sent_list:
        phrase_list = []
        check_word_list = []
        all_phrase_list = new_gen_phrases(nlp, sent, phrase_type)
        for i in all_phrase_list[0]:
            phrase_list.append(i)
        for i in all_phrase_list[1]:
            check_word_list.append(i)
        # try:
        #
        # except:
        #     continue
        s_inf = acquire_dependence_tree(sent, nlp)
        c_lemma_list = [s_inf[i][3] for i in range(len(s_inf))]
        c_text_list = [s_inf[i][2] for i in range(len(s_inf))]
        # context = sent.lower()
    # sent_list = context.split('.')
        c_list = sent.split(' ')
        if ch_type == 'NOUN':
            ch_type_list = [s_inf[i][2] for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][-1] in ['root','obl','obj','iobj','nsubj','nsubj:pass']
                            ]
            ch_type_list1 = []
            for ch in ch_type_list:
                flag = 1
                for j in phrase_list:
                    if ch in j:
                        flag = 0
                        break
                if flag == 1:
                    ch_type_list1.append(ch)
            ch_type_list = ch_type_list1
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][-1] in ['root','obl','obj','iobj','nsubj','nsubj:pass']]
        elif ch_type == 'VERB':
            # ch_type_list = [s_inf[i][3] for i in range(len(s_inf)-1) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            # ch_type_index = [i for i in range(len(s_inf)) if s_inf[i][-4]==ch_type and s_inf[i][3] not in not_word]
            ch_type_list = [s_inf[i][2] for i in range(len(s_inf)-1) if
                            s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
                                s_inf) and s_inf[i][2] not in check_word_list]
            ch_type_index = [i for i in range(len(s_inf)-1) if
                            s_inf[i][-4] == ch_type and s_inf[i][3] not in not_word and s_inf[i][3] not in judge_verb(
                                s_inf) and s_inf[i][2] not in check_word_list]
            try:
                if s_inf[-1][-4] == ch_type :
                    ch_type_list.append(s_inf[-1][2])
                    # ch_type_index.append(len(s_inf))
            except:
                pass
            # ch_type_list1 = []
            # for i in ch_type_index:
            #     flag = 1
            #     for j in phrase_list:
            #         if s_inf[i][2] in j:
            #             flag = 0
            #             break
            #     if flag == 1:
            #         ch_type_list1.append(s_inf[i][2])
            # ch_type_list = ch_type_list1

        word_dict = Counter(ch_type_list)
        word_1_dict = dict(filter(lambda x: x[1]==1, word_dict.items()))
        ch_words = word_1_dict.keys()
#        print(c_text_list)
#        print(ch_type_index)
#        print(ch_type_list)
#        print(ch_words)
    # phrase_list = [' '.join(a) for a in phrase_list]
        b = 1
        for word in ch_words:
            ch_sent = sent
            prompt1 = f"Next I will give you a sentence and a word in this sentence, " \
                                                "I need you to generate a question and its answer must be the word ." \
                                                "I will give you two examples: the first one ,for sentence " \
                                                "'The Witch took a coffin and threw it with contempt into a ditch' and word 'contempt', " \
                                                "the generated question can be 'What does the witch threw a coffin into a ditch with?'; " \
                                                "the second one,for sentence " \
                                                "'War would be practised no more as soldiers turned their swords into ploughshares on red anvils' " \
                                                "and word 'war', the generated question can be " \
                                                "'What would be practised no more as soldiers turned their swords into ploughshares on red anvils' " \
                                                f"Like the above example, generate the question for sentence '{ch_sent}' and word '{word}', " \
                                                f"you can just output the question without any prompts."\
                if ch_type == 'NOUN' else f"Next I will give you a sentence and a word in this sentence, " \
                                                "I need you to generate a question and its answer must be the word ." \
                                                "I will give you two examples: the first one ,for sentence " \
                                                "'Max and Kate are met by the police moments after' and word 'met', " \
                                                "the generated question can be 'What are Max and Kate done by the police?'; " \
                                                "the second one,for sentence " \
                                                "'They manage to open the door , but a dangerous security system prevents them from going any farther' " \
                                                "and word 'prevents', the generated question can be " \
                                                "'What does the dangerous security system do to them from going any farther?' " \
                                                f"Like the above example, generate the question for sentence '{ch_sent}' and word '{word}', " \
                                                f"you can just output the question without any prompts."
            ask_dataset_1.append([context, word,ch_sent, prompt1])

            sample_list1 = choose_sim_answer(model, ran_list, word, ch_num=2,method_type=1)
            print(sample_list1)
            try:
            	prompt2 = f"Next I will give you a sentence and a word in this sentence,I need you to generate a question and its answer must be the word .Here are two examples of the answer and the generated question." \
                      f"(1) answer:'{sample_list1[0][1]}',question:'{sample_list1[0][0]}';(2) answer:'{sample_list1[1][1]}',question:'{sample_list1[1][0]}'.Like the above question in the example, generate new question for sentence '{ch_sent}' and word '{word}'," \
                      f"you should just output the question without any prompts or the answer."
            	ask_dataset_2.append([context, word, ch_sent, prompt2])
            except:
            	pass
            sample_list2 = choose_sim_answer(model, ran_list, word, ch_num=2,method_type=2)
            print(sample_list2)
            try:
            	prompt3 = f"Next I will give you a sentence and a word in this sentence,I need you to generate a question and its answer must be the word .Here are two examples of the answer and the generated question." \
                      f"(1) answer:'{sample_list2[0][1]}',question:'{sample_list2[0][0]}';(2) answer:'{sample_list2[1][1]}',question:'{sample_list2[1][0]}'.Like the above question in the example, generate new question for sentence '{ch_sent}' and word '{word}'," \
                      f"you should just output the question without any prompts or the answer."
            	ask_dataset_3.append([context, word, ch_sent, prompt3])
            except:
            	pass
    ad_dataframe_1 = pd.DataFrame(ask_dataset_1)
    ad_dataframe_2 = pd.DataFrame(ask_dataset_2)
    ad_dataframe_3 = pd.DataFrame(ask_dataset_3)
    ad_dataframe_1.to_csv(sav_pro_file + ch_type + '_1n.csv',mode='a',index=False, header=False)
    ad_dataframe_2.to_csv(sav_pro_file + ch_type + '_2n.csv',mode='a', index=False, header=False)
    ad_dataframe_3.to_csv(sav_pro_file + ch_type + '_3n.csv',mode='a', index=False, header=False)
    print(f'本context信息提取完成，共{len(ask_dataset_1)}条')

    # for i in range(len(ask_dataset_1)):
    #     try:
    #         q1 = ask_through_llm(ask_dataset_1[i][0], ask_dataset_1[i][1], ask_dataset_1[i][2], ask_dataset_1[i][3])
    #         q2 = ask_through_llm(ask_dataset_2[i][0], ask_dataset_2[i][1], ask_dataset_2[i][2], ask_dataset_2[i][3])
    #         q3 = ask_through_llm(ask_dataset_3[i][0], ask_dataset_3[i][1], ask_dataset_3[i][2], ask_dataset_3[i][3])
    #         flag = (check_answer(q1, ask_dataset_1[i][1], 'words', ch_type) and check_from_ask(ask_dataset_1[i][2], ask_dataset_1[i][1], q1, 'words', ch_type))
    #         if flag:
    #             re_list1.append([ask_dataset_1[i][2], ask_dataset_1[i][1], q1, ask_dataset_1[i][0]])
    #         flag = (check_answer(q2, ask_dataset_2[i][1], 'words', ch_type) and check_from_ask(ask_dataset_2[i][2], ask_dataset_2[i][1], q2, 'words', ch_type))
    #         if flag:
    #             re_list2.append([ask_dataset_2[i][2], ask_dataset_2[i][1], q1, ask_dataset_2[i][0]])
    #         flag = (check_answer(q3, ask_dataset_3[i][1], 'words', ch_type) and check_from_ask(ask_dataset_3[i][2], ask_dataset_3[i][1], q3, 'words', ch_type))
    #         if flag:
    #             re_list3.append([ask_dataset_3[i][2], ask_dataset_3[i][1], q1, ask_dataset_3[i][0]])
    #     except:
    #         continue
    # for r in re_list1:
    #     writer1.writerow(r)
    # for r in re_list2:
    #     writer2.writerow(r)
    # for r in re_list3:
    #     writer3.writerow(r)
    a = 1



def new_search_phrase(context,nlp,ch_type=None,prompt_type=None,thread_id=None):
    phrase_type = ''
    if ch_type == 'NOUN':
        phrase_type = 'NP'
    elif ch_type == 'VERB':
        phrase_type = 'VP'

    sent_list = context.split('.')
    sent_len = [len(sent.split(' ')) for sent in sent_list]
    return_list = []
    for sent in sent_list:
        if len(sent.split(' '))<10:
            continue
        s_inf = acquire_dependence_tree(sent, nlp)
        word_list = [s[2] for s in s_inf]
        # context = sent.lower()
        try:
            raw_phrase = new_gen_phrases(nlp, sent, phrase_type)[0]
            raw_phrase = [r.split(' ') for r in raw_phrase]
            phrase_list = [r for r in raw_phrase if len(r) < 8]
        except:
            continue
        phrase_list = [' '.join(a) for a in phrase_list]

        for phrase in phrase_list:
            prompt = generate_question_llm(context, phrase, sent, prompt_type)
            return_list.append([sent, context, phrase, prompt])
    return return_list


def new_search_info(context, cond, ch_type, prompt_type=None):
    if cond == 'phrases':
        return new_search_phrase(context, nlp, ch_type, prompt_type)
    elif cond == 'words':
        return new_search_word(context, nlp, ch_type, prompt_type)



def contexts_to_prompts(random_text_list,cond,type,sim_model,prompt_type=None):
    # all_prompt 返回为[sent,context,answer,prompt]
    processes = []
    # num_workers = 4  # 要创建的进程数量
    #
    # # 创建多个进程
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     process_with_args = partial(new_search_info, cond=cond, ch_type=type)
    #     results = pool.map(process_with_args, random_text_list)
    #
    # print('All contexts processed:')
    all_prompt = []
    # for result in results:
    #     all_prompt.extend(result)

    max_threads = 10
    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(new_search_info, context, cond, type, prompt_type) for i, context in
                       enumerate(random_text_list)]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in as_completed(futures):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None:
                    all_prompt.extend(result)
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")

    # print("All info processed.")
    # for c in random_text_list:
    #     # search_once_word(c, nlp, writer, ch_type=type)
    #     if cond == 'phrases':
    #         a = 1
    #         # search_c_phrase(c, "me_test\\ran_1000_data.csv", nlp,writer3=writer3, ch_type=type)
    #         # search_c_phrase_1(c, nlp, sim_model, ran_list, ch_type=type, sav_pro_file ='p_inf_phrase_')
    #         all_prompt.extend(new_search_phrase(c, nlp, ch_type=type))
    #     elif cond == 'words':
    #         all_prompt.extend(new_search_word(c, nlp, ch_type=type))
    #         # search_once_word1_1(c, nlp, sim_model, ran_list, ch_type=type, sav_pro_file='p_inf_word_')
    #         # search_once_word(c, "me_test\\ran_1000_data.csv", nlp, writer3=writer3, ch_type=type)
    return all_prompt
api_key = 'your_key'
def new_context_main(random_text_list,w_file1=None,raw_w_file=None,nlp=None, type=None, cond=None,prompt_type=None):
    start_time = time.time()
    dataset = w_file1.split('/')[-1].split('_')[0]
    fee1 = get_account_balance(api_key)
    sim_model = SimCSE("simcse")
    # contexts_to_prompts返回[[sent, context, phrase, prompt],...]
    basic_info_list = contexts_to_prompts(random_text_list, cond, type, sim_model, prompt_type)
    # prompts = basic_info_list[:][-1]
    # get_threadpool_gpt返回[[sentence,context,answer,prompt,q_str],...]
    results = get_threadpool_gpt(basic_info_list)
    # sim_model.build_index(a_list)
    q_info_list = []
    # random_text_list = random.sample(result_text, num)
    with open(w_file1, "w", newline='', encoding='utf-8') as csvfile1, \
        open(raw_w_file, "w", newline='', encoding='utf-8') as csvfile2:
        writer = csv.writer(csvfile1)
        raw_writer = csv.writer(csvfile2)
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
                    result_copy.extend([str_question,str_reason])
                    q_info_list.append(result_copy)
                except:
                    continue
            a = 1

        # [sent, context, g_answer, prompt, answer, str_question, str_reason, llm_answer]

        new_q_info_list = check_from_ask_1(q_info_list, sim_model, cond, type, nlp, raw_writer)
        writer.writerows(new_q_info_list)
    end_time = time.time()
    during_time = end_time - start_time
    fee2 = get_account_balance(api_key)
    fee = fee2 - fee1
    with open('experiment/fact_data/fee.txt', 'a') as p:
        p.write(
            f'dataset:{dataset},info_type:grammar_info,type:{type},cond:{cond},length:{len(new_q_info_list)},time:{datetime.datetime.now()},'
            f'fee:{fee}' + '\n')
    with open('experiment/fact_data/time.txt', 'a') as p:
        p.write(f'dataset:{dataset},info_type:grammar_info,type:{type},cond:{cond},length:{len(new_q_info_list)},time:{datetime.datetime.now()},'
                f'during_time:{during_time}'+'\n')



def get_random_context(num,r_file,w_file1=None,w_file2=None,w_file3=None,nlp=None, type=None, cond=None, data_file=None, w_file1_1=None,w_file2_1=None,w_file3_1=None):
    with open(r_file, mode='r',encoding='utf-8',newline='') as f:
        raw = f.readlines()
        result_text = []
        for r in raw :
            result_text.append(r)
        f.close()

    raw = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1')
    q_list = list(raw['question'])
    a_list = list(raw['answer'])
    a_list = [a.split('///')[0] for a in a_list if '///' in a]
    print(a_list)
    sim_model = SimCSE("simcse")
    # sim_model.build_index(a_list)
    raw_list = []
    for i in range(len(q_list)):
        if a_list[i] != '<No Answer>':
            raw_list.append([q_list[i], a_list[i]])
    ran_list = random.sample(raw_list, len(raw_list))

    random_text_list = random.sample(result_text, num)
    with open(w_file1, "w",newline='',encoding='utf-8') as csvfile1, \
            open(w_file2, "w", newline='', encoding='utf-8') as csvfile2 , \
            open(w_file3, "w", newline='', encoding='utf-8') as csvfile3:
            # open(w_file1_1, "w", newline='', encoding='utf-8') as csvfile1_1, \
            #     open(w_file2_1, "w", newline='', encoding='utf-8') as csvfile2_1, \
            #     open(w_file3_1, "w", newline='', encoding='utf-8') as csvfile3_1:
        writer1 = csv.writer(csvfile1)
        writer2 = csv.writer(csvfile2)
        writer3 = csv.writer(csvfile3)
        # writer1_1 = csv.writer(csvfile1_1)
        # writer2_1 = csv.writer(csvfile2_1)
        # writer3_1 = csv.writer(csvfile3_1)
        # writer1 = None
        # writer2 = None
        # writer3 = None
        for c in random_text_list:
            # search_once_word(c, nlp, writer, ch_type=type)
            if cond == 'phrases':
                a = 1
                # search_c_phrase(c, "me_test\\ran_1000_data.csv", nlp,writer3=writer3, ch_type=type)
                # search_c_phrase_1(c, nlp, sim_model, ran_list, ch_type=type, sav_pro_file ='p_inf_phrase_')

            elif cond == 'words':
                new_search_word(c,nlp,ch_type=type)
                # search_once_word1_1(c, nlp, sim_model, ran_list, ch_type=type, sav_pro_file='p_inf_word_')
                # search_once_word(c, "me_test\\ran_1000_data.csv", nlp, writer3=writer3, ch_type=type)
        if cond == 'phrases':
            # search_c_phrase(c, "me_test\\ran_1000_data.csv", nlp,writer3=writer3, ch_type=type)
            # search_c_phrase_1(c, "raw_dataset/nat_origin_qa.csv", nlp, writer1, writer2, writer3, ch_type=type,model=model,sav_pro_file = 'p_inf_phrase_')
            # try:
            search_c_phrase_2('p_inf_phrase_', ch_type=type, writer1=writer1, writer2=writer2, writer3=writer3,nlp=nlp,sim_model=sim_model)
            # except:
            #     csvfile1.close()
            #     csvfile2.close()
            #     csvfile3.close()
        elif cond == 'words':
            # search_once_word1(c, "raw_dataset/nat_origin_qa.csv", nlp, writer1, writer2, writer3, ch_type=type,model=model,sav_pro_file='p_inf_word_')
            try:
                search_once_word2('p_inf_word_', ch_type=type, writer1=writer1, writer2=writer2, writer3=writer3,nlp=nlp,sim_model=sim_model)
            except:
                csvfile1.close()
                csvfile2.close()
                csvfile3.close()

def get_random_sample(num,r_file,w_file):
    l = []
    with open(r_file, mode='r',encoding='ISO-8859-1',newline='') as f:
        raw = csv.reader(f,delimiter=',')
        for row in raw:
            if len(row) != 0:
                l.append(row)
    ran_num = num if len(l)> num else len(l)
    ran_num_list = random.sample(l,ran_num)

    a = 1
    # raw = list(pd.read_csv(file, sep=',',encoding='ISO-8859-1'))
    # ran_index_list = random.sample(raw,100)
    # ran_list = raw[ran_index_list]
    with open(w_file, "w",encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for c in ran_num_list:
            writer.writerow(c)

def sample_data(data_file,num,w_file):
    header = ['question','answer']
    raw = pd.read_csv(data_file, sep=',', encoding='ISO-8859-1', index_col=None)
    # with open(data_file, mode='r',encoding='ISO-8859-1',newline='') as f:
    #     raw = f.readlines()
    q_list = list(raw['question'])
    a_list = list(raw['answer'])
    raw_list = []
    for i in range(len(q_list)):
        if a_list[i] != '<No Answer>' and a_list[i] != None:
            raw_list.append([q_list[i], a_list[i]])
    ran_list = random.sample(raw_list, num)
    with open(w_file, "w",encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile,header)
        writer.writerow(header)
        for c in ran_list:
            writer.writerow(c)

def read_gpt_answer(file,nlp, p):
    with open(file, mode='r') as f:
        raw = f.readlines()
        result_text = []
        for r in raw :
            s = r.index('.')+2 if '.' in r[:5] else 0
            # e = r.index('\\')
            result_text.append(r[s:-1])
    gen_sent_pair = []

    with open("gpt_result_1.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        for sent in result_text:
            p.apply_async(judge_sent,args=(sent,nlp,writer,))
            # result = judge_sent(sent, nlp)
            # for r in b:
            #     writer.writerow(r)

            # gen_sent_pair.append(r)
        # result_text = [r[r.index('.')+2:r.index('\\n')] ]
    # output = pd.DataFrame(gen_sent_pair)
    # output.to_csv("gpt_result_2.csv",sep=',',header=None,index=None)

def add_context_to_qa(data_root,type,con_file):
    # 为生成的qa查找对应的context，为之后生成送入sut的格式做准备
    file = data_root + 'try_' + type + '.csv'
    with open(con_file, mode='r', encoding='utf-8') as con_f:
        context_list = con_f.readlines()
    raw = pd.read_csv(file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    qa_s_c_list = []
    c_list = []
    for i in range(len(raw)):
        sent, answer, question = raw[i][0], raw[i][1], raw[i][2]
        for context in context_list:
            if sent in context:
                # context = context.replace(sent,'!!!'+sent+'!!!')
                c_list.append(context)
                qa_s_c_list.append([context, answer, question, sent])
                break
    print(len(qa_s_c_list) - len(raw))
    data = pd.DataFrame(qa_s_c_list)
    sav_file = data_root + 'new_try_' + type + '.csv'
    data.to_csv(sav_file, sep=',', header=None, index=None)

def get_random_sample(num,r_file,w_file):
    l = []
    with open(r_file, mode='r',encoding='ISO-8859-1',newline='') as f:
        raw = csv.reader(f,delimiter=',')
        for row in raw:
            if len(row) != 0 and len(row[3].split(' '))>8:
                l.append(row)
    ran_num = num if len(l)> num else len(l)
    ran_num_list = random.sample(l,ran_num)

    a = 1
    # raw = list(pd.read_csv(file, sep=',',encoding='ISO-8859-1'))
    # ran_index_list = random.sample(raw,100)
    # ran_list = raw[ran_index_list]
    with open(w_file, "w",encoding='utf-8',newline='') as csvfile:
        writer = csv.writer(csvfile)
        for c in ran_num_list:
            writer.writerow(c)
def merge_csv(dataset,file_tail):
    csv_files = [f'experiment/fact_data/{dataset}_v_p_{file_tail}.csv',
                 f'experiment/fact_data/ {dataset}_n_p_{file_tail}.csv',
                 f'experiment/fact_data/{dataset}_v_w_{file_tail}.csv',
                 f'experiment/fact_data/{dataset}_n_w_{file_tail}.csv'
                 ]
    df_list = []
    for file in csv_files:
        try:
            f = pd.read_csv(file,sep=',', encoding='ISO-8859-1', index_col=None, header=None)
            df_list.append(f)
        except:
            continue

    # df_list = [pd.read_csv(file,sep=',', encoding='ISO-8859-1', index_col=None, header=None) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.sample(len(combined_df))
    combined_df.to_csv(f'experiment/fact_data/{dataset}_all_{file_tail}.csv', sep=',', header=None, index=None)

def qa_to_sut(in_file,out_file=None):
    raw = pd.read_csv(in_file, sep=',', encoding='ISO-8859-1', index_col=None,header=None)
    raw = raw.values.tolist()
    a = []
    for r in raw:
        b = []
        # for i in range(len(r)):
        #     r[i].replace('\n','')
        # r[3]=r[3][:-1]
        b.append(r[2] + '\\n' + r[0])
        # if b[0].remove():
        #     b[0] = b[0][:-1]
        b.append(r[1])
        a.append(b)
    data = open(out_file, 'w', encoding='utf-8',newline=None)
    for c in a:
        print(c[0][:-2] + "\t" + c[1], file=data)
model_name = "D:/huggingface_model/unifiedqa-v2-t5-large-1251000"  # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
unifiedqa_model = T5ForConditionalGeneration.from_pretrained(model_name)
def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = unifiedqa_model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
def use_unifiedqa(input_file,information_file,output_file):
    w_data = []
    raw = pd.read_csv(information_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None)
    raw = raw.values.tolist()
    with open(input_file, "r", encoding='utf-8') as f:
        # f = f.readlines()
        i = 0
        for line in f:
            try:
                question, answer = line.split("\t")
                if answer[-1] == '\n':
                    answer = answer[:-1]
                sut_answer = run_model(question)[0]
                sent = raw[i][-1]
                w_data.append([sent, question, answer, sut_answer])
                i = i + 1
            except:
                i = i+1
                continue
        data = pd.DataFrame(w_data)
        data.to_csv(output_file,sep=',', header=None, index=None)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--type",
    #     default=None,
    #     type=str,
    #     required=False,
    #     help=""
    # )
    # parser.add_argument(
    #     "--cond",
    #     default=None,
    #     type=str,
    #     required=False,
    #     help=""
    # )
    #
    # args = parser.parse_args()

    file_tail = '1'
    # w_embedding = sbert_model.encode(a1)
    # context_embedding = sbert_model.encode(a2)
    # sim = util.dot_score(w_embedding, context_embedding)
    # print(sim)
    # merge_csv('nat',file_tail)
    # merge_csv('squad2',file_tail)
    # merge_csv('boolq',file_tail)
    #     b = 1
    # c_inf = acquire_dependence_tree(context, nlp)
    # search_once_word(context,nlp,writer,ch_type='NOUN')
    # with open('boolq_dev_context.txt', mode='r', encoding='utf-8', newline='') as f:
    #     raw = f.readlines()
    #     result_text = []
    #     for r in raw:
    #         result_text.append(r)
    #     f.close()
    # # random_text_list = random.sample(result_text, 10)
    # random_text_list = result_text[30:40]
    # start_time = time.time()
    new_context_main(random_text_list, f'experiment/fact_data/boolq_v_p_{file_tail}.csv', f'experiment/fact_data/boolq_v_p_raw_{file_tail}.csv',
                     nlp=nlp, type='VERB', cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/boolq_n_p_{file_tail}.csv', f'experiment/fact_data/boolq_n_p_raw_{file_tail}.csv',
    #                  nlp=nlp, type='NOUN', cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/boolq_v_w_{file_tail}.csv', f'experiment/fact_data/boolq_v_w_raw_{file_tail}.csv',
    #                  nlp=nlp, type='VERB', cond='words',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/boolq_n_w_{file_tail}.csv', f'experiment/fact_data/boolq_n_w_raw_{file_tail}.csv',
    #                  nlp=nlp, type='NOUN', cond='words', prompt_type=1)
    # with open('squad2_dev_context.txt', mode='r', encoding='utf-8', newline='') as f:
    #     raw = f.readlines()
    #     result_text = []
    #     for r in raw:
    #         result_text.append(r)
    #     f.close()
    # random_text_list = result_text[30:40]
    # new_context_main(random_text_list, f'experiment/fact_data/squad2_v_p_{file_tail}.csv', f'experiment/fact_data/squad2_v_p_raw_{file_tail}.csv',nlp = nlp,type = 'VERB',cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/squad2_n_p_{file_tail}.csv', f'experiment/fact_data/squad2_n_p_raw_{file_tail}.csv',nlp=nlp, type='NOUN', cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/squad2_v_w_{file_tail}.csv', f'experiment/fact_data/squad2_v_w_raw_{file_tail}.csv', nlp=nlp, type='VERB', cond='words',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/squad2_n_w_{file_tail}.csv', f'experiment/fact_data/squad2_n_w_raw_{file_tail}.csv', nlp=nlp, type='NOUN', cond='words',prompt_type=1)
    # with open('nat_dev_context.txt', mode='r', encoding='utf-8', newline='') as f:
    #     raw = f.readlines()
    #     result_text = []
    #     for r in raw:
    #         result_text.append(r)
    #     f.close()
    # # random_text_list = random.sample(result_text, 10)
    # random_text_list = result_text[30:40]
    # new_context_main(random_text_list, f'experiment/fact_data/nat_v_p_{file_tail}.csv', f'experiment/fact_data/nat_v_p_raw_{file_tail}.csv',
    #                  nlp=nlp, type='VERB', cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/nat_n_p_{file_tail}.csv', f'experiment/fact_data/nat_n_p_raw_{file_tail}.csv',
    #                  nlp=nlp, type='NOUN', cond='phrases',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/nat_v_w_{file_tail}.csv', f'experiment/fact_data/nat_v_w_raw_{file_tail}.csv',
    #                  nlp=nlp, type='VERB', cond='words',prompt_type=1)
    # new_context_main(random_text_list, f'experiment/fact_data/nat_n_w_{file_tail}.csv', f'experiment/fact_data/nat_n_w_raw_{file_tail}.csv',
    #                  nlp=nlp, type='NOUN', cond='words',prompt_type=1)
    #
    # merge_csv('nat', file_tail)
    # merge_csv('squad2', file_tail)
    # merge_csv('boolq', file_tail)
    #
    qa_to_sut_inf(f'experiment/fact_data/squad2_all_{file_tail}.csv', f'experiment/sut_test/squad2_all_{file_tail}.tsv',5000)
    # qa_to_sut_inf(f'experiment/fact_data/squad2_all_{file_tail}.csv', f'experiment/sut_test/squad2_all_{file_tail}.tsv')
    # qa_to_sut_inf(f'experiment/fact_data/nat_all_{file_tail}.csv', f'experiment/sut_test/nat_all_{file_tail}.tsv')
    # qa_to_sut_inf(f'experiment/fact_data/boolq_all_{file_tail}.csv', f'experiment/sut_test/boolq_all_{file_tail}.tsv')
    # use_unifiedqa_inf(f'experiment/sut_test/squad2_all_{file_tail}.tsv', f'experiment/fact_data/squad2_all_{file_tail}.csv',
    #                   f'experiment/sut_output/squad2_all_output_{file_tail}.csv')
    # use_unifiedqa_inf(f'experiment/sut_test/nat_all_{file_tail}.tsv', f'experiment/fact_data/nat_all_{file_tail}.csv',
    #                   f'experiment/sut_output/nat_all_output_{file_tail}.csv')
    # use_unifiedqa_inf(f'experiment/sut_test/boolq_all_{file_tail}.tsv', f'experiment/fact_data/boolq_all_{file_tail}.csv',
    #                   f'experiment/sut_output/boolq_all_output_{file_tail}.csv')
    # # # file_tail = '2'
    # merge_all_csv_for_con(f'experiment/sut_output/inf_all_output_{file_tail}.csv','inf',file_tail)
    judge_consistency(f'experiment/sut_output/squad2_all_output_{file_tail}.csv', f'experiment/final_result/squad2_final_inf_{file_tail}.csv', 1)


    #
    # # sent1 = "Who is the main character?"
    # # sent2 = "Whom is the main character?"
    # # s_inf_1 = acquire_dependence_tree(sent1, nlp)
    # # s_inf_2 = acquire_dependence_tree(sent2, nlp)
    # # check_answer([sent1,sent2],'SVC',s_inf_1,s_inf_2)
    # # sent_list = []
    # # p = Pool(3)
    # # with open('gpt_result_data_try.txt', mode='r') as f:
    # #     raw = f.readlines()
    # #     result_text = []
    # #     for r in raw :
    # #         s = r.index('.')+2 if '.' in r[:5] else 0
    # #         # e = r.index('\\')
    # #         result_text.append(r[s:-1])
    # #     f.close()
    # gen_sent_pair = []

    # with open("gpt_result_conj.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     for q in result_text:
    #         judge_sent(q, nlp, writer)
    # p.close()
    # p.join()


