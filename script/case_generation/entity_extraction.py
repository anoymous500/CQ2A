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
# from openai import OpenAI
from multiprocessing import Pool
import argparse
import numpy as np
from tqdm import tqdm
# from relation_extraction import use_unifiedqa_inf, judge_consistency, merge_all_csv_for_con

from model_access import sim_model,nlp
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from nltk.corpus import wordnet
import string
import csv
from func_timeout import func_timeout
from func_timeout import FunctionTimedOut
import os
from nltk.tree import Tree
from model_access import sim_model
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


def acquire_dependence_tree(sentence,nlp):

    doc = nlp(sentence)
    # print(*[
    #     f'id: {word.id}\tfeats: {word.feats if word.feats else "_"}\tword: {word.text}\tlemma: {word.lemma}\tupos: {word.upos}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
    #     for sent in doc.sentences for word in sent.words], sep='\n')
    return [[word.id, word.feats, word.text, word.lemma, word.upos, word.head,sent.words[word.head - 1].text if word.head > 0 else "root",word.deprel] for sent in doc.sentences for word in sent.words]

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



def new_search_phrase(context,nlp,ch_type=None):
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
            # prompt = generate_question_llm(context, phrase, sent, prompt_type)
            return_list.append([sent, context, phrase])
    return return_list

not_word = ['generate','continue','become','remain','deem','let','take','express','conclude','give','consider','demand','resolve',
            'cause','do','turn','discover','realize','start','convince','be','involve','begin','choose','ensue','get','endeavour','assume','insinuate','have']

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
            # prompt = generate_question_llm(context, word, sent, prompt_type)
            return_list.append([sent, context, word])
    return return_list

def new_search_info(context, cond, ch_type, prompt_type=None):
    if cond == 'phrases':
        return new_search_phrase(context, nlp, ch_type)
    elif cond == 'words':
        return new_search_word(context, nlp, ch_type)

def entity_extraction(dataset):
    with open(f'{dataset}_dev_context.txt', mode='r', encoding='utf-8', newline='') as f:
        raw = f.readlines()
        result_text = []
        for r in raw:
            result_text.append(r)
        f.close()
    for cond in ['phrases','words']:
        for ch_type in ['NOUN','VERB']:
            all_result = []
            info_file_name = f'../../data/result/relation_info/{dataset}/entity_info_{dataset}_{cond}_{ch_type}.csv'
            # for result in results:
            #     all_prompt.extend(result)

            max_threads = 10
            with ThreadPoolExecutor(max_threads) as executor:
                # 提交任务到线程池中
                futures = [executor.submit(new_search_info, context, cond, ch_type) for i, context in
                           enumerate(result_text)]

                # 处理结果，as_completed 会在任务完成后返回结果
                for future in as_completed(futures):
                    try:
                        result = future.result()  # 获取每个线程的结果
                        if result is not None:
                            all_result.extend(result)
                        # print(result)
                    except Exception as e:
                        print(f"Error occurred while processing result: {e}")
            q_dataframe = pd.DataFrame(all_result)
            q_dataframe.to_csv(info_file_name, mode='w', index=False, header=False)

if __name__ == '__main__':
    entity_extraction('your_dataset')