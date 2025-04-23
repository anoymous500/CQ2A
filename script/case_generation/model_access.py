from openai import OpenAI
from openai import RateLimitError
import threading
import torch
from transformers import pipeline
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import torch
from simcse import SimCSE
import stanza
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

print(os.cpu_count())

sim_model = SimCSE("simcse")

nlp = stanza.Pipeline(lang='en')

model_name = "D:/huggingface_model/unifiedqa-v2-t5-large-1251000"  # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
unifiedqa_model = T5ForConditionalGeneration.from_pretrained(model_name)
# device = torch.device('cuda:0')
# unifiedqa_model.to(device)

OPENAI_KEY = 'your_key'
client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = unifiedqa_model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)

def get_local_Llama(prompt):
    model_id = "D:\huggingface_model\Llama-3.2-1B-Instruct"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(pipe(prompt,max_new_tokens=1024)['generated_text'])


def connect_gpt_prompt_cov(prompt=None,client=None,thread_id=None):
    try:
        # print(f"Thread {thread_id} sending request with prompt: {prompt}")
        # OPENAI_KEY = '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'
        # client = OpenAI(
        #     # This is the default and can be omitted
        #     api_key=OPENAI_KEY,
        # )

        # prompt = '你好'
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        answer = response.choices[0].message.content.strip()

        return answer
        # print(f"Thread {thread_id} received response: {response.choices[0].message['content']}")
    except RateLimitError as e:
        print(f"Thread {thread_id}: RateLimitError encountered. Waiting before retrying...")
        retry_after = int(e.headers.get("Retry-After", 60))  # 从响应头中获取 Retry-After，默认60秒
        time.sleep(retry_after)  # 等待一段时间后重试
        return connect_gpt_prompt(prompt, client, thread_id)  # 重试请求
    except Exception as e:
        print(f"Thread {thread_id}: Other error occurred: {e}")
        return None

def connect_gpt_prompt(prompt=None,client=None,thread_id=None):
    try:
        # prompt = '你好'
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-mini-2024-07-18",
            temperature=0,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )

        answer = response.choices[0].message.content.strip()

        return answer
        # print(f"Thread {thread_id} received response: {response.choices[0].message['content']}")
    except RateLimitError as e:
        print(f"Thread {thread_id}: RateLimitError encountered. Waiting before retrying...")
        retry_after = int(e.headers.get("Retry-After", 60))  # 从响应头中获取 Retry-After，默认60秒
        time.sleep(retry_after)  # 等待一段时间后重试
        return connect_gpt_prompt(prompt, client, thread_id)  # 重试请求
    except Exception as e:
        print(f"Thread {thread_id}: Other error occurred: {e}")
        return None

def connect_gpt_prompt_ask(prompt=None,client=None,thread_id=None):
    # 以列表形式传参
    try:

        # prompt = '你好'
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-mini",
            temperature=0.7,
            n=5,
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
        choices = response.choices
        answer_list = []
        for choice in choices:
            answer_list.append(choice.message.content.strip())
        answer_set = set(answer_list)
        answer_count = [answer_list.count(a) for a in answer_set]
        answer_dict = list(zip(answer_set,answer_count))
        answer = sorted(answer_dict, key=lambda x:x[1], reverse=True)[0][0]
        return answer
        # print(f"Thread {thread_id} received response: {response.choices[0].message['content']}")
    except RateLimitError as e:
        print(f"Thread {thread_id}: RateLimitError encountered. Waiting before retrying...")
        retry_after = int(e.headers.get("Retry-After", 60))  # 从响应头中获取 Retry-After，默认60秒
        time.sleep(retry_after)  # 等待一段时间后重试
        return connect_gpt_prompt(prompt, client, thread_id)  # 重试请求
    except Exception as e:
        print(f"Thread {thread_id}: Other error occurred: {e}")
        return None

def connect_gpt_info(prompt_info=None,client=None,thread_id=None):
    # 以列表形式传参
    try:
        # prompt = '你好'
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            temperature=0,
            model="gpt-4o-mini",
            n=5,
            messages=[
                {"role": "user", "content": prompt_info[-1]},
            ]
        )

        answer = response.choices[0].message.content.strip()
        prompt_info.append(answer)
        # print(response.id)
        return prompt_info
        # print(f"Thread {thread_id} received response: {response.choices[0].message['content']}")
    except RateLimitError as e:
        print(f"Thread {thread_id}: RateLimitError encountered. Waiting before retrying...")
        retry_after = int(e.headers.get("Retry-After", 60))  # 从响应头中获取 Retry-After，默认60秒
        time.sleep(retry_after)  # 等待一段时间后重试
        return connect_gpt_info(prompt_info, client, thread_id)  # 重试请求
    except Exception as e:
        print(f"Thread {thread_id}: Other error occurred: {e}")
        prompt_info.append(None)
        return prompt_info

def connect_gpt_info_ask(prompt_info=None,client=None,thread_id=None):
    # 以列表形式传参
    try:

        # prompt = '你好'
        response = client.chat.completions.create(
            # model="gpt-3.5-turbo-0125",
            model="gpt-4o-mini",
            temperature=0.7,
            n=5,
            messages=[
                {"role": "user", "content": prompt_info[-1]},
            ]
        )
        choices = response.choices
        answer_list = []
        for choice in choices:
            answer_list.append(choice.message.content.strip())
        answer_set = set(answer_list)
        answer_count = [answer_list.count(a) for a in answer_set]
        answer_dict = list(zip(answer_set,answer_count))
        answer = sorted(answer_dict, key=lambda x:x[1], reverse=True)[0][0]
        # answer = response.choices[0].message.content.strip()
        prompt_info.append(answer)
        # print(response.id)
        return prompt_info
        # print(f"Thread {thread_id} received response: {response.choices[0].message['content']}")
    except RateLimitError as e:
        print(f"Thread {thread_id}: RateLimitError encountered. Waiting before retrying...")
        retry_after = int(e.headers.get("Retry-After", 60))  # 从响应头中获取 Retry-After，默认60秒
        time.sleep(retry_after)  # 等待一段时间后重试
        return connect_gpt_info(prompt_info, client, thread_id)  # 重试请求
    except Exception as e:
        print(f"Thread {thread_id}: Other error occurred: {e}")
        prompt_info.append(None)
        return prompt_info

# 线程工作函数
def thread_worker(prompt, thread_id):
    connect_gpt_info(prompt, thread_id)

def get_threadpool_gpt_prompt(prompts_info_list):
    # 创建并启动多个线程,针对输入为纯prompt
    threads = []
    OPENAI_KEY = 'your_key'

    client = OpenAI(
        # This is the default and can be omitted
        api_key=OPENAI_KEY,
    )

    max_threads = 10  # 同时运行的最大线程数
    results = []

    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(connect_gpt_info, prompt_info, client, i) for i, prompt_info in enumerate(prompts_info_list)]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in as_completed(futures):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None:
                    results.append(result)
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")

    print("All prompts processed.")
    return results

def get_threadpool_gpt(prompts_info_list,num=None,pause_writer=None,writer=None):
    # pause_writer为输出结果时即时记录的文件
    # 创建并启动多个线程
    threads = []

    max_threads = 10 if num is None else num  # 同时运行的最大线程数
    results = []

    with ThreadPoolExecutor(max_threads) as executor:
        # 提交任务到线程池中
        futures = [executor.submit(connect_gpt_info, prompt_info, client, i) for i, prompt_info in enumerate(prompts_info_list)]

        # 处理结果，as_completed 会在任务完成后返回结果
        for future in as_completed(futures):
            try:
                result = future.result()  # 获取每个线程的结果
                if result is not None and result[0] is not None:
                    results.append(result)
                    if writer is not None:
                        writer.writerow(result)
                    if pause_writer is not None:
                        pause_writer.writerow(result[:-1])
                # print(result)
            except Exception as e:
                print(f"Error occurred while processing result: {e}")

    print("All prompts processed.")
    return results
def get_account_balance(api_key):
    response = requests.get(
            f"https://api.openai.com/v1/usage?date={str(datetime.datetime.today()).split(' ')[0]}",
            headers={"Authorization": "Bearer " + api_key,
        "Content-Type": "application/json"}
                     # 'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'},
            # cookies='%7B%22distinct_id%22%3A%20%22user-aOXlEOfbwOnsnQBCsQRr1GJQ%22%2C%22%24device_id%22%3A%20%22191ad976864170-0a7c9a6f4f7bc4-26001e51-1bcab9-191ad97686517a1%22%2C%22%24initial_referrer%22%3A%20%22%24direct%22%2C%22%24initial_referring_domain%22%3A%20%22%24direct%22%2C%22%24user_id%22%3A%20%22user-aOXlEOfbwOnsnQBCsQRr1GJQ%22%7D'
        )

    if response.status_code == 200:
        balance_info = response.json()
    else:
        return {"error": response.text}
    total_input_token = 0
    total_output_token = 0
    time_list = []
    for bal in balance_info['data']:
        total_input_token += bal['n_context_tokens_total']
        total_output_token += bal['n_generated_tokens_total']
        time_list.append(datetime.datetime.fromtimestamp(bal['aggregation_timestamp']))
    fee = total_input_token / 1000000 * 0.15 + total_output_token / 1000000 * 0.6
    return fee


if __name__ == '__main__':
    pass