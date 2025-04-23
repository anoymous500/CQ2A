import copy
import logging
import os
from concurrent.futures import ThreadPoolExecutor,as_completed
import re
from model_access import *
import pandas as pd

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
        if len(sent.split(' ')) > 8:
            prompt_list.append([sent, context, prompt.format(sent)])
    thread_num = 5

    prompt_list_1 = get_threadpool_gpt(prompt_list, thread_num)
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
                prompt_info_list.append(k[:-1] + [str_relation, str_explain])
                # [sent, context, prompt, str_relation, str_explain]

            a = 1
        a = 1
    # logger = logging.getLogger()
    # logger.info(f'extract {len(fin_answer)} relation triples from this context.')
    # ad_dataframe = pd.DataFrame(fin_answer)
    # ad_dataframe.to_csv(w_path, mode='a', index=False, header=False)
    return prompt_info_list

def rel_extraction(dataset):
    with open(f'{dataset}_dev_context.txt', mode='r', encoding='utf-8', newline='') as f:
        raw = f.readlines()
        result_text = []
        for r in raw:
            result_text.append(r)
        f.close()
    max_threads = 10   # 同时运行的最大线程数
    results = []
    all_info = []
    info_file_name = f'../../data/result/relation_info/{dataset}/rel_info_{dataset}.csv'
    # contexts_to_prompts返回[[sent, context, phrase, prompt],...]
    if os.path.exists(info_file_name):
        load_data = pd.read_csv(info_file_name)
        all_info = load_data.values.tolist()
    else:
        with ThreadPoolExecutor(max_threads) as executor:
            # 提交任务到线程池中
            futures = [executor.submit(relation_extraction_s, context) for i, context in
                       enumerate(result_text)]

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
if __name__ == '__main__':
    rel_extraction('your_dataset')