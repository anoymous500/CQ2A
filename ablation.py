import configparser
import random

import pandas as pd

config = configparser.ConfigParser()

config.read(f'../../config.ini')

def select_ran(dataset,info_type):
    for file_tail in ['raw', 'rule_filter', 'all_filter']:
        raw_file = f'../../data/result/case/{dataset}/{info_type}_info_{dataset}_{file_tail}_q.csv'
        ran_num = int(config['ablation']['rand_num'])
        sav_file = f'../../data/result/experiment_data/ablation/{dataset}/{info_type}_abalation_{dataset}_{file_tail}_ran_{ran_num}.csv'
        data = pd.read_csv(raw_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
        ran_data = random.sample(data, ran_num)
        data = pd.DataFrame(ran_data)
        data.to_csv(sav_file, mode='w', index=False, header=False)

if __name__ == '__main__':
    for t in ['entity', 'rel']:
        select_ran('your_dataset', t)