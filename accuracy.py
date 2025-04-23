import configparser
import random

import pandas as pd

config = configparser.ConfigParser()

config.read(f'../../config.ini')

def select_ran(dataset,info_type):
    raw_file = f'../../data/result/test_result/bug_result/{info_type}_bug_result_{dataset}_all_filter.csv'
    ran_num = int(config['accuracy']['rand_num'])
    sav_file = f'../../data/result/experiment_data/accuracy/{dataset}/{info_type}_accuracy_{dataset}_ran_{ran_num}.csv'
    data = pd.read_csv(raw_file, sep=',', encoding='ISO-8859-1', index_col=None, header=None).values.tolist()
    ran_data = random.sample(data, ran_num)
    data = pd.DataFrame(ran_data)
    data.to_csv(sav_file, mode='w', index=False, header=False)
if __name__ == '__main__':
    for t in ['entity', 'rel']:
        select_ran('your_dataset', t)