def data_process(dataset):
    # change your data type
    raw_data_path = f'../../data/dataset/raw/{dataset}...'
    context_path = f'../../data/dataset/processed_data/{dataset}...'
    q_path = f'../../data/dataset/processed_data/{dataset}...'
    # extract context and q_pair from your dataset, save contexts in the form of 'txt' in context_path
    # and save q_pairs in the form of 'q \\n c \t a' in q_path
if __name__ == '__main__':
    data_process('your_dataset')