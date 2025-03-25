from datasets import load_dataset


if __name__ == '__main__':
    datafiles = './data/orz_dataset/generation.parquet'
    data = load_dataset('parquet', data_files=datafiles)['train']
    print(data['prompt'][0])
    print(data['responses'][0])