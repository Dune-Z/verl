import argparse
from datasets import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafiles', type=str, default='./data/mix-math/generation.parquet')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    data = load_dataset('parquet', data_files=args.datafiles)[args.split]
    print(data['reward_model'][0])
    print(data.column_names)
 