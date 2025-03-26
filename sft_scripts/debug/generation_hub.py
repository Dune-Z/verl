import argparse
from datasets import load_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', action='store_true')
    parser.add_argument('--datafiles', type=str, default='./data/orz_dataset/generation.parquet')
    parser.add_argument('--hub', type=str, default='Ethan-Z/orz-generation-debug')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    if args.push:
        data = load_dataset('parquet', data_files=args.datafiles)
        data.push_to_hub(args.hub, private=False)
    else:
        data = load_dataset(args.hub)[args.split]
        data.to_parquet(args.datafiles)
        