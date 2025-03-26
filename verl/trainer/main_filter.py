import argparse
from datasets import load_dataset
from verl.utils.reward_score.math_r1 import compute_score_val


def filter_and_select_response(example):
    responses = example['responses']
    gt = example['reward_model']['ground_truth']
    for response in responses:
        if compute_score_val(response, gt) == 1.0:
            return {'responses': response}
    return {'responses': None}  # Mark for removal later


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafiles', type=str, default='./data/mix-math/generation.parquet')
    parser.add_argument('--output_files', type=str, default='./data/mix-math/generation-filtered.parquet')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    datasets = load_dataset('parquet', data_files=args.datafiles)[args.split]
    datasets = datasets.map(filter_and_select_response)
    datasets = datasets.filter(lambda x: x['responses'] is not None)
    datasets.to_parquet(args.output_files)

 