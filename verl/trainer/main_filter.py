import argparse
from datasets import load_dataset
from verl.utils.reward_score.math_r1 import compute_score_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafiles', type=str, default='./data/mix-math/generation.parquet')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    datasets = load_dataset('parquet', data_files=args.datafiles)[args.split]
    for data in datasets:
        responses = data['responses']
        gt = data['reward_model']['ground_truth']
        idxes = []
        for idx, response in enumerate(responses):
            print(response)
            score = compute_score_val(response, gt)
            if score == 1.0:
                idxes.append(idx)
                break

 