import argparse
from datasets import load_dataset
from verl.utils.reward_score.math_r1 import compute_score_val


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datafiles', type=str, default='./data/mix-math/generation.parquet')
    parser.add_argument('--output_files', type=str, default='./data/mix-math/generation-filtered.parquet')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()
    datasets = load_dataset('parquet', data_files=args.datafiles)[args.split]
    remove_idx = []
    for dp, data in enumerate(datasets):
        responses = data['responses']
        gt = data['reward_model']['ground_truth']
        idxes = []
        for idx, response in enumerate(responses):
            print(response)
            score = compute_score_val(response, gt)
            if score == 1.0:
                idxes.append(idx)
                break
        if len(idxes) == 0:
            # remove this data
            remove_idx.append(dp)
        else:
            # only keep responses[idxes[0]]
            response = responses[idxes[0]]
            datasets['responses'][dp] = response
    
    # remove data
    datasets = datasets.select(indices=[i for i in range(len(datasets)) if i not in remove_idx])
    # print number of data
    print(len(datasets))

 