from datasets import concatenate_datasets, load_from_disk, Dataset, load_dataset
import math
from typing import List
import argparse
def create_mixed_dataset(
        dataset_paths: List[str],
        copies: List[float]):
    copied_datasets = []
    num_rows = []
    keep_columns = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    #import pdb;pdb.set_trace()
    datasets = [load_dataset("parquet", data_files=path)['train'].select_columns(keep_columns) for path in dataset_paths]
    for dataset, num_copy in zip(datasets, copies):
        n_dataset =concatenate_datasets([dataset] * num_copy)
        num_rows.append(len(n_dataset))
        copied_datasets.append(n_dataset)
    merged_dataset = concatenate_datasets(copied_datasets)
    sampling_ratios = [i/len(merged_dataset) for i in num_rows]
    print(f"Data rows: {num_rows}, Sampling ratios: {sampling_ratios}")
    return merged_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/train.parquet')
    dataset_list = [
        "data/math_r1_dataset/train.parquet",
        "data/aime_train/train.parquet",
        "data/still_30k_train/train.parquet"
    ]
    copies = [1, 42, 1]
    args = parser.parse_args()

    merged_dataset = create_mixed_dataset(dataset_list, copies)
    merged_dataset.to_parquet(args.local_dir)



