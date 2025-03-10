# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets
import argparse
from utils import copy, makedirs
import json


def make_format(example, idx):
    # example now is a dictionary with a 'messages' key
    messages = example['messages']
    human_message = messages[0]
    assistant_message = messages[1]
    
    question = human_message.get('value', None) if human_message.get('from') == 'human' else None
    
    answer = None
    if assistant_message.get('from') == 'assistant':
        ground_truth = assistant_message.get('ground_truth', None)
        if isinstance(ground_truth, dict):
            answer = ground_truth.get('value', None)
    
    if not question or not answer:
        return None
        
    return {
        "data_source": "orz_dataset",
        "prompt": [
            {"content": question + " " + "Let's think step by step and output the final answer within \\boxed{}.", "role": "user"}
        ],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule",
        },
        "extra_info": {
            "index": idx,
            "split": "train"
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--sample_start_idx", default=0, type=int)
    parser.add_argument("--sample_end_idx", default=999999999, type=int)
    parser.add_argument("--data_remote_dir", default='Tonic/OpenReasonerZero', type=str)
    args = parser.parse_args()

    print("Loading the local dataset...", flush=True)
    from pathlib import Path

    file_path = Path(os.path.join(args.local_dir, 'train.parquet'))

    if file_path.exists() and file_path.suffix == ".parquet":
        print("file existed")
        return

    with open('./data_preprocess/orz_math_57k_collected.json', 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    formatted_data = [{"messages": item} for item in raw_data]
    dataset = datasets.Dataset.from_list(formatted_data)
    
    train_dataset = dataset
    train_dataset = train_dataset.select(range(args.sample_start_idx, min(args.sample_end_idx, len(train_dataset))))
    
    train_dataset = train_dataset.map(make_format, with_indices=True)

    print(f"len of training dataset is {len(train_dataset)}")
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)

if __name__ == '__main__':
    main()
