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

from verl import DataProto
from verl.utils.reward_score import my_compute_score
import torch


class MyRewardManager:
    """Yuanxin's reward manager.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, offset=0, scale=1.) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = my_compute_score
        self.scale = scale
        self.offset = offset

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        # batched scoring
        prompt_ids = data.batch['prompts']
        prompt_length = prompt_ids.shape[-1]

        response_ids = data.batch['responses']
        valid_response_length = data.batch['attention_mask'][:, prompt_length:].sum(dim=-1)
        sequences_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_info = data.non_tensor_batch.get('extra_info', [None] * len(data_sources))

        assert len(sequences_str) == len(ground_truth) == len(data_sources) == len(extra_info)

        # Sequential scoring instead of parallel
        scores = []
        for completion, reference, task, task_extra_info in zip(sequences_str, ground_truth, data_sources, extra_info):
            try:
                result = self.compute_score(task, completion, reference, task_extra_info)
                if isinstance(result, (int, float, bool)):
                    scores.append(float(result))
                elif isinstance(result, (list, tuple)) and isinstance(result[0], (int, float, bool)):
                    scores.append(float(result[0]))
                else:
                    scores.append(0.0)
            except Exception as e:
                print(f"Error processing completion at index: {e}")
                scores.append(0.0)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_length[i].item() - 1] = self.scale * scores[i] + self.offset

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(sequences_str[i])

        return reward_tensor
