set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY

MODEL_NAME=Open-Reasoner-Zero/Open-Reasoner-Zero-7B
PROJECT_NAME=Open-Reasoner-Zero-7B
EXPERIMENT_NAME=${MODEL_NAME}-sft-gsm8k
# SAVE_LOCAL_DIR=/checkpoints/hongpaul-sandbox/r1/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_LOCAL_DIR=./outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}

rm -rf ./data/gsm8k
python data_preprocess/gsm8k.py --local_dir ./data/gsm8k


CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --standalone \
    --master_port=29501 \
    --nproc_per_node=4 \
    --nnodes=1 \
    -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/gsm8k/train.parquet \
        data.val_files=./data/gsm8k/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        +data.prompt_dict_keys=['question'] \
        +data.response_dict_keys=['answer'] \
        data.train_batch_size=4 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL_NAME} \
        trainer.project_name=gsm8k-sft \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.total_epochs=4 \
        trainer.logger=['console','wandb']

