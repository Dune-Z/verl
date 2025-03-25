set -x
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY

MODEL_NAME=Open-Reasoner-Zero/Open-Reasoner-Zero-7B
PROJECT_NAME=Open-Reasoner-Zero-7B
EXPERIMENT_NAME=${MODEL_NAME}-sft-generation
# SAVE_LOCAL_DIR=/checkpoints/hongpaul-sandbox/r1/${PROJECT_NAME}/${EXPERIMENT_NAME}
SAVE_LOCAL_DIR=./outputs/${PROJECT_NAME}/${EXPERIMENT_NAME}
TASK_NAMES=("orz_aime2024" "orz_gpqa_diamond" "orz_math500")
REMOTE_DATA_PATH=PRIME-RL/Eurus-2-RL-Data

### preprocess the dataset
# DATA_PATHS=()
# for TASK_NAME in "${TASK_NAMES[@]}"; do
#     echo "Processing task: $TASK_NAME"
    
#     if [ -z "${START_IDX:-}" ]; then
#         DATA_PATH_SUFF=${TASK_NAME}
#         python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --data_remote_dir $REMOTE_DATA_PATH
#     else
#         DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}
#         python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX --data_remote_dir $REMOTE_DATA_PATH
#     fi
#     DATA_PATHS+=("./data/$DATA_PATH_SUFF")
# done
# echo "Combined tasks: ${TASK_NAMES[@]}"
# python3 data_preprocess/combine_parquet.py --data_dirs ${DATA_PATHS[@]} --output_dir ./data/combined

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,8 torchrun \
    --standalone \
    --master_port=29501 \
    --nproc_per_node=8 \
    --nnodes=1 \
    -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/orz_dataset/generation.parquet \
        data.val_files=./data/orz_dataset/generation.parquet \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL_NAME} \
        trainer.project_name=orz-sft \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.total_epochs=4 \
        trainer.logger=['console','wandb'] \
        optim.lr=1e-6

