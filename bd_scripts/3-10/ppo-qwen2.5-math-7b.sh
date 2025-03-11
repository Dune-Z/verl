wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
uv pip install -r requirements.txt --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
uv uninstall wandb
uv pip install wandb --no-cache-dir --link-mode=copy
uv pip install math_verify --no-build-isolation --link-mode=copy
export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
wandb login --relogin $WANDB_API_KEY

TASK_NAMES=("orz_aime2024" "orz_gpqa_diamond" "orz_math500")

# comment START_IDX and END_IDX if you want to use the whole dataset for the training
sft_loss_coef=0
REMOTE_DATA_PATH=PRIME-RL/Eurus-2-RL-Data
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=Qwen2.5-Math-7B
MODEL_NAME=Qwen/Qwen2.5-Math-7B
EXPERIMENT_NAME=ppo
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}


### preprocess the dataset
DATA_PATHS=()
for TASK_NAME in "${TASK_NAMES[@]}"; do
    echo "Processing task: $TASK_NAME"
    
    if [ -z "${START_IDX:-}" ]; then
        DATA_PATH_SUFF=${TASK_NAME}
        python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --data_remote_dir $REMOTE_DATA_PATH
    else
        DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}
        python3 data_preprocess/${TASK_NAME}.py --local_dir ./data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX --data_remote_dir $REMOTE_DATA_PATH
    fi
    DATA_PATHS+=("./data/$DATA_PATH_SUFF")
done
echo "Combined tasks: ${TASK_NAMES[@]}"
python3 data_preprocess/combine_parquet.py --data_dirs ${DATA_PATHS[@]} --output_dir ./data/combined

python3 data_preprocess/orz_dataset.py --local_dir ./data/orz_dataset

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    actor_rollout_ref.actor.sft_loss_coef=${sft_loss_coef} \
    algorithm.reward_scale=1. \
    algorithm.reward_offset=0 \
    algorithm.adv_estimator=gae \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=prime \
    data.custom_temp_dir=$HOME/tmp/ray/  \
    data.train_files=./data/orz_dataset/train.parquet \
    data.val_files=./data/combined/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=3500 \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=24000 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=${MODEL_NAME} \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_max_token_len_per_gpu=72000 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15
