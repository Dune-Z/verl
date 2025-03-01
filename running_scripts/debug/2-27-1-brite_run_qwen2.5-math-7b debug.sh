#set -x
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

### task name can be selected from [gsm8k, math_dataset, opencoder]
TASK_NAME=prime
MATH_ONLY=False
# comment START_IDX and END_IDX if you want to use the whole dataset for the training
sft_loss_coef=0.2
REMOTE_DATA_PATH=PRIME-RL/Eurus-2-RL-Data
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=Qwen2.5-Math-0.5B
MODEL_NAME=Qwen/Qwen2.5-0.5B
EXPERIMENT_NAME=DEBUG
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}

### preprocess the dataset
if [ -z "${START_IDX:-}" ]; then
    DATA_PATH_SUFF=${TASK_NAME}_${MATH_ONLY}
    python3 data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF --data_remote_dir $REMOTE_DATA_PATH --math_only $MATH_ONLY
else
    DATA_PATH_SUFF=${TASK_NAME}_${START_IDX}_${END_IDX}_${MATH_ONLY}
    python3 data_preprocess/${TASK_NAME}.py --local_dir $HOME/data/$DATA_PATH_SUFF --sample_start_idx $START_IDX --sample_end_idx $END_IDX --data_remote_dir $REMOTE_DATA_PATH --math_only $MATH_ONLY
fi

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
# yifei's key
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
python3 -m verl.trainer.main_ppo \
    algorithm.reward_scale=1. \
    algorithm.reward_offset=-1. \
    algorithm.adv_estimator=grpo \
    reward_model.reward_manager=prime \
    data.custom_temp_dir=$HOME/tmp/ray/  \
    data.train_files=$HOME/data/$DATA_PATH_SUFF/train.parquet \
    data.val_files=$HOME/data/$DATA_PATH_SUFF/test.parquet \
    data.train_batch_size=1024 \
    data.val_batch_size=1024 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    actor_rollout_ref.model.path=${MODEL_NAME} \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4800 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${SAVE_LOCAL_DIR} \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=15 \
    trainer.test_freq=15 \
    trainer.total_epochs=1 $@