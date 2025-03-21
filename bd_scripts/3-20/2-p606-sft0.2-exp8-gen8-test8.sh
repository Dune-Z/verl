wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
uv pip install -r requirements.txt --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
uv uninstall wandb
uv pip install wandb --no-cache-dir --link-mode=copy
#export EXPERIMENT_NAME=qwen-7b-math-606
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY

sft_loss_coef=0.2
sft_loss_exp_ceof=8
SAMPLING_TIME_TEST=8
REMOTE_DATA_PATH=PRIME-RL/Eurus-2-RL-Data


SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=Qwen2.5-7B_Mix-Math
MODEL_NAME=Qwen/Qwen2.5-7B
EXPERIMENT_NAME=grpo_0_test_gen_8_test_${SAMPLING_TIME_TEST}
SAVE_LOCAL_DIR=/checkpoints/hongpaul-sandbox/r1/${PROJECT_NAME}/${EXPERIMENT_NAME}

echo "Processing task: math_r1_dataset"
python3 data_preprocess/math_r1_dataset.py
echo "Processing task: still_30k"
python3 data_preprocess/still_30k.py
echo "Processing task: aime_train_dataset"
python3 data_preprocess/aime_train_dataset.py
echo "Processing task: create_math_data_mix"
python3 data_preprocess/create_math_data_mix.py
echo "Processing task: aime_24_dataset"
python3 data_preprocess/aime_24_dataset.py
echo "Processing task: math_r1_500"
python3 data_preprocess/math_r1_500.py

echo "start training"
python3 -m verl.trainer.main_ppo \
        actor_rollout_ref.actor.sft_loss_coef=${sft_loss_coef} \
        actor_rollout_ref.actor.sft_loss_exp_coef=${sft_loss_exp_ceof} \
        algorithm.adv_estimator=grpo \
        trainer.test_sample_n=${SAMPLING_TIME_TEST} \
        data.custom_temp_dir=$HOME/tmp/ray/  \
        data.train_files=data/train.parquet \
        data.val_files=['data/aime_2024/test.parquet','data/math_r1_500/test.parquet'] \
        data.train_batch_size=512 \
        data.val_batch_size=256 \
        data.max_prompt_length=512 \
        data.max_response_length=8192 \
        actor_rollout_ref.model.path=$MODEL_NAME \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.logger=['console','wandb'] \
        trainer.default_hdfs_dir=null \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=20 \
        trainer.total_epochs=5 $@
