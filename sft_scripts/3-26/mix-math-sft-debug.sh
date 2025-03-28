uv pip install transformers==4.47.1 deepspeed --link-mode=copy
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
wandb login --relogin $WANDB_API_KEY
export HF_TOKEN=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

PROJECT_NAME=Qwen2.5-7B_Mix-Math-yt
MODEL_NAME=Qwen/Qwen2.5-7B
EXPERIMENT_NAME=${MODEL_NAME}-rs-debug
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}

CUDA_VISIBLE_DEVICES=8,9 torchrun --standalone --nproc_per_node=2 --nnodes=1 -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/mix-math/generation-debug.parquet \
        data.val_files=./data/mix-math/generation-debug.parquet \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.train_batch_size=64 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL_NAME} \
        model.lora_rank=32 \
        model.lora_alpha=128 \
        trainer.project_name=orz-sft \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.total_epochs=1 \
        trainer.logger=['console','wandb'] \
        optim.lr=1e-6 \
        ulysses_sequence_parallel_size=2 \
        use_remove_padding=true \
        trainer.hub_model_id=Yuanxin-Liu/baseline-rs-debug-lora