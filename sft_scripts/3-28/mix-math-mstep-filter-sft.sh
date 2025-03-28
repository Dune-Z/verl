set -x
export HF_TOKEN=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
export WANDB_API_KEY=6f9e1eaf73cd08b4f0cd4674c7856201f2453428
wandb login --relogin $WANDB_API_KEY
MODEL_NAME=Yuanxin-Liu/Qwen2.5-7B_Mix-Math-yt-rbt-grpo_0_exp0_gen_8_test_8_clip_ratio_0_outer_kl-320
PROJECT_NAME=Qwen2.5-7B_Mix-Math-yt
EXPERIMENT_NAME=${MODEL_NAME}-rs
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}

# python3 sft_scripts/debug/generation_hub.py \
#     --datafiles ./data/mix-math/generation.parquet \
#     --hub Yuanxin-Liu/mix-math-7b-Qwen-rs

python3 -m verl.trainer.main_filter \
    --datafiles ./data/mix-math/filtered.parquet \
    --output_files ./data/mix-math/generation-filtered.parquet

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 -m verl.trainer.fsdp_sft_trainer \
        data.train_files=./data/mix-math/generation-filtered.parquet \
        data.val_files=./data/mix-math/generation-filtered.parquet \
        data.prompt_key=prompt \
        data.response_key=responses \
        data.train_batch_size=8 \
        data.micro_batch_size_per_gpu=1 \
        model.partial_pretrain=${MODEL_NAME} \
        model.lora_rank=32 \
        model.lora_alpha=128 \
        trainer.project_name=${PROJECT_NAME} \
        trainer.experiment_name=${EXPERIMENT_NAME} \
        trainer.default_local_dir=${SAVE_LOCAL_DIR} \
        trainer.total_epochs=4 \
        trainer.logger=['console','wandb'] \
        optim.lr=1e-6 \
        ulysses_sequence_parallel_size=2 \
        use_remove_padding=true