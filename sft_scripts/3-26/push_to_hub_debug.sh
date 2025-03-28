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

for CHECKPOINT in ${SAVE_LOCAL_DIR}/global_step_*; do
    STEP=$(basename $CHECKPOINT)  # Extracts "global_step_X"
    HUB_MODEL_ID="Yuanxin-Liu/mix-math-7b-Qwen-rs-debug-${STEP}"

    echo "Pushing checkpoint: $STEP to Hugging Face Hub at $HUB_MODEL_ID"

    python sft_scripts/3-26/push_to_hub.py \
        --model_name_or_path ${MODEL_NAME} \
        --adapter_path ${CHECKPOINT} \
        --hub_model_id ${HUB_MODEL_ID}
done