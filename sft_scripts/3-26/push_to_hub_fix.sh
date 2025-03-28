set -x
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
uv pip install -r requirements.txt --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
uv uninstall wandb
uv pip install wandb --no-cache-dir --link-mode=copy
uv pip install math_verify --no-build-isolation --link-mode=copy
uv pip install transformers==4.47.1 deepspeed
source briter/bin/activate
export HF_TOKEN=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
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY
MODEL_NAME=Qwen/Qwen2.5-7B
PROJECT_NAME=Qwen2.5-7B_Mix-Math-yt
EXPERIMENT_NAME=${MODEL_NAME}-rs
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}

for CHECKPOINT in ${SAVE_LOCAL_DIR}/global_step_*; do
    STEP=$(basename $CHECKPOINT)  # Extracts "global_step_X"
    HUB_MODEL_ID="Yuanxin-Liu/mix-math-7b-Qwen-rs-baseline-${STEP}"

    echo "Pushing checkpoint: $STEP to Hugging Face Hub at $HUB_MODEL_ID"

    python sft_scripts/3-26/push_to_hub.py \
        --model_name_or_path ${MODEL_NAME} \
        --adapter_path ${CHECKPOINT} \
        --hub_model_id ${HUB_MODEL_ID}
done