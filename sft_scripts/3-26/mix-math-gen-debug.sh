set -x
wget -qO- https://astral.sh/uv/install.sh | sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
uv pip install -r requirements.txt --link-mode=copy
uv pip install flash_attn --no-build-isolation --link-mode=copy
uv uninstall wandb
uv pip install wandb --no-cache-dir --link-mode=copy
uv pip install math_verify --no-build-isolation --link-mode=copy
uv pip install transformers==4.47.1 deepspeed
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY
export HF_PATH=Yuanxin-Liu/${PROJECT_NAME}-${EXPERIMENT_NAME}

MODEL_NAME=Qwen/Qwen2.5-7B

python3 data_preprocess/math_r1_dataset.py
python3 data_preprocess/still_30k.py
python3 data_preprocess/aime_train_dataset.py
python3 data_preprocess/create_math_data_mix.py
python3 data_preprocess/aime_24_dataset.py
python3 data_preprocess/math_r1_500.py

if [ -d "./data/mix-math" ]; then
    rm -rf ./data/mix-math
fi
python3 data_preprocess/create_math_data_mix.py --local_dir data/mix-math/train.parquet --sample_start_idx 0 --sample_end_idx 1024


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=./data/mix-math/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=2 \
    data.output_path=./data/mix-math/generation-debug.parquet \
    model.path=$MODEL_NAME \
    +model.trust_remote_code=True \
    rollout.temperature=1.0 \
    rollout.top_k=50 \
    rollout.top_p=0.7 \
    rollout.prompt_length=1024 \
    rollout.response_length=8192 \
    rollout.tensor_model_parallel_size=2 \
    rollout.gpu_memory_utilization=0.8

python3 sft_scripts/debug/generation_hub.py --push \
    --datafiles ./data/mix-math/generation-debug.parquet \
    --hub Yuanxin-Liu/mix-math-7b-Qwen-rs-debug