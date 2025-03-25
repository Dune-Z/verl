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

PROJECT_NAME=generation_and_sft
MODEL_NAME=Open-Reasoner-Zero/Open-Reasoner-Zero-7B
EXPERIMENT_NAME=${MODEL_NAME}-sft-generation
SAVE_LOCAL_DIR_PREFIX='checkpoints/'
SAVE_LOCAL_DIR=${SAVE_LOCAL_DIR_PREFIX}${PROJECT_NAME}/${EXPERIMENT_NAME}

# if ./data/orz_dataset exists, remove it
if [ -d "./data/orz_dataset" ]; then
    rm -rf ./data/orz_dataset
fi
python3 data_preprocess/orz_dataset.py --local_dir ./data/orz_dataset

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=8 \
    data.path=./data/orz_dataset/train.parquet \
    data.prompt_key=prompt \
    data.n_samples=1 \
    data.output_path=./data/orz_dataset/generation.parquet \
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
    --datafiles ./data/orz_dataset/generation.parquet \
    --hub Ethan-Z/orz-generation-orz

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --standalone --nproc_per_node=8 --nnodes=1 -m verl.trainer.fsdp_sft_trainer \
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
        optim.lr=1e-6 \
        ulysses_sequence_parallel_size=2 \
        use_remove_padding=true