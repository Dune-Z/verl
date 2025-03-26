set -x
export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY

MODEL_NAME=Open-Reasoner-Zero/Open-Reasoner-Zero-7B

python3 data_preprocess/math_r1_dataset.py
python3 data_preprocess/still_30k.py
python3 data_preprocess/aime_train_dataset.py
python3 data_preprocess/create_math_data_mix.py
python3 data_preprocess/aime_24_dataset.py
python3 data_preprocess/math_r1_500.py

if [ -d "./data/mix-math" ]; then
    rm -rf ./data/mix-math
fi
python3 data_preprocess/create_math_data_mix.py --local_dir data/mix/train.parquet
python3 data_preprocess/orz_dataset.py --local_dir ./data/orz_dataset --sample_start_idx 0 --sample_end_idx 129


CUDA_VISIBLE_DEVICES=0,1 python3 -m verl.trainer.main_generation \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=2 \
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

