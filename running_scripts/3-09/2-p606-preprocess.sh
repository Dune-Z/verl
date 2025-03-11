

export EXPERIMENT_NAME=qwen-7b-math-606
export VLLM_ATTENTION_BACKEND=XFORMERS

export WANDB_API_KEY=d61cd005c38e0e1e27d921c951303410316ac718
wandb login --relogin $WANDB_API_KEY

sft_loss_coef=0
REMOTE_DATA_PATH=PRIME-RL/Eurus-2-RL-Data


SAVE_LOCAL_DIR_PREFIX='checkpoints/'
PROJECT_NAME=Qwen2.5-7B_Mix-Math
MODEL_NAME=Qwen/Qwen2.5-7B
EXPERIMENT_NAME=grpo
SAVE_LOCAL_DIR=/checkpoints/hongpaul-sandbox/r1/${PROJECT_NAME}/${EXPERIMENT_NAME}

python3 data_preprocess/math_r1_dataset.py
python3 data_preprocess/still_30k.py
python3 data_preprocess/aime_train_dataset.py
python3 data_preprocess/create_math_data_mix.py
python3 data_preprocess/aime_24_dataset.py
python3 data_preprocess/math_r1_500.py
