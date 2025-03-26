export HF_TOKEN=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
MODEL_NAME=Qwen/Qwen2.5-7B

python3 sft_scripts/debug/generation_hub.py \
    --datafiles ./data/mix-math/generation.parquet \
    --hub Yuanxin-Liu/mix-math-7b-rs-debug

python3 -m verl.trainer.main_filter