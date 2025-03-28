import argparse
from transformers import AutoModelForCausalLM


def push_to_hub(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.load_adapter(args.adapter_path)
    model.push_to_hub(
        repo_id=args.hub_model_id,
        commit_message="Add adapter",
        private=False,
    )

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Path to the pre-trained model or model identifier from Hugging Face Hub",
        default="Qwen/Qwen2.5-7B",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        required=True,
        help="Path to the adapter to be loaded into the model",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        required=True,
        help="The name of the model repository on Hugging Face Hub",
    )
    args = parser.parse_args()
    
    push_to_hub(args)
