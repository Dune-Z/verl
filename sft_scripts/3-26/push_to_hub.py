from transformers import AutoModelForCausalLM


def push_to_hub(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # load adaptor
