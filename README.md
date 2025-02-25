# BRiTE-R verl

## Installation

Install `uv` via [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).  

```sh
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
```

Next, install vllm

```sh
uv pip install vllm==0.7.1 --link-mode=copy
```

This will also install PyTorch `v2.5.1` and it is very important to use this version since the `vLLM` binaries are compiled for it. Then install the rest of the dependencies by running the following command:
```sh
uv pip install -r requirements.txt --link-mode=copy
```

At last, install `flash_attn` by running the following command:

```sh
uv pip install flash_attn --no-build-isolation --link-mode=copy
```

## Example

To run an example experiment, you can run the following command:

```sh
bash running_scripts/brite_run_qwen2.5-math-7b.sh
```

