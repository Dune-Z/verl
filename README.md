# BRiTE-R verl

## Installation

Install `uv` in BRiTER verl directory. [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).  

```sh
git clone https://github.com/Dune-Z/verl.git && cd verl
uv venv briter --python 3.11 && source briter/bin/activate && uv pip install --upgrade pip --link-mode=copy
```

Then install the rest of the dependencies.

```sh
uv pip install -r requirements.txt --link-mode=copy
```

At last, install `flash_attn`.

```sh
uv pip install flash_attn --no-build-isolation --link-mode=copy
```

## Example

To run an example experiment, you can run the following command:

```sh
bash running_scripts/brite_run_qwen2.5-math-7b.sh
```

