# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

### Data pipeline


We use a BPE tokenizer with a vocabulary size of 10,000 for the TinyStories dataset.

Train BPE: uv run cs336_basics/train_bpe.py

Preprocess Text: Use the multi-processing script to convert .txt to memory-mapped .npy arrays.

```sh
uv run tokenize_data.py --input data/train.txt --output data/train.npy
```

## Training Experiments

### Model Specifications (17M Parameters)
Our "Base Model" follows these architectural constraints:
- **Dimensions**: $d_{model}=512$, $d_{ff}=1344$
- **Structure**: 4 layers, 16 attention heads
- **Context**: 256 tokens
- **Positional Encoding**: RoPE with $\theta=10000$

### Training Options

#### Option A: Low-Resource (Apple Silicon / MPS)
Optimized for MacBook M1/M2/M3 Pro/Max. This config processes ~40M tokens.
- **Target Val Loss**: < 2.00
- **Acceleration**: Enabled via `torch.compile(backend="aot_eager")`

```bash
uv run python cs336_basics/train.py --device mps --learning_rate 1e-3 --max_iters 5000
```

#### Option B: High-Performance (NVIDIA CUDA)
Recommended for H100, A40, or RTX 4090 to reach the full 327M tokens goal.
- **Target Val Loss**: < 1.45
- **Optimization**: TF32 matmul precision enabled

```bash
uv run python cs336_basics/train.py --device cuda --batch_size 128 --max_iters 10000 --learning_rate 1e-3
```