# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT is a minimal PyTorch implementation for training and finetuning GPT models. The codebase is intentionally simple: ~300 lines for training (`train.py`) and ~300 lines for the model (`model.py`).

**Note:** This repo is deprecated in favor of [nanochat](https://github.com/karpathy/nanochat).

## Common Commands

### Data Preparation
```bash
# Character-level Shakespeare (quick start)
python data/shakespeare_char/prepare.py

# BPE-tokenized Shakespeare (for finetuning GPT-2)
python data/shakespeare/prepare.py

# OpenWebText (large dataset, takes time)
python data/openwebtext/prepare.py
```

### Training
```bash
# Single GPU training with config file
python train.py config/train_shakespeare_char.py

# Single GPU with command-line overrides
python train.py config/train_shakespeare_char.py --device=mps --compile=False

# Multi-GPU training with DDP (8 GPUs)
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

# CPU training (for macbooks without MPS)
python train.py config/train_shakespeare_char.py --device=cpu --compile=False
```

### Sampling/Inference
```bash
# Sample from trained model
python sample.py --out_dir=out-shakespeare-char

# Sample from pretrained GPT-2
python sample.py --init_from=gpt2-xl --start="Hello, world"

# Sample with prompt from file
python sample.py --start=FILE:prompt.txt
```

### Evaluation/Benchmarking
```bash
# Evaluate pretrained GPT-2 models
python train.py config/eval_gpt2.py

# Benchmark training loop performance
python bench.py
```

## Architecture

### Core Files
- **model.py**: Complete GPT model definition with `GPTConfig` dataclass and `GPT` nn.Module. Includes:
  - `CausalSelfAttention`: Multi-head attention with Flash Attention support (PyTorch 2.0+)
  - `MLP`: Feed-forward network with GELU activation
  - `Block`: Transformer block (LayerNorm -> Attention -> LayerNorm -> MLP)
  - `GPT.from_pretrained()`: Load OpenAI GPT-2 weights
  - `GPT.generate()`: Autoregressive text generation

- **train.py**: Training loop with DDP support, gradient accumulation, mixed precision, and wandb logging

- **sample.py**: Text generation from trained checkpoints or pretrained GPT-2

- **configurator.py**: Simple config system that allows overriding any global variable via config files or `--key=value` CLI args

### Configuration System
Config files in `config/` are plain Python that override global variables. Use them as:
```bash
python train.py config/train_shakespeare_char.py --batch_size=32
```
The config file runs first, then CLI args override. Key training parameters:
- `n_layer`, `n_head`, `n_embd`: Model architecture
- `batch_size`, `block_size`: Training batch and context length
- `learning_rate`, `max_iters`, `warmup_iters`: Optimization
- `init_from`: `'scratch'`, `'resume'`, or `'gpt2'`/`'gpt2-medium'`/`'gpt2-large'`/`'gpt2-xl'`
- `device`: `'cuda'`, `'cpu'`, or `'mps'`
- `compile`: Enable torch.compile (requires PyTorch 2.0)

### Data Format
Training data is stored as memory-mapped numpy arrays of token IDs:
- `data/<dataset>/train.bin`: Training tokens (uint16)
- `data/<dataset>/val.bin`: Validation tokens (uint16)
- `data/<dataset>/meta.pkl`: Optional vocab metadata (for char-level models)

## Key Implementation Details

- Weight tying between token embeddings and output projection (`lm_head`)
- Cosine learning rate decay with linear warmup
- AdamW optimizer with separate weight decay groups (2D params decay, others don't)
- GradScaler for float16 training
- Model checkpoints saved to `out_dir/ckpt.pt` with model state, optimizer state, and config
