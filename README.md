# pytorch-quality-bench

PyTorch quality benchmark experiments. Runs on any CUDA GPU; the reference
numbers in this repo were measured on an A10G 22GB (e.g. AWS g5, SPCS).
Five experiments cover torch.compile graph breaks, INT4 quantization quality
regression, KV-cache OOM, LLM-as-judge agreement, and vLLM vs torch.compile.

| Script | Topic | Main flags |
|--------|-------|------------|
| `exp1-compile-graph-breaks.py` | torch.compile running slower than eager due to graph breaks | none |
| `exp2-int4-quality-regression.py` | FP16 vs INT4 (torchao) quality comparison | `--model`, `--n_tasks` |
| `exp3-kv-cache-oom.py` | Context-length sweep to find the KV-cache OOM point | `--model`, `--quantize`, `--contexts` |
| `exp4-llm-judge-cohen-kappa.py` | Cohen's kappa between LLM judges (offline recorded data) | none |
| `exp5-vllm-vs-compile.py` | vLLM vs torch.compile native throughput | `--runtime`, `--model`, `--batch`, `--max_new_tokens` |

## Setup

```bash
# Local Mac (installs CPU/MPS wheels)
uv sync

# HF token required for Gemma / Llama
huggingface-cli login
```

GPU runs use Docker (NGC container) and work on any host with the NVIDIA
Container Toolkit — a cloud instance, an on-prem box, or a container platform
such as SPCS. Do **not** run `uv sync` inside the container — it would clobber
the NGC-optimized torch build.

```bash
# Image for exp1-4
docker build -f Dockerfile.exp -t pqb-exp .

# Image for exp5 (vLLM) — vllm pins its own torch, so it gets a separate image
docker build -f Dockerfile.vllm -t pqb-vllm .

# Example run
docker run --rm --gpus all pqb-exp python exp1-compile-graph-breaks.py
```

## Run samples

### exp1: torch.compile graph breaks

```bash
python exp1-compile-graph-breaks.py

# With verbose graph-break logs
TORCH_LOGS=graph_breaks python exp1-compile-graph-breaks.py
```

### exp2: INT4 quantization quality regression (FP16 vs INT4)

```bash
# Default: Qwen2.5-3B (fast iteration), 20 tasks
python exp2-int4-quality-regression.py

# Qwen2.5-7B
python exp2-int4-quality-regression.py --model Qwen/Qwen2.5-7B-Instruct --n_tasks 50

# Gemma-2-9B
python exp2-int4-quality-regression.py --model google/gemma-2-9b-it --n_tasks 50

# Llama-3.1-8B
python exp2-int4-quality-regression.py --model meta-llama/Llama-3.1-8B-Instruct --n_tasks 50
```

### exp3: KV-cache OOM sweep

```bash
# Default: Llama-3.1-8B FP16, contexts = 1K 2K 4K 8K 16K 32K
python exp3-kv-cache-oom.py

# Re-sweep with torchao INT4 to free headroom
python exp3-kv-cache-oom.py --quantize

# Qwen2.5-7B
python exp3-kv-cache-oom.py --model Qwen/Qwen2.5-7B-Instruct
python exp3-kv-cache-oom.py --model Qwen/Qwen2.5-7B-Instruct --quantize

# Gemma-2-9B is expected to OOM in FP16 -> INT4 required
python exp3-kv-cache-oom.py --model google/gemma-2-9b-it --quantize

# Custom context lengths
python exp3-kv-cache-oom.py --contexts 1024 4096 16384
```

### exp4: LLM-as-judge Cohen's kappa

```bash
# Offline analysis of recorded judge outputs (no GPU needed)
python exp4-llm-judge-cohen-kappa.py
```

### exp5: vLLM vs torch.compile native

```bash
# torch.compile native (works in either image, pqb-exp or pqb-vllm)
python exp5-vllm-vs-compile.py --runtime torch

# vLLM (only available in the pqb-vllm image)
python exp5-vllm-vs-compile.py --runtime vllm

# Custom model / batch (defaults: Qwen2.5-7B, batch=8, max_new_tokens=128)
python exp5-vllm-vs-compile.py --runtime vllm --model meta-llama/Llama-3.1-8B-Instruct --batch 16
```

### Run everything (run_all.sh)

Runs each script in sequence, captures per-script logs under
`logs/<timestamp>/<script>.log`, and prints a pass/fail summary.

```bash
# Default list (exp1-4 + exp5 --runtime torch)
./run_all.sh

# Run specific commands (one argument = one command)
./run_all.sh "exp3-kv-cache-oom.py --model google/gemma-2-9b-it --quantize"

# Inside the vllm image, always pass arguments (exp1-4 files are absent there)
./run_all.sh "exp5-vllm-vs-compile.py --runtime vllm"

# Local Mac via uv
PYTHON="uv run python" ./run_all.sh "exp4-llm-judge-cohen-kappa.py"
```

## VRAM cheat sheet (example: A10G 22GB)

Figures below are for an A10G 22GB; GPUs with more VRAM (L4 24GB, A100, etc.)
have correspondingly more headroom.

| Model | FP16 weights | INT4 weights | A10G 22GB @ 16K ctx (FP16) |
|-------|-------------|-------------|----------------------------|
| Qwen2.5-7B | 15.23 GB | 5.28 GB | ✅ 21.28/22 GB (measured) |
| Gemma-2-9B | ~17 GB (est) | ~6 GB (est) | ⚠️ FP16 OOM risk -> **INT4 required** |
| Llama-3.1-8B | ~16 GB (est) | ~5.5 GB (est) | ⚠️ 18–20 GB (est) |

A full 3-model comparison takes roughly 3–4 hours on a single A10G.
