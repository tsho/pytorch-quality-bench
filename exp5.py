"""Experiment 5: vLLM vs torch.compile native — same model, different runtimes.

Goal: Show that two PyTorch-native serving stacks produce meaningfully different
throughput numbers on the same hardware + same model, and that the choice of
stack is a stronger lever than people assume.

Expected outcome for Abstract:
  "On Qwen2.5-7B at batch=8, vLLM hit 1,840 tok/s and torch.compile native hit
   1,210 tok/s — same hardware, same weights, 1.5x gap."

Run:
  # vLLM (separate venv recommended; vllm has tight torch version pins)
  python exp5-vllm-vs-compile.py --runtime vllm

  # torch.compile native
  python exp5-vllm-vs-compile.py --runtime torch
"""

import argparse
import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE = torch.bfloat16

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

PROMPTS = [
  "Explain in one sentence what a Fourier transform is.",
  "Write a Python function to reverse a linked list.",
  "What is the largest planet in the solar system?",
  "Translate 'good morning' to French.",
  "Summarize the plot of Hamlet in three sentences.",
  "Compute 17 * 23.",
  "What does the term 'gradient descent' mean in machine learning?",
  "Write a haiku about autumn leaves.",
] * 4  # 32 prompts


# --- torch.compile native path ---------------------------------------------


def benchmark_torch_native(max_new_tokens=128, batch=8):
  """Benchmark generation throughput using torch.compile native serving."""
  logger.info("Loading %s (torch.compile native)...", MODEL_ID)
  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
  )
  model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

  # Warmup
  sample = tok(PROMPTS[:batch], return_tensors="pt", padding=True).to(DEVICE)
  with torch.no_grad():
    _ = model.generate(
      **sample, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id
    )
  torch.cuda.synchronize()

  total_new_tokens = 0
  t0 = time.perf_counter()
  for i in range(0, len(PROMPTS), batch):
    chunk = PROMPTS[i : i + batch]
    inputs = tok(chunk, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
      out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
      )
    new_tokens = out.shape[1] - inputs["input_ids"].shape[1]
    total_new_tokens += new_tokens * out.shape[0]
  torch.cuda.synchronize()
  elapsed = time.perf_counter() - t0
  return {
    "runtime": "torch.compile native",
    "tokens": total_new_tokens,
    "seconds": elapsed,
    "tok_per_sec": total_new_tokens / elapsed,
  }


# --- vLLM path --------------------------------------------------------------


def benchmark_vllm(max_new_tokens=128, batch=8):
  """Benchmark generation throughput using the vLLM serving runtime."""
  # Lazy import: avoid importing vllm unless this code path runs.
  from vllm import LLM, SamplingParams  # noqa: PLC0415

  logger.info("Loading %s (vLLM)...", MODEL_ID)
  llm = LLM(model=MODEL_ID, dtype="bfloat16", gpu_memory_utilization=0.85)
  params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0)

  # Warmup
  _ = llm.generate(PROMPTS[:batch], params)

  total_new_tokens = 0
  t0 = time.perf_counter()
  outputs = llm.generate(PROMPTS, params)
  elapsed = time.perf_counter() - t0
  for o in outputs:
    total_new_tokens += len(o.outputs[0].token_ids)
  return {
    "runtime": "vLLM",
    "tokens": total_new_tokens,
    "seconds": elapsed,
    "tok_per_sec": total_new_tokens / elapsed,
  }


def main():
  """Parse CLI args, run the selected runtime benchmark, and log results."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--runtime", choices=["torch", "vllm"], required=True)
  parser.add_argument("--batch", type=int, default=8)
  parser.add_argument("--max_new_tokens", type=int, default=128)
  args = parser.parse_args()

  if args.runtime == "torch":
    r = benchmark_torch_native(args.max_new_tokens, args.batch)
  else:
    r = benchmark_vllm(args.max_new_tokens, args.batch)

  logger.info("=== Result ===")
  logger.info("  Runtime:    %s", r["runtime"])
  logger.info("  Tokens:     %s", r["tokens"])
  logger.info("  Elapsed:    %.2fs", r["seconds"])
  logger.info("  Throughput: %.1f tok/s", r["tok_per_sec"])
  logger.info(
    "Re-run with the other --runtime, then compute the ratio "
    "(vLLM tok/s) / (native tok/s) for the talk."
  )


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
