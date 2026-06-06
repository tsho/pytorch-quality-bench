"""Experiment 3: KV-cache OOM on A10G 24GB.

Goal: Show that an 8B model fits at FP16 on A10G 24GB *until* you ask for
realistic context lengths; the KV-cache eats the remaining VRAM.

Expected outcome for Abstract:
  "Llama-3-8B FP16 weights = 16 GB. At 8K context, KV-cache adds 9 GB
   → OOM on A10G. INT4 (torchao) frees enough headroom to reach 16K context."

Run:
  python exp3-kv-cache-oom.py --model meta-llama/Llama-3.1-8B-Instruct
  python exp3-kv-cache-oom.py --quantize  # use torchao INT4 to fit
"""

import argparse
import gc
import logging

import torch
from torchao.quantization import int4_weight_only, quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def vram_gb() -> float:
  """Return the currently allocated CUDA memory in gigabytes."""
  return torch.cuda.memory_allocated() / 1e9


def try_context(model, tok, n_tokens: int) -> dict:
  """Run a forward pass with `n_tokens` of context and report VRAM / OOM."""
  prompt = " ".join(["hello"] * n_tokens)[: n_tokens * 6]  # rough char→token
  inputs = tok(
    prompt, return_tensors="pt", truncation=True, max_length=n_tokens
  ).to(DEVICE)
  actual = inputs["input_ids"].shape[1]
  torch.cuda.reset_peak_memory_stats()
  try:
    with torch.no_grad():
      model(
        **inputs,
        use_cache=True,  # allocate full KV cache
        output_hidden_states=False,
      )
    return {
      "context": actual,
      "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9,
      "ok": True,
    }
  except torch.cuda.OutOfMemoryError as e:
    return {
      "context": actual,
      "vram_peak_gb": None,
      "ok": False,
      "err": str(e)[:120],
    }


def main():
  """Parse CLI args and sweep context lengths to find the KV-cache OOM point."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
  parser.add_argument(
    "--quantize",
    action="store_true",
    help="Apply torchao INT4 weight-only quantization",
  )
  parser.add_argument(
    "--contexts",
    nargs="+",
    type=int,
    default=[1024, 2048, 4096, 8192, 16384, 32768],
  )
  args = parser.parse_args()

  logger.info("Loading %s...", args.model)
  tok = AutoTokenizer.from_pretrained(args.model)
  model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype=DTYPE,
    device_map=DEVICE,
  )
  logger.info("Weights only: %.2f GB VRAM", vram_gb())

  if args.quantize:
    quantize_(model, int4_weight_only())
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("After INT4 quantize: %.2f GB VRAM", vram_gb())

  label = "INT4" if args.quantize else "FP16"
  logger.info("=== Context length sweep (%s) ===", label)
  rows = []
  for n in args.contexts:
    r = try_context(model, tok, n)
    if r["ok"]:
      logger.info(
        "  ctx=%6d  peak=%6.2f GB  ✅", r["context"], r["vram_peak_gb"]
      )
    else:
      logger.warning("  ctx=%6d  OOM ❌  (%s)", r["context"], r["err"])
    rows.append(r)

  last_ok = next((r["context"] for r in reversed(rows) if r["ok"]), None)
  logger.info("Max context that fit: %s tokens (%s)", last_ok, label)


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
