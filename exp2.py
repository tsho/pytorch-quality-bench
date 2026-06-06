"""Experiment 2: INT4 (torchao) quality regression vs FP16.

Goal: Show that aggressive quantization can hurt downstream task accuracy
even when perplexity looks fine.

Expected outcome for Abstract:
  "INT4 saved 65% of VRAM but dropped GSM8K accuracy from 42% → 28%."

We use Qwen2.5-3B-Instruct as a small but realistic model that fits on A10G.
For 8B models swap MODEL_ID and lower BATCH if needed.

Run:
  python exp2-int4-quality-regression.py --n_tasks 50
"""

import argparse
import gc
import logging
import time

import torch

# torchao quantization API
from torchao.quantization import int4_weight_only, quantize_
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

torch.manual_seed(42)

MODEL_ID = (
  "Qwen/Qwen2.5-3B-Instruct"  # 3B for fast iteration; swap to 7B/8B for talk
)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16

# Number of initial eval examples to log verbosely for inspection.
LOG_SAMPLE_LIMIT = 5

# Tiny GSM8K-style probe set. For talk: load actual GSM8K from datasets.
# Keep deterministic and self-contained for repro.
PROBE = [
  {
    "q": "If Alice has 3 apples and buys 4 more, how many does she have?",
    "a": "7",
  },
  {
    "q": "A book costs $12. Bob has $50. How many books can Bob buy?",
    "a": "4",
  },
  {
    "q": "A train travels 60 km in 30 min. What is the speed in km/h?",
    "a": "120",
  },
  {
    "q": "5! = ? (Compute 5 factorial)",
    "a": "120",
  },
  {
    "q": "What is 17 + 25?",
    "a": "42",
  },
  # Add more for talk; this script accepts --n_tasks to repeat probe.
]


def load_model(quantize: bool):
  """Load the model and tokenizer, optionally quantizing weights to INT4."""
  tok = AutoTokenizer.from_pretrained(MODEL_ID)
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    device_map=DEVICE,
  )
  label = "FP16/BF16"
  if quantize:
    quantize_(model, int4_weight_only())
    label = "INT4 (torchao)"
  return model, tok, label


def generate_answer(model, tok, question: str, max_new_tokens=128) -> str:
  """Greedily generate the model's answer text for a single question."""
  prompt = (
    "Solve the problem. Give only the final numeric answer on the last line.\n"
    f"Q: {question}\nA: "
  )
  inputs = tok(prompt, return_tensors="pt").to(DEVICE)
  with torch.no_grad():
    out = model.generate(
      **inputs,
      max_new_tokens=max_new_tokens,
      do_sample=False,
      pad_token_id=tok.eos_token_id,
    )
  return tok.decode(
    out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
  )


def grade(answer: str, expected: str) -> bool:
  """Return True if the expected number appears among the answer's digits."""
  # Liberal grading: match any digit sequence containing expected.
  digits = "".join(ch if ch.isdigit() else " " for ch in answer).split()
  return expected in digits


def run_eval(model, tok, n_tasks: int):
  """Evaluate the model over n_tasks probes and return accuracy and timing."""
  correct = 0
  total = 0
  latencies = []
  for i in range(n_tasks):
    task = PROBE[i % len(PROBE)]
    t0 = time.perf_counter()
    ans = generate_answer(model, tok, task["q"])
    latencies.append((time.perf_counter() - t0) * 1000)
    ok = grade(ans, task["a"])
    correct += int(ok)
    total += 1
    if i < LOG_SAMPLE_LIMIT:
      logger.info(
        "  [%s] Q: %s\n      A: %s  expected=%s  ok=%s",
        i,
        task["q"],
        ans.strip()[:120],
        task["a"],
        ok,
      )
  return {
    "accuracy": correct / total,
    "p50_ms": sorted(latencies)[len(latencies) // 2],
    "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
    "vram_peak_gb": torch.cuda.max_memory_allocated() / 1e9
    if DEVICE == "cuda"
    else 0.0,
  }


def main():
  """Run the FP16 vs INT4 evaluation and log the regression report."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--n_tasks", type=int, default=20)
  args = parser.parse_args()

  results = {}

  # --- FP16 baseline -----------------------------------------------------
  logger.info("=== Loading %s as FP16/BF16 ===", MODEL_ID)
  torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
  model, tok, label = load_model(quantize=False)
  logger.info("VRAM after load: %.2f GB", torch.cuda.memory_allocated() / 1e9)
  logger.info("=== Eval: %s ===", label)
  results["fp16"] = run_eval(model, tok, args.n_tasks)
  logger.info("%s", results["fp16"])
  del model
  gc.collect()
  torch.cuda.empty_cache() if DEVICE == "cuda" else None

  # --- INT4 (torchao) ----------------------------------------------------
  logger.info("=== Loading %s and quantizing to INT4 ===", MODEL_ID)
  torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
  model, tok, label = load_model(quantize=True)
  logger.info(
    "VRAM after quantize: %.2f GB", torch.cuda.memory_allocated() / 1e9
  )
  logger.info("=== Eval: %s ===", label)
  results["int4"] = run_eval(model, tok, args.n_tasks)
  logger.info("%s", results["int4"])

  # --- Report ------------------------------------------------------------
  fp = results["fp16"]
  iq = results["int4"]
  logger.info("=== Regression report ===")
  logger.info(
    "  Accuracy:    FP16 %5.1f%%  →  INT4 %5.1f%%  (Δ %+.1fpp)",
    fp["accuracy"] * 100,
    iq["accuracy"] * 100,
    (iq["accuracy"] - fp["accuracy"]) * 100,
  )
  logger.info(
    "  P50 latency: FP16 %6.1fms →  INT4 %6.1fms",
    fp["p50_ms"],
    iq["p50_ms"],
  )
  logger.info(
    "  Peak VRAM:   FP16 %5.2f GB → INT4 %5.2f GB  (%.0f%% reduction)",
    fp["vram_peak_gb"],
    iq["vram_peak_gb"],
    (1 - iq["vram_peak_gb"] / fp["vram_peak_gb"]) * 100,
  )


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
