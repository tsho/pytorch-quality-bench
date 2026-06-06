"""Experiment 1: torch.compile graph breaks demonstration.

Goal: Show what happens when torch.compile encounters Python control flow that
forces graph breaks, and how to detect them.

Expected outcome for Abstract:
  "We saw a model with 7 graph breaks run *slower* than eager mode despite
  compile."

Run:
  python exp1-compile-graph-breaks.py

Optional verbose graph-break logs:
  TORCH_LOGS=graph_breaks python exp1-compile-graph-breaks.py
"""

import logging
import time

import torch
import torch._dynamo as dynamo
from torch import nn

logger = logging.getLogger(__name__)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Element-count threshold above which the unlikely warning branch fires.
LARGE_TENSOR_NUMEL = 10_000_000


# --- Two models: one clean, one break-prone ---------------------------------


class CleanModel(nn.Module):
  """Pure tensor ops — torch.compile produces a single graph."""

  def __init__(self, hidden=1024):
    """Build three stacked linear layers of width ``hidden``."""
    super().__init__()
    self.l1 = nn.Linear(hidden, hidden)
    self.l2 = nn.Linear(hidden, hidden)
    self.l3 = nn.Linear(hidden, hidden)

  def forward(self, x):
    """Apply the linear layers with ReLU activations and return the output."""
    x = torch.relu(self.l1(x))
    x = torch.relu(self.l2(x))
    return self.l3(x)


class BreakProneModel(nn.Module):
  """Data-dependent control flow forces graph breaks on every forward."""

  def __init__(self, hidden=1024):
    """Build three stacked linear layers of width ``hidden``."""
    super().__init__()
    self.l1 = nn.Linear(hidden, hidden)
    self.l2 = nn.Linear(hidden, hidden)
    self.l3 = nn.Linear(hidden, hidden)

  def forward(self, x):
    """Run the layers with data-dependent control flow that breaks graphs."""
    x = torch.relu(self.l1(x))

    # BAD #1: data-dependent branch
    if x.sum().item() > 0:
      x = self.l2(x)
    else:
      x = -self.l2(x)

    # BAD #2: Python-level iteration over tensor values
    scale = 1.0
    for v in x.mean(dim=0).tolist()[:3]:
      scale *= 1.0 + v * 1e-4

    # BAD #3: log / side effect in hot path
    if x.numel() > LARGE_TENSOR_NUMEL:
      logger.warning("(unlikely branch)")

    return self.l3(x) * scale


# --- Benchmark helpers ------------------------------------------------------


def benchmark(model, x, n_warmup=5, n_iter=20):
  """Return mean forward latency in ms after warmup, over ``n_iter`` runs."""
  for _ in range(n_warmup):
    _ = model(x)
  if DEVICE == "cuda":
    torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(n_iter):
    _ = model(x)
  if DEVICE == "cuda":
    torch.cuda.synchronize()
  return (time.perf_counter() - t0) / n_iter * 1000  # ms


def explain_breaks(model, x):
  """torch._dynamo.explain returns a structured report of graph breaks."""
  report = dynamo.explain(model)(x)
  return report


def main():
  """Benchmark both models, inspect graph breaks, and return the results."""
  logger.info("Device: %s", DEVICE)
  x = torch.randn(32, 1024, device=DEVICE)

  clean = CleanModel().to(DEVICE)
  breaky = BreakProneModel().to(DEVICE)

  # --- Clean model -------------------------------------------------------
  eager_clean = benchmark(clean, x)
  compiled_clean = benchmark(torch.compile(clean), x)

  # --- Break-prone model -------------------------------------------------
  eager_breaky = benchmark(breaky, x)
  compiled_breaky = benchmark(torch.compile(breaky), x)

  # --- Graph break inspection -------------------------------------------
  logger.info("=== Graph break analysis (break-prone model) ===")
  rep = explain_breaks(BreakProneModel().to(DEVICE), x)
  logger.info("  Graph breaks reported: %s", rep.graph_break_count)
  logger.info("  Graphs produced:       %s", rep.graph_count)
  logger.info("  Op count (total):      %s", rep.op_count)

  # --- Report ------------------------------------------------------------
  logger.info("=== Latency (ms / forward) ===")
  logger.info("  Clean    eager    : %7.3f", eager_clean)
  logger.info(
    "  Clean    compiled : %7.3f  (%.2fx speedup)",
    compiled_clean,
    eager_clean / compiled_clean,
  )
  logger.info("  Break    eager    : %7.3f", eager_breaky)
  logger.info(
    "  Break    compiled : %7.3f  (%.2fx speedup)",
    compiled_breaky,
    eager_breaky / compiled_breaky,
  )

  logger.info("=== Takeaway ===")
  if compiled_breaky >= eager_breaky:
    logger.warning("  ⚠️  torch.compile SLOWER than eager due to graph breaks.")
  else:
    logger.info(
      "  ✅ compile won by %.2fx even with breaks.",
      eager_breaky / compiled_breaky,
    )

  return {
    "eager_clean_ms": eager_clean,
    "compiled_clean_ms": compiled_clean,
    "eager_breaky_ms": eager_breaky,
    "compiled_breaky_ms": compiled_breaky,
    "graph_breaks": rep.graph_break_count,
  }


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
