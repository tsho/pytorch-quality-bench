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

  def __init__(self, hidden: int = 1024):
    """Build three stacked linear layers of width ``hidden``.

    Args:
      hidden: Feature width of every linear layer (input and output).
    """
    super().__init__()
    self.l1 = nn.Linear(hidden, hidden)
    self.l2 = nn.Linear(hidden, hidden)
    self.l3 = nn.Linear(hidden, hidden)

  def forward(self, data: torch.Tensor) -> torch.Tensor:
    """Apply the linear layers with ReLU activations.

    Args:
      data: Input tensor of shape ``(batch, hidden)``.

    Returns:
      Output tensor of shape ``(batch, hidden)``.
    """
    data = torch.relu(self.l1(data))
    data = torch.relu(self.l2(data))
    return self.l3(data)


class BreakProneModel(nn.Module):
  """Data-dependent control flow forces graph breaks on every forward."""

  def __init__(self, hidden: int = 1024):
    """Build three stacked linear layers of width ``hidden``.

    Args:
      hidden: Feature width of every linear layer (input and output).
    """
    super().__init__()
    self.l1 = nn.Linear(hidden, hidden)
    self.l2 = nn.Linear(hidden, hidden)
    self.l3 = nn.Linear(hidden, hidden)

  def forward(self, data: torch.Tensor) -> torch.Tensor:
    """Run the layers with data-dependent control flow that breaks graphs.

    Each marked ``BAD`` block forces torch.compile to split the graph: a
    data-dependent branch, Python-level iteration over tensor values, and a
    logging side effect in the hot path.

    Args:
      data: Input tensor of shape ``(batch, hidden)``.

    Returns:
      Output tensor of shape ``(batch, hidden)``, scaled by a Python float
      derived from the data (one of the break sources).
    """
    data = torch.relu(self.l1(data))

    # BAD #1: data-dependent branch
    if data.sum().item() > 0:
      data = self.l2(data)
    else:
      data = -self.l2(data)

    # BAD #2: Python-level iteration over tensor values
    scale = 1.0
    for v in data.mean(dim=0).tolist()[:3]:
      scale *= 1.0 + v * 1e-4

    # BAD #3: log / side effect in hot path
    if data.numel() > LARGE_TENSOR_NUMEL:
      logger.warning("(unlikely branch)")

    return self.l3(data) * scale


# --- Benchmark helpers ------------------------------------------------------


def benchmark(
  model: nn.Module,
  data: torch.Tensor,
  n_warmup: int = 5,
  n_iter: int = 20,
) -> float:
  """Measure the mean forward-pass latency of ``model``.

  Runs ``n_warmup`` un-timed forwards first (lets torch.compile finish
  compiling and caches warm up), then times ``n_iter`` forwards. On CUDA,
  synchronizes around the timed region so kernel launches are not measured
  asynchronously.

  Args:
    model: Model to benchmark (eager or torch.compile-wrapped).
    data: Input tensor passed to every forward call.
    n_warmup: Number of un-timed warmup iterations.
    n_iter: Number of timed iterations to average over.

  Returns:
    Mean latency of one forward pass, in milliseconds.
  """
  for _ in range(n_warmup):
    _ = model(data)
  if DEVICE == "cuda":
    torch.cuda.synchronize()
  t0 = time.perf_counter()
  for _ in range(n_iter):
    _ = model(data)
  if DEVICE == "cuda":
    torch.cuda.synchronize()
  return (time.perf_counter() - t0) / n_iter * 1000  # ms


def explain_breaks(model: nn.Module, data: torch.Tensor):
  """Trace ``model`` with torch._dynamo.explain and report graph breaks.

  Args:
    model: Un-compiled model to analyze.
    data: Example input used for tracing.

  Returns:
    A ``torch._dynamo`` ExplainOutput with ``graph_break_count``,
    ``graph_count``, and ``op_count`` attributes.
  """
  report = dynamo.explain(model)(data)
  return report


def main() -> dict[str, float]:
  """Benchmark both models eager vs compiled and inspect graph breaks.

  Returns:
    Latency results in ms (``eager_clean_ms``, ``compiled_clean_ms``,
    ``eager_breaky_ms``, ``compiled_breaky_ms``) plus ``graph_breaks``,
    the break count reported by torch._dynamo.explain.
  """
  logger.info("Device: %s", DEVICE)
  data = torch.randn(32, 1024, device=DEVICE)

  clean = CleanModel().to(DEVICE)
  breaky = BreakProneModel().to(DEVICE)

  # --- Clean model -------------------------------------------------------
  eager_clean = benchmark(clean, data)
  compiled_clean = benchmark(torch.compile(clean), data)

  # --- Break-prone model -------------------------------------------------
  eager_breaky = benchmark(breaky, data)
  compiled_breaky = benchmark(torch.compile(breaky), data)

  # --- Graph break inspection -------------------------------------------
  logger.info("=== Graph break analysis (break-prone model) ===")
  rep = explain_breaks(BreakProneModel().to(DEVICE), data)
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
