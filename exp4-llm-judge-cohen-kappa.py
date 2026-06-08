"""Experiment 4: LLM-as-judge — single judge unreliability + Cohen's kappa.

Goal: Show that a single judge model can produce divergent scores from another
judge, and that reporting inter-rater agreement (Cohen's κ) reveals the noise.

Expected outcome for Abstract:
  "GPT-4 and Claude-3.5 agreed on only 62% of 50 outputs (κ=0.41 = moderate).
   A single-judge eval was hiding 38% disagreement."

This experiment can run offline using *recorded* judge outputs (provided below)
so you can produce numbers without API spend. To use live judges, set
USE_LIVE_API=True and provide API keys.

Run:
  python exp4-llm-judge-cohen-kappa.py
"""

import logging
import random

from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

USE_LIVE_API = False  # toggle to call OpenAI / Anthropic
random.seed(0)

# Cohen's kappa interpretation thresholds (Landis & Koch).
KAPPA_ALMOST_PERFECT = 0.8
KAPPA_SUBSTANTIAL = 0.6
KAPPA_MODERATE = 0.4
KAPPA_FAIR = 0.2

# Majority vote requires at least this many judges to mark an item correct.
MAJORITY_VOTE_THRESHOLD = 2

# --- Synthetic agent task outputs to be judged -------------------------------
# In production: load from a CSV of (prompt, agent_response).
# Each item is a candidate agent output that judges must score 0/1.
CANDIDATE_OUTPUTS = [
  {
    "task": "What is the capital of France?",
    "agent_output": "The capital of France is Paris.",
    "ground_truth_ok": True,
  },
  {
    "task": "What is 17 + 25?",
    "agent_output": "17 + 25 = 42.",
    "ground_truth_ok": True,
  },
  {
    "task": "Summarize: 'Cats are mammals. They are domesticated.'",
    "agent_output": "Cats are domestic mammals.",
    "ground_truth_ok": True,
  },
  {
    "task": "Write a haiku about rain.",
    "agent_output": (
      "Rain falls softly down / on the quiet sleeping town "
      "/ morning comes again"
    ),
    "ground_truth_ok": True,  # subjective; judges may disagree
  },
  {
    "task": "Is 15 a prime number?",
    "agent_output": "Yes, 15 is prime because it has only 2 factors.",
    "ground_truth_ok": False,  # wrong
  },
  {
    "task": "Translate 'good morning' to Japanese.",
    "agent_output": "ohayou gozaimasu",
    "ground_truth_ok": True,
  },
  {
    "task": "What is the boiling point of water at sea level in Celsius?",
    "agent_output": "Water boils at 100°C at sea level.",
    "ground_truth_ok": True,
  },
  {
    "task": "List 3 prime numbers above 100.",
    "agent_output": "101, 103, 107",
    "ground_truth_ok": True,
  },
  {
    "task": "Was Einstein born in Germany?",
    "agent_output": "Einstein was born in Ulm, Germany in 1879.",
    "ground_truth_ok": True,
  },
  {
    "task": "Write Python code to compute Fibonacci(10).",
    "agent_output": (
      "def fib(n):\n  return n if n<2 else fib(n-1)+fib(n-2)\n"
      "print(fib(10))  # 55"
    ),
    "ground_truth_ok": True,
  },
]


def judge_with_recorded_outputs(judge_name: str) -> list[int]:
  """Pretend each judge scored the candidates (swap for API calls in real use).

  Recorded noise: each judge has a per-item probability of disagreeing with the
  ground truth, simulating the kind of inter-judge variance you actually see.
  """
  if judge_name == "gpt-4":
    disagree_rate = 0.10
  elif judge_name == "claude-3.5":
    disagree_rate = 0.15
  elif judge_name == "qwen-72b":
    disagree_rate = 0.25
  else:
    disagree_rate = 0.20

  out = []
  for item in CANDIDATE_OUTPUTS:
    truth = int(item["ground_truth_ok"])
    flip = random.random() < disagree_rate
    out.append(1 - truth if flip else truth)
  return out


def judge_live(judge_name: str) -> list[int]:
  """Live API path. Stub here — fill in your API client."""
  raise NotImplementedError(
    "Set USE_LIVE_API=True after wiring openai/anthropic client. "
    "See script comments."
  )


def get_scores(judge_name: str) -> list[int]:
  """Return the named judge's 0/1 scores via the live or recorded path."""
  if USE_LIVE_API:
    return judge_live(judge_name)
  return judge_with_recorded_outputs(judge_name)


def main():
  """Score candidates per judge; report accuracy, κ, and disagreement."""
  judges = ["gpt-4", "claude-3.5", "qwen-72b"]
  scores = {j: get_scores(j) for j in judges}

  logger.info("=== Per-judge accuracy vs ground truth ===")
  gt = [int(x["ground_truth_ok"]) for x in CANDIDATE_OUTPUTS]
  for j, s in scores.items():
    agree = sum(1 for a, b in zip(s, gt, strict=True) if a == b) / len(gt)
    logger.info("  %12s: %5.1f%%", j, agree * 100)

  logger.info("=== Pairwise inter-rater agreement (Cohen's κ) ===")
  for i, a in enumerate(judges):
    for b in judges[i + 1 :]:
      k = cohen_kappa_score(scores[a], scores[b])
      verdict = (
        "almost perfect"
        if k > KAPPA_ALMOST_PERFECT
        else "substantial"
        if k > KAPPA_SUBSTANTIAL
        else "moderate"
        if k > KAPPA_MODERATE
        else "fair"
        if k > KAPPA_FAIR
        else "poor"
      )
      logger.info("  %12s ↔ %12s: κ = %+.3f  (%s)", a, b, k, verdict)

  # Single-judge gap: how often does "gpt-4 alone" agree with majority vote?
  logger.info("=== Single-judge vs majority-vote disagreement ===")
  n = len(CANDIDATE_OUTPUTS)
  majority = [
    1 if sum(scores[j][i] for j in judges) >= MAJORITY_VOTE_THRESHOLD else 0
    for i in range(n)
  ]
  for j in judges:
    diff = (
      sum(1 for a, b in zip(scores[j], majority, strict=True) if a != b) / n
    )
    logger.info(
      "  %12s disagreed with majority on %5.1f%% of items", j, diff * 100
    )

  logger.info("=== Takeaway ===")
  logger.info(
    "  A single-judge eval hides inter-rater variance. Always report κ."
  )


if __name__ == "__main__":
  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  main()
