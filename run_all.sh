#!/usr/bin/env bash
# Run every experiment script sequentially, capturing each one's output to its
# own log file under logs/<timestamp>/, and print a pass/fail summary.
#
# Usage:
#   ./run_all.sh                                  # run the default command list
#   ./run_all.sh "exp3-kv-cache-oom.py --quantize"  # run only the given command(s)
#   PYTHON="uv run python" ./run_all.sh           # local (Mac) run via uv
#
# Each positional argument is one full "script + flags" string. Exit code is
# non-zero if any script failed; failures do NOT stop the remaining scripts.
set -u

cd "$(dirname "$0")"

PYTHON_BIN="${PYTHON:-python}"
LOG_DIR="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

# exp5 --runtime vllm is intentionally absent from the defaults: vllm only
# exists in the Dockerfile.vllm image. Run it there with:
#   ./run_all.sh "exp5-vllm-vs-compile.py --runtime vllm"
DEFAULT_CMDS=(
  "exp1-compile-graph-breaks.py"
  "exp2-int4-quality-regression.py"
  "exp3-kv-cache-oom.py"
  "exp4-llm-judge-cohen-kappa.py"
  "exp5-vllm-vs-compile.py --runtime torch"
)

if [ "$#" -gt 0 ]; then
  CMDS=("$@")
else
  CMDS=("${DEFAULT_CMDS[@]}")
fi

SUMMARY=()
overall=0

for cmd in "${CMDS[@]}"; do
  script_name="${cmd%% *}"
  log_file="$LOG_DIR/${script_name%.py}.log"
  echo "==> $PYTHON_BIN $cmd"
  echo "    log: $log_file"
  start=$(date +%s)

  # Word-splitting of $cmd is intentional (script name + flags).
  # shellcheck disable=SC2086
  $PYTHON_BIN $cmd 2>&1 | tee "$log_file"
  rc=${PIPESTATUS[0]}

  elapsed=$(( $(date +%s) - start ))
  if [ "$rc" -eq 0 ]; then
    SUMMARY+=("OK    ${elapsed}s  $cmd")
  else
    SUMMARY+=("FAIL($rc)  ${elapsed}s  $cmd")
    overall=1
  fi
  echo
done

echo "================ summary ================"
printf '%s\n' "${SUMMARY[@]}"
echo "logs: $LOG_DIR/"
exit "$overall"
