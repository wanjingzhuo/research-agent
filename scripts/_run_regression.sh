#!/usr/bin/env bash
set -uo pipefail
cd /home/wanjz/research-agent
source .venv/bin/activate

declare -a QUESTIONS=(
  "What is the price of gold today, and what moved it over the past month?"
  "What is the current price of Bitcoin, and what has driven its movement this week?"
  "What are the latest developments in AI regulation this month?"
  "今年的PSLE数学考纲跟去年相比有哪些变动？"
  "What is 1 + 1?"
)

for i in "${!QUESTIONS[@]}"; do
  n=$((i+1))
  q="${QUESTIONS[$i]}"
  echo "=================================================================="
  echo "[regression] starting run $n/5: $q"
  echo "=================================================================="
  python research_agent.py "$q" --eval > "_regression_logs/log_${n}.txt" 2>&1
  echo "[regression] finished run $n/5, exit code $?"
  tail -n 5 "_regression_logs/log_${n}.txt"
done

echo "[regression] ALL RUNS COMPLETE"
