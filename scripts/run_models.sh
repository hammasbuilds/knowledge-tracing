#!/usr/bin/env bash
# The LLM arm, end to end: qwen2.5:14b-instruct predicts correctness from the
# answer-history text for the frozen sample in results/assist09/llm_jobs.jsonl,
# then it is scored against the classical models on exactly the same rows.
#
#   scripts/run_models.sh --dry-run   # job list, cached count, call estimate; no model
#   scripts/run_models.sh             # the real run (needs the GPU free)
#
# Every generation is cached under results/llm_cache/, so a killed run resumes.
set -euo pipefail
cd "$(dirname "$0")/.."
unset VIRTUAL_ENV

MODEL="${MODEL:-qwen2.5:14b-instruct}"
HOST="${OLLAMA_HOST_URL:-http://127.0.0.1:11434}"
OUT="results/assist09"
MIN_FREE_GB="${MIN_FREE_GB:-12}"
DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

[ -f "$OUT/llm_jobs.jsonl" ] || {
  echo "missing $OUT/llm_jobs.jsonl - build it with:" >&2
  echo "  uv run kt llm build --data data/raw/skill_builder_data_original.csv" >&2
  exit 1
}

echo "== job list"
echo "  1. kt llm run    model=$MODEL host=$HOST jobs=$(wc -l < "$OUT/llm_jobs.jsonl")"
echo "  2. kt llm score  -> $OUT/llm_arm.json"
plan=$(uv run --quiet kt llm run --model "$MODEL" --host "$HOST" --out "$OUT" --dry-run)
echo "$plan" | sed 's/^/  /'
calls=$(echo "$plan" | grep -o 'needed: [0-9]*' | grep -o '[0-9]*')
echo "  estimate: ${calls} calls x ~3 s (14B model, one consumer GPU) = ~$(( calls * 3 / 60 )) min"

if [ "$DRY" = 1 ]; then
  echo "(dry run: no model was called)"
  exit 0
fi

echo "== resource check"
free_gb=$(powershell -NoProfile -Command "[math]::Floor((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB)" 2>/dev/null \
  || awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)
echo "  free RAM: ${free_gb} GB (need ${MIN_FREE_GB})"
if [ "${free_gb:-0}" -lt "$MIN_FREE_GB" ]; then
  echo "not enough free RAM; refusing to start" >&2; exit 1
fi
if command -v nvidia-smi >/dev/null; then
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  echo "  GPU memory used: ${used}/${total} MiB"
  if [ "$used" -gt $(( total / 4 )) ]; then
    echo "GPU is busy (another job holds >25% of its memory); refusing to start" >&2; exit 1
  fi
fi
curl -s -m 10 "$HOST/api/tags" | grep -q "\"$MODEL\"" || {
  echo "ollama at $HOST does not list $MODEL (ollama pull $MODEL)" >&2; exit 1
}

echo "== run"
uv run kt llm run --model "$MODEL" --host "$HOST" --out "$OUT"
uv run kt llm score --model "$MODEL" --out "$OUT"
