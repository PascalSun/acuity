#!/bin/zsh
# Scaled register control: evaluate all 6 closed models on the rewritten set,
# plus gpt-4o-mini on the canonical forms (its canonical baseline never existed).
set -e
cd /Users/pascal/DrSun/KAIA/Talk2Metadata
PY=.venv/bin/python

for spec in \
  "openai:gpt-4.1-2025-04-14" \
  "openai:gpt-4.1-mini" \
  "openai:gpt-4o-mini" \
  "anthropic:claude-sonnet-4-5-20250929" \
  "anthropic:claude-haiku-4-5-20251001" \
  "gemini:gemini-2.5-flash"
do
  echo "=== $spec ==="
  $PY scripts/py/e2_resolution_eval_retry.py --benchmark spider \
    --set-dir data/spider/e2_rewrite_scaled --set-name rewrite_scaled \
    --models "$spec" --api-workers 4 \
    --output-dir data/spider/e2_rewrite_scaled_eval
done

echo "=== gpt-4o-mini canonical (baseline fill) ==="
$PY scripts/py/e2_resolution_eval_retry.py --benchmark spider \
  --set-dir data/spider/e2_rewrite_scaled --set-name rewrite_scaled_canonical \
  --models "openai:gpt-4o-mini" --api-workers 4 --use-canonical \
  --output-dir data/spider/e2_rewrite_scaled_eval

echo "ALL EVALS DONE"
