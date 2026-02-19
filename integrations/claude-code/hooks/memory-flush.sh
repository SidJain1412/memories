#!/bin/bash
# memory-flush.sh — PreCompact hook (async)
# Aggressive extraction before context compaction.
# Same as memory-extract.sh but with context=pre_compact.

set -euo pipefail

MEMORIES_URL="${MEMORIES_URL:-http://localhost:8900}"
MEMORIES_API_KEY="${MEMORIES_API_KEY:-}"

INPUT=$(cat)
MESSAGES=$(echo "$INPUT" | jq -r '.messages // empty')
if [ -z "$MESSAGES" ]; then
  exit 0
fi

CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"')
PROJECT=$(basename "$CWD")

BODY=$(jq -nc --arg msgs "$MESSAGES" --arg src "claude-code/$PROJECT" '{"messages": $msgs, "source": $src, "context": "pre_compact"}')
curl -sf -X POST "$MEMORIES_URL/memory/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MEMORIES_API_KEY" \
  -d "$BODY" \
  > /dev/null 2>&1 || true
