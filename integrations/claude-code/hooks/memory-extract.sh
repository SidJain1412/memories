#!/bin/bash
# memory-extract.sh — Stop hook (async)
# Extracts facts from the last exchange and stores via AUDN pipeline.
# Requires Memories service with extraction enabled (EXTRACT_PROVIDER set).

set -euo pipefail

# Load from dedicated env file — avoids requiring shell profile changes
[ -f "${MEMORIES_ENV_FILE:-$HOME/.config/memories/env}" ] && . "${MEMORIES_ENV_FILE:-$HOME/.config/memories/env}"

MEMORIES_URL="${MEMORIES_URL:-http://localhost:8900}"
MEMORIES_API_KEY="${MEMORIES_API_KEY:-}"

INPUT=$(cat)
STOP_REASON=$(echo "$INPUT" | jq -r '.stop_reason // "end_turn"')

# Only extract on normal completions
if [ "$STOP_REASON" != "end_turn" ]; then
  exit 0
fi

MESSAGES=$(echo "$INPUT" | jq -r '.messages // empty')
if [ -z "$MESSAGES" ]; then
  exit 0
fi

CWD=$(echo "$INPUT" | jq -r '.cwd // "unknown"')
PROJECT=$(basename "$CWD")

# POST to extraction endpoint (fire-and-forget, async hook)
BODY=$(jq -nc --arg msgs "$MESSAGES" --arg src "claude-code/$PROJECT" '{"messages": $msgs, "source": $src, "context": "stop"}')
curl -sf -X POST "$MEMORIES_URL/memory/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MEMORIES_API_KEY" \
  -d "$BODY" \
  > /dev/null 2>&1 || true
