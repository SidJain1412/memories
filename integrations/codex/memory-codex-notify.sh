#!/bin/bash
# memory-codex-notify.sh — Codex notify hook (after-agent)
# Receives Codex legacy notify payload as argv[1] JSON and enqueues extraction.

set -euo pipefail

MEMORIES_URL="${MEMORIES_URL:-http://localhost:8900}"
MEMORIES_API_KEY="${MEMORIES_API_KEY:-}"

PAYLOAD="${1:-}"
if [ -z "$PAYLOAD" ]; then
  PAYLOAD="$(cat 2>/dev/null || true)"
fi

if [ -z "$PAYLOAD" ]; then
  exit 0
fi

if ! echo "$PAYLOAD" | jq -e . >/dev/null 2>&1; then
  exit 0
fi

CWD=$(echo "$PAYLOAD" | jq -r '.cwd // "unknown"')
PROJECT=$(basename "$CWD")

USER_LINES=$(echo "$PAYLOAD" | jq -r '
  (."input-messages" // .input_messages // [])
  | map(select(type == "string" and length > 0))
  | map("User: " + .)
  | join("\n")
')
ASSISTANT_LINE=$(echo "$PAYLOAD" | jq -r '
  (."last-assistant-message" // .last_assistant_message // "")
  | if type == "string" then . else "" end
')

MESSAGES="$USER_LINES"
if [ -n "$ASSISTANT_LINE" ]; then
  if [ -n "$MESSAGES" ]; then
    MESSAGES="$MESSAGES"$'\n'
  fi
  MESSAGES="$MESSAGES""Assistant: $ASSISTANT_LINE"
fi

if [ -z "$MESSAGES" ]; then
  exit 0
fi

curl -sf -X POST "$MEMORIES_URL/memory/extract" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MEMORIES_API_KEY" \
  -d "{\"messages\": $(echo "$MESSAGES" | jq -Rs), \"source\": \"codex/$PROJECT\", \"context\": \"after_agent\"}" \
  > /dev/null 2>&1 || true
