"""Tests for Codex notify hook integration script."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from socketserver import TCPServer
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "codex"
    / "memory-codex-notify.sh"
)


@contextmanager
def _capture_server():
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            content_length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(content_length).decode("utf-8")
            requests.append(
                {
                    "path": self.path,
                    "headers": dict(self.headers.items()),
                    "body": body,
                }
            )
            self.send_response(202)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"job_id":"job-1"}')

        def log_message(self, format: str, *args: object) -> None:  # noqa: A003
            return

    with TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address
            yield f"http://{host}:{port}", requests
        finally:
            server.shutdown()
            thread.join()


def _run_hook(memories_url: str, payload: dict[str, object], api_key: str = "") -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "MEMORIES_URL": memories_url,
            "MEMORIES_API_KEY": api_key,
        }
    )
    return subprocess.run(
        [str(SCRIPT), json.dumps(payload)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_codex_notify_posts_extraction_payload() -> None:
    assert SCRIPT.exists(), f"missing script: {SCRIPT}"

    payload = {
        "type": "agent-turn-complete",
        "thread-id": "thread-1",
        "turn-id": "turn-1",
        "cwd": "/Users/example/memories",
        "input-messages": ["use qdrant in prod", "remember we chose docker compose"],
        "last-assistant-message": "done, updated docker compose and docs",
    }

    with _capture_server() as (memories_url, requests):
        result = _run_hook(memories_url, payload, api_key="abc123")

    assert result.returncode == 0
    assert len(requests) == 1
    request = requests[0]
    assert request["path"] == "/memory/extract"
    headers = request["headers"]
    assert headers.get("X-API-Key") == "abc123"

    body = json.loads(str(request["body"]))
    assert body["context"] == "after_agent"
    assert body["source"] == "codex/memories"
    assert "User: use qdrant in prod" in body["messages"]
    assert "User: remember we chose docker compose" in body["messages"]
    assert "Assistant: done, updated docker compose and docs" in body["messages"]


def test_codex_notify_skips_when_no_messages() -> None:
    assert SCRIPT.exists(), f"missing script: {SCRIPT}"

    payload = {
        "type": "agent-turn-complete",
        "thread-id": "thread-1",
        "turn-id": "turn-1",
        "cwd": "/Users/example/memories",
        "input-messages": [],
        "last-assistant-message": "",
    }

    with _capture_server() as (memories_url, requests):
        result = _run_hook(memories_url, payload)

    assert result.returncode == 0
    assert requests == []
