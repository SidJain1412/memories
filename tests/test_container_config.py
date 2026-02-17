"""Container configuration regression tests."""
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_runtime_stage_does_not_install_curl() -> None:
    dockerfile = _read("Dockerfile")
    marker = "# ---- Runtime stage: copy only what we need ----"
    assert marker in dockerfile

    runtime_stage = dockerfile.split(marker, 1)[1]
    assert "curl" not in runtime_stage


def test_dockerfile_healthcheck_uses_python_probe() -> None:
    dockerfile = _read("Dockerfile")
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD python -c "import sys,urllib.request;' in dockerfile
    assert "http://localhost:8000/health" in dockerfile


def test_compose_healthchecks_use_python_probe() -> None:
    for compose_file in ("docker-compose.yml", "docker-compose.snippet.yml"):
        contents = _read(compose_file)
        assert "healthcheck:" in contents
        assert 'test: ["CMD", "python", "-c", "import sys,urllib.request;' in contents
        assert '"curl"' not in contents
