import os

import pytest

RECORDED_BASE_URL = "https://one.chat-yu.net/v1"
RECORDED_MODEL = "opencode/glm-5.2"


@pytest.fixture(scope="module")
def vcr_config():
    """Keep credentials out of cassettes and store readable bodies."""
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key", "cookie"],
        "decode_compressed_response": True,
    }


@pytest.fixture
def pinned_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin endpoint/model to the values the cassettes were recorded with.

    Replay matches on request URI, so the base URL must equal the recorded
    one even when the local .env has drifted. The API key only needs to be
    real at record time; during replay no request leaves the machine.
    """
    monkeypatch.setenv("LLM_BASE_URL", RECORDED_BASE_URL)
    monkeypatch.setenv("LLM_MODEL", RECORDED_MODEL)
    monkeypatch.setenv("LLM_API_KEY", os.environ.get("LLM_API_KEY", "test-key"))
