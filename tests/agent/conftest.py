import json
import os
from typing import Any, Final
from urllib.parse import urlsplit, urlunsplit

import pytest
from vcr import VCR
from vcr.request import Request

from tests.agent.recording import CASSETTE_ORIGIN, RECORDED_MODEL

os.environ["LANGSMITH_TRACING"] = "false"

_MISSING: Final = object()
_SHOWN_CHARS: Final = 300


def _mask_origin(request: Request) -> Request:
    parsed = urlsplit(request.uri)
    request.uri = CASSETTE_ORIGIN + urlunsplit(parsed._replace(scheme="", netloc=""))
    return request


def _keep_body_headers(response: dict[str, Any]) -> dict[str, Any]:
    # VCR cannot decode zstd: such bodies stay compressed and need their encoding.
    response["headers"] = {
        k: v
        for k, v in response["headers"].items()
        if k.lower() in {"content-type", "content-encoding"}
    }
    return response


def _shown(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= _SHOWN_CHARS else text[:_SHOWN_CHARS] + "..."


def _difference(live: Any, recorded: Any, path: str) -> str:
    if isinstance(live, dict) and isinstance(recorded, dict):
        for key in sorted(live.keys() | recorded.keys()):
            a, b = live.get(key, _MISSING), recorded.get(key, _MISSING)
            if a is _MISSING or b is _MISSING:
                side = "recorded" if a is _MISSING else "live"
                return f"{path}.{key} only in the {side} request"
            if a != b:
                return _difference(a, b, f"{path}.{key}")
    if isinstance(live, list) and isinstance(recorded, list):
        for i, (a, b) in enumerate(zip(live, recorded, strict=False)):
            if a != b:
                return _difference(a, b, f"{path}[{i}]")
        if len(live) != len(recorded):
            return f"{path} has {len(live)} items, recorded {len(recorded)}"
    if isinstance(live, str) and isinstance(recorded, str):
        at = next(
            (i for i, (a, b) in enumerate(zip(live, recorded, strict=False)) if a != b),
            min(len(live), len(recorded)),
        )
        start = max(at - _SHOWN_CHARS // 2, 0)
        end = at + _SHOWN_CHARS // 2
        return (
            f"{path} differs at character {at}: "
            f"live {_shown(live[start:end])} != recorded {_shown(recorded[start:end])}"
        )
    return f"{path}: live {_shown(live)} != recorded {_shown(recorded)}"


def _json_body(live: Request, recorded: Request) -> None:
    a = json.loads(live.body) if live.body else None
    b = json.loads(recorded.body) if recorded.body else None
    if a != b:
        raise AssertionError(_difference(a, b, "body"))


def pytest_recording_configure(vcr: VCR) -> None:
    vcr.register_matcher("json_body", _json_body)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key", "cookie", "host"],
        "decode_compressed_response": True,
        "before_record_request": _mask_origin,
        "before_record_response": _keep_body_headers,
    }


@pytest.fixture
def pinned_llm_env(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> None:
    """Pin the model to the one the cassettes were recorded with."""
    if record_mode == "none":
        monkeypatch.setenv("LLM_BASE_URL", f"{CASSETTE_ORIGIN}/v1")
    monkeypatch.setenv("LLM_MODEL", RECORDED_MODEL)
    monkeypatch.setenv("LLM_API_KEY", os.environ.get("LLM_API_KEY", "test-key"))
