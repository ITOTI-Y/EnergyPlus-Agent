import os
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import pytest
from idfpy.models import (
    Material,
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)
from langgraph.runtime import Runtime
from vcr.request import Request

from src.agent.state import SimContext

CASSETTE_ORIGIN = "https://llm.invalid"
RECORDED_MODEL = "google/gemini-3.8-flash"


def _mask_origin(request: Request) -> Request:
    """Swap the real endpoint's scheme and host for a placeholder.

    VCR runs this on live requests before matching as well as before saving,
    so replay matches whatever endpoint the local .env names.
    """
    parsed = urlsplit(request.uri)
    request.uri = CASSETTE_ORIGIN + urlunsplit(parsed._replace(scheme="", netloc=""))
    return request


def _keep_body_headers(response: dict[str, Any]) -> dict[str, Any]:
    """Drop response headers, which carry gateway, proxy and cookie details.

    content-encoding stays because VCR cannot decode zstd bodies, so they are
    stored compressed and the client still has to decompress them on replay.
    """
    response["headers"] = {
        k: v
        for k, v in response["headers"].items()
        if k.lower() in {"content-type", "content-encoding"}
    }
    return response


@pytest.fixture(scope="module")
def vcr_config():
    """Keep credentials and the endpoint out of cassettes; store readable bodies."""
    return {
        "filter_headers": ["authorization", "api-key", "x-api-key", "cookie", "host"],
        "decode_compressed_response": True,
        "before_record_request": _mask_origin,
        "before_record_response": _keep_body_headers,
    }


@pytest.fixture
def pinned_llm_env(monkeypatch: pytest.MonkeyPatch, record_mode: str) -> None:
    """Pin the model to the one the cassettes were recorded with.

    Replay matches on request path, so in replay-only mode the base URL is set
    to the placeholder to keep a local .env with a different path from
    breaking it. Recording uses the endpoint and key from the local .env.
    """
    if record_mode == "none":
        monkeypatch.setenv("LLM_BASE_URL", f"{CASSETTE_ORIGIN}/v1")
    monkeypatch.setenv("LLM_MODEL", RECORDED_MODEL)
    monkeypatch.setenv("LLM_API_KEY", os.environ.get("LLM_API_KEY", "test-key"))


@pytest.fixture
def brick() -> Material:
    return Material(
        name="Brick_100mm",
        roughness="MediumRough",
        thickness=0.1,
        conductivity=0.89,
        density=1920.0,
        specific_heat=790.0,
    )


@pytest.fixture
def constant_schedule() -> Callable[[str, str, float], ScheduleCompact]:
    def build(name: str, type_limits: str, value: float) -> ScheduleCompact:
        return ScheduleCompact(
            name=name,
            schedule_type_limits_name=type_limits,
            data=[
                ScheduleCompactDataItem(field="Through: 12/31"),
                ScheduleCompactDataItem(field="For: AllDays"),
                ScheduleCompactDataItem(field="Until: 24:00"),
                ScheduleCompactDataItem(field=str(value)),
            ],
        )

    return build


@pytest.fixture
def fraction_limits() -> ScheduleTypeLimits:
    return ScheduleTypeLimits(
        name="Fraction",
        lower_limit_value=0.0,
        upper_limit_value=1.0,
        numeric_type="Continuous",
    )


@pytest.fixture
def runtime() -> Runtime[SimContext]:
    """Graph runtime without the reference library."""
    return Runtime(context=SimContext(epw_path=Path("data/weather/Shenzhen.epw")))
