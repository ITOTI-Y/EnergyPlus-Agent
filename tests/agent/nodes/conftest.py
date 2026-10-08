import os
from collections.abc import Callable
from pathlib import Path

import pytest
from idfpy.models.constructions import Material
from idfpy.models.schedules import (
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
)
from langgraph.runtime import Runtime

from src.agent.state import SimContext

RECORDED_BASE_URL = "https://one.chat-yu.net/v1"
RECORDED_MODEL = "google/gemini-3.8-flash"


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
