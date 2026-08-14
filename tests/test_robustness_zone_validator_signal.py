"""Tests for the zone-validator failure signal in the robustness harness.

zone.py writes a "[zone-validator] ..." AIMessage when the zone completeness
validator exhausts its retry budget. The structured validation_errors field
is overwritten by cross_ref_foundations_node on the next graph step
(last-write-wins reducer), so the robustness harness must recover this
signal by scanning the zone node's message stream — otherwise a model that
silently produced 0 zones would look clean to the report.
"""

from pathlib import Path

from langchain_core.messages import AIMessage

from tests.agent.test_run_robustness import CaseHarness, CaseSpecSchema


def _make_harness(tmp_path: Path) -> CaseHarness:
    case = CaseSpecSchema(case_id="t", data={}, case_dir=tmp_path)
    return CaseHarness(case=case, output_dir=tmp_path)


def test_zone_validator_failure_is_captured_from_message_stream(tmp_path):
    """A "[zone-validator] ..." AIMessage in the zone node update must set
    zone_validator_failed and parse the trailing reasons."""
    harness = _make_harness(tmp_path)

    harness.event_handler(
        "zone",
        {
            "messages": [
                AIMessage(content="[zone] created some zones"),
                AIMessage(
                    content=(
                        "[zone-validator] Zone validation failed after 3 rounds: "
                        "0 zones created but specs require 8; "
                        "duplicate name 'Core' appears twice"
                    )
                ),
            ]
        },
    )

    assert harness.result.zone_validator_failed is True
    joined = " | ".join(harness.result.zone_validator_reasons)
    assert "0 zones created but specs require 8" in joined
    assert "duplicate name 'Core'" in joined


def test_zone_node_without_validator_message_keeps_flag_false(tmp_path):
    """A normal zone node update (no [zone-validator] prefix) must not flip
    the flag — the validator either approved or never exhausted."""
    harness = _make_harness(tmp_path)

    harness.event_handler(
        "zone",
        {"messages": [AIMessage(content="[zone] created 8 zones")]},
    )

    assert harness.result.zone_validator_failed is False
    assert harness.result.zone_validator_reasons == []


def test_non_zone_nodes_are_ignored_for_validator_signal(tmp_path):
    """Only the 'zone' node carries the validator message; an identical
    prefix appearing elsewhere (e.g. a copied message in a later phase)
    must not falsely trigger the flag."""
    harness = _make_harness(tmp_path)

    harness.event_handler(
        "hvac",
        {"messages": [AIMessage(content="[zone-validator] leaked")]},
    )

    assert harness.result.zone_validator_failed is False
