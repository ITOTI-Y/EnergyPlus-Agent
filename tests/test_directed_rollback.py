"""Tests for directed-rollback helpers (validate + self-repair scoping).

These pin the contract that:
- classify_errors groups validate_references() messages by the phase
  that OWNS the broken reference
- earliest_phase picks the upstream-most owner so a single rollback hop
  can resolve cascading downstream refs
- inject_validation_errors is a no-op when there are no errors and
  otherwise appends a clearly-marked global-error block
- _errors_owned_by_phase (used by self-repair during rollback) narrows
  convergence to THIS phase's errors, ignoring out-of-scope phases
"""

from src.agent.nodes._share import (
    PIPELINE_ORDER,
    _errors_owned_by_phase,
    classify_errors,
    earliest_phase,
    inject_validation_errors,
)


def _sample_errors() -> list[str]:
    return [
        "Surface 'W1' references construction 'C1' which does not exist.",
        "Surface 'W2' references zone 'Z9' which does not exist.",
        "Fenestration 'F1' references building surface 'W9' which does not exist.",
        "Thermostat 'T1' references heating setpoint schedule 'S9' which does not exist.",
        "Ideal load system references zone 'Z9' which does not exist.",
        "People 'P1' references schedule 'S8' which does not exist.",
        "Lights 'L1' references zone 'Z9' which does not exist.",
        "Construction 'C1' references material 'M9' which does not exist.",
    ]


def test_classify_errors_groups_by_owner_phase() -> None:
    grouped = classify_errors(_sample_errors())
    assert set(grouped) == {
        "surface",
        "fenestration",
        "hvac",
        "people",
        "lights",
        "construction",
    }
    # surface owns both of its references
    assert len(grouped["surface"]) == 2
    # hvac owns thermostat + ideal load system
    assert len(grouped["hvac"]) == 2


def test_classify_errors_drops_unclassifiable() -> None:
    # An error phrasing we don't recognize is silently dropped — caller
    # falls back to full re-intake rather than guessing a wrong phase.
    grouped = classify_errors(["Something weird happened."])
    assert grouped == {}


def test_earliest_phase_picks_upstream_most() -> None:
    # When errors span construction + surface + hvac, we roll back to
    # construction first (fixing upstream often resolves downstream).
    grouped = classify_errors(_sample_errors())
    assert earliest_phase(set(grouped)) == "construction"


def test_earliest_phase_respects_pipeline_order() -> None:
    # people comes before lights in PIPELINE_ORDER
    assert earliest_phase({"lights", "people"}) == "people"
    # surface before fenestration
    assert earliest_phase({"fenestration", "surface"}) == "surface"
    # empty -> None
    assert earliest_phase(set()) is None


def test_pipeline_order_is_monotonic() -> None:
    assert PIPELINE_ORDER[0] == "zone"
    assert PIPELINE_ORDER[-1] == "lights"
    # No duplicates
    assert len(set(PIPELINE_ORDER)) == len(PIPELINE_ORDER)


def test_inject_validation_errors_is_noop_on_empty() -> None:
    assert inject_validation_errors("SPECS", []) == "SPECS"


def test_inject_validation_errors_appends_block() -> None:
    errors = [
        "Surface 'W1' references construction 'C1' which does not exist.",
    ]
    injected = inject_validation_errors("SPECS", errors)
    assert injected.startswith("SPECS")
    assert "GLOBAL VALIDATION ERRORS" in injected
    assert errors[0] in injected


def test_errors_owned_by_phase_narrows_scope() -> None:
    errors = _sample_errors()
    # Surface phase only owns the two surface errors
    assert len(_errors_owned_by_phase(errors, "surface")) == 2
    # Lights phase only owns the one lights error
    assert len(_errors_owned_by_phase(errors, "lights")) == 1
    # Zone phase owns none of these (it has no outbound refs to validate)
    assert _errors_owned_by_phase(errors, "zone") == []
    # Material phase owns none — its only error surfaces via "Construction"
    assert _errors_owned_by_phase(errors, "material") == []
