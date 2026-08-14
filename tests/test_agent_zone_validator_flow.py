from idfpy.models.thermal_zones import Zone
from langchain_core.messages import AIMessage

from src.agent.nodes import zone as zone_module
from src.agent.state import AgentState, IntakeOutput
from src.mcp.state import ConfigState


def _intake() -> IntakeOutput:
    return IntakeOutput.model_validate(
        {
            "building": {"Name": "Test_Building"},
            "site_location": {
                "Name": "Test_Location",
                "Latitude": 22.5,
                "Longitude": 114.0,
                "Time Zone": 8.0,
                "Elevation": 10.0,
            },
            "zone_specs": "Create two zones: F1_Office and F1_Corridor.",
            "material_specs": "",
            "schedule_specs": "",
            "construction_specs": "",
            "surface_specs": "",
            "fenestration_specs": "",
            "hvac_specs": "",
            "people_specs": "",
            "lights_specs": "",
        }
    )


class _FakeZoneAgent:
    def __init__(self, local):
        self.local = local
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload)
        if not self.local.idf.has("Zone", "F1_Corridor"):
            self.local.idf.add(Zone(name="F1_Corridor"))
        return {"messages": [AIMessage(content="repaired zones")]}


def test_zone_agent_reinvokes_main_agent_after_validator_reject(monkeypatch):
    fake_agents = []
    current_local = {"value": None}

    def fake_build_agent(**kwargs):
        agent = _FakeZoneAgent(current_local["value"])
        fake_agents.append(agent)
        return agent

    def fake_invoke_with_self_repair(agent, local_config, specs, **kwargs):
        current_local["value"] = local_config
        agent.local = local_config
        local_config.idf.add(Zone(name="F1_Office"))
        return {"messages": [AIMessage(content="created one zone")]}

    verdicts = iter(
        [
            (
                "rejected",
                ["Specs require zone 'F1_Corridor' but it was not created."],
            ),
            ("approved", None),
        ]
    )

    monkeypatch.setattr(zone_module, "build_agent", fake_build_agent)
    monkeypatch.setattr(
        zone_module, "invoke_with_self_repair", fake_invoke_with_self_repair
    )
    monkeypatch.setattr(
        zone_module, "run_zone_validator", lambda *args, **kwargs: next(verdicts)
    )

    out = zone_module.zone_agent(AgentState(intake_output=_intake()))

    zones = out["config_state"].idf.all_of_type("Zone")
    assert set(zones) == {"F1_Office", "F1_Corridor"}
    assert len(fake_agents[0].calls) == 1
    feedback = fake_agents[0].calls[0]["messages"][-1].content
    assert "F1_Corridor" in feedback
    assert out["validation_errors"] == []


def test_zone_agent_stops_after_validator_retry_budget(monkeypatch):
    repair_calls = []

    class AlwaysNoopAgent:
        def invoke(self, payload):
            repair_calls.append(payload)
            return {"messages": [AIMessage(content="still incomplete")]}

    def fake_invoke_with_self_repair(agent, local_config, specs, **kwargs):
        return {"messages": [AIMessage(content="created no zones")]}

    monkeypatch.setattr(zone_module, "build_agent", lambda **kwargs: AlwaysNoopAgent())
    monkeypatch.setattr(
        zone_module, "invoke_with_self_repair", fake_invoke_with_self_repair
    )
    monkeypatch.setattr(
        zone_module,
        "run_zone_validator",
        lambda *args, **kwargs: (
            "rejected",
            ["0 zones created but specs require zones."],
        ),
    )

    monkeypatch.setattr(
        ConfigState,
        "validate_references",
        lambda self: ["Current cross-reference error."],
    )

    out = zone_module.zone_agent(AgentState(intake_output=_intake()))

    assert len(repair_calls) == zone_module.MAX_ZONE_VALIDATION_ROUNDS
    assert out["validation_errors"] == [
        "Current cross-reference error.",
        f"Zone validation failed after {zone_module.MAX_ZONE_VALIDATION_ROUNDS} rounds: "
        "0 zones created but specs require zones.",
    ]


def test_zone_agent_clears_consumed_upstream_request(monkeypatch):
    captured_specs = {}

    def fake_invoke_with_self_repair(agent, local_config, specs, **kwargs):
        captured_specs["specs"] = specs
        local_config.idf.add(Zone(name="Missing_Zone"))
        return {"messages": [AIMessage(content="created upstream zone")]}

    monkeypatch.setattr(
        zone_module, "build_agent", lambda **kwargs: _FakeZoneAgent(None)
    )
    monkeypatch.setattr(
        zone_module, "invoke_with_self_repair", fake_invoke_with_self_repair
    )
    monkeypatch.setattr(
        zone_module, "run_zone_validator", lambda *args, **kwargs: ("approved", None)
    )

    state = AgentState(
        intake_output=_intake(),
        upstream_request={
            "target": "zone",
            "specs": "Please create Zone 'Missing_Zone'.",
        },
    )

    out = zone_module.zone_agent(state)

    assert "Missing_Zone" in captured_specs["specs"]
    assert out["upstream_request"] == {}
