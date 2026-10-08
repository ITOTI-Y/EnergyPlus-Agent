import json

from src.agent.nodes.intake import _tool
from src.agent.state import IntakeOutput, IntakePatch


def test_intake_tools_offer_no_empty_choices():
    # Gemini rejects "" in tool enums; idfpy uses it for blank fields.
    for schema in (IntakeOutput, IntakePatch):
        tool = json.dumps(_tool(schema))

        assert '"enum"' in tool
        assert '""' not in tool
