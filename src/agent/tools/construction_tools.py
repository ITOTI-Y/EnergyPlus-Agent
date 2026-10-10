from idfpy.models.constructions import Construction
from langchain_core.tools import BaseTool

from src.agent.tools._share import list_tool, model_tool, ok
from src.agent.tools.material_tools import list_materials_tool
from src.modeling import objects
from src.modeling.envelope import construction_from_layers
from src.state.config_state import ConfigState


def make_construction_tools(config: ConfigState) -> list[BaseTool]:
    idf = config.idf

    @model_tool
    def create_construction(name: str, layers: list[str]) -> str:
        """Create a Construction as an ordered list of material layers.

        Args:
            name: Unique construction name (e.g., 'ExtWall_Brick').
            layers: 1 to 10 existing material names, from outside to inside.
        """
        construction = objects.create(idf, construction_from_layers(name, layers))
        return ok(
            f"Construction '{name}' created.",
            construction.model_dump(exclude_none=True),
        )

    @model_tool
    def get_construction(name: str) -> str:
        """Read a construction by name."""
        construction = objects.get(idf, Construction, name)
        return ok(
            f"Construction '{name}' read.", construction.model_dump(exclude_none=True)
        )

    @model_tool
    def delete_construction(name: str) -> str:
        """Delete a construction; refused while surfaces or fenestration use it."""
        objects.delete(idf, objects.get(idf, Construction, name), name)
        return ok(f"Construction '{name}' deleted.")

    return [
        create_construction,
        list_tool(idf, "list_constructions", Construction, "List all constructions."),
        get_construction,
        delete_construction,
        list_materials_tool(config),
    ]
