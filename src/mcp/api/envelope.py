from typing import Literal

from fastmcp import FastMCP
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import objects
from src.modeling.envelope import (
    Roughness,
    VertexSchema,
    all_materials,
    construction_from_layers,
    fenestration_from_vertices,
    fenestration_vertices,
    find_material,
    layer_fields,
    surface_geometry,
)
from src.state.config_state import ConfigState

type SurfaceType = Literal["Wall", "Floor", "Roof", "Ceiling"]
type BoundaryCondition = Literal["Outdoors", "Ground", "Surface", "Zone", "Adiabatic"]
type SunExposure = Literal["SunExposed", "NoSun"]
type WindExposure = Literal["WindExposed", "NoWind"]
type FenestrationType = Literal["Window", "Door", "GlassDoor"]


def _register_materials(mcp: FastMCP, state: ConfigState) -> None:
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_standard_material(
        name: str,
        roughness: Roughness,
        thickness: float,
        conductivity: float,
        density: float,
        specific_heat: float,
    ) -> Outcome:
        """Create a Material with thermal mass.

        Args:
            name: Unique material name.
            roughness: Surface roughness.
            thickness: Meters.
            conductivity: W/(m*K).
            density: kg/m^3.
            specific_heat: J/(kg*K).
        """
        material = Material(
            name=name,
            roughness=roughness,
            thickness=thickness,
            conductivity=conductivity,
            density=density,
            specific_heat=specific_heat,
        )
        return f"Material '{name}' created.", dump(objects.create(idf, material))

    @tool
    def create_no_mass_material(
        name: str, roughness: Roughness, thermal_resistance: float
    ) -> Outcome:
        """Create a Material:NoMass defined by its thermal resistance (m^2*K/W)."""
        material = MaterialNoMass(
            name=name, roughness=roughness, thermal_resistance=thermal_resistance
        )
        return f"Material '{name}' created.", dump(objects.create(idf, material))

    @tool
    def create_air_gap_material(name: str, thermal_resistance: float) -> Outcome:
        """Create a Material:AirGap for opaque constructions (m^2*K/W)."""
        material = MaterialAirGap(name=name, thermal_resistance=thermal_resistance)
        return f"Material '{name}' created.", dump(objects.create(idf, material))

    @tool
    def create_glazing_material(
        name: str,
        u_factor: float,
        solar_heat_gain_coefficient: float,
        visible_transmittance: float | None = None,
    ) -> Outcome:
        """Create a WindowMaterial:SimpleGlazingSystem.

        Args:
            name: Unique material name.
            u_factor: W/(m^2*K).
            solar_heat_gain_coefficient: 0-1.
            visible_transmittance: 0-1.
        """
        material = WindowMaterialSimpleGlazingSystem(
            name=name,
            u_factor=u_factor,
            solar_heat_gain_coefficient=solar_heat_gain_coefficient,
            visible_transmittance=visible_transmittance,
        )
        return f"Material '{name}' created.", dump(objects.create(idf, material))

    @tool
    def get_material(name: str) -> Outcome:
        """Read a material of any type by name."""
        material = find_material(idf, name)
        return f"Material '{name}' read.", {
            "type": material.idf_object_type(),
            **dump(material),
        }

    @tool
    def update_standard_material(
        name: str,
        new_name: str | None = None,
        roughness: Roughness | None = None,
        thickness: float | None = None,
        conductivity: float | None = None,
        density: float | None = None,
        specific_heat: float | None = None,
    ) -> Outcome:
        """Update a Material; a new name is applied to every construction using it."""
        material = objects.update(
            idf,
            objects.get(idf, Material, name),
            given(
                name=new_name,
                roughness=roughness,
                thickness=thickness,
                conductivity=conductivity,
                density=density,
                specific_heat=specific_heat,
            ),
        )
        return f"Material '{name}' updated.", dump(material)

    @tool
    def update_no_mass_material(
        name: str,
        new_name: str | None = None,
        roughness: Roughness | None = None,
        thermal_resistance: float | None = None,
    ) -> Outcome:
        """Update a Material:NoMass; omitted fields stay unchanged."""
        material = objects.update(
            idf,
            objects.get(idf, MaterialNoMass, name),
            given(
                name=new_name,
                roughness=roughness,
                thermal_resistance=thermal_resistance,
            ),
        )
        return f"Material '{name}' updated.", dump(material)

    @tool
    def update_air_gap_material(
        name: str, new_name: str | None = None, thermal_resistance: float | None = None
    ) -> Outcome:
        """Update a Material:AirGap; omitted fields stay unchanged."""
        material = objects.update(
            idf,
            objects.get(idf, MaterialAirGap, name),
            given(name=new_name, thermal_resistance=thermal_resistance),
        )
        return f"Material '{name}' updated.", dump(material)

    @tool
    def update_glazing_material(
        name: str,
        new_name: str | None = None,
        u_factor: float | None = None,
        solar_heat_gain_coefficient: float | None = None,
        visible_transmittance: float | None = None,
    ) -> Outcome:
        """Update a WindowMaterial:SimpleGlazingSystem; omitted fields stay."""
        material = objects.update(
            idf,
            objects.get(idf, WindowMaterialSimpleGlazingSystem, name),
            given(
                name=new_name,
                u_factor=u_factor,
                solar_heat_gain_coefficient=solar_heat_gain_coefficient,
                visible_transmittance=visible_transmittance,
            ),
        )
        return f"Material '{name}' updated.", dump(material)

    @tool
    def delete_material(name: str) -> Outcome:
        """Delete a material; refused while a construction uses it."""
        objects.delete(idf, find_material(idf, name), name)
        return f"Material '{name}' deleted.", None

    @tool
    def list_materials() -> Outcome:
        """List all materials with their EnergyPlus type."""
        items = [{"type": m.idf_object_type(), **dump(m)} for m in all_materials(idf)]
        return f"Listed {len(items)} materials.", items


def _register_constructions(mcp: FastMCP, state: ConfigState) -> None:
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_construction(name: str, layers: list[str]) -> Outcome:
        """Create a construction from 1 to 10 material names, outside to inside."""
        construction = objects.create(idf, construction_from_layers(name, layers))
        return f"Construction '{name}' created.", dump(construction)

    @tool
    def get_construction(name: str) -> Outcome:
        """Read a construction by name."""
        return (
            f"Construction '{name}' read.",
            dump(objects.get(idf, Construction, name)),
        )

    @tool
    def update_construction(
        name: str, new_name: str | None = None, layers: list[str] | None = None
    ) -> Outcome:
        """Rename a construction or replace its layers (outside to inside)."""
        changes = given(name=new_name)
        if layers is not None:
            changes |= layer_fields(layers)
        construction = objects.update(
            idf, objects.get(idf, Construction, name), changes
        )
        return f"Construction '{name}' updated.", dump(construction)

    @tool
    def delete_construction(name: str) -> Outcome:
        """Delete a construction; refused while surfaces or fenestration use it."""
        objects.delete(idf, objects.get(idf, Construction, name), name)
        return f"Construction '{name}' deleted.", None

    @tool
    def list_constructions() -> Outcome:
        """List all constructions."""
        return "Listed constructions.", objects.dumps(idf.all_of_type(Construction))


def _register_surfaces(mcp: FastMCP, state: ConfigState) -> None:
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_surface(
        name: str,
        surface_type: SurfaceType,
        construction_name: str,
        zone_name: str,
        outside_boundary_condition: BoundaryCondition,
        vertices: list[VertexSchema],
        sun_exposure: SunExposure = "NoSun",
        wind_exposure: WindExposure = "NoWind",
        outside_boundary_condition_object: str | None = None,
    ) -> Outcome:
        """Create a BuildingSurface:Detailed.

        Args:
            name: Unique surface name.
            surface_type: Wall, Floor, Roof or Ceiling.
            construction_name: Existing construction.
            zone_name: Existing zone the surface belongs to.
            outside_boundary_condition: What the outside face sees.
            vertices: >= 3 vertices in meters, counter-clockwise seen from outside.
            sun_exposure: SunExposed for outdoor walls and roofs.
            wind_exposure: WindExposed for outdoor walls and roofs.
            outside_boundary_condition_object: Partner surface name for a
                Surface boundary, adjacent zone name for a Zone boundary.
        """
        surface = BuildingSurfaceDetailed.model_validate(
            {
                "name": name,
                "surface_type": surface_type,
                "construction_name": construction_name,
                "zone_name": zone_name,
                "outside_boundary_condition": outside_boundary_condition,
                "outside_boundary_condition_object": outside_boundary_condition_object,
                "sun_exposure": sun_exposure,
                "wind_exposure": wind_exposure,
                **surface_geometry(vertices),
            }
        )
        return f"Surface '{name}' created.", dump(objects.create(idf, surface))

    @tool
    def get_surface(name: str) -> Outcome:
        """Read a surface by name."""
        return (
            f"Surface '{name}' read.",
            dump(objects.get(idf, BuildingSurfaceDetailed, name)),
        )

    @tool
    def update_surface(
        name: str,
        new_name: str | None = None,
        surface_type: SurfaceType | None = None,
        construction_name: str | None = None,
        zone_name: str | None = None,
        outside_boundary_condition: BoundaryCondition | None = None,
        outside_boundary_condition_object: str | None = None,
        sun_exposure: SunExposure | None = None,
        wind_exposure: WindExposure | None = None,
        vertices: list[VertexSchema] | None = None,
    ) -> Outcome:
        """Update a surface; a new name is applied to its fenestration and partner."""
        changes = given(
            name=new_name,
            surface_type=surface_type,
            construction_name=construction_name,
            zone_name=zone_name,
            outside_boundary_condition=outside_boundary_condition,
            outside_boundary_condition_object=outside_boundary_condition_object,
            sun_exposure=sun_exposure,
            wind_exposure=wind_exposure,
        )
        if vertices is not None:
            changes |= surface_geometry(vertices)
        surface = objects.update(
            idf, objects.get(idf, BuildingSurfaceDetailed, name), changes
        )
        return f"Surface '{name}' updated.", dump(surface)

    @tool
    def delete_surface(name: str) -> Outcome:
        """Delete a surface; refused while fenestration or a partner uses it."""
        objects.delete(idf, objects.get(idf, BuildingSurfaceDetailed, name), name)
        return f"Surface '{name}' deleted.", None

    @tool
    def list_surfaces() -> Outcome:
        """List all building surfaces."""
        return "Listed surfaces.", objects.dumps(
            idf.all_of_type(BuildingSurfaceDetailed)
        )


def _register_fenestration(mcp: FastMCP, state: ConfigState) -> None:
    idf = state.idf
    tool = model_tool(mcp)

    @tool
    def create_fenestration_surface(
        name: str,
        surface_type: FenestrationType,
        construction_name: str,
        building_surface_name: str,
        vertices: list[VertexSchema],
        multiplier: int = 1,
    ) -> Outcome:
        """Create a FenestrationSurface:Detailed on a parent surface.

        Args:
            name: Unique fenestration name.
            surface_type: Window, Door or GlassDoor.
            construction_name: Existing construction.
            building_surface_name: Existing parent surface.
            vertices: 3 or 4 vertices in meters on the parent surface plane,
                counter-clockwise seen from outside.
            multiplier: Count of identical openings represented (>= 1).
        """
        fenestration = fenestration_from_vertices(
            vertices,
            name=name,
            surface_type=surface_type,
            construction_name=construction_name,
            building_surface_name=building_surface_name,
            multiplier=float(multiplier),
        )
        return (
            f"Fenestration '{name}' created.",
            dump(objects.create(idf, fenestration)),
        )

    @tool
    def get_fenestration_surface(name: str) -> Outcome:
        """Read a fenestration surface by name."""
        return (
            f"Fenestration '{name}' read.",
            dump(objects.get(idf, FenestrationSurfaceDetailed, name)),
        )

    @tool
    def update_fenestration_surface(
        name: str,
        new_name: str | None = None,
        surface_type: FenestrationType | None = None,
        construction_name: str | None = None,
        building_surface_name: str | None = None,
        multiplier: int | None = None,
        vertices: list[VertexSchema] | None = None,
    ) -> Outcome:
        """Update a fenestration surface; omitted fields stay unchanged."""
        changes = given(
            name=new_name,
            surface_type=surface_type,
            construction_name=construction_name,
            building_surface_name=building_surface_name,
            multiplier=None if multiplier is None else float(multiplier),
        )
        if vertices is not None:
            changes |= fenestration_vertices(vertices)
        fenestration = objects.update(
            idf, objects.get(idf, FenestrationSurfaceDetailed, name), changes
        )
        return f"Fenestration '{name}' updated.", dump(fenestration)

    @tool
    def delete_fenestration_surface(name: str) -> Outcome:
        """Delete a fenestration surface."""
        objects.delete(idf, objects.get(idf, FenestrationSurfaceDetailed, name), name)
        return f"Fenestration '{name}' deleted.", None

    @tool
    def list_fenestration_surfaces() -> Outcome:
        """List all fenestration surfaces."""
        return "Listed fenestration.", objects.dumps(
            idf.all_of_type(FenestrationSurfaceDetailed)
        )


def register_envelope_tools(mcp: FastMCP, state: ConfigState) -> None:
    """Register material, construction, surface and fenestration tools."""
    _register_materials(mcp, state)
    _register_constructions(mcp, state)
    _register_surfaces(mcp, state)
    _register_fenestration(mcp, state)
