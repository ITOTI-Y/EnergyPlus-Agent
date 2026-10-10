from typing import Literal

from fastmcp import FastMCP
from idfpy.models.constructions import (
    Construction,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    WindowMaterialGas,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
)
from idfpy.models.thermal_zones import (
    BuildingSurfaceDetailed,
    FenestrationSurfaceDetailed,
)

from src.mcp.api.common import Outcome, dump, given, model_tool
from src.modeling import geometry, objects
from src.modeling.envelope import (
    MaterialObject,
    Roughness,
    VertexSchema,
    all_materials,
    check_construction_fits,
    check_material_name_free,
    checked_construction,
    construction_kind,
    fenestration_from_vertices,
    find_material,
    layer_changes,
    surface_geometry,
)
from src.modeling.fenestration import (
    add_fenestration,
    remove_fenestration,
    update_fenestration,
)
from src.modeling.geometry import PlanPointSchema, ZoneConstructions
from src.modeling.surfaces import SurfaceSpecSchema, add_surface, add_surfaces
from src.state.config_state import ConfigState

type SurfaceType = Literal["Wall", "Floor", "Roof", "Ceiling"]
type BoundaryCondition = Literal["Outdoors", "Ground", "Surface", "Zone", "Adiabatic"]
type SunExposure = Literal["SunExposed", "NoSun"]
type WindExposure = Literal["WindExposed", "NoWind"]
type FenestrationType = Literal["Window", "Door", "GlassDoor"]
type GasType = Literal["Air", "Argon", "Krypton", "Xenon"]


def _register_materials(mcp: FastMCP, state: ConfigState) -> None:
    idf = state.idf
    tool = model_tool(mcp)

    def create(material: MaterialObject) -> Outcome:
        check_material_name_free(idf, material)
        return f"Material '{material.name}' created.", dump(
            objects.create(idf, material)
        )

    @tool
    def create_standard_material(
        name: str,
        roughness: Roughness,
        thickness: float,
        conductivity: float,
        density: float,
        specific_heat: float,
        thermal_absorptance: float | None = None,
        solar_absorptance: float | None = None,
        visible_absorptance: float | None = None,
    ) -> Outcome:
        """Create a Material with thermal mass.

        Args:
            name: Unique material name, of any material type.
            roughness: Surface roughness.
            thickness: Meters.
            conductivity: W/(m*K).
            density: kg/m^3.
            specific_heat: J/(kg*K).
            thermal_absorptance: Long-wave emittance; EnergyPlus default 0.9.
            solar_absorptance: Default 0.7.
            visible_absorptance: Default 0.7.
        """
        material = Material(
            name=name,
            roughness=roughness,
            thickness=thickness,
            conductivity=conductivity,
            density=density,
            specific_heat=specific_heat,
            **given(
                thermal_absorptance=thermal_absorptance,
                solar_absorptance=solar_absorptance,
                visible_absorptance=visible_absorptance,
            ),
        )
        return create(material)

    @tool
    def create_no_mass_material(
        name: str,
        roughness: Roughness,
        thermal_resistance: float,
        thermal_absorptance: float | None = None,
        solar_absorptance: float | None = None,
        visible_absorptance: float | None = None,
    ) -> Outcome:
        """Create a Material:NoMass defined by its thermal resistance (m^2*K/W).

        Absorptances left out take the EnergyPlus defaults 0.9 / 0.7 / 0.7.
        """
        material = MaterialNoMass(
            name=name,
            roughness=roughness,
            thermal_resistance=thermal_resistance,
            **given(
                thermal_absorptance=thermal_absorptance,
                solar_absorptance=solar_absorptance,
                visible_absorptance=visible_absorptance,
            ),
        )
        return create(material)

    @tool
    def create_air_gap_material(name: str, thermal_resistance: float) -> Outcome:
        """Create a Material:AirGap for opaque constructions (m^2*K/W).

        Not for windows: separate panes with create_window_gas_material.
        """
        material = MaterialAirGap(name=name, thermal_resistance=thermal_resistance)
        return create(material)

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
        return create(material)

    @tool
    def create_window_glazing_material(
        name: str,
        thickness: float,
        solar_transmittance: float = 0.775,
        solar_reflectance: float = 0.071,
        visible_transmittance: float = 0.881,
        visible_reflectance: float = 0.080,
        conductivity: float = 0.9,
    ) -> Outcome:
        """Create one glass pane (WindowMaterial:Glazing); defaults are clear glass.

        Args:
            name: Unique material name.
            thickness: Meters.
            solar_transmittance: At normal incidence, 0-1.
            solar_reflectance: At normal incidence, both faces, 0-1.
            visible_transmittance: At normal incidence, 0-1.
            visible_reflectance: At normal incidence, both faces, 0-1.
            conductivity: W/(m*K).
        """
        material = WindowMaterialGlazing(
            name=name,
            optical_data_type="SpectralAverage",
            thickness=thickness,
            solar_transmittance_at_normal_incidence=solar_transmittance,
            front_side_solar_reflectance_at_normal_incidence=solar_reflectance,
            back_side_solar_reflectance_at_normal_incidence=solar_reflectance,
            visible_transmittance_at_normal_incidence=visible_transmittance,
            front_side_visible_reflectance_at_normal_incidence=visible_reflectance,
            back_side_visible_reflectance_at_normal_incidence=visible_reflectance,
            conductivity=conductivity,
        )
        return create(material)

    @tool
    def create_window_gas_material(
        name: str, thickness: float, gas_type: GasType = "Air"
    ) -> Outcome:
        """Create the gas layer between two panes (WindowMaterial:Gas), thickness in m."""
        material = WindowMaterialGas(name=name, gas_type=gas_type, thickness=thickness)
        return create(material)

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
        """Create a construction from 1 to 10 material names, outside to inside.

        Opaque constructions use opaque materials only. Window constructions
        are one SimpleGlazingSystem, or glazing layers with exactly one window
        gas layer between each pair, starting and ending with glazing.
        """
        construction = objects.create(idf, checked_construction(idf, name, layers))
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
        construction = objects.get(idf, Construction, name)
        changes = given(name=new_name)
        if layers is not None:
            changes |= layer_changes(idf, construction, layers)
        objects.update(idf, construction, changes)
        return f"Construction '{name}' updated.", dump(construction)

    @tool
    def delete_construction(name: str) -> Outcome:
        """Delete a construction; refused while surfaces or fenestration use it."""
        objects.delete(idf, objects.get(idf, Construction, name), name)
        return f"Construction '{name}' deleted.", None

    @tool
    def list_constructions() -> Outcome:
        """List all constructions with their kind: window, opaque or mixed."""
        return "Listed constructions.", [
            {"kind": construction_kind(c), **dump(c)}
            for c in idf.all_of_type(Construction).values()
        ]


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
            construction_name: Existing opaque construction.
            zone_name: Existing zone the surface belongs to.
            outside_boundary_condition: What the outside face sees.
            vertices: >= 3 vertices in meters, counter-clockwise seen from outside.
            sun_exposure: SunExposed for outdoor walls and roofs.
            wind_exposure: WindExposed for outdoor walls and roofs.
            outside_boundary_condition_object: Partner surface name for a
                Surface boundary (it may be created later; an existing one is
                linked back), adjacent zone name for a Zone boundary.
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
        return f"Surface '{name}' created.", [
            dump(s) for s in add_surface(idf, surface)
        ]

    @tool
    def create_zone_geometry(
        zone_name: str,
        plan: list[PlanPointSchema],
        floor_z: float,
        height: float,
        exterior_wall_construction: str,
        roof_construction: str,
        ground_floor_construction: str,
        interior_wall_construction: str,
        interior_floor_construction: str,
    ) -> Outcome:
        """Create all surfaces of a zone by extruding its floor plan.

        Faces touching another zone's faces become interzone pairs, in any
        creation order and for zones of different height or misaligned
        storeys. The top is a flat roof; use create_surfaces for sloped ones.

        Args:
            zone_name: Existing zone.
            plan: Floor plan corners (X, Y in meters), in order around the zone.
            floor_z: Floor level in meters; 0 for the ground floor.
            height: Floor-to-ceiling height in meters.
            exterior_wall_construction: Outdoor walls.
            roof_construction: Roof.
            ground_floor_construction: Floor on the ground or above outdoor air.
            interior_wall_construction: Walls shared with another zone.
            interior_floor_construction: Floors and ceilings shared with another zone.
        """
        result = geometry.create_zone_geometry(
            idf,
            zone_name,
            plan,
            floor_z,
            height,
            ZoneConstructions(
                exterior_wall=exterior_wall_construction,
                roof=roof_construction,
                ground_floor=ground_floor_construction,
                interior_wall=interior_wall_construction,
                interior_floor=interior_floor_construction,
            ),
        )
        message = " ".join(
            [
                f"Zone '{zone_name}' extruded into {len(result.created)} surfaces.",
                *result.notes,
            ]
        )
        return message, {"created": result.created, "replaced": result.replaced}

    @tool
    def create_surfaces(surfaces: list[SurfaceSpecSchema]) -> Outcome:
        """Create several surfaces; each succeeds or fails on its own.

        Interzone partners may be later entries of the same list. Failed
        entries are reported by position so only they need resending.
        """
        outcome = add_surfaces(idf, surfaces)
        return (
            f"Created {len(outcome.created)} surfaces, {len(outcome.failed)} failed.",
            {"created": outcome.created, "failed": outcome.failed},
        )

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
        surface = objects.get(idf, BuildingSurfaceDetailed, name)
        if construction_name is not None or surface_type is not None:
            check_construction_fits(
                idf,
                construction_name or surface.construction_name,
                surface_type or surface.surface_type,
            )
        objects.update(idf, surface, changes)
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
            construction_name: A window construction for Window and GlassDoor,
                an opaque one for Door.
            building_surface_name: Existing parent surface. On a wall shared
                with another zone, the matching opening in that zone is
                created as '<name>_Partner'.
            vertices: 3 or 4 vertices in meters on the parent surface plane
                and inside its outline; the order is corrected to match the
                parent surface.
            multiplier: Count of identical openings represented (>= 1).
        """
        created, flipped = add_fenestration(
            idf,
            fenestration_from_vertices(
                vertices,
                name=name,
                surface_type=surface_type,
                construction_name=construction_name,
                building_surface_name=building_surface_name,
                multiplier=float(multiplier),
            ),
        )
        note = " Vertex order reversed to match the surface." if flipped else ""
        return f"Fenestration '{name}' created.{note}", [dump(f) for f in created]

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
        construction_name: str | None = None,
        multiplier: int | None = None,
    ) -> Outcome:
        """Rename an opening or change its construction or multiplier.

        Construction and multiplier also apply to an interzone partner. To
        move or reshape an opening, delete it and create it again.
        """
        updated = update_fenestration(
            idf,
            name,
            new_name=new_name,
            construction_name=construction_name,
            multiplier=None if multiplier is None else float(multiplier),
        )
        return f"Fenestration '{name}' updated.", [dump(f) for f in updated]

    @tool
    def delete_fenestration_surface(name: str) -> Outcome:
        """Delete an opening, and its partner when it is an interzone opening."""
        return f"Deleted {', '.join(remove_fenestration(idf, name))}.", None

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
