from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, SupportsIndex

import yaml
from idfpy import IDF
from idfpy.models import (
    Building,
    BuildingSurfaceDetailed,
    BuildingSurfaceDetailedVerticesItem,
    Construction,
    FenestrationSurfaceDetailed,
    GlobalGeometryRules,
    HVACTemplateThermostat,
    HVACTemplateZoneIdealLoadsAirSystem,
    Lights,
    Material,
    MaterialAirGap,
    MaterialNoMass,
    OutputControlTableStyle,
    OutputDiagnostics,
    OutputDiagnosticsDiagnosticsItem,
    OutputTableSummaryReports,
    OutputTableSummaryReportsReportsItem,
    OutputVariable,
    OutputVariableDictionary,
    People,
    RunPeriod,
    ScheduleCompact,
    ScheduleCompactDataItem,
    ScheduleTypeLimits,
    SimulationControl,
    SiteLocation,
    Timestep,
    Version,
    WindowMaterialGlazing,
    WindowMaterialSimpleGlazingSystem,
    Zone,
)
from loguru import logger
from omegaconf import OmegaConf
from pydantic import Field, PrivateAttr

from src.mcp.geometry import (
    NORMAL_DOT_TOLERANCE,
    surface_normal,
    surface_vertices,
)
from src.mcp.interface import ConfigSummary
from src.validator import (
    BaseSchema,
    BuildingSchema,
    ConstructionSchema,
    FenestrationSurfaceSchema,
    GlobalGeometryRulesSchema,
    HVACSchema,
    LightSchema,
    MaterialSchema,
    OutputControlTableStyleSchema,
    OutputDiagnosticsSchema,
    OutputTableSummaryReportsSchema,
    OutputVariableDictionarySchema,
    OutputVariableSchema,
    PeopleSchema,
    RunPeriodSchema,
    ScheduleCollectionSchema,
    SimulationControlSchema,
    SiteLocationSchema,
    SurfaceSchema,
    ZoneSchema,
)


def _snake(name: str) -> str:
    return (
        name.replace(":", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace("(", "_")
        .replace(")", "_")
        .replace(" ", "_")
        .lower()
    )


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    snake_lookup = {_snake(str(k)): v for k, v in data.items()}
    for key in keys:
        sk = _snake(key)
        if sk in snake_lookup:
            return snake_lookup[sk]
    return default


def _clean_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def _yes_no(value: Any) -> Any:
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return value


def _as_items(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, list):
        return [v for v in value if isinstance(v, dict)]
    if isinstance(value, dict):
        return [value]
    return []


def _vertices(value: Any) -> list[dict[str, float]]:
    result: list[dict[str, float]] = []
    for vertex in value or []:
        if isinstance(vertex, dict):
            result.append(
                {
                    "X": float(_get(vertex, "X", "x")),
                    "Y": float(_get(vertex, "Y", "y")),
                    "Z": float(_get(vertex, "Z", "z")),
                }
            )
        elif isinstance(vertex, (list, tuple)) and len(vertex) >= 3:
            result.append(
                {"X": float(vertex[0]), "Y": float(vertex[1]), "Z": float(vertex[2])}
            )
    return result


def _idf_values(idf: IDF, *object_types: str) -> list[Any]:
    values: list[Any] = []
    seen: set[int] = set()
    for object_type in object_types:
        try:
            objs = idf.all_of_type(object_type)
        except Exception:
            continue
        for obj in objs.values():
            marker = id(obj)
            if marker not in seen:
                seen.add(marker)
                values.append(obj)
    return values


def _idf_has(idf: IDF, name: str, *object_types: str) -> bool:
    return any(
        getattr(obj, "name", None) == name for obj in _idf_values(idf, *object_types)
    )


def _field_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    return dict(getattr(obj, "__dict__", {}))


# Root directory that user-facing IDF/YAML exports are confined to. Exports are
# resolved against this and rejected if they escape it, preventing directory
# traversal (e.g. "../../etc/passwd" or absolute paths outside the project).
_EXPORT_ROOT = Path(__file__).resolve().parent.parent.parent


def _safe_output_path(output_path: str | Path) -> Path:
    """Resolve an export path and confine it to the project root.

    Raises ``ValueError`` if the resolved path escapes ``_EXPORT_ROOT``. This
    guards MCP-facing ``export_idf`` / ``export_yaml`` endpoints from clients
    that pass traversal sequences or absolute paths outside the workspace.
    """
    raw = Path(output_path)
    base = raw if raw.is_absolute() else (_EXPORT_ROOT / raw)
    try:
        resolved = base.resolve(strict=False)
        resolved.relative_to(_EXPORT_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"Refusing to write outside the project directory: {output_path!r} "
            f"resolves to {resolved}, which is outside {_EXPORT_ROOT}."
        ) from exc
    return resolved


class ConfigState(BaseSchema):
    """MCP configuration state backed by an ``idfpy.IDF`` object.

    The public Pydantic fields are retained for compatibility with the
    existing LangGraph state merge code. MCP tools, resources, YAML loading
    and simulation now operate on ``self.idf`` directly instead of generating
    a YAML dict as an intermediate representation.
    """

    building: BuildingSchema | None = Field(default=None, alias="Building")
    site_location: SiteLocationSchema | None = Field(
        default=None, alias="Site:Location"
    )
    zones: list[ZoneSchema] = Field(default_factory=list, alias="Zone")
    materials: list[MaterialSchema] = Field(default_factory=list, alias="Material")
    constructions: list[ConstructionSchema] = Field(
        default_factory=list, alias="Construction"
    )
    surfaces: list[SurfaceSchema] = Field(
        default_factory=list, alias="BuildingSurface:Detailed"
    )
    fenestrations: list[FenestrationSurfaceSchema] = Field(
        default_factory=list, alias="FenestrationSurface:Detailed"
    )
    schedules: ScheduleCollectionSchema = Field(
        default_factory=ScheduleCollectionSchema, alias="Schedule"
    )
    people: list[PeopleSchema] = Field(default_factory=list, alias="People")
    lights: list[LightSchema] = Field(default_factory=list, alias="Light")
    hvac: HVACSchema = Field(default_factory=HVACSchema, alias="HVAC")
    simulation_control: SimulationControlSchema = Field(
        default_factory=SimulationControlSchema, alias="SimulationControl"
    )
    global_geometry_rules: GlobalGeometryRulesSchema = Field(
        default_factory=GlobalGeometryRulesSchema, alias="GlobalGeometryRules"
    )
    run_period: RunPeriodSchema = Field(
        default_factory=RunPeriodSchema, alias="RunPeriod"
    )
    output_variable_dictionary: OutputVariableDictionarySchema = Field(
        default_factory=OutputVariableDictionarySchema,
        alias="Output:VariableDictionary",
    )
    output_diagnostics: OutputDiagnosticsSchema = Field(
        default_factory=OutputDiagnosticsSchema, alias="Output:Diagnostics"
    )
    output_table_summary_reports: OutputTableSummaryReportsSchema = Field(
        default_factory=OutputTableSummaryReportsSchema,
        alias="Output:Table:SummaryReports",
    )
    output_variable: list[OutputVariableSchema] = Field(
        default_factory=list, alias="Output:Variable"
    )
    output_control_table_style: OutputControlTableStyleSchema = Field(
        default_factory=OutputControlTableStyleSchema, alias="OutputControl:Table:Style"
    )

    seed_idf_text: str = Field(default="", repr=False)
    """Serialized IDF text carried inside ConfigState for revision turns.

    LangGraph's START-boundary input coercion strips ConfigState's
    PrivateAttr ``_idf`` (the loaded seed model) before any node runs.
    This declared field survives that coercion, and
    :func:`merge_config_state` rebuilds ``_idf`` from it when the merged
    state's ``_idf`` is otherwise empty. This is the authoritative
    recovery path for revision-turn seed models.

    Note: must NOT use ``exclude=True`` — that would make model_dump drop
    it, and LangGraph's Pydantic-based checkpoint serialization would
    then strip it just like _idf.
    """

    _idf: IDF | None = PrivateAttr(default=None)

    def model_post_init(self, __context: Any) -> None:
        if self._idf is None:
            self._set_idf_private(IDF())

    def _set_idf_private(self, idf: IDF) -> None:
        """Assign ``_idf`` into ``__pydantic_private__`` directly.

        BaseSchema enables ``validate_assignment=True`` AND declares
        ``_idf`` as a class-level annotation, so a plain ``self._idf = ...``
        goes through Pydantic's assignment handler and silently fails to
        store the value. ``object.__setattr__`` would store it in
        ``__dict__`` instead of ``__pydantic_private__``, breaking
        pickling (Pydantic's ``__reduce_ex__`` only serializes
        ``__pydantic_private__``). Writing the PrivateAttr dict directly
        satisfies both constraints.
        """
        priv = self.__pydantic_private__
        if priv is None:
            priv = {}
            object.__setattr__(self, "__pydantic_private__", priv)
        priv["_idf"] = idf

    def _has_idf_objects(self) -> bool:
        """True if the backing IDF contains any typed model objects."""
        idf = self._idf
        if idf is None:
            return False
        try:
            return any(
                idf.all_of_type(t)
                for t in (
                    "Zone",
                    "Material",
                    "Material:NoMass",
                    "Material:AirGap",
                    "WindowMaterial:SimpleGlazingSystem",
                    "WindowMaterial:Glazing",
                    "Construction",
                    "BuildingSurface:Detailed",
                    "FenestrationSurface:Detailed",
                    "Schedule:Compact",
                    "ScheduleTypeLimits",
                    "HVACTemplate:Thermostat",
                    "HVACTemplate:Zone:IdealLoadsAirSystem",
                    "People",
                    "Lights",
                )
            )
        except Exception:
            return False

    def recover_idf_from_seed(self) -> bool:
        """Rebuild ``_idf`` from ``seed_idf_text`` if it's been stripped.

        LangGraph's input coercion at the graph START (and at each
        checkpoint write) strips the PrivateAttr ``_idf`` from
        ConfigState. On revision turns the seed model is carried as the
        declared ``seed_idf_text`` field (which survives coercion), and
        this method rebuilds ``_idf`` from it. Returns True if recovery
        happened. Safe to call at the top of every phase node — no-op
        when ``_idf`` already has objects or no seed text is present.

        Note: we assign via ``__pydantic_private__`` directly because
        ``BaseSchema`` enables ``validate_assignment=True`` AND declares
        ``_idf`` as a class-level annotation. A plain ``self._idf = ...``
        goes through Pydantic's assignment handler and silently fails to
        store the value, while ``object.__setattr__`` stores it in
        ``__dict__`` (not ``__pydantic_private__``), which breaks
        pickling because Pydantic's ``__reduce_ex__`` only knows about
        ``__pydantic_private__``. Writing the PrivateAttr dict directly
        satisfies both constraints.
        """
        if self._has_idf_objects() or not self.seed_idf_text:
            return False
        import tempfile

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".idf", delete=False
            ) as tf:
                tf.write(self.seed_idf_text)
                tf.flush()
                tmp_path = Path(tf.name)
            loaded = IDF.load(tmp_path)
            new_idf = IDF.from_dict(loaded.to_dict())
            self._set_idf_private(new_idf)
            return True
        except Exception:
            return False
        finally:
            if tmp_path is not None:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()

    def __reduce_ex__(self, protocol: SupportsIndex):
        """Pickle protocol: serialize IDF as text to avoid weakref.

        idfpy's IDF holds weakref internals that break pickle. We intercept
        pickling: capture IDF as text + model_dump, rebuild via IDF.from_dict
        on unpickle. This makes ConfigState embeddable in AgentState/dict for
        the LangGraph checkpointer.
        """
        import tempfile

        idf_text = ""
        if self._idf is not None:
            with tempfile.NamedTemporaryFile(suffix=".idf", delete=False) as tf:
                self._idf.save(Path(tf.name))
                idf_text = Path(tf.name).read_text(encoding="utf-8")
                os.unlink(tf.name)
        return (_reconstruct_config_state, (idf_text, self.model_dump(by_alias=True)))

    def clone(self) -> ConfigState:
        """Deep copy that produces a pickle-safe ConfigState.

        idfpy IDF objects loaded via ``IDF.load`` hold weakref internals that
        break ``pickle`` / ``model_copy(deep=True)`` / ``copy.deepcopy`` —
        which crashes LangGraph's InMemorySaver checkpointer on revision turns
        (where config_state is rebuilt from a saved IDF).

        This method copies all Pydantic fields normally (nested schemas keep
        their types) and rebuilds the IDF via ``IDF().from_dict(...)``, which
        produces a fresh, weakref-free IDF that pickles cleanly. Use ``clone()``
        instead of ``model_copy(deep=True)`` wherever a ConfigState is mutated
        in-place (phase agents, simulate, revise).
        """
        new = self.__class__(**self.model_dump(by_alias=True, exclude_defaults=False))
        # seed_idf_text is excluded=True so model_dump drops it; carry it
        # manually so phase-node clones can still recover _idf from it.
        new.seed_idf_text = self.seed_idf_text
        if self._idf is not None:
            new._set_idf_private(IDF.from_dict(self._idf.to_dict()))
        else:
            new._set_idf_private(IDF())
        return new

    @property
    def idf(self) -> IDF:
        if self._idf is None:
            idf = IDF()
            self._set_idf_private(idf)
            return idf
        return self._idf

    def new_idf(self) -> None:
        self._set_idf_private(IDF())

    def to_yaml_dict(self) -> dict[str, Any]:
        """Serialize the current IDF contents into a YAML-friendly dict."""
        data: dict[str, Any] = {}
        singleton_map = {
            "Version": ("Version",),
            "SimulationControl": ("SimulationControl",),
            "Building": ("Building",),
            "Timestep": ("Timestep",),
            "Site:Location": ("Site:Location",),
            "RunPeriod": ("RunPeriod",),
            "GlobalGeometryRules": ("GlobalGeometryRules",),
            "Output:VariableDictionary": ("Output:VariableDictionary",),
            "Output:Diagnostics": ("Output:Diagnostics",),
            "Output:Table:SummaryReports": ("Output:Table:SummaryReports",),
            "OutputControl:Table:Style": ("OutputControl:Table:Style",),
        }
        for yaml_key, object_types in singleton_map.items():
            values = _idf_values(self.idf, *object_types)
            if values:
                data[yaml_key] = _field_dict(values[0])

        grouped_map = {
            "Zone": ("Zone",),
            "Material": (
                "Material",
                "Material:NoMass",
                "MaterialNoMass",
                "Material:AirGap",
                "MaterialAirGap",
                "WindowMaterial:SimpleGlazingSystem",
                "WindowMaterial:Glazing",
            ),
            "Construction": ("Construction",),
            "BuildingSurface:Detailed": ("BuildingSurface:Detailed",),
            "FenestrationSurface:Detailed": ("FenestrationSurface:Detailed",),
            "People": ("People",),
            "Light": ("Lights", "Light"),
            "Output:Variable": ("Output:Variable",),
        }
        for yaml_key, object_types in grouped_map.items():
            values = [_field_dict(obj) for obj in _idf_values(self.idf, *object_types)]
            if values:
                data[yaml_key] = values

        schedule = {}
        type_limits = [
            _field_dict(obj) for obj in _idf_values(self.idf, "ScheduleTypeLimits")
        ]
        schedules = [
            _field_dict(obj)
            for obj in _idf_values(self.idf, "Schedule:Compact", "ScheduleCompact")
        ]
        if type_limits:
            schedule["ScheduleTypeLimits"] = type_limits
        if schedules:
            schedule["Schedule:Compact"] = schedules
        if schedule:
            data["Schedule"] = schedule

        hvac = {}
        thermostats = [
            _field_dict(obj) for obj in _idf_values(self.idf, "HVACTemplate:Thermostat")
        ]
        ideal_loads = [
            _field_dict(obj)
            for obj in _idf_values(self.idf, "HVACTemplate:Zone:IdealLoadsAirSystem")
        ]
        if thermostats:
            hvac["HVACTemplate:Thermostat"] = thermostats
        if ideal_loads:
            hvac["HVACTemplate:Zone:IdealLoadsAirSystem"] = ideal_loads
        if hvac:
            data["HVAC"] = hvac

        if not data:
            return self.model_dump(
                by_alias=True, exclude_none=True, serialize_as_any=True
            )
        return data

    def export_yaml(self, output_path: str | Path) -> None:
        path = _safe_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(config=self.to_yaml_dict(), f=path)
        logger.info("Exported YAML-like IDF snapshot to {}", path)

    def save_idf(self, output_path: str | Path) -> Path:
        path = _safe_output_path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.sync_legacy_fields_to_idf()
        self.idf.save(path)
        logger.info("Saved IDF to {}", path)
        return path

    def load_idf(self, input_path: str | Path) -> None:
        # Rebuild via IDF().from_dict to avoid weakref internals that
        # IDF.load introduces — those break pickle / deepcopy and crash the
        # LangGraph checkpointer on revision turns.
        loaded = IDF.load(Path(input_path))
        self._set_idf_private(IDF.from_dict(loaded.to_dict()))

    @classmethod
    def load_yaml(cls, input_path: str | Path) -> ConfigState:
        state = cls()
        state.load_yaml_into_idf(input_path)
        return state

    def load_yaml_into_idf(self, input_path: str | Path) -> None:
        path = Path(input_path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        self.clear()
        self.add_yaml_data(data)
        logger.info("Loaded YAML directly into IDF from {}", path)

    def add_yaml_data(self, data: dict[str, Any]) -> None:
        self._add_global_settings(data)
        self._add_materials(data)
        self._add_constructions(data)
        self._add_zones(data)
        self._add_surfaces(data)
        self._add_fenestrations(data)
        self._add_schedules(data)
        self._add_hvac(data)
        self._add_people(data)
        self._add_lights(data)
        self._add_outputs(data)

    def clear(self) -> None:
        self.new_idf()
        # Drop any carried seed so a subsequent recover_idf_from_seed() can't
        # restore the model that was just cleared (relevant on revision turns
        # or after load_idf()).
        self.seed_idf_text = ""
        self.building = None
        self.site_location = None
        self.zones.clear()
        self.materials.clear()
        self.constructions.clear()
        self.surfaces.clear()
        self.fenestrations.clear()
        self.schedules.schedule_type_limits.clear()
        self.schedules.schedules.clear()
        self.people.clear()
        self.lights.clear()
        self.hvac.thermostats.clear()
        self.hvac.ideal_loads_systems.clear()
        self.global_geometry_rules = GlobalGeometryRulesSchema()
        self.simulation_control = SimulationControlSchema()
        self.run_period = RunPeriodSchema()
        self.output_variable_dictionary = OutputVariableDictionarySchema()
        self.output_diagnostics = OutputDiagnosticsSchema()
        self.output_table_summary_reports = OutputTableSummaryReportsSchema()
        self.output_control_table_style = OutputControlTableStyleSchema()
        self.output_variable = []

    def update_from(self, other: ConfigState) -> None:
        self._set_idf_private(other.idf)
        for field_name in self.model_fields:
            setattr(self, field_name, getattr(other, field_name))

    def get_summary(self) -> ConfigSummary:
        return ConfigSummary(
            building=self._first_dump("Building"),
            site_location=self._first_dump("Site:Location"),
            zones_count=len(_idf_values(self.idf, "Zone")),
            materials_count=len(
                _idf_values(
                    self.idf,
                    "Material",
                    "Material:NoMass",
                    "MaterialNoMass",
                    "Material:AirGap",
                    "MaterialAirGap",
                    "WindowMaterial:SimpleGlazingSystem",
                    "WindowMaterial:Glazing",
                )
            ),
            constructions_count=len(
                _idf_values(self.idf, "Construction", "Construction:AirBoundary")
            ),
            surfaces_count=len(_idf_values(self.idf, "BuildingSurface:Detailed")),
            fenestrations_count=len(
                _idf_values(self.idf, "FenestrationSurface:Detailed")
            ),
            schedules_count=len(
                _idf_values(self.idf, "Schedule:Compact", "ScheduleCompact")
            ),
            hvac_thermostats_count=len(
                _idf_values(self.idf, "HVACTemplate:Thermostat")
            ),
            hvac_ideal_loads_count=len(
                _idf_values(self.idf, "HVACTemplate:Zone:IdealLoadsAirSystem")
            ),
            simulation_control=self._first_dump("SimulationControl"),
            run_period=self._first_dump("RunPeriod"),
            global_geometry_rules=self._first_dump("GlobalGeometryRules"),
        )

    def validate_references(self) -> list[str]:
        errors: list[str] = []
        material_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(
                self.idf,
                "Material",
                "Material:NoMass",
                "MaterialNoMass",
                "Material:AirGap",
                "MaterialAirGap",
                "WindowMaterial:SimpleGlazingSystem",
                "WindowMaterial:Glazing",
            )
        }
        # Glazing (window) material names only. EnergyPlus treats a
        # construction as a "window construction" iff at least one of its
        # layers is a WindowMaterial:* object — the agent may create either
        # SimpleGlazingSystem (whole-window equivalent) or Glazing (a true
        # per-pane glass layer). Used to flag fenestrations
        # (Window / GlassDoor / TubularDaylight*) whose construction is
        # opaque, which makes EnergyPlus abort with "has an opaque surface
        # construction; it should have a window construction".
        glazing_material_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(
                self.idf,
                "WindowMaterial:SimpleGlazingSystem",
                "WindowMaterial:Glazing",
            )
        }
        # SimpleGlazingSystem is a WHOLE-WINDOW model: it collapses the entire
        # fenestration into a single U-factor/SHGC/VT. It therefore must be
        # the ONLY layer of a Construction. Combining it with other layers
        # (e.g. Glass + AirGap + Glass) gives EnergyPlus no per-pane optical
        # data, so SolveForWindowTemperatures diverges and the run aborts
        # with a Fatal "Convergence error ... new temperature = NaN". Window
        # Material:Glazing, by contrast, is a real per-pane layer and may be
        # composed with gas gaps in multi-pane assemblies. Names only; used
        # by the sole-layer check below.
        simple_glazing_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "WindowMaterial:SimpleGlazingSystem")
        }
        # Construction layer fields (outside_layer + layer_2..layer_10).
        # Defined once here so both the glazing-construction computation and
        # the dangling-material check below share it.
        layer_fields = [
            "outside_layer",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
            "layer_6",
            "layer_7",
            "layer_8",
            "layer_9",
            "layer_10",
        ]
        construction_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "Construction", "Construction:AirBoundary")
        }
        # Constructions that qualify as "window constructions" — at least one
        # layer points at a WindowMaterial:SimpleGlazingSystem. Computed once
        # here so the fenestration check below is O(1) per surface.
        glazing_construction_names: set[str] = set()
        for const in _idf_values(self.idf, "Construction"):
            for lf in layer_fields:
                layer = getattr(const, lf, None)
                if layer and layer in glazing_material_names:
                    glazing_construction_names.add(const.name)
                    break
        # Layer-less open-air constructions used on interior subsurfaces
        # (doors/windows between two zones) to avoid the "invalid blank
        # Outside Boundary Condition Object" severe error. A subsurface on a
        # 'Surface'-boundary parent must be one of these.
        airboundary_construction_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "Construction:AirBoundary")
        }
        surface_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "BuildingSurface:Detailed")
        }
        # Parent surface name -> its outside boundary condition, so the
        # fenestration check can tell interior ('Surface') parents apart.
        surface_boundary: dict[str, str] = {
            getattr(obj, "name", ""): getattr(obj, "outside_boundary_condition", "")
            for obj in _idf_values(self.idf, "BuildingSurface:Detailed")
        }
        zone_names = {getattr(obj, "name", "") for obj in _idf_values(self.idf, "Zone")}
        schedule_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "Schedule:Compact", "ScheduleCompact")
        }
        thermostat_names = {
            getattr(obj, "name", "")
            for obj in _idf_values(self.idf, "HVACTemplate:Thermostat")
        }

        for const in _idf_values(self.idf, "Construction"):
            const_layers = [getattr(const, lf, None) for lf in layer_fields]
            for layer in const_layers:
                if layer and layer not in material_names:
                    errors.append(
                        f"Construction '{const.name}' references material '{layer}' which does not exist."
                    )
            # SimpleGlazingSystem is a whole-window equivalent (U/SHGC/VT
            # only). If it appears as one of several layers, EnergyPlus
            # cannot solve per-pane glass temperatures and aborts with a
            # Fatal convergence error. Require it to be the sole layer.
            # The error is prefixed "Construction '" so _ERROR_PATTERNS
            # routes it to the construction phase for self-repair. A
            # multi-pane window should instead use WindowMaterial:Glazing
            # layers (create_glazing_layer_material), which carry true
            # per-pane optical/thermal data and may be composed with gas gaps.
            simple_layers = [
                layer
                for layer in const_layers
                if layer and layer in simple_glazing_names
            ]
            if simple_layers and any(
                layer for layer in const_layers if layer not in simple_glazing_names
            ):
                errors.append(
                    f"Construction '{const.name}' uses WindowMaterial:"
                    f"SimpleGlazingSystem ({', '.join(simple_layers)}) together "
                    f"with other layers. SimpleGlazingSystem is a whole-window "
                    f"model (U/SHGC/VT only) and must be the ONLY layer — "
                    f"combining it with gas gaps / extra panes makes EnergyPlus "
                    f"abort with a Fatal convergence error. Either rebuild this "
                    f"construction as a single SimpleGlazingSystem layer, or "
                    f"replace the per-pane glass with WindowMaterial:Glazing "
                    f"(create_glazing_layer_material) which can be composed "
                    f"with Material:AirGap in a multi-pane assembly."
                )

        for surface in _idf_values(self.idf, "BuildingSurface:Detailed"):
            if surface.construction_name not in construction_names:
                errors.append(
                    f"Surface '{surface.name}' references construction '{surface.construction_name}' which does not exist."
                )
            if surface.zone_name not in zone_names:
                errors.append(
                    f"Surface '{surface.name}' references zone '{surface.zone_name}' which does not exist."
                )
            # Auto-correct flipped Floor / Roof / Ceiling normals. EnergyPlus
            # expects (under the Counterclockwise vertex-entry convention) a
            # Floor's outward normal to point DOWN (tilt 180) and a
            # Roof/Ceiling's to point UP (tilt 0). When the LLM lists floor
            # corners in plan-view CCW order the normal ends up pointing +Z,
            # which EnergyPlus flags as "Floor is upside down! Tilt=0.0,
            # should be near 180". We silently reverse the vertex order to
            # flip the normal — no error, no repair round (mirrors how window
            # normals are auto-aligned in fenestration_tools). Tolerates near-
            # horizontal surfaces; only clearly-wrong normals are flipped.
            if surface.surface_type in ("Floor", "Roof", "Ceiling"):
                verts = surface_vertices(surface)
                if len(verts) >= 3:
                    _nx, _ny, nz = surface_normal(verts)
                    # Only flip when the normal is decisively off the expected
                    # vertical axis (|z| well above the tolerance), so a
                    # tilted/sloped surface is left untouched.
                    flipped = False
                    if (
                        surface.surface_type == "Floor" and nz > NORMAL_DOT_TOLERANCE
                    ) or (
                        surface.surface_type in ("Roof", "Ceiling")
                        and nz < -NORMAL_DOT_TOLERANCE
                    ):
                        flipped = True
                    if flipped and getattr(surface, "vertices", None):
                        surface.vertices.reverse()

        # Fenestration surface_types that EnergyPlus requires to be backed by
        # a window (glazing) construction. A plain Door is the only type that
        # may use an opaque construction.
        _glazing_surface_types = {
            "Window",
            "GlassDoor",
            "TubularDaylightDiffuser",
            "TubularDaylightDome",
        }
        for fen in _idf_values(self.idf, "FenestrationSurface:Detailed"):
            if fen.construction_name not in construction_names:
                errors.append(
                    f"Fenestration '{fen.name}' references construction '{fen.construction_name}' which does not exist."
                )
            elif (
                fen.surface_type in _glazing_surface_types
                and fen.construction_name not in glazing_construction_names
                # An AirBoundary construction is the legitimate construction
                # for interior windows/glass-doors (open-air passage between
                # zones); it has no WindowMaterial, so the glazing check must
                # NOT flag it. The dedicated interzone check below governs it.
                and fen.construction_name not in airboundary_construction_names
            ):
                # EnergyPlus aborts: "FenestrationSurface:Detailed has an
                # opaque surface construction; it should have a window
                # construction". The construction owns the layer composition,
                # so this is routed at construction via _ERROR_PATTERNS.
                errors.append(
                    f"Construction '{fen.construction_name}' is referenced by "
                    f"fenestration '{fen.name}' ({fen.surface_type}) but has no "
                    f"WindowMaterial layer — it is opaque. Rebuild this "
                    f"construction to use a WindowMaterial:SimpleGlazingSystem "
                    f"glazing layer."
                )
            if fen.building_surface_name not in surface_names:
                errors.append(
                    f"Fenestration '{fen.name}' references building surface '{fen.building_surface_name}' which does not exist."
                )
            elif (
                surface_boundary.get(fen.building_surface_name) == "Surface"
                and fen.construction_name not in airboundary_construction_names
                and not getattr(fen, "outside_boundary_condition_object", None)
            ):
                # Parent is a zone-separating wall ('Surface' boundary). A
                # regular layered subsurface construction here leaves its own
                # Outside Boundary Condition Object blank, which EnergyPlus
                # rejects ("invalid blank Outside Boundary Condition Object").
                # The fix is to model the interior door/window with a
                # Construction:AirBoundary. Routed to construction (which owns
                # construction creation — only it can build the AirBoundary);
                # the error string starts with "Construction '" so _ERROR_PATTERNS
                # matches the ("Construction '", "construction") rule.
                errors.append(
                    f"Construction '{fen.construction_name}' used by fenestration "
                    f"'{fen.name}' on interior wall "
                    f"'{fen.building_surface_name}' (Surface boundary) is not a "
                    f"Construction:AirBoundary. Interior subsurfaces must use "
                    f"a Construction:AirBoundary (open-air) construction."
                )

        for ils in _idf_values(self.idf, "HVACTemplate:Zone:IdealLoadsAirSystem"):
            if ils.zone_name not in zone_names:
                errors.append(
                    f"Ideal load system references zone '{ils.zone_name}' which does not exist."
                )
            if ils.template_thermostat_name not in thermostat_names:
                errors.append(
                    f"Ideal load system references thermostat '{ils.template_thermostat_name}' which does not exist."
                )
            avail = getattr(ils, "system_availability_schedule_name", None)
            if avail and avail not in schedule_names:
                errors.append(
                    f"Ideal load system for zone '{ils.zone_name}' references availability schedule '{avail}' which does not exist."
                )

        for thermostat in _idf_values(self.idf, "HVACTemplate:Thermostat"):
            if thermostat.heating_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references heating setpoint schedule '{thermostat.heating_setpoint_schedule_name}' which does not exist."
                )
            if thermostat.cooling_setpoint_schedule_name not in schedule_names:
                errors.append(
                    f"Thermostat '{thermostat.name}' references cooling setpoint schedule '{thermostat.cooling_setpoint_schedule_name}' which does not exist."
                )

        for people in _idf_values(self.idf, "People"):
            zone = getattr(people, "zone_or_zonelist_or_space_or_spacelist_name", "")
            if zone and zone not in zone_names:
                errors.append(
                    f"People '{people.name}' references zone '{zone}' which does not exist."
                )
            for field in (
                "number_of_people_schedule_name",
                "activity_level_schedule_name",
            ):
                sched = getattr(people, field, None)
                if sched and sched not in schedule_names:
                    errors.append(
                        f"People '{people.name}' references schedule '{sched}' which does not exist."
                    )

        for light in _idf_values(self.idf, "Lights", "Light"):
            zone = getattr(light, "zone_or_zonelist_or_space_or_spacelist_name", None)
            zone = zone or getattr(
                light, "zone_or_zone_list_or_space_or_space_list_name", ""
            )
            if zone and zone not in zone_names:
                errors.append(
                    f"Lights '{light.name}' references zone '{zone}' which does not exist."
                )
            sched = getattr(light, "schedule_name", None)
            if sched and sched not in schedule_names:
                errors.append(
                    f"Lights '{light.name}' references schedule '{sched}' which does not exist."
                )

        return errors

    def sync_legacy_fields_to_idf(self) -> None:
        """Mirror compatibility Pydantic fields into IDF when they were used."""
        legacy = self.model_dump(
            by_alias=True, exclude_none=True, serialize_as_any=True
        )
        if legacy:
            self.add_yaml_data(legacy)

    def _first_dump(self, *object_types: str) -> dict[str, Any] | None:
        values = _idf_values(self.idf, *object_types)
        return _field_dict(values[0]) if values else None

    def _add_global_settings(self, data: dict[str, Any]) -> None:
        if "Version" in data and not _idf_values(self.idf, "Version"):
            raw = data["Version"]
            version = _get(
                raw, "Version Identifier", "version", default=self.idf.version
            )
            self.idf.add(Version(version_identifier=version))

        raw = data.get("SimulationControl")
        if raw and not _idf_values(self.idf, "SimulationControl"):
            self.idf.add(
                SimulationControl(
                    **_clean_kwargs(
                        {
                            "do_zone_sizing_calculation": _yes_no(
                                _get(raw, "Do Zone Sizing Calculation")
                            ),
                            "do_system_sizing_calculation": _yes_no(
                                _get(raw, "Do System Sizing Calculation")
                            ),
                            "do_plant_sizing_calculation": _yes_no(
                                _get(raw, "Do Plant Sizing Calculation")
                            ),
                            "run_simulation_for_sizing_periods": _yes_no(
                                _get(raw, "Run Simulation for Sizing Periods")
                            ),
                            "run_simulation_for_weather_file_run_periods": _yes_no(
                                _get(raw, "Run Simulation for Weather File Run Periods")
                            ),
                            "do_hvac_sizing_simulation_for_sizing_periods": _yes_no(
                                _get(
                                    raw, "Do HVAC Sizing Simulation for Sizing Periods"
                                )
                            ),
                            "maximum_number_of_hvac_sizing_simulation_passes": _get(
                                raw, "Maximum Number of HVAC Sizing Simulation Passes"
                            ),
                        }
                    )
                )
            )

        raw = data.get("Building")
        if raw and not _idf_values(self.idf, "Building"):
            self.idf.add(
                Building(
                    **_clean_kwargs(
                        {
                            "name": _get(raw, "Name", "name"),
                            "north_axis": _get(raw, "North Axis", default=0.0),
                            "terrain": _get(raw, "Terrain", default="Suburbs"),
                            "loads_convergence_tolerance_value": _get(
                                raw, "Loads Convergence Tolerance Value"
                            ),
                            "temperature_convergence_tolerance_value": _get(
                                raw, "Temperature Convergence Tolerance Value"
                            ),
                            "solar_distribution": _get(raw, "Solar Distribution"),
                            "maximum_number_of_warmup_days": _get(
                                raw, "Maximum Number of Warmup Days"
                            ),
                            "minimum_number_of_warmup_days": _get(
                                raw, "Minimum Number of Warmup Days"
                            ),
                        }
                    )
                )
            )

        raw = data.get("Timestep")
        if raw and not _idf_values(self.idf, "Timestep"):
            self.idf.add(
                Timestep(
                    number_of_timesteps_per_hour=_get(
                        raw, "Number of Timesteps per Hour", default=4
                    )
                )
            )

        raw = data.get("Site:Location")
        if raw and not _idf_values(self.idf, "Site:Location"):
            self.idf.add(
                SiteLocation(
                    **_clean_kwargs(
                        {
                            "name": _get(raw, "Name", "name"),
                            "latitude": _get(raw, "Latitude"),
                            "longitude": _get(raw, "Longitude"),
                            "time_zone": _get(raw, "Time Zone"),
                            "elevation": _get(raw, "Elevation"),
                        }
                    )
                )
            )

        raw = data.get("RunPeriod")
        if raw and not _idf_values(self.idf, "RunPeriod"):
            self.idf.add(
                RunPeriod(
                    **_clean_kwargs(
                        {
                            "name": _get(raw, "Name", default="Run Period 1"),
                            "begin_month": _get(raw, "Begin Month"),
                            "begin_day_of_month": _get(raw, "Begin Day of Month"),
                            "begin_year": _get(raw, "Begin Year"),
                            "end_month": _get(raw, "End Month"),
                            "end_day_of_month": _get(raw, "End Day of Month"),
                            "end_year": _get(raw, "End Year"),
                            "day_of_week_for_start_day": _get(
                                raw, "Day of Week for Start Day"
                            ),
                            "use_weather_file_holidays_and_special_days": _yes_no(
                                _get(raw, "Use Weather File Holidays and Special Days")
                            ),
                            "use_weather_file_daylight_saving_period": _yes_no(
                                _get(raw, "Use Weather File Daylight Saving Period")
                            ),
                            "apply_weekend_holiday_rule": _yes_no(
                                _get(raw, "Apply Weekend Holiday Rule")
                            ),
                            "use_weather_file_rain_indicators": _yes_no(
                                _get(raw, "Use Weather File Rain Indicators")
                            ),
                            "use_weather_file_snow_indicators": _yes_no(
                                _get(raw, "Use Weather File Snow Indicators")
                            ),
                        }
                    )
                )
            )

        raw = data.get("GlobalGeometryRules")
        if raw and not _idf_values(self.idf, "GlobalGeometryRules"):
            self.idf.add(
                GlobalGeometryRules(
                    **_clean_kwargs(
                        {
                            "starting_vertex_position": _get(
                                raw, "Starting Vertex Position"
                            ),
                            "vertex_entry_direction": _get(
                                raw, "Vertex Entry Direction"
                            ),
                            "coordinate_system": _get(raw, "Coordinate System"),
                        }
                    )
                )
            )

    def _add_materials(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("Material")):
            name = _get(raw, "Name", "name")
            if not name:
                continue
            material_type = _get(raw, "Type", "type")
            if material_type is None:
                if _get(raw, "U-Factor", "U Factor", "u_factor") is not None:
                    material_type = "Glazing"
                elif (
                    _get(
                        raw,
                        "Thermal_Resistance",
                        "Thermal Resistance",
                        "thermal_resistance",
                    )
                    is not None
                    and _get(raw, "Roughness", "roughness") is not None
                ):
                    material_type = "NoMass"
                elif (
                    _get(
                        raw,
                        "Thermal_Resistance",
                        "Thermal Resistance",
                        "thermal_resistance",
                    )
                    is not None
                ):
                    material_type = "AirGap"
                else:
                    material_type = "Standard"
            material_type = str(material_type)
            if _idf_has(
                self.idf,
                name,
                "Material",
                "Material:NoMass",
                "MaterialNoMass",
                "Material:AirGap",
                "MaterialAirGap",
                "WindowMaterial:SimpleGlazingSystem",
                "WindowMaterial:Glazing",
            ):
                continue
            if material_type == "Standard":
                self.idf.add(
                    Material(
                        name=name,
                        roughness=_get(raw, "Roughness"),
                        thickness=_get(raw, "Thickness"),
                        conductivity=_get(raw, "Conductivity"),
                        density=_get(raw, "Density"),
                        specific_heat=_get(raw, "Specific_Heat", "Specific Heat"),
                    )
                )
            elif material_type == "NoMass":
                self.idf.add(
                    MaterialNoMass(
                        name=name,
                        roughness=_get(raw, "Roughness"),
                        thermal_resistance=_get(
                            raw, "Thermal_Resistance", "Thermal Resistance"
                        ),
                    )
                )
            elif material_type == "AirGap":
                self.idf.add(
                    MaterialAirGap(
                        name=name,
                        thermal_resistance=_get(
                            raw, "Thermal_Resistance", "Thermal Resistance"
                        ),
                    )
                )
            elif material_type == "Glazing":
                self.idf.add(
                    WindowMaterialSimpleGlazingSystem(
                        name=name,
                        u_factor=_get(raw, "U-Factor", "U Factor"),
                        solar_heat_gain_coefficient=_get(
                            raw,
                            "Solar_Heat_Gain_Coefficient",
                            "Solar Heat Gain Coefficient",
                        ),
                        visible_transmittance=_get(
                            raw, "Visible_Transmittance", "Visible Transmittance"
                        ),
                    )
                )
            elif material_type == "GlazingLayer":
                # A true per-pane glass layer (WindowMaterial:Glazing). Unlike
                # the whole-window SimpleGlazingSystem above, this carries
                # thickness + optical/thermal data per pane and MAY be composed
                # with gas gaps in multi-pane assemblies. optical_data_type is
                # fixed to SpectralAverage for the simplified YAML path.
                self.idf.add(
                    WindowMaterialGlazing(
                        **_clean_kwargs(
                            {
                                "name": name,
                                "optical_data_type": _get(
                                    raw,
                                    "Optical_Data_Type",
                                    "Optical Data Type",
                                    default="SpectralAverage",
                                ),
                                "thickness": _get(raw, "Thickness"),
                                "solar_transmittance_at_normal_incidence": _get(
                                    raw,
                                    "Solar_Transmittance_at_Normal_Incidence",
                                    "Solar Transmittance at Normal Incidence",
                                ),
                                "front_side_solar_reflectance_at_normal_incidence": _get(
                                    raw,
                                    "Front_Side_Solar_Reflectance_at_Normal_Incidence",
                                    "Front Side Solar Reflectance at Normal Incidence",
                                ),
                                "back_side_solar_reflectance_at_normal_incidence": _get(
                                    raw,
                                    "Back_Side_Solar_Reflectance_at_Normal_Incidence",
                                    "Back Side Solar Reflectance at Normal Incidence",
                                ),
                                "visible_transmittance_at_normal_incidence": _get(
                                    raw,
                                    "Visible_Transmittance_at_Normal_Incidence",
                                    "Visible Transmittance at Normal Incidence",
                                ),
                                "front_side_visible_reflectance_at_normal_incidence": _get(
                                    raw,
                                    "Front_Side_Visible_Reflectance_at_Normal_Incidence",
                                    "Front Side Visible Reflectance at Normal Incidence",
                                ),
                                "back_side_visible_reflectance_at_normal_incidence": _get(
                                    raw,
                                    "Back_Side_Visible_Reflectance_at_Normal_Incidence",
                                    "Back Side Visible Reflectance at Normal Incidence",
                                ),
                                "infrared_transmittance_at_normal_incidence": _get(
                                    raw,
                                    "Infrared_Transmittance_at_Normal_Incidence",
                                    "Infrared Transmittance at Normal Incidence",
                                ),
                                "front_side_infrared_hemispherical_emissivity": _get(
                                    raw,
                                    "Front_Side_Infrared_Hemispherical_Emissivity",
                                    "Front Side Infrared Hemispherical Emissivity",
                                ),
                                "back_side_infrared_hemispherical_emissivity": _get(
                                    raw,
                                    "Back_Side_Infrared_Hemispherical_Emissivity",
                                    "Back Side Infrared Hemispherical Emissivity",
                                ),
                                "conductivity": _get(raw, "Conductivity"),
                                "dirt_correction_factor": _get(
                                    raw,
                                    "Dirt_Correction_Factor",
                                    "Dirt Correction Factor",
                                ),
                                "solar_diffusing": _get(
                                    raw, "Solar_Diffusing", "Solar Diffusing"
                                ),
                            }
                        )
                    )
                )

    def _add_constructions(self, data: dict[str, Any]) -> None:
        layer_fields = [
            "outside_layer",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
            "layer_6",
            "layer_7",
            "layer_8",
            "layer_9",
            "layer_10",
        ]
        for raw in _as_items(data.get("Construction")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "Construction"):
                continue
            layers = _get(raw, "Layers", "layers", default=[])
            kwargs = {"name": name}
            if layers:
                for idx, layer in enumerate(layers[: len(layer_fields)]):
                    kwargs[layer_fields[idx]] = layer
            else:
                for field in layer_fields:
                    value = _get(raw, field)
                    if value:
                        kwargs[field] = value
            self.idf.add(Construction(**kwargs))

    def _add_zones(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("Zone")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "Zone"):
                continue
            self.idf.add(
                Zone(
                    **_clean_kwargs(
                        {
                            "name": name,
                            "direction_of_relative_north": _get(
                                raw, "Direction of Relative North", default=0.0
                            ),
                            "x_origin": _get(raw, "X Origin", default=0.0),
                            "y_origin": _get(raw, "Y Origin", default=0.0),
                            "z_origin": _get(raw, "Z Origin", default=0.0),
                            "type": _get(raw, "Type"),
                            "multiplier": _get(raw, "Multiplier"),
                            "ceiling_height": _get(raw, "Ceiling Height"),
                            "volume": _get(raw, "Volume"),
                            "floor_area": _get(raw, "Floor Area"),
                            "zone_inside_convection_algorithm": _get(
                                raw, "Zone Inside Convection Algorithm"
                            ),
                            "zone_outside_convection_algorithm": _get(
                                raw, "Zone Outside Convection Algorithm"
                            ),
                            "part_of_total_floor_area": _yes_no(
                                _get(raw, "Part of Total Floor Area")
                            ),
                        }
                    )
                )
            )

    def _add_surfaces(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("BuildingSurface:Detailed")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "BuildingSurface:Detailed"):
                continue
            verts = _vertices(_get(raw, "Vertices", "vertices", default=[]))
            vertex_items = [
                BuildingSurfaceDetailedVerticesItem(
                    vertex_x_coordinate=v["X"],
                    vertex_y_coordinate=v["Y"],
                    vertex_z_coordinate=v["Z"],
                )
                for v in verts
            ]
            self.idf.add(
                BuildingSurfaceDetailed(
                    **_clean_kwargs(
                        {
                            "name": name,
                            "surface_type": _get(raw, "Surface Type"),
                            "construction_name": _get(raw, "Construction Name"),
                            "zone_name": _get(raw, "Zone Name"),
                            "space_name": _get(raw, "Space Name"),
                            "outside_boundary_condition": _get(
                                raw, "Outside Boundary Condition"
                            ),
                            "outside_boundary_condition_object": _get(
                                raw, "Outside Boundary Condition Object"
                            ),
                            "sun_exposure": _get(raw, "Sun Exposure"),
                            "wind_exposure": _get(raw, "Wind Exposure"),
                            "view_factor_to_ground": _get(raw, "View Factor to Ground"),
                            "number_of_vertices": len(verts)
                            if verts
                            else _get(raw, "Number of Vertices"),
                            "vertices": vertex_items,
                        }
                    )
                )
            )

    def _add_fenestrations(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("FenestrationSurface:Detailed")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "FenestrationSurface:Detailed"):
                continue
            verts = _vertices(_get(raw, "Vertices", "vertices", default=[]))
            kwargs = _clean_kwargs(
                {
                    "name": name,
                    "surface_type": _get(raw, "Surface Type"),
                    "construction_name": _get(raw, "Construction Name"),
                    "building_surface_name": _get(raw, "Building Surface Name"),
                    "outside_boundary_condition_object": _get(
                        raw, "Outside Boundary Condition Object"
                    ),
                    "frame_and_divider_name": _get(raw, "Frame and Divider Name"),
                    "multiplier": _get(raw, "Multiplier"),
                    "view_factor_to_ground": _get(raw, "View Factor to Ground"),
                    "number_of_vertices": len(verts)
                    if verts
                    else _get(raw, "Number of Vertices"),
                }
            )
            for idx, vertex in enumerate(verts, start=1):
                kwargs[f"vertex_{idx}_x_coordinate"] = vertex["X"]
                kwargs[f"vertex_{idx}_y_coordinate"] = vertex["Y"]
                kwargs[f"vertex_{idx}_z_coordinate"] = vertex["Z"]
            self.idf.add(FenestrationSurfaceDetailed(**kwargs))

    def _add_schedules(self, data: dict[str, Any]) -> None:
        schedule_data = data.get("Schedule") or {}
        for raw in _as_items(schedule_data.get("ScheduleTypeLimits")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "ScheduleTypeLimits"):
                continue
            lower = _get(raw, "Lower Limit Value")
            upper = _get(raw, "Upper Limit Value")
            self.idf.add(
                ScheduleTypeLimits(
                    **_clean_kwargs(
                        {
                            "name": name,
                            "lower_limit_value": None if lower == "" else lower,
                            "upper_limit_value": None if upper == "" else upper,
                            "numeric_type": _get(raw, "Numeric Type"),
                            "unit_type": _get(raw, "Unit Type"),
                        }
                    )
                )
            )

        for raw in _as_items(schedule_data.get("Schedule:Compact")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(
                self.idf, name, "Schedule:Compact", "ScheduleCompact"
            ):
                continue
            flat_data = _flatten_schedule_data(_get(raw, "Data", "data", default=[]))
            self.idf.add(
                ScheduleCompact(
                    name=name,
                    schedule_type_limits_name=_get(raw, "Schedule Type Limits Name"),
                    data=[
                        ScheduleCompactDataItem(field=str(item)) for item in flat_data
                    ],
                )
            )

    def _add_hvac(self, data: dict[str, Any]) -> None:
        hvac_data = data.get("HVAC") or {}
        for raw in _as_items(hvac_data.get("HVACTemplate:Thermostat")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "HVACTemplate:Thermostat"):
                continue
            self.idf.add(
                HVACTemplateThermostat(
                    name=name,
                    heating_setpoint_schedule_name=_get(
                        raw, "Heating Setpoint Schedule Name"
                    ),
                    cooling_setpoint_schedule_name=_get(
                        raw, "Cooling Setpoint Schedule Name"
                    ),
                )
            )

        existing_zones = {
            getattr(obj, "zone_name", None)
            for obj in _idf_values(self.idf, "HVACTemplate:Zone:IdealLoadsAirSystem")
        }
        for raw in _as_items(hvac_data.get("HVACTemplate:Zone:IdealLoadsAirSystem")):
            zone_name = _get(raw, "Zone Name", "zone_name")
            if not zone_name or zone_name in existing_zones:
                continue
            self.idf.add(
                HVACTemplateZoneIdealLoadsAirSystem(
                    zone_name=zone_name,
                    template_thermostat_name=_get(raw, "Template Thermostat Name"),
                    system_availability_schedule_name=_get(
                        raw, "System Availability Schedule Name"
                    ),
                )
            )
            existing_zones.add(zone_name)

    def _add_people(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("People")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "People"):
                continue
            self.idf.add(
                People(
                    **_clean_kwargs(
                        {
                            "name": name,
                            "zone_or_zonelist_or_space_or_spacelist_name": _get(
                                raw, "Zone or ZoneList or Space or SpaceList Name"
                            ),
                            "number_of_people_schedule_name": _get(
                                raw, "Number of People Schedule Name"
                            ),
                            "number_of_people_calculation_method": _get(
                                raw, "Number of People Calculation Method"
                            ),
                            "number_of_people": _get(raw, "Number of People"),
                            "people_per_floor_area": _get(raw, "People per Floor Area"),
                            "floor_area_per_person": _get(raw, "Floor Area per Person"),
                            "fraction_radiant": _get(raw, "Fraction Radiant"),
                            "sensible_heat_fraction": _get(
                                raw, "Sensible Heat Fraction"
                            ),
                            "activity_level_schedule_name": _get(
                                raw, "Activity Level Schedule Name"
                            ),
                            "carbon_dioxide_generation_rate": _get(
                                raw, "Carbon Dioxide Generation Rate"
                            ),
                            "enable_ashrae_55_comfort_warnings": _yes_no(
                                _get(raw, "Enable ASHRAE 55 Comfort Warnings")
                            ),
                            "mean_radiant_temperature_calculation_type": _get(
                                raw, "Mean Radiant Temperature Calculation Type"
                            ),
                            "surface_name_angle_factor_list_name": _get(
                                raw, "Surface Name Angle Factor List Name"
                            ),
                            "work_efficiency_schedule_name": _get(
                                raw, "Work Efficiency Schedule Name"
                            ),
                            "clothing_insulation_calculation_method": _get(
                                raw, "Clothing Insulation Calculation Method"
                            ),
                            "clothing_insulation_calculation_method_schedule_name": _get(
                                raw,
                                "Clothing Insulation Calculation Method Schedule Name",
                            ),
                            "clothing_insulation_schedule_name": _get(
                                raw, "Clothing Insulation Schedule Name"
                            ),
                            "air_velocity_schedule_name": _get(
                                raw, "Air Velocity Schedule Name"
                            ),
                        }
                    )
                )
            )

    def _add_lights(self, data: dict[str, Any]) -> None:
        for raw in _as_items(data.get("Light") or data.get("Lights")):
            name = _get(raw, "Name", "name")
            if not name or _idf_has(self.idf, name, "Lights", "Light"):
                continue
            self.idf.add(
                Lights(
                    **_clean_kwargs(
                        {
                            "name": name,
                            "zone_or_zonelist_or_space_or_spacelist_name": _get(
                                raw,
                                "Zone or ZoneList or Space or SpaceList Name",
                                "Zone or ZoneList or SpaceList Name",
                            ),
                            "schedule_name": _get(raw, "Schedule Name"),
                            "design_level_calculation_method": _get(
                                raw, "Design Level Calculation Method"
                            ),
                            "lighting_level": _get(raw, "Lighting Level"),
                            "watts_per_floor_area": _get(raw, "Watts per Floor Area"),
                            "watts_per_person": _get(raw, "Watts per Person"),
                            "return_air_fraction": _get(raw, "Return Air Fraction"),
                            "fraction_radiant": _get(raw, "Fraction Radiant"),
                            "fraction_visible": _get(raw, "Fraction Visible"),
                            "fraction_replaceable": _get(raw, "Fraction Replaceable"),
                            "end_use_subcategory": _get(raw, "End Use Subcategory"),
                        }
                    )
                )
            )

    def _add_outputs(self, data: dict[str, Any]) -> None:
        raw = data.get("Output:VariableDictionary")
        if raw and not _idf_values(self.idf, "Output:VariableDictionary"):
            self.idf.add(OutputVariableDictionary(key_field=_get(raw, "Key Field")))

        raw = data.get("Output:Diagnostics")
        if raw and not _idf_values(self.idf, "Output:Diagnostics"):
            key = _get(raw, "Key 1", "Key", "key_1")
            if key:
                self.idf.add(
                    OutputDiagnostics(
                        diagnostics=[OutputDiagnosticsDiagnosticsItem(key=key)]
                    )
                )

        raw = data.get("Output:Table:SummaryReports")
        if raw and not _idf_values(self.idf, "Output:Table:SummaryReports"):
            report = _get(raw, "Report 1 Name", "Report Name", "report_1_name")
            if report:
                self.idf.add(
                    OutputTableSummaryReports(
                        reports=[
                            OutputTableSummaryReportsReportsItem(report_name=report)
                        ]
                    )
                )

        raw = data.get("OutputControl:Table:Style")
        if raw and not _idf_values(self.idf, "OutputControl:Table:Style"):
            self.idf.add(
                OutputControlTableStyle(
                    **_clean_kwargs(
                        {
                            "column_separator": _get(raw, "Column Separator"),
                            "unit_conversion": _get(raw, "Unit Conversion"),
                        }
                    )
                )
            )

        existing_output_variables = {
            (
                getattr(obj, "key_value", None),
                getattr(obj, "variable_name", None),
                getattr(obj, "reporting_frequency", None),
            )
            for obj in _idf_values(self.idf, "Output:Variable")
        }
        for raw in _as_items(data.get("Output:Variable")):
            identity = (
                _get(raw, "Key Value", default="*"),
                _get(raw, "Variable Name"),
                _get(raw, "Reporting Frequency", default="Hourly"),
            )
            if identity in existing_output_variables:
                continue
            self.idf.add(
                OutputVariable(
                    key_value=identity[0],
                    variable_name=identity[1],
                    reporting_frequency=identity[2],
                )
            )
            existing_output_variables.add(identity)


def _flatten_schedule_data(data: Any) -> list[str]:
    if not data:
        return []
    if isinstance(data, list) and all(isinstance(item, str) for item in data):
        return data
    result: list[str] = []
    for item in data:
        if not isinstance(item, dict):
            result.append(str(item))
            continue
        through = _get(item, "Through")
        if through:
            result.append(f"Through: {through}")
        for day in _get(item, "Days", default=[]) or []:
            day_type = _get(day, "For")
            if day_type:
                result.append(f"For: {day_type}")
            for time_row in _get(day, "Times", default=[]) or []:
                until = _get(time_row, "Until", default={})
                time = _get(until, "Time")
                value = _get(until, "Value")
                if time is not None and value is not None:
                    result.append(f"Until: {time}, {value}")
    return result


def _reconstruct_config_state(idf_text: str, fields: dict) -> ConfigState:
    """Rebuild a ConfigState from pickled IDF text + field dict."""
    cs = ConfigState(**fields)
    if idf_text:
        import tempfile

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".idf", delete=False
            ) as tf:
                tf.write(idf_text)
                tf.flush()
                tmp_path = Path(tf.name)
            loaded = IDF.load(tmp_path)
            cs._set_idf_private(IDF.from_dict(loaded.to_dict()))
        finally:
            if tmp_path is not None:
                with contextlib.suppress(Exception):
                    tmp_path.unlink()
    else:
        cs._set_idf_private(IDF())
    return cs
