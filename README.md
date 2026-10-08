![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ITOTI-Y/EnergyPlus-Agent?utm_source=oss&utm_medium=github&utm_campaign=ITOTI-Y%2FEnergyPlus-Agent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ITOTI-Y/EnergyPlus-Agent)

# EnergyPlus Agent System

## Overview

EnergyPlus Agent turns a natural-language building brief, optionally accompanied by architectural drawings, into a validated EnergyPlus model and runs the simulation. The system is built in Python around three layers:

- **Multi-phase LangGraph agent**: an LLM-driven graph that reads the brief, splits it into per-domain tasks, builds zones, materials, schedules, constructions, surfaces, fenestration, HVAC and internal loads through tool calls, cross-checks references, pauses for human approval, and finally runs EnergyPlus.
- **MCP server**: a FastMCP server exposing the same building-configuration CRUD tools and workflow tools over stdio, HTTP, SSE or streamable-HTTP, so any MCP client (for example Claude Desktop) can assemble a model interactively.
- **idfpy model and EnergyPlus runner**: every tool edits one [idfpy](https://github.com/ITOTI-Y/idfpy) model, which is saved as IDF or epJSON and simulated by a runner around the EnergyPlus CLI.

A reference library of materials, constructions and schedules from the DOE prototype buildings, searched by meaning (Qwen3-Embedding-8B + Qdrant), and SQLite data tools for standard materials, constructions, schedules and design days complete the toolset.

## Key Features

### Multi-phase agent (LangGraph)
- **Intake**: one structured LLM call parses text and images into `IntakeOutput`, which carries the `Building` and `Site:Location` objects, every zone as a prism (floor plan, floor level, height and the constructions of its faces), and natural-language task specs for the other phases.
- **Phased construction with parallelism**: independent object types are built by separate ReAct sub-agents; zones and their surfaces are built in code from the intake output. Zone, material and schedule run in parallel; construction, surface and fenestration run sequentially because of their dependencies; HVAC, people, lights and equipment run in parallel again.
- **Parallel-safe state**: a reducer (`merge_config_state`) unions the idfpy models written by concurrent phases; on a name conflict the later branch wins.
- **Shared model operations**: agent tools and MCP tools are thin adapters over `src/modeling`, which rejects missing references and duplicate names at the call, applies updates atomically, renames references along with an object, and refuses to delete an object that others still reference. Tool arguments are typed models with declared fields, so a rejected call names the exact field (for example `vertices.0.X: Field required`).
- **Envelope rules at the tool boundary**: constructions are checked as they are created. Opaque layers never mix with window layers, a SimpleGlazingSystem stands alone, and multi-pane glazing alternates glass and `WindowMaterial:Gas`. Each surface and opening accepts only a construction of the matching kind, and the list tools show the kind. Openings get their vertex order corrected, are rejected when off their wall, and on an interzone wall get a mirrored partner in the adjacent zone. The two faces of an interzone wall can be created in either order and are linked to each other.
- **Schedule rules at the tool boundary**: a `Schedule:Compact` is rejected unless its periods end at 12/31, every day ends at 24:00, and every period gives all twelve day types a value (weekends, holidays, design days and custom days included), through `AllDays` or blocks ending with `AllOtherDays`. EnergyPlus only warns about missing day types and runs those days without the intended values.
- **Zone geometry by extrusion**: the surface phase, and the MCP tool `create_zone_geometry`, build a zone's walls, floor and flat roof from its floor plan, floor level and height. Faces touching another zone's faces become interzone pairs in any creation order, also for zones of different height and storeys whose plans do not line up (overlaps computed with shapely). The ceiling side, or the wall of the zone already there, gets the interior construction with its layers reversed (`<name>_Reversed`), as EnergyPlus expects of the two faces of one element. Concave faces are cut into convex parts, and a concave zone switches solar distribution to `FullExterior`. Sloped roofs and walls are left to the surface phase's LLM, which replaces flat faces through `create_surfaces`, where each entry succeeds or fails on its own.
- **Failure-loop guard**: every phase agent stops when the same tool call fails three times or ten calls fail in a row, logs each failure, and reports the last error as the phase summary instead of retrying until the LLM budget runs out.
- **Structured validation**: `src/modeling/validation.py` reports each problem as a `ModelIssue` tied to an object type, name and field. Sources are idfpy's reference check, geometric checks (fenestration reversed, off its parent surface's plane or outside its outline, which EnergyPlus would only warn about), empty models, phases that created nothing, and EnergyPlus Severe and Fatal messages, which are tied to the first object they quote. `src/agent/phases.py` maps object types to the phase that owns them; after its run each phase repairs the problems in its own objects, and every phase agent also receives read-only `list_*` tools to inspect what earlier phases created.
- **Partial retries**: after a pass, problems go to the phases that own them. If those phases can fix them from their unchanged task, only they and the phases depending on their objects run again, with the problems as feedback (zone -> surface -> fenestration and loads; material -> construction -> surface; schedule -> loads). Otherwise intake receives the brief, its previous output and the problems, and returns a patch of the fields to change; the phases fed by changed fields, the phases owning the problems, and their dependants run again. Before a pass, the objects of the phases about to run are removed; the other phases keep their objects and are not prompted. Intake revises at most twice before a human reviews the model, and EnergyPlus Severe errors after approval go through the same loop.
- **Human-in-the-loop approval**: the validate node raises a LangGraph `interrupt()` with a configuration summary and any errors. Approval continues to simulation; free-text feedback goes to intake as a correction and is handled like a revision.
- **Multimodal input**: PNG, JPEG, WebP and GIF drawings are passed to the intake LLM as base64 image parts alongside the text brief.
- **Tool-call tracing**: `TraceCollector` wraps every tool call in the ReAct subgraphs and records name, arguments, result and success flag per phase, intended as fine-tuning data. A script also exports full LangSmith run trees to local JSON.
- **Provider-agnostic LLM**: `src/configs/llm.yaml` selects provider, model, temperature and token budget; Anthropic and OpenAI integrations are bundled. `AGENT_LANGUAGE` switches the narrative language of all agent output while EnergyPlus identifiers stay ASCII.

### MCP server
- **FastMCP framework** with `stdio`, `http`, `sse` and `streamable-http` transports.
- **Full CRUD tool set** for Building, Location, Zone, Surface, Material, Construction, Fenestration, Schedule, HVAC, People, Lights and ElectricEquipment; update tools take an optional `new_name` that is applied to every reference.
- **Workflow tools** for model export and load (IDF or epJSON), cross-reference validation, simulation and summary.
- **Resource endpoints** exposing the current configuration and its summary.

### Model and simulation
- **Default objects**: a new model starts with `Version`, `SimulationControl`, `Timestep`, `GlobalGeometryRules`, an annual `RunPeriod` and the summary-report outputs.
- **Design days**: before a run, the annual heating 99.6% and cooling 0.4% design days are imported from the `.ddy` file next to the EPW (`Shenzhen.epw` → `Shenzhen.ddy`) unless the model already has design days; a missing `.ddy` stops the run with an error.
- **Ground contact**: floors with a `Ground` boundary are simulated on an uninsulated `Foundation:Kiva` slab, with each floor's exposed perimeter measured from the outdoor walls above its edges; Kiva derives the soil temperatures from the weather file. A model that defines `Site:GroundTemperature:BuildingSurface` keeps the fixed-temperature boundary.
- **Runner**: each run gets its own directory; EnergyPlus runs with `-x` (ExpandObjects, so `HVACTemplate` objects are expanded) and `-r` (ReadVarsESO), and `eplusout.err` is parsed into structured Warning, Severe and Fatal messages.
- **Default output variables**: when no `Output:Variable` is configured, the simulate step adds an hourly monitoring set (zone temperature and humidity, ideal-loads heating and cooling energy, lighting and people energy, facility HVAC electricity) so results are actually recorded.

### Prototype reference library
- **Sources**: the DOE commercial prototypes (ASHRAE 90.1-2019 and 2022, 16 building types, 19 locations from 0A to 8) and residential prototypes (IECC 2021 and 2024, single- and multifamily on four foundations, 16 climate zones) from energycodes.gov: 864 models.
- **Upgrade**: models are reduced to materials, constructions and schedules and set to EnergyPlus 26.1; the 22.1 to 26.1 transition rules change none of these objects, and running the transition programs on reduced models gave identical objects.
- **Library**: equal objects are merged with every building type, standard, climate zone and use they appear in (1,519 materials, 1,553 constructions, 1,434 schedules). Constructions carry their layer materials; schedules of any kind (`Schedule:Compact`, or `Schedule:Year` over weekly and hourly schedules) become the nested Through/For/Until periods the schedule tool accepts.
- **Search**: each object has a bilingual description embedded with Qwen3-Embedding-8B; Qdrant filters by kind and by the site's ASHRAE 169 climate zones, computed from the weather file (CDD10 and HDD18 of daily means, neighbour zones within 10% of a threshold, moisture regime from precipitation when the file has it). For Shenzhen this gives 1A and 2A.
- **Agent use**: when configured, the material, construction and schedule phases get `find_reference_*` tools and a prompt section; values and names from the specification still take precedence. A configured but unreachable service stops `run-agent` before any LLM call.
- **Storage**: the library is one SQLite file with embeddings, published to a private Hugging Face dataset; the Qdrant collection can be rebuilt from it without embedding again.

### Data tools
- SQLite-backed management of standard materials, no-mass materials, constructions, compact schedules, schedule type limits and design days.
- Partial updates via an `UNSET` sentinel so only provided columns are written.

## Project Structure

```
EnergyPlus-Agent/
├── src/
│   ├── agent/                        # LangGraph multi-phase agent
│   │   ├── graph.py                  # build_graph(): topology + in-memory checkpointer
│   │   ├── state.py                  # AgentState, IntakeOutput, SimContext, merge_config_state
│   │   ├── phases.py                 # Phase -> owned object types, issue routing
│   │   ├── react.py                  # 3-node ReAct subgraph (llm -> tools -> llm)
│   │   ├── runner.py                 # run_session(), interactive_approval(), auto_approval()
│   │   ├── llm.py                    # create_llm() from src/configs/llm.yaml
│   │   ├── trace.py                  # TraceCollector and per-phase trace registry
│   │   ├── _share.py                 # AGENT_LANGUAGE directive, constants
│   │   ├── nodes/                    # intake, plan_rerun, zone, material, schedule, construction,
│   │   │                             # surface, fenestration, hvac, people, lights, equipment,
│   │   │                             # cross_ref, validate, simulate
│   │   └── tools/                    # make_*_tools() closures over src/modeling
│   ├── mcp/                          # MCP server
│   │   ├── server.py                 # FastMCP entry point
│   │   ├── interface.py              # ToolResponse
│   │   ├── api/                      # Tool registration grouped by domain
│   │   │   ├── core.py               # Building, Location, Zone
│   │   │   ├── envelope.py           # Material, Construction, Surface, Fenestration
│   │   │   ├── schedule.py           # ScheduleTypeLimits, Schedule:Compact
│   │   │   ├── hvac.py               # Thermostat, IdealLoadsAirSystem
│   │   │   ├── loads.py              # People, Lights, ElectricEquipment
│   │   │   ├── workflow.py           # model export/load, validate, simulate, summary, clear
│   │   │   ├── resources.py          # config://current, config://summary
│   │   │   └── common.py             # model_tool(): (message, data) -> MCP response
│   │   └── tools/workflow.py         # WorkflowTool: model I/O, validation, simulation
│   ├── state/
│   │   ├── config_state.py           # ConfigState (idfpy model, save/load, summary)
│   │   └── defaults.py               # Default objects and design-day import
│   ├── modeling/                     # Model operations shared by agent and MCP tools
│   │   ├── objects.py                # create / get / update / delete with reference checks
│   │   ├── envelope.py               # Materials, layers, vertex input
│   │   ├── schedules.py              # Nested Through/For/Until input -> Schedule:Compact
│   │   ├── hvac.py                   # Ideal loads systems keyed by zone
│   │   ├── validation.py             # ModelIssue from references, geometry and EnergyPlus
│   │   ├── ground.py                 # Kiva slab foundations and exposed perimeters
│   │   ├── surfaces.py               # Base surfaces, interzone pairs, batch creation
│   │   ├── geometry.py               # Zone extrusion and pairing of shared faces
│   │   ├── fenestration.py           # Opening placement, orientation and partners
│   │   └── errors.py                 # Rejections reported to tool callers
│   ├── runner/
│   │   └── runner.py                 # run_energyplus and eplusout.err parsing
│   ├── reference/                    # Prototype reference library: download, extract, library, climate, index, search, hub
│   ├── database/datatools/           # SQLite data tools
│   ├── configs/
│   │   ├── config.py                 # LLMConfig
│   │   └── llm.yaml                  # Agent LLM settings
│   └── utils/logging.py              # Loguru setup
├── scripts/
│   ├── run_demo.py                   # End-to-end agent demo with auto-approval
│   ├── export_trace.py               # Run demo and dump LangSmith run trees to output/traces/
│   └── _share.py                     # Demo building briefs
├── tests/                            # Mirrors src/
├── data/
│   ├── weather/Shenzhen.{epw,ddy}    # TMYx 2011-2025 weather and design days
│   ├── schemas/                      # Example models (epJSON)
│   └── examples/EP_Agent_data.db     # Example SQLite database
├── docker/                           # Dockerfile (nrel/energyplus:25.1.0) + docker-compose.yml
├── docs/_dev/                        # Design documents
├── output/                           # Models, logs, traces, simulation results
├── main.py                           # Typer CLI
└── pyproject.toml
```

## Technology Stack

### Runtime requirements
- **Python 3.12+**
- **EnergyPlus 26.1** on `PATH` (matches the pinned `idfpy==26.1.*`)
- **uv** for dependency management

### Main dependencies
| Library | Version | Purpose |
|---|---|---|
| **langgraph** | >=1.1.3 | Agent graph, checkpointing, interrupts |
| **langchain** | >=1.2.13 | Chat model factory, tools, messages |
| **langchain-anthropic** / **langchain-openai** | >=1.4.0 / >=1.1.12 | Bundled LLM providers |
| **fastmcp** | >=2.14.1 | MCP server framework |
| **idfpy** | ==26.1.* | Typed EnergyPlus objects, IDF and epJSON I/O, reference checks |
| **pydantic** | >=2.11.7 | Schema validation |
| **qdrant-client** | >=1.17.1 | Vector database of the reference library |
| **openai** | >=1.100 | Client of the OpenAI-compatible embedding service |
| **huggingface-hub** | >=0.35 | Publishing and pulling the reference library |
| **pydantic-settings** | >=2.11 | Reference library settings from the environment |
| **omegaconf** | >=2.3.0 | LLM settings with env interpolation |
| **typer** | >=0.20.1 | CLI |
| **shapely** | >=2.2.0 | Polygon overlap and splitting for zone geometry |
| **loguru** | >=0.7.3 | Logging |

Development extras: `pytest`, `pytest-recording`, `ruff`, `ty`, `pre-commit`, `langsmith`, `grandalf`.

## Quick Start

### Installation

```bash
git clone https://github.com/ITOTI-Y/EnergyPlus-Agent.git
cd EnergyPlus-Agent
uv sync
```

Simulation needs an EPW weather file and a `.ddy` design-day file with the same stem beside it; the repository ships `data/weather/Shenzhen.epw` and `data/weather/Shenzhen.ddy`.

### Environment variables

Copy `.env.example` to `.env` and fill in what you need:

```env
# Agent LLM (used with src/configs/llm.yaml)
LLM_API_KEY=
LLM_BASE_URL=            # optional, for OpenAI-compatible gateways

# Prototype reference library (both URLs, or neither)
REFERENCE_QDRANT_URL=      # e.g. http://pan-office:6333
REFERENCE_EMBEDDING_URL=   # OpenAI-compatible, e.g. http://pan-office:8000/v1
# REFERENCE_QDRANT_API_KEY, REFERENCE_EMBEDDING_API_KEY, REFERENCE_COLLECTION,
# REFERENCE_EMBEDDING_MODEL, REFERENCE_HUB_REPO, REFERENCE_LIBRARY_PATH: optional

# Optional LangSmith tracing
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=
LANGSMITH_PROJECT=
LANGSMITH_TRACING=
```

`AGENT_LANGUAGE` (default `English`) controls the language of agent narrative text.

### LLM configuration

`src/configs/llm.yaml` selects the model used by every agent phase:

```yaml
provider: "anthropic"          # any provider supported by LangChain init_chat_model
base_url: ${oc.env:LLM_BASE_URL,null}
api_key: ${oc.env:LLM_API_KEY,null}
model_name: "gpt-5.4"
temperature: 0.7
max_tokens: 64000
```

### Run the agent

```bash
# Text brief only
uv run main.py run-agent "Design a 5-zone office in Shenzhen, 10m x 8m x 3m, ..." \
  --epw data/weather/Shenzhen.epw

# With drawings (repeat --image for several files)
uv run main.py run-agent "Office building described in the drawings" \
  --epw data/weather/Shenzhen.epw \
  --image floorplan.png --image elevation.png \
  --output-dir output/run1 --thread-id run1
```

The command stops at the validate step, prints a configuration summary and any validation problems, and waits for input. Type `y` to approve and simulate, or type feedback text to send the graph back to intake with your correction. The generated IDF and the EnergyPlus results are written to a new `run_*` directory under `--output-dir`.

Two scripts cover non-interactive use:

```bash
uv run python scripts/run_demo.py       # auto-approves when there are no errors
uv run python scripts/export_trace.py   # interactive approval, dumps run trees to output/traces/
```

### MCP server

```bash
uv run main.py mcp-server                                                # stdio
uv run main.py mcp-server --transport http --host 0.0.0.0 --port 8000    # HTTP
```

Claude Desktop configuration:

```json
{
  "mcpServers": {
    "energyplus-agent": {
      "command": "uv",
      "args": ["--directory", "/path/to/EnergyPlus-Agent", "run", "main.py", "mcp-server"]
    }
  }
}
```

### Reference library

The embedding service is vLLM serving Qwen3-Embedding-8B with `--runner pooling` (OpenAI-compatible `/v1/embeddings`); Qdrant runs as usual. With both URLs set:

```bash
uv run main.py reference pull      # library from the Hugging Face dataset, or:
uv run main.py reference build     # download the prototypes and build it (about 30 s)
uv run main.py reference embed     # embed entries without an embedding
uv run main.py reference load      # replace the Qdrant collection
uv run main.py reference publish   # upload the library to the private dataset
```

### Docker

```bash
cd docker
docker compose up -d    # builds on nrel/energyplus:25.1.0 and serves the MCP server on port 8000
```

## Agent Architecture

```
START -> intake -> plan_rerun        (removes the objects of the phases about to run)
                       |
     +-----+-----+          phase 1, parallel; phases not pending return at once
     v     v     v
   zone material schedule
     |     |     |
     +-----+-----+
           v
  cross_ref_foundations --[errors]--> validate
           | (clean)
      construction -> surface -> fenestration
                                     |
                         +-----+-----+-----+   phase 3, parallel
                         v     v     v     v
                       hvac people lights equipment
                         |     |     |     |
                         +-----+-----+-----+
                                     v
                             cross_ref_complete
                                     v
                                 validate --[phases can fix it]--> plan_rerun
                                     |----[needs new specs, <= 2]--> intake
                                     +--[interrupt]--> approved -> simulate -> END
                                                    |                 +--[Severe]--> validate
                                                    +-- feedback -> intake
```

- **State**: `AgentState` holds the message list (intake conversation and phase summaries only), the user brief, image paths, `ConfigState`, `IntakeOutput`, validation errors, problems found while building in code, the phases pending in this pass with their feedback, and the retry counters. `plan_rerun` writes the model with LangGraph's `Overwrite`, since the parallel-branch reducer is a union and would bring removed objects back.
- **Phase agents**: each phase is a compiled ReAct subgraph with `parallel_tool_calls=False`, working on a local copy of `ConfigState` and returning only its delta. Tool-call history stays inside the subgraph and is captured by `TraceCollector`.
- **Checkpointing**: `InMemorySaver` with a pickle serializer, so the idfpy model survives the interrupt round-trip.
- **Runtime context**: `SimContext` carries the EPW path and output directory; `RunnableConfig` carries the `thread_id`.

## MCP Server Tools

### Core
| Tool | Description |
|------|-------------|
| `create_building` / `get_building` / `update_building` / `delete_building` / `list_buildings` | Building CRUD |
| `create_location` / `get_location` / `update_location` / `delete_location` / `list_locations` | Site location CRUD |
| `create_zone` / `get_zone` / `update_zone` / `delete_zone` / `list_zones` | Zone CRUD |

### Envelope
| Tool | Description |
|------|-------------|
| `create_standard_material` / `create_no_mass_material` / `create_air_gap_material` / `create_glazing_material` / `create_window_glazing_material` / `create_window_gas_material` | Create materials by type; the last two build multi-pane windows |
| `get_material` / `update_*_material` / `delete_material` / `list_materials` | Material read, update, delete, list |
| `create_construction` / `get_construction` / `update_construction` / `delete_construction` / `list_constructions` | Construction CRUD |
| `create_zone_geometry` | Extrude a zone from its floor plan and pair faces shared with other zones |
| `create_surfaces` | Create several surfaces, each succeeding or failing on its own |
| `create_surface` / `get_surface` / `update_surface` / `delete_surface` / `list_surfaces` | Building surface CRUD |
| `create_fenestration_surface` / `get_fenestration_surface` / `update_fenestration_surface` / `delete_fenestration_surface` / `list_fenestration_surfaces` | Window and door CRUD |

### Schedule
| Tool | Description |
|------|-------------|
| `create_schedule_type_limits` / `get_schedule_type_limits` / `update_schedule_type_limits` / `delete_schedule_type_limits` / `list_schedule_type_limits` | ScheduleTypeLimits CRUD |
| `create_schedule_compact` / `get_schedule_compact` / `update_schedule_compact` / `delete_schedule_compact` / `list_schedule_compacts` | Schedule:Compact CRUD |

### HVAC
| Tool | Description |
|------|-------------|
| `create_hvac_thermostat` / `get_hvac_thermostat` / `update_hvac_thermostat` / `delete_hvac_thermostat` / `list_hvac_thermostats` | Thermostat CRUD |
| `create_hvac_ideal_loads_system` / `get_hvac_ideal_loads_system` / `update_hvac_ideal_loads_system` / `delete_hvac_ideal_loads_system` / `list_hvac_ideal_loads_systems` | Ideal loads air system CRUD |

### Loads
| Tool | Description |
|------|-------------|
| `create_people` / `get_people` / `update_people` / `delete_people` / `list_people` | People CRUD |
| `create_light` / `get_light` / `update_light` / `delete_light` / `list_lights` | Lights CRUD |
| `create_equipment` / `get_equipment` / `update_equipment` / `delete_equipment` / `list_equipment` | ElectricEquipment (plug load) CRUD |

### Workflow
| Tool | Description |
|------|-------------|
| `export_model` | Save the current model; the suffix `.idf` or `.epJSON` selects the format |
| `load_model` | Replace the current model with an IDF or epJSON file |
| `validate_config` | Check references, fenestration placement and completeness |
| `run_simulation` | Validate, add design days and Kiva foundations, write the IDF and run EnergyPlus |
| `get_summary` | Return object counts |
| `clear_all` | Reset the configuration |

### Resources
| Resource | Description |
|----------|-------------|
| `config://current` | Full current model as epJSON |
| `config://summary` | Configuration summary |

## Example Models

`data/schemas/` holds epJSON models that `load_model` reads directly: `building_schema.epJSON` (two zones with ideal loads, people and lights), `example/L_shape.epJSON` (L-shaped geometry with windows), `example/complex_building.epJSON` and `office_building_5f_atrium_transit.epJSON` (multi-zone offices with ideal loads). EnergyPlus expands `HVACTemplate` objects only from IDF input, so simulate these models through `run_simulation`, which writes IDF.

## CLI Reference

| Command | Description |
|---------|-------------|
| `uv run main.py run-agent "<brief>" --epw <file> [--image <file>]... [--output-dir <dir>] [--thread-id <id>]` | Run the multi-phase agent end to end with interactive approval |
| `uv run main.py mcp-server [--transport] [--host] [--port]` | Start the MCP server |
| `uv run main.py reference build\|embed\|load\|publish\|pull` | Build, embed, index and share the prototype reference library |
| `energyplus-mcp` | MCP server entry point installed by `pyproject.toml` |

## Testing

```bash
uv run pytest
```

`tests/` mirrors `src/`. Tests that run EnergyPlus are skipped when `energyplus` is not on `PATH`; phase-agent tests replay recorded cassettes.

## Roadmap

### Done
- idfpy-backed model with IDF and epJSON import and export, default objects and design-day import
- EnergyPlus runner with per-run directories, ExpandObjects and ReadVarsESO, and structured `eplusout.err` parsing
- FastMCP server with full CRUD, workflow tools, resources, multi-transport support, CLI and Docker
- Prototype reference library (DOE commercial and residential models) with semantic search by climate zone, used by the material, construction and schedule phases
- SQLite data tools for materials, constructions, schedules and design days
- LangGraph multi-phase agent: structured intake, parallel phase sub-agents, parallel-safe state merge, phase-scoped self-repair, partial reruns and intake patches, human-in-the-loop approval, simulation
- Multimodal intake (text + drawings) with zone prisms extruded and paired in code
- Tool-call trace collection and LangSmith trace export

### Planned
- Simulation result parsing and visualization
- Broader HVAC coverage beyond `HVACTemplate` ideal loads
- Persistent checkpointing and resumable sessions
- Fine-tuning pipeline built on the collected traces

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'feat: add AmazingFeature'`)
4. Push the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Code style
- Python 3.12+ features
- Ruff for linting and formatting (configuration in `pyproject.toml`)
- Use idfpy model classes for EnergyPlus objects instead of new schemas
- Keep docstrings and comments focused on non-obvious behavior
- Make sure `uv run pytest` passes

## Contact

- Project home: [https://github.com/ITOTI-Y/EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)
- Issues: [https://github.com/ITOTI-Y/EnergyPlus-Agent/issues](https://github.com/ITOTI-Y/EnergyPlus-Agent/issues)
