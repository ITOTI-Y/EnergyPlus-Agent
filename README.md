![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ITOTI-Y/EnergyPlus-Agent?utm_source=oss&utm_medium=github&utm_campaign=ITOTI-Y%2FEnergyPlus-Agent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ITOTI-Y/EnergyPlus-Agent)

# EnergyPlus Agent System

## Overview

EnergyPlus Agent turns a natural-language building brief, optionally accompanied by architectural drawings, into a validated EnergyPlus model and runs the simulation. The system is built in Python around three layers:

- **Multi-phase LangGraph agent**: an LLM-driven graph that reads the brief, splits it into per-domain tasks, builds zones, materials, schedules, constructions, surfaces, fenestration, HVAC and internal loads through tool calls, cross-checks references, pauses for human approval, and finally runs EnergyPlus.
- **MCP server**: a FastMCP server exposing the same building-configuration CRUD tools and workflow tools over stdio, HTTP, SSE or streamable-HTTP, so any MCP client (for example Claude Desktop) can assemble a model interactively.
- **YAML to IDF conversion and validation**: Pydantic schemas validate every EnergyPlus object, 11 converters map the YAML configuration to an IDF file via `eppy`, and a runner executes EnergyPlus.

A RAG knowledge base (Gemini Embedding + Qdrant) and SQLite data tools for standard materials, constructions, schedules and design days complete the toolset.

## Key Features

### Multi-phase agent (LangGraph)
- **Intake**: one structured LLM call parses text and images into `IntakeOutput`, which carries the `Building` and `Site:Location` objects, natural-language task specs for each downstream phase, and per-zone axis-aligned box geometry hints (`ZoneGeometry`).
- **Phased construction with parallelism**: independent object types are built by separate ReAct sub-agents. Zone, material and schedule run in parallel; construction, surface and fenestration run sequentially because of their dependencies; HVAC, people and lights run in parallel again.
- **Parallel-safe state**: a field-level reducer (`merge_config_state`) unions the `ConfigState` written by concurrent phases, keyed by object identity.
- **Cross-reference self-repair**: after each phase group, `ConfigState.validate_references()` checks that every referenced zone, material, construction, surface and schedule exists. Each phase agent also receives read-only `list_*` tools so it can inspect what earlier phases created.
- **Human-in-the-loop approval**: the validate node raises a LangGraph `interrupt()` with a configuration summary and any errors. Approval continues to simulation; free-text feedback loops back to intake.
- **Multimodal input**: PNG, JPEG, WebP and GIF drawings are passed to the intake LLM as base64 image parts alongside the text brief.
- **Tool-call tracing**: `TraceCollector` wraps every tool call in the ReAct subgraphs and records name, arguments, result and success flag per phase, intended as fine-tuning data. A script also exports full LangSmith run trees to local JSON.
- **Provider-agnostic LLM**: `src/configs/llm.yaml` selects provider, model, temperature and token budget; Anthropic and OpenAI integrations are bundled. `AGENT_LANGUAGE` switches the narrative language of all agent output while EnergyPlus identifiers stay ASCII.

### MCP server
- **FastMCP framework** with `stdio`, `http`, `sse` and `streamable-http` transports.
- **Full CRUD tool set** for Building, Location, Zone, Surface, Material, Construction, Fenestration, Schedule, HVAC, People and Lights.
- **Workflow tools** for YAML export and load, cross-reference validation, simulation and summary.
- **Resource endpoints** exposing the current configuration and its summary.

### YAML to IDF conversion
- **Strict validation**: 31 Pydantic schema classes cover every supported EnergyPlus object, including geometry closure and vertex-order checks.
- **11 converters** map validated YAML sections to IDF objects with `eppy`; the runner invokes EnergyPlus with `-x` (ExpandObjects, so `HVACTemplate` objects are expanded) and `-r` (ReadVarsESO, so CSV output is produced).
- **Default output variables**: when no `Output:Variable` is configured, the simulate step adds an hourly monitoring set (zone temperature and humidity, ideal-loads heating and cooling energy, lighting and people energy, facility HVAC electricity) so results are actually recorded.

### RAG knowledge base
- **Async embedding pipeline** built on Gemini Embedding and Qdrant.
- **Rate limiting, concurrency control and retry** on 429 / `RESOURCE_EXHAUSTED`.
- **Incremental sync** with deletion of stale vectors.
- **Typed results** using dataclasses (`QdrantData`, `RowRecord`, `VectorizedResult`).

### Data tools
- SQLite-backed management of standard materials, no-mass materials, constructions, compact schedules, schedule type limits and design days.
- Partial updates via an `UNSET` sentinel so only provided columns are written.

## Project Structure

```
EnergyPlus-Agent/
├── src/
│   ├── agent/                        # LangGraph multi-phase agent
│   │   ├── graph.py                  # build_graph(): topology + in-memory checkpointer
│   │   ├── state.py                  # AgentState, IntakeOutput, ZoneGeometry, SimContext, merge_config_state
│   │   ├── react.py                  # 3-node ReAct subgraph (llm -> tools -> llm)
│   │   ├── runner.py                 # run_session(), interactive_approval(), auto_approval()
│   │   ├── llm.py                    # create_llm() from src/configs/llm.yaml
│   │   ├── trace.py                  # TraceCollector and per-phase trace registry
│   │   ├── _share.py                 # IDD path, AGENT_LANGUAGE directive, constants
│   │   ├── nodes/                    # intake, zone, material, schedule, construction,
│   │   │                             # surface, fenestration, hvac, people, lights,
│   │   │                             # cross_ref, validate, simulate
│   │   └── tools/                    # make_*_tools() closures wrapping the MCP Tool classes
│   ├── mcp/                          # MCP server
│   │   ├── server.py                 # FastMCP entry point
│   │   ├── state.py                  # ConfigState (in-memory configuration + cross-ref validation)
│   │   ├── interface.py              # Tool interfaces and response models
│   │   ├── api/                      # Tool registration grouped by domain
│   │   │   ├── core.py               # Building, Location, Zone
│   │   │   ├── envelope.py           # Material, Construction, Surface, Fenestration
│   │   │   ├── schedule.py           # ScheduleTypeLimits, Schedule:Compact
│   │   │   ├── hvac.py               # Thermostat, IdealLoadsAirSystem
│   │   │   ├── loads.py              # People, Lights
│   │   │   ├── workflow.py           # export, load, validate, simulate, summary, clear
│   │   │   ├── resources.py          # config://current, config://summary
│   │   │   └── common.py             # Shared helpers
│   │   └── tools/                    # Tool implementations (one class per object type)
│   ├── converters/                   # YAML -> IDF converters (11 + base class)
│   ├── validator/
│   │   └── data_model.py             # 31 Pydantic schema classes
│   ├── runner/
│   │   └── runner.py                 # EnergyPlusRunner
│   ├── rag/                          # RAG pipeline (rag.py, embedding.py, vector.py, chunk.py)
│   ├── database/datatools/           # SQLite data tools
│   ├── configs/
│   │   ├── config.py                 # EmbeddingConfig, LLMConfig
│   │   ├── llm.yaml                  # Agent LLM settings
│   │   └── embedding.yaml            # Embedding model settings
│   ├── utils/logging.py              # Loguru setup
│   └── converter_manager.py          # ConverterManager
├── scripts/
│   ├── run_demo.py                   # End-to-end agent demo with auto-approval
│   ├── export_trace.py               # Run demo and dump LangSmith run trees to output/traces/
│   └── _share.py                     # Demo building briefs
├── tests/
│   ├── test_merge.py                 # ConfigState reducer tests
│   └── test_zone_agent.py            # Zone phase agent test (requires an LLM)
├── data/
│   ├── dependencies/Energy+.idd      # EnergyPlus IDD
│   ├── weather/Shenzhen.epw          # Example weather file
│   ├── schemas/                      # Example YAML configurations
│   └── examples/EP_Agent_data.db     # Example SQLite database
├── docker/                           # Dockerfile (nrel/energyplus:25.1.0) + docker-compose.yml
├── docs/_dev/                        # Design documents
├── output/                           # IDF, YAML, logs, traces, simulation results
├── main.py                           # Typer CLI
└── pyproject.toml
```

## Technology Stack

### Runtime requirements
- **Python 3.12+**
- **EnergyPlus 25.1.0+** on `PATH`
- **uv** for dependency management

### Main dependencies
| Library | Version | Purpose |
|---|---|---|
| **langgraph** | >=1.1.3 | Agent graph, checkpointing, interrupts |
| **langchain** | >=1.2.13 | Chat model factory, tools, messages |
| **langchain-anthropic** / **langchain-openai** | >=1.4.0 / >=1.1.12 | Bundled LLM providers |
| **fastmcp** | >=2.14.1 | MCP server framework |
| **eppy** | >=0.5.63 | IDF manipulation |
| **pydantic** | >=2.11.7 | Schema validation |
| **google-genai** | >=1.68.0 | Gemini Embedding API |
| **qdrant-client** | >=1.17.1 | Vector database client |
| **omegaconf** | >=2.3.0 | YAML configuration with env interpolation |
| **typer** | >=0.20.1 | CLI |
| **numpy** / **scipy** / **trimesh** | — | Geometry validation |
| **loguru** | >=0.7.3 | Logging |
| **pyyaml** | >=6.0.2 | YAML parsing |

Development extras: `pytest`, `langsmith`, `grandalf`.

## Quick Start

### Installation

```bash
git clone https://github.com/ITOTI-Y/EnergyPlus-Agent.git
cd EnergyPlus-Agent
uv sync
```

Make sure `data/dependencies/Energy+.idd` exists and that an EPW weather file is available (the repository ships `data/weather/Shenzhen.epw`).

### Environment variables

Copy `.env.example` to `.env` and fill in what you need:

```env
# Agent LLM (used with src/configs/llm.yaml)
LLM_API_KEY=
LLM_BASE_URL=            # optional, for OpenAI-compatible gateways

# RAG embedding pipeline
QDRANT_ENDPOINT=
QDRANT_API_KEY=
GEMINI_API_KEY=

# Optional LangSmith tracing
LANGSMITH_API_KEY=
LANGSMITH_ENDPOINT=
LANGSMITH_PROJECT=
LANGSMITH_TRACING=
```

`AGENT_LANGUAGE` (default `English`) controls the language of agent narrative text. The `embedding` command requires all three of `QDRANT_ENDPOINT`, `QDRANT_API_KEY` and `GEMINI_API_KEY` to be set.

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

The command stops at the validate step, prints a configuration summary and any cross-reference errors, and waits for input. Type `y` to approve and simulate, or type feedback text to send the graph back to intake with your correction. Results, the exported YAML and the generated IDF are written under `--output-dir`.

Two scripts cover non-interactive use:

```bash
uv run python scripts/run_demo.py       # auto-approves when there are no errors
uv run python scripts/export_trace.py   # interactive approval, dumps run trees to output/traces/
```

### Convert a YAML configuration directly

```bash
uv run main.py convert-idf   # data/schemas/building_schema.yaml -> output/idf/*.idf + simulation
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

### RAG index

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

uv run main.py embedding --collection energyplus_database --db-path data/examples/EP_Agent_data.db
```

### Docker

```bash
cd docker
docker compose up -d    # builds on nrel/energyplus:25.1.0 and serves the MCP server on port 8000
```

## Agent Architecture

```
START -> intake
           |
     +-----+-----+          phase 1, parallel
     v     v     v
   zone material schedule
     |     |     |
     +-----+-----+
           v
  cross_ref_foundations --[errors]--> validate
           | (clean)
      construction -> surface -> fenestration
                                     |
                               +-----+-----+   phase 3, parallel
                               v     v     v
                             hvac  people lights
                               |     |     |
                               +-----+-----+
                                     v
                             cross_ref_complete
                                     v
                                 validate --[interrupt]--> approved -> simulate -> END
                                     |
                                     +-- rejected / feedback -> intake
```

- **State**: `AgentState` holds the message list (intake conversation and phase summaries only), the user brief, image paths, `ConfigState`, `IntakeOutput`, validation errors and a retry counter.
- **Phase agents**: each phase is a compiled ReAct subgraph with `parallel_tool_calls=False`, working on a local copy of `ConfigState` and returning only its delta. Tool-call history stays inside the subgraph and is captured by `TraceCollector`.
- **Geometry**: the surface phase builds a canonical six-surface box for each zone from `ZoneGeometry` (origin, width, depth, height, exterior wall faces, floor and roof boundary conditions).
- **Checkpointing**: `InMemorySaver` with a pickle serializer, so nested Pydantic subclasses survive the interrupt round-trip.
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
| `create_standard_material` / `create_no_mass_material` / `create_air_gap_material` / `create_glazing_material` | Create materials by type |
| `get_material` / `update_*_material` / `delete_material` / `list_materials` | Material read, update, delete, list |
| `create_construction` / `get_construction` / `update_construction` / `delete_construction` / `list_constructions` | Construction CRUD |
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

### Workflow
| Tool | Description |
|------|-------------|
| `export_yaml` | Export the current configuration as YAML |
| `load_yaml` | Load a YAML configuration |
| `validate_config` | Run all cross-reference checks |
| `run_simulation` | Validate, export YAML, convert to IDF and run EnergyPlus |
| `get_summary` | Return object counts |
| `clear_all` | Reset the configuration |

### Resources
| Resource | Description |
|----------|-------------|
| `config://current` | Full current configuration as YAML |
| `config://summary` | Configuration summary |

## YAML Configuration

The YAML file mirrors EnergyPlus objects section by section:

- **SimulationControl**, **Timestep**, **RunPeriod**, **GlobalGeometryRules**
- **Building**, **Site:Location**
- **Material** (standard, no-mass, air gap, glazing), **Construction**
- **Zone**, **BuildingSurface:Detailed**, **FenestrationSurface:Detailed**
- **ScheduleTypeLimits**, **Schedule:Compact**
- **HVACTemplate:Thermostat**, **HVACTemplate:Zone:IdealLoadsAirSystem**
- **People**, **Lights**
- **Output:Variable**, **Output:VariableDictionary**, **Output:Diagnostics**, **Output:Table:SummaryReports**, **OutputControl:Table:Style**

Every section is validated by a matching Pydantic schema before conversion. Examples live in `data/schemas/`.

## CLI Reference

| Command | Description |
|---------|-------------|
| `uv run main.py run-agent "<brief>" --epw <file> [--image <file>]... [--output-dir <dir>] [--thread-id <id>]` | Run the multi-phase agent end to end with interactive approval |
| `uv run main.py convert-idf` | Convert `data/schemas/building_schema.yaml` to IDF and simulate |
| `uv run main.py mcp-server [--transport] [--host] [--port]` | Start the MCP server |
| `uv run main.py embedding --collection <name> --db-path <path>` | Build the RAG index |
| `energyplus-mcp` | MCP server entry point installed by `pyproject.toml` |

## Testing

```bash
uv run pytest
```

`tests/test_merge.py` exercises the state reducer without network access. `tests/test_zone_agent.py` drives the zone phase agent and needs a configured LLM.

## Roadmap

### Done
- YAML schema and IDF converters with Pydantic validation, geometry checks and cross-reference validation
- EnergyPlus runner with ExpandObjects and ReadVarsESO enabled; sizing periods on by default
- FastMCP server with full CRUD, workflow tools, resources, multi-transport support, CLI and Docker
- Async RAG pipeline with rate limiting, retry, incremental sync and typed results
- SQLite data tools for materials, constructions, schedules and design days
- LangGraph multi-phase agent: structured intake, parallel phase sub-agents, parallel-safe state merge, cross-reference self-repair, human-in-the-loop approval, simulation
- Multimodal intake (text + drawings) and box-recipe zone geometry
- Tool-call trace collection and LangSmith trace export

### Planned
- Simulation result parsing and visualization
- Broader HVAC coverage beyond `HVACTemplate` ideal loads
- Agent-side use of the RAG knowledge base and data tools during construction
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
- Add or extend Pydantic schemas for any new EnergyPlus object
- Keep docstrings and comments focused on non-obvious behavior
- Make sure `uv run pytest` passes

## Contact

- Project home: [https://github.com/ITOTI-Y/EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)
- Issues: [https://github.com/ITOTI-Y/EnergyPlus-Agent/issues](https://github.com/ITOTI-Y/EnergyPlus-Agent/issues)
