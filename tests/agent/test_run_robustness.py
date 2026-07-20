"""EnergyPlus Agent multi-case robustness (first-pass IDF success) benchmark.

Measures how reliably the agent produces a *runnable* EnergyPlus IDF in
one shot, with zero auto-rollback rounds and no human intervention,
across a corpus of building description cases.

A case counts as a **first-pass success** only if all of the following hold:

1. The validate interrupt fires with no cross-ref errors AND
   ``retry_count == 0`` (the agent never rolled back).
2. After auto-approval the simulation actually runs, EnergyPlus produces
   output artifacts, the IDF has real geometry, and ``eplusout.err``
   contains no Fatal/Severe lines (warnings are tolerated).

If the first validate hit has errors or the agent rolled back, the case
still runs to completion (the harness always approves) and is tagged
``recovered`` if it simulates OK in the end, or ``failed`` otherwise.

Outputs are written to ``output/agent_robustness/<timestamp>/``:
``results.json`` (full per-case record), ``summary.json`` (aggregate
stats), ``report.md`` (human-readable tables) and one ``run.log`` per case.

This benchmark calls an LLM and runs EnergyPlus, so it never runs as an
automatic pytest case (it defines no test functions). Execute it manually
from the repo root:

    uv run python -m tests.agent.test_run_robustness
    uv run python -m tests.agent.test_run_robustness --only office/large
    uv run python -m tests.agent.test_run_robustness --repeat 5 --temperature 0.0
"""

import contextlib
import json
import re
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Final

from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field
from typer import Exit, Option, Typer

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.agent.trace import export_traces, reset_traces
from src.mcp.state import ConfigState, _idf_values
from src.results.err_parser import extract_errors

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT: Final = REPO_ROOT / "tests" / "agent" / "test_data" / "text_only"
DEFAULT_EPW: Final = REPO_ROOT / "data" / "weather" / "Shenzhen.epw"
RESULTS_DIR: Final = REPO_ROOT / "output" / "agent_robustness"
LLM_YAML: Final = REPO_ROOT / "src" / "configs" / "llm.yaml"

ZONE_VALIDATOR_PREFIX: Final = "[zone-validator]"


class CaseSpecSchema(BaseModel):
    """One discovered test case (a ``testdata_prompt.json`` directory)."""

    case_id: str
    data: dict[str, Any]
    case_dir: Path
    category: str | None = None
    scale: str | None = None


class NodeTimingSchema(BaseModel):
    """Wall-clock duration of one node execution (re-runs appear twice)."""

    node: str
    duration_s: float


class SimRoundSchema(BaseModel):
    """One simulate round's ``eplusout.err`` snapshot.

    ``fixed_vs_prev`` / ``new_vs_prev`` compare severe-error multisets with
    the previous round, showing what the agent repaired and what it newly
    introduced across the simulate -> revise loop.
    """

    round_no: int
    success: bool
    fatal: int
    severe: int
    warning: int
    severe_lines: list[str] = Field(default_factory=list)
    fatal_lines: list[str] = Field(default_factory=list)
    fixed_vs_prev: list[str] = Field(default_factory=list)
    new_vs_prev: list[str] = Field(default_factory=list)


class IdfGeometrySchema(BaseModel):
    """Zone/surface counts extracted from the IDF used by simulate."""

    zones: int = 0
    surfaces: int = 0
    zones_without_surfaces: list[str] = Field(default_factory=list)
    error: str | None = None


class CaseResultSchema(BaseModel):
    """Everything recorded about a single case run.

    ``first_pass`` is the end-to-end definition: the agent built a usable
    IDF in one shot AND EnergyPlus ran it without Error-level messages.
    See :meth:`CaseHarness.finalize`.
    """

    case_id: str
    case_name: str = ""
    prompt_json: str = ""
    repeat_index: int = 0

    first_pass: bool = False
    model_clean_first: bool = False
    recovered: bool = False
    simulation_ok: bool = False
    reached_validate: bool = False
    reached_simulate: bool = False

    rollback_rounds: int = 0
    validate_hits: int = 0
    rebuild_count: int = 0
    first_validate_errors: int = 0
    cross_ref_errors_seen: list[str] = Field(default_factory=list)
    validate_summary: dict[str, Any] | None = None

    node_sequence: list[str] = Field(default_factory=list)
    node_counts: dict[str, int] = Field(default_factory=dict)
    node_timings: list[NodeTimingSchema] = Field(default_factory=list)
    phase_total_s: dict[str, float] = Field(default_factory=dict)

    phase_traces: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)
    phase_tool_stats: dict[str, dict[str, int]] = Field(default_factory=dict)

    idf_path: str | None = None
    output_files: list[str] = Field(default_factory=list)
    err_fatal_count: int = 0
    err_severe_count: int = 0
    err_warning_count: int = 0
    err_has_error_level: bool = False
    idf_zone_count: int = 0
    idf_surface_count: int = 0
    idf_zones_without_surfaces: list[str] = Field(default_factory=list)
    sim_rounds: list[SimRoundSchema] = Field(default_factory=list)

    elapsed_s: float = 0.0
    error: str | None = None
    simulate_message: str | None = None
    zone_validator_failed: bool = False
    zone_validator_reasons: list[str] = Field(default_factory=list)


class CaseStabilitySchema(BaseModel):
    """First-pass stability of one case across repetitions."""

    first_pass_count: int
    repetitions: int
    first_pass_rate: float


class BenchmarkSummarySchema(BaseModel):
    """Aggregate statistics over all runs of one benchmark invocation."""

    timestamp: str
    epw: str
    temperature: float | None
    repeat: int
    total_runs: int
    first_pass_ok: int
    first_pass_rate: float
    model_clean_first_ok: int
    model_clean_first_rate: float
    recovered_ok: int
    simulation_ok: int
    reached_simulate: int
    failed: int
    crashed: int
    err_with_error_level: int
    avg_rollback_rounds: float
    avg_rebuild_count: float
    avg_validate_hits: float
    avg_first_validate_errors: float
    avg_elapsed_s: float
    total_err_fatal: int
    total_err_severe: int
    total_err_warning: int
    zone_validator_failed: int
    avg_phase_s: dict[str, float] = Field(default_factory=dict)
    per_case_stability: dict[str, CaseStabilitySchema] = Field(default_factory=dict)


def discover_cases(root: Path = TEST_DATA_ROOT) -> list[CaseSpecSchema]:
    """Recursively load every ``testdata_prompt.json`` under ``root``.

    The ``case_id`` is the case directory's path relative to ``root`` in
    POSIX form (e.g. ``"residential/large/case_03"``), unique across the
    tree; ``--only`` matches prefixes against it. ``category`` and ``scale``
    are sliced from the first two path segments when present so the report
    can aggregate per building type / scale.
    """
    cases: list[CaseSpecSchema] = []
    for p in sorted(root.rglob("testdata_prompt.json")):
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        rel = p.parent.relative_to(root)
        parts = rel.parts
        cases.append(
            CaseSpecSchema(
                case_id=rel.as_posix(),
                data=data,
                case_dir=p.parent.resolve(),
                category=parts[0] if len(parts) >= 1 else None,
                scale=parts[1] if len(parts) >= 2 else None,
            )
        )
    return cases


def build_user_prompt(case: CaseSpecSchema) -> str:
    """Render a testdata_prompt.json into a natural-language description.

    The agent's intake node expects free-form text plus optional images;
    the structured JSON fields become a compact English description so the
    same prompt works regardless of the LLM language setting. Both the
    legacy numeric fields and the extended descriptive fields (footprint,
    orientation, layout, ...) are supported; a rich ``Description`` block,
    when present, is appended verbatim.
    """
    d = case.data
    name = d.get("TestName") or f"building_{case.case_id}"
    lines = [
        "Please build an EnergyPlus IDF model for the following building.",
        f"- Name: {name}",
        f"- Location: {d.get('Building location', 'Shenzhen')}",
        f"- Building type: {d.get('Building type', 'Office')}",
    ]
    for label, key in (
        ("- Total floor area", "Floor area"),
        ("- Number of floors", "Number of floors"),
        (
            "- Thermal zones per floor",
            "Number of thermal zones per floor of the building",
        ),
        ("- Total thermal zones", "Number of total thermal zones in the building"),
        ("- Footprint width (east-west, m)", "Footprint width (east-west, m)"),
        ("- Footprint depth (north-south, m)", "Footprint depth (north-south, m)"),
        ("- Floor-to-floor height (m)", "Floor-to-floor height (m)"),
        ("- Orientation (deg from north)", "Orientation (degrees from north)"),
        ("- Window-to-wall ratio", "Window-to-wall ratio"),
        ("- Space layout", "Space layout"),
    ):
        val = d.get(key)
        if val:
            lines.append(f"{label}: {val}")
    desc = d.get("Description")
    if desc:
        lines.append("")
        lines.append(desc.strip())
    else:
        lines.append(
            "Create all zones, materials, constructions, surfaces, fenestrations, "
            "HVAC (ideal loads), people, lights and schedules so the resulting "
            "IDF can run in EnergyPlus without errors. "
            "Derive the zone layout and geometry from the counts above; keep the "
            "footprint rectangular and dimensions reasonable for the stated area."
        )
    return "\n".join(lines)


def collect_images(case: CaseSpecSchema) -> list[str]:
    """Collect the optional drawing paths declared in the prompt JSON."""
    keys = (
        "Top view path of the building",
        "Front view path of the building",
        "Building side view path",
        "Path of the supplementary plan example drawing for the building",
    )
    imgs: list[str] = []
    for k in keys:
        v = case.data.get(k, "")
        if not v:
            continue
        p = Path(v)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            imgs.append(str(p))
        else:
            logger.warning("[{}] image not found, skipping: {}", case.case_id, v)
    return imgs


class CaseHarness:
    """Per-case state machine that drives ``run_session`` and records metrics.

    ``interrupt_handler`` is what the validate gate calls (auto-decide and
    record); ``event_handler`` records the executed node sequence.
    """

    def __init__(
        self, case: CaseSpecSchema, output_dir: Path, repeat_index: int = 0
    ) -> None:
        self.case = case
        self.output_dir = output_dir
        self.result = CaseResultSchema(
            case_id=case.case_id,
            case_name=case.data.get("TestName", ""),
            prompt_json=str(case.case_dir),
            repeat_index=repeat_index,
        )
        self._validated_count = 0
        self._first_errors: list[str] = []
        self._first_summary: dict[str, Any] | None = None
        # Timing anchor at harness creation (~case start) so even the FIRST
        # node (intake) gets a duration = first_callback_time - anchor.
        self._last_node_t = time.perf_counter()

    def interrupt_handler(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Record the validate verdict and ALWAYS approve.

        By the time the agent interrupts a human its *automatic* directed
        rollback is exhausted, so the model is as good as it will get.
        Rejecting would route back to a full rebuild, which resets
        ``retry_count`` and is NOT bounded by ``MAX_RETRIES`` — it can loop
        forever on a persistent error. First-pass success is decided purely
        by the first validate hit (errors == 0 AND retry_count == 0);
        approving lets every case reach simulate so ``simulation_ok`` /
        ``recovered`` are always measurable.
        """
        self._validated_count += 1
        self.result.validate_hits = self._validated_count
        self.result.reached_validate = True

        errors = list(payload.get("errors", []) or [])
        summary = payload.get("summary", {}) or {}

        if self._validated_count == 1:
            self._first_errors = errors
            self._first_summary = summary if isinstance(summary, dict) else None
            self.result.first_validate_errors = len(errors)

        self.result.cross_ref_errors_seen.extend(errors)
        if isinstance(summary, dict):
            self.result.rollback_rounds = max(
                self.result.rollback_rounds, int(summary.get("retry_count", 0))
            )
        return {"approved": True}

    def event_handler(self, node: str, update: dict[str, Any]) -> None:
        """Record timing, sequence and diagnostic signals for one node run."""
        if node in ("__interrupt__", "__end__", "__start__"):
            return
        # This callback fires AFTER the node finished, so the elapsed time
        # since the previous callback is this node's wall-clock duration.
        now = time.perf_counter()
        duration = round(now - self._last_node_t, 3)
        self._last_node_t = now
        self.result.node_timings.append(
            NodeTimingSchema(node=node, duration_s=duration)
        )
        self.result.phase_total_s[node] = round(
            self.result.phase_total_s.get(node, 0.0) + duration, 3
        )

        self.result.node_sequence.append(node)
        self.result.node_counts[node] = self.result.node_counts.get(node, 0) + 1
        # A re-entry into intake/revise means the agent started over.
        if node in ("intake", "revise"):
            self.result.rebuild_count += 1
        if node == "zone":
            self._scan_zone_validator_signal(update)
        if node == "simulate":
            self.result.reached_simulate = True
            for m in update.get("messages") or []:
                content = getattr(m, "content", None) or str(m)
                if content:
                    self.result.simulate_message = str(content)
                    break
            self._record_sim_round()

    def _scan_zone_validator_signal(self, update: dict[str, Any]) -> None:
        """Recover the zone-validator exhaustion signal from the messages.

        zone.py emits a ``[zone-validator] ...`` AIMessage ONLY when the
        validator exhausted its retry budget. The structured
        ``validation_errors`` channel is NOT reliable here: the next graph
        step (cross_ref_foundations) overwrites it via a last-write-wins
        reducer, so the message stream is the only durable carrier.
        """
        for m in update.get("messages") or []:
            content = getattr(m, "content", "") or ""
            if isinstance(content, str) and content.startswith(ZONE_VALIDATOR_PREFIX):
                self.result.zone_validator_failed = True
                reasons_text = content[len(ZONE_VALIDATOR_PREFIX) :].strip()
                if reasons_text:
                    self.result.zone_validator_reasons = [
                        s.strip() for s in reasons_text.split(";") if s.strip()
                    ]

    def _record_sim_round(self) -> None:
        """Snapshot this simulate round's eplusout.err into ``sim_rounds``.

        EnergyPlus emits one line per offending object, so the SAME error
        text can repeat N times (e.g. 15 windows with bad winding). A plain
        set diff would collapse those and hide a 15 -> 9 reduction, so the
        vs-previous comparison uses a Counter (multiset) diff to reflect
        both error TYPES changing and error COUNTS changing.
        """
        info = extract_errors(self.output_dir / "eplusout.err")
        entry = SimRoundSchema(
            round_no=len(self.result.sim_rounds) + 1,
            success=not info["has_error_level"],
            fatal=info["fatal"],
            severe=info["severe"],
            warning=info["warning"],
            severe_lines=info["severe_lines"],
            fatal_lines=info["fatal_lines"],
        )
        if self.result.sim_rounds:
            prev = Counter(self.result.sim_rounds[-1].severe_lines)
            curr = Counter(entry.severe_lines)
            entry.fixed_vs_prev = [
                f"{msg}  (-{prev[msg] - curr[msg]})"
                for msg in sorted(prev)
                if curr[msg] < prev[msg]
            ]
            entry.new_vs_prev = [
                f"{msg}  (+{curr[msg] - prev[msg]})"
                for msg in sorted(curr)
                if curr[msg] > prev[msg]
            ]
        self.result.sim_rounds.append(entry)

    def finalize(self) -> None:
        """Compute the final verdicts from every recorded signal."""
        # Pure modeling-quality signal: clean on the FIRST validate hit and
        # no automatic rollback at all.
        self.result.model_clean_first = (
            not self._first_errors and self.result.rollback_rounds == 0
        )

        # Simulation success needs four independent signals: the simulate
        # message reports a real idf path, EnergyPlus produced result
        # artifacts, the err file has no Error-level lines, and the IDF has
        # real geometry (an empty 0-zone building can "complete" cleanly).
        msg = self.result.simulate_message or ""
        has_idf_in_msg = "idf=" in msg.lower() and "error" not in msg.lower()
        produced_output, out_files, err_path = _check_simulation_output(self.output_dir)
        self.result.output_files = out_files
        err_info = extract_errors(err_path)
        self.result.err_fatal_count = err_info["fatal"]
        self.result.err_severe_count = err_info["severe"]
        self.result.err_warning_count = err_info["warning"]
        self.result.err_has_error_level = err_info["has_error_level"]

        if "idf=" in msg:
            token = msg.split("idf=", 1)[1].strip().split()[0]
            self.result.idf_path = token or None

        geometry_ok = False
        if self.result.idf_path:
            geom = _inspect_idf_geometry(Path(self.result.idf_path))
            self.result.idf_zone_count = geom.zones
            self.result.idf_surface_count = geom.surfaces
            self.result.idf_zones_without_surfaces = geom.zones_without_surfaces
            geometry_ok = (
                geom.zones > 0 and geom.surfaces > 0 and not geom.zones_without_surfaces
            )

        self.result.simulation_ok = (
            produced_output
            and has_idf_in_msg
            and geometry_ok
            and not self.result.err_has_error_level
        )
        self.result.first_pass = (
            self.result.model_clean_first and self.result.simulation_ok
        )
        self.result.recovered = not self.result.first_pass and self.result.simulation_ok

        if self._first_summary is not None:
            self.result.validate_summary = self._first_summary

    def snapshot_traces(self) -> None:
        """Capture the per-phase tool-call traces (best-effort)."""
        try:
            traces = export_traces()
        except Exception as e:  # pragma: no cover - tracing is best-effort
            logger.warning("trace export failed for case {}: {}", self.case.case_id, e)
            return
        self.result.phase_traces = traces
        self.result.phase_tool_stats = {
            phase: {
                "calls": len(entries),
                "succeeded": sum(1 for e in entries if e.get("success")),
                "failed": sum(1 for e in entries if not e.get("success")),
            }
            for phase, entries in traces.items()
        }


def _check_simulation_output(output_dir: Path) -> tuple[bool, list[str], Path | None]:
    """Return ``(ok, output_files, err_path)`` for a simulation output dir.

    ``ok`` is True only if EnergyPlus wrote a completion marker AND at least
    one time-series result artifact. ``eplustbl.csv`` alone is not enough:
    an empty 0-zone building can produce table/audit files without any real
    simulated zone data.
    """
    if not output_dir.exists():
        return False, [], None
    artifacts: list[str] = []
    err_path: Path | None = None
    has_end = False
    has_timeseries = False
    for p in output_dir.glob("**/*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name in (
            "eplusout.end",
            "eplusout.sql",
            "eplustbl.csv",
            "eplusout.csv",
            "eplusout.eso",
        ):
            artifacts.append(str(p))
        if name == "eplusout.end":
            has_end = True
        if name in ("eplusout.csv", "eplusout.eso"):
            has_timeseries = True
        if name == "eplusout.err":
            err_path = p
    return has_end and has_timeseries, artifacts, err_path


def _inspect_idf_geometry(idf_path: Path) -> IdfGeometrySchema:
    """Return zone/surface counts for the IDF used by simulate.

    Catches empty-building false positives independently of EnergyPlus'
    exit status and output-file presence.
    """
    result = IdfGeometrySchema()
    if not idf_path.exists():
        return result
    try:
        state = ConfigState()
        state.load_idf(idf_path)
        zones = _idf_values(state.idf, "Zone")
        surfaces = _idf_values(state.idf, "BuildingSurface:Detailed")
        result.zones = len(zones)
        result.surfaces = len(surfaces)
        surfaces_by_zone: dict[str, int] = {}
        for surface in surfaces:
            zone_name = getattr(surface, "zone_name", "")
            if zone_name:
                surfaces_by_zone[zone_name] = surfaces_by_zone.get(zone_name, 0) + 1
        result.zones_without_surfaces = [
            name
            for zone in zones
            if (name := getattr(zone, "name", ""))
            and surfaces_by_zone.get(name, 0) == 0
        ]
    except Exception as exc:
        result.error = str(exc)
    return result


def _render_case_top_view(harness: CaseHarness, out_base: Path) -> None:
    """Render a bird's-eye PNG for the case's produced IDF, if any.

    Best-effort: a render failure only logs a warning and never affects the
    test verdict. The IDF path comes from the simulate message; when missing
    or stale, fall back to globbing ``*.idf`` under the simulation output dir.
    """
    idf_path: Path | None = None
    if harness.result.idf_path and Path(harness.result.idf_path).exists():
        idf_path = Path(harness.result.idf_path)
    else:
        hits = sorted(harness.output_dir.glob("**/*.idf"))
        if hits:
            idf_path = hits[-1]
    if idf_path is None:
        logger.info("[{}] no IDF found; skipping top-view render", harness.case.case_id)
        return
    try:
        # Imported lazily: the render stack pulls in matplotlib and parses
        # geometry at import time; a broken renderer must not break the
        # benchmark (or pytest collection of this module).
        from tests.agent.render_top_view import render_top_view

        out_png = out_base / "top_view.png"
        render_top_view(idf_path, out_png)
        logger.info("[{}] top-view rendered -> {}", harness.case.case_id, out_png)
    except Exception as e:
        logger.warning("[{}] top-view render failed: {}", harness.case.case_id, e)


def run_one_case(
    case: CaseSpecSchema,
    epw: Path,
    results_root: Path,
    timestamp: str,
    repeat_index: int = 0,
    use_images: bool = False,
    data_root: Path | None = None,
) -> CaseResultSchema:
    """Run one (case, repetition) pair and return its full result record."""
    # Mirror the data tree into the results tree: results/<ts>/<rel>/rep_<i>.
    if data_root is not None:
        try:
            rel = case.case_dir.resolve().relative_to(data_root.resolve())
        except ValueError:
            rel = Path(case.case_id)
    else:
        rel = Path(case.case_id)
    out_base = results_root / timestamp / rel / f"rep_{repeat_index}"
    harness = CaseHarness(case, out_base / "sim_out", repeat_index=repeat_index)

    log_path = out_base / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handler_id = logger.add(
        log_path, level="DEBUG", encoding="utf-8", mode="w", enqueue=True
    )

    logger.info("===== CASE {} (rep {}) start =====", case.case_id, repeat_index)
    t0 = time.perf_counter()
    try:
        # A fresh graph per case isolates the InMemorySaver checkpointer so
        # state does not leak between cases / repetitions.
        graph = build_graph()
        reset_traces()

        initial = AgentState(
            user_input=build_user_prompt(case),
            image_paths=(collect_images(case) if use_images else []),
        )
        context = SimContext(epw_path=epw, output_dir=harness.output_dir)
        config: RunnableConfig = {
            "configurable": {
                "thread_id": f"robust_{timestamp}_{case.case_id}_{repeat_index}"
            }
        }

        run_session(
            graph,
            initial,
            context,
            config,
            on_interrupt=harness.interrupt_handler,
            on_event=harness.event_handler,
        )
        harness.snapshot_traces()
        harness.finalize()
        _render_case_top_view(harness, out_base)
    except Exception as e:
        harness.result.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        logger.exception("CASE {} crashed", case.case_id)
    finally:
        harness.result.elapsed_s = round(time.perf_counter() - t0, 2)
        logger.info(
            "===== CASE {} done: first_pass={} sim_ok={} rounds={} ({}s) =====",
            case.case_id,
            harness.result.first_pass,
            harness.result.simulation_ok,
            harness.result.rollback_rounds,
            harness.result.elapsed_s,
        )
        with contextlib.suppress(ValueError):
            logger.remove(log_handler_id)

    return harness.result


def build_summary(
    results: list[CaseResultSchema],
    timestamp: str,
    epw: Path,
    temperature: float | None,
    repeat: int,
) -> BenchmarkSummarySchema:
    """Aggregate per-run results into one benchmark summary."""
    n = len(results) or 1

    phase_accum: dict[str, float] = {}
    for r in results:
        for phase, secs in r.phase_total_s.items():
            phase_accum[phase] = phase_accum.get(phase, 0.0) + secs
    avg_phase_s = {
        phase: round(total / n, 2)
        for phase, total in sorted(
            phase_accum.items(), key=lambda kv: kv[1], reverse=True
        )
    }

    # Per-case first-pass stability across repetitions: a case that passes
    # every time is stable; flaky cases are the interesting signal.
    per_case: dict[str, list[bool]] = {}
    for r in results:
        per_case.setdefault(r.case_id, []).append(r.first_pass)
    per_case_stability = {
        cid: CaseStabilitySchema(
            first_pass_count=sum(v),
            repetitions=len(v),
            first_pass_rate=round(sum(v) / len(v), 4) if v else 0.0,
        )
        for cid, v in sorted(per_case.items())
    }

    return BenchmarkSummarySchema(
        timestamp=timestamp,
        epw=str(epw),
        temperature=temperature,
        repeat=repeat,
        total_runs=len(results),
        first_pass_ok=sum(r.first_pass for r in results),
        first_pass_rate=round(sum(r.first_pass for r in results) / n, 4),
        model_clean_first_ok=sum(r.model_clean_first for r in results),
        model_clean_first_rate=round(sum(r.model_clean_first for r in results) / n, 4),
        recovered_ok=sum(r.recovered for r in results),
        simulation_ok=sum(r.simulation_ok for r in results),
        reached_simulate=sum(r.reached_simulate for r in results),
        failed=sum(not r.first_pass and not r.recovered for r in results),
        crashed=sum(r.error is not None for r in results),
        err_with_error_level=sum(r.err_has_error_level for r in results),
        avg_rollback_rounds=round(sum(r.rollback_rounds for r in results) / n, 2),
        avg_rebuild_count=round(sum(r.rebuild_count for r in results) / n, 2),
        avg_validate_hits=round(sum(r.validate_hits for r in results) / n, 2),
        avg_first_validate_errors=round(
            sum(r.first_validate_errors for r in results) / n, 2
        ),
        avg_elapsed_s=round(sum(r.elapsed_s for r in results) / n, 2),
        total_err_fatal=sum(r.err_fatal_count for r in results),
        total_err_severe=sum(r.err_severe_count for r in results),
        total_err_warning=sum(r.err_warning_count for r in results),
        zone_validator_failed=sum(r.zone_validator_failed for r in results),
        avg_phase_s=avg_phase_s,
        per_case_stability=per_case_stability,
    )


def render_report(
    results: list[CaseResultSchema],
    summary: BenchmarkSummarySchema,
) -> str:
    """Render the human-readable markdown report."""
    lines = [
        f"# Agent Robustness Benchmark — {summary.timestamp}",
        "",
        f"- EPW: `{summary.epw}`  |  Temperature: `{summary.temperature}`  "
        f"|  Repetitions/case: **{summary.repeat}**  "
        f"|  Total runs: **{summary.total_runs}**",
        "",
        "## Aggregate",
        "",
        f"- **一次成功 (first_pass, end-to-end): {summary.first_pass_ok}"
        f" → {summary.first_pass_rate * 100:.1f}%**",
        "  _(首次建模零交叉错误 + 零回滚 + 模拟跑通且 err 无 Error 级)_",
        f"- 建模一次干净 (model_clean_first, 纯建模口径): "
        f"{summary.model_clean_first_ok} → "
        f"{summary.model_clean_first_rate * 100:.1f}%",
        f"- 跑通模拟 (simulation_ok): {summary.simulation_ok}  "
        f"(其中 recovered 回滚后跑通: {summary.recovered_ok})",
        f"- 失败 (failed): {summary.failed}  |  崩溃 (crashed): "
        f"{summary.crashed}  |  err 含 Error 级: {summary.err_with_error_level}",
        f"- zone-validator 耗尽 (zone_validator_failed): "
        f"{summary.zone_validator_failed}  "
        f"_(zone 阶段未能满足 specs，validation_errors 被 cross_ref 覆盖，"  # noqa: RUF001
        f"故从 [zone-validator] message 恢复)_",
        f"- 平均: 回滚 {summary.avg_rollback_rounds} 轮  |  "
        f"重建 {summary.avg_rebuild_count} 次  |  "
        f"validate 命中 {summary.avg_validate_hits} 次  |  "
        f"首次错误 {summary.avg_first_validate_errors} 条  |  "
        f"耗时 {summary.avg_elapsed_s} s",
        f"- err 统计: Fatal {summary.total_err_fatal}  |  "
        f"Severe {summary.total_err_severe}  |  "
        f"Warning {summary.total_err_warning}",
        "- 各阶段平均耗时(s): "
        + "  ".join(f"{p} {s}" for p, s in summary.avg_phase_s.items()),
        "",
        "## Per-run detail",
        "",
        "| Case | rep | first_pass | model_clean | recovered | sim_ok "
        "| rollback | rebuild | validate hits | err@1st | F/S/W | zfail "
        "| time(s) | slowest phase |",
        "|------|-----|-----------|-------------|-----------|---------"
        "|----------|---------|-----------|---------|-------|-------"
        "|---------|---------------|",
    ]
    for r in results:
        if r.phase_total_s:
            slowest_phase, slowest_s = max(
                r.phase_total_s.items(), key=lambda kv: kv[1]
            )
            slowest = f"{slowest_phase} {slowest_s}s"
        else:
            slowest = "-"
        zfail = "⚠️" if r.zone_validator_failed else ""
        lines.append(
            f"| {r.case_id} | {r.repeat_index} "
            f"| {'✅' if r.first_pass else '❌'} "
            f"| {'✅' if r.model_clean_first else '❌'} "
            f"| {'✅' if r.recovered else '-'} "
            f"| {'✅' if r.simulation_ok else '❌'} "
            f"| {r.rollback_rounds} | {r.rebuild_count} | {r.validate_hits} "
            f"| {r.first_validate_errors} "
            f"| {r.err_fatal_count}/{r.err_severe_count}/{r.err_warning_count} "
            f"| {zfail} "
            f"| {r.elapsed_s} | {slowest} |"
        )
    if summary.repeat > 1:
        lines += [
            "",
            "## Per-case stability (across repetitions)",
            "",
            "| Case | first_pass / reps | rate |",
            "|------|------------------|------|",
        ]
        for cid, info in summary.per_case_stability.items():
            lines.append(
                f"| {cid} | {info.first_pass_count} / {info.repetitions} "
                f"| {info.first_pass_rate * 100:.0f}% |"
            )
    lines.append("")
    lines.append("> `errors@1st-validate` = cross-ref errors at the agent's first")
    lines.append("> validate hit (0 ⇒ clean first pass; matches `first_pass`).")

    # Only cases with more than one simulate round (i.e. a rollback fired)
    # are interesting for the error-evolution section.
    multi_round = [r for r in results if len(r.sim_rounds) > 1]
    if multi_round:
        lines += [
            "",
            "## Simulation error evolution (simulate→revise loop)",
            "",
            "Each row = one simulate round within a case. `fixed`/`new` are vs the",
            "previous round's severe errors — they show what the agent repaired and",
            "what it newly introduced each rollback.",
            "",
        ]
        for r in multi_round:
            lines.append(
                f"### {r.case_id} (rep {r.repeat_index}, "
                f"{len(r.sim_rounds)} simulate rounds)"
            )
            lines.append("")
            lines.append(
                "| round | success | fatal | severe | warning "
                "| fixed vs prev | new vs prev |"
            )
            lines.append(
                "|-------|---------|-------|--------|---------"
                "|---------------|-------------|"
            )
            for s in r.sim_rounds:
                lines.append(
                    f"| {s.round_no} | {'✅' if s.success else '❌'} "
                    f"| {s.fatal} | {s.severe} | {s.warning} "
                    f"| {len(s.fixed_vs_prev)} | {len(s.new_vs_prev)} |"
                )
            for s in r.sim_rounds:
                if not s.fixed_vs_prev and not s.new_vs_prev:
                    continue
                lines.append("")
                lines.append(f"<details><summary>round {s.round_no} detail</summary>")
                lines.append("")
                if s.fixed_vs_prev:
                    lines.append(f"**Fixed ({len(s.fixed_vs_prev)}):**")
                    lines.extend(f"- {e}" for e in s.fixed_vs_prev)
                if s.new_vs_prev:
                    lines.append(f"**Newly introduced ({len(s.new_vs_prev)}):**")
                    lines.extend(f"- {e}" for e in s.new_vs_prev)
                lines.append("")
                lines.append("</details>")
            lines.append("")
    return "\n".join(lines)


def write_reports(
    results: list[CaseResultSchema],
    timestamp: str,
    results_root: Path,
    epw: Path,
    temperature: float | None = None,
    repeat: int = 1,
) -> None:
    """Write results.json, summary.json and report.md, then echo the report."""
    out_dir = results_root / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "results.json").write_text(
        json.dumps(
            [r.model_dump(mode="json") for r in results],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = build_summary(results, timestamp, epw, temperature, repeat)
    (out_dir / "summary.json").write_text(
        summary.model_dump_json(indent=2), encoding="utf-8"
    )

    report = render_report(results, summary)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    logger.info("reports written to {}", out_dir)
    print("\n" + report.split("\n", 1)[1])


def set_llm_temperature(temperature: float) -> str | None:
    """Rewrite ``temperature:`` in src/configs/llm.yaml in place.

    The phase agents each call ``create_llm()`` with no arguments, and
    ``create_llm`` re-reads ``llm.yaml`` on every call — there is no shared
    LLM singleton to inject into, so pinning the sampling temperature for a
    whole benchmark run means editing the YAML before the first agent runs.
    ``DEFAULT_TEMPERATURE`` from .env is deliberately ignored: the benchmark
    temperature is controlled solely by ``--temperature`` so runs are
    reproducible and explicit.

    Returns:
        The original temperature value (as a string) so the caller can
        restore it afterward, even if the run crashes. None (file untouched)
        if no ``temperature:`` line is found.
    """
    text = LLM_YAML.read_text(encoding="utf-8")
    m = re.search(r"^(\s*temperature:\s*)(\S+)\s*$", text, flags=re.MULTILINE)
    if not m:
        logger.warning(
            "no `temperature:` line found in {}; leaving it unchanged", LLM_YAML
        )
        return None
    original = m.group(2)
    LLM_YAML.write_text(
        text[: m.start()] + f"{m.group(1)}{temperature}" + text[m.end() :],
        encoding="utf-8",
    )
    logger.info("llm.yaml temperature: {} -> {}", original, temperature)
    return original


def restore_llm_temperature(original: str | None) -> None:
    """Inverse of :func:`set_llm_temperature`."""
    if original is None:
        return
    text = LLM_YAML.read_text(encoding="utf-8")
    m = re.search(r"^(\s*temperature:\s*)(\S+)\s*$", text, flags=re.MULTILINE)
    if not m:
        return
    LLM_YAML.write_text(
        text[: m.start()] + f"{m.group(1)}{original}" + text[m.end() :],
        encoding="utf-8",
    )
    logger.info("llm.yaml temperature restored to {}", original)


app: Final = Typer(add_completion=False)


@app.command()
def run(
    epw: Annotated[Path, Option("--epw", help="EPW weather file.")] = DEFAULT_EPW,
    only: Annotated[
        str,
        Option(
            "--only",
            help="Comma-separated case-id prefixes to run, e.g. 'residential', "
            "'office/large', 'retail/small/case_03'. Default: all discovered "
            "cases.",
        ),
    ] = "",
    data_root: Annotated[
        Path,
        Option(
            "--data-root",
            help="Directory holding <case>/testdata_prompt.json subfolders.",
        ),
    ] = TEST_DATA_ROOT,
    images: Annotated[
        bool,
        Option(
            "--images",
            help="Pass the building drawings declared in each "
            "testdata_prompt.json to the agent as multimodal input. OFF by "
            "default: the corpus is text-only. Turn on only with a "
            "vision-capable LLM.",
        ),
    ] = False,
    temperature: Annotated[
        float | None,
        Option(
            "--temperature",
            help="Override the LLM sampling temperature for this run (written "
            "into src/configs/llm.yaml, restored on exit). Does NOT read "
            "DEFAULT_TEMPERATURE from .env.",
        ),
    ] = None,
    repeat: Annotated[
        int,
        Option(
            "--repeat",
            help="How many times to run each case. Repeat > 1 yields per-case "
            "stability stats in the report.",
        ),
    ] = 1,
) -> None:
    """Run the EnergyPlus Agent first-pass-success robustness benchmark."""
    if not epw.exists():
        logger.error("EPW not found: {}", epw)
        raise Exit(2)
    if not data_root.exists():
        logger.error("test data root not found: {}", data_root)
        raise Exit(2)
    if repeat < 1:
        logger.error("--repeat must be >= 1, got {}", repeat)
        raise Exit(2)

    all_cases = discover_cases(data_root)
    if only:
        prefixes = tuple(s.strip() for s in only.split(",") if s.strip())
        cases = [c for c in all_cases if c.case_id.startswith(prefixes)]
        if not cases:
            logger.error(
                "no cases matched --only prefixes {}; available ids: {}",
                prefixes,
                [c.case_id for c in all_cases][:10],
            )
            raise Exit(2)
    else:
        cases = all_cases
    if not cases:
        logger.error("no cases to run")
        raise Exit(2)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    total_runs = len(cases) * repeat
    logger.info(
        "running {} case(s) x {} rep(s) = {} run(s); epw={}; ts={}; temp={}; images={}",
        len(cases),
        repeat,
        total_runs,
        epw,
        timestamp,
        temperature,
        "ON" if images else "OFF (text-only)",
    )

    original_temp: str | None = None
    if temperature is not None:
        original_temp = set_llm_temperature(temperature)

    results: list[CaseResultSchema] = []
    try:
        run_idx = 0
        for case in cases:
            for rep in range(repeat):
                run_idx += 1
                logger.info(
                    "[{}/{}] case id={} rep={}",
                    run_idx,
                    total_runs,
                    case.case_id,
                    rep,
                )
                results.append(
                    run_one_case(
                        case,
                        epw,
                        RESULTS_DIR,
                        timestamp,
                        repeat_index=rep,
                        use_images=images,
                        data_root=data_root,
                    )
                )
    finally:
        if temperature is not None:
            restore_llm_temperature(original_temp)

    write_reports(
        results,
        timestamp,
        RESULTS_DIR,
        epw,
        temperature=temperature,
        repeat=repeat,
    )


if __name__ == "__main__":
    app()
