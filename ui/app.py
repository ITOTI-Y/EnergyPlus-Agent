"""Gradio web UI for testing the EnergyPlus Agent."""

from __future__ import annotations

import contextlib
import json
import os
import queue
import threading
import time
import traceback
from collections.abc import Generator
from html import escape
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_core.runnables import RunnableConfig

from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.mcp.state import ConfigState
from src.results import charts as result_charts
from src.results import load_results, parse_idf_geometry
from src.results.charts import ZONE_ALL
from src.results.solar import resolve_surface_solar
from src.utils.logging import setup_logger
from ui.idf_viewer import build_idf_3d_model

setup_logger(level="WARNING")

# Dropdown label -> internal metric key used by charts.zone_energy_3d
_METRIC_MAP: dict[str, str] = {
    "Cooling Load (kWh)": "cooling",
    "Heating Load (kWh)": "heating",
    "Annual Average Temperature (deg C)": "temperature",
    "Lighting Energy (kWh)": "lighting",
}
_METRIC_OPTIONS = list(_METRIC_MAP.keys())
_ENGLISH_UI_JS = r"""
function forceLightWorkbench() {
  document.documentElement.style.colorScheme = "light";
  document.documentElement.classList.remove("dark");
  if (document.body) {
    document.body.classList.remove("dark");
    document.body.style.colorScheme = "light";
  }
  const container = document.querySelector(".gradio-container");
  if (container) {
    container.classList.remove("dark");
    container.style.colorScheme = "light";
  }
}

function forceEnglishUi() {
  forceLightWorkbench();
  const replacements = [
    ["\u5c06\u6587\u4ef6\u62d6\u653e\u5230\u6b64\u5904", "Drop files here"],
    ["\u70b9\u51fb\u4e0a\u4f20", "Click to upload"],
    ["- \u6216 -", "- or -"],
    ["\u6216", "or"],
    ["\u901a\u8fc7 API \u4f7f\u7528", "Use via API"],
    ["\u6807\u5fd7", "badge"],
    ["\u5355\u9009\u6846", ""],
    ["\u56fe\u8868", ""],
    ["Textbox", ""],
  ];

  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  for (const node of nodes) {
    let next = node.nodeValue || "";
    for (const pair of replacements) {
      next = next.split(pair[0]).join(pair[1]);
    }
    if (next !== node.nodeValue) node.nodeValue = next;
  }
}

function selectInspectorTab(label) {
  const tabs = Array.from(document.querySelectorAll("#inspector-tabs button[role='tab']"));
  const target = tabs.find((tab) => (tab.textContent || "").trim() === label);
  if (target && target.getAttribute("aria-selected") !== "true") target.click();
}

function flashWorkbenchToast(text) {
  let toast = document.querySelector("#workbench-toast");
  if (!toast) return;
  toast.textContent = text;
  toast.classList.add("show");
  window.setTimeout(() => toast.classList.remove("show"), 1600);
}

function bindWorkbenchBehavior() {
  if (window.__energyPlusWorkbenchBound) return;
  window.__energyPlusWorkbenchBound = true;

  document.addEventListener("click", (event) => {
    const button = event.target.closest("button");
    if (!button) return;
    const label = (button.textContent || "").trim();
    if (label === "Send" || label === "Load charts") {
      window.setTimeout(() => selectInspectorTab("Results"), 420);
    }
    if (label === "Reload model") {
      window.setTimeout(() => selectInspectorTab("Model"), 120);
    }
    if (label === "New simulation") {
      window.setTimeout(() => selectInspectorTab("Model"), 420);
    }
  });
}

function runWorkbenchPolish() {
  if (!document.body) return;
  forceEnglishUi();
  bindWorkbenchBehavior();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", runWorkbenchPolish, { once: true });
} else {
  runWorkbenchPolish();
}

let polishRuns = 0;
const polishTimer = window.setInterval(() => {
  runWorkbenchPolish();
  polishRuns += 1;
  if (polishRuns >= 8) window.clearInterval(polishTimer);
}, 500);
"""


DEFAULT_EPW = Path("data/weather/Shenzhen.epw")
OUTPUT_DIR = Path("output/ui")
SESSIONS_ROOT = OUTPUT_DIR / "sessions"


# ---------------------------------------------------------------------------
# Session management - each simulation run writes to / reads from its own
# self-contained directory under ``output/ui/sessions/<session_id>/``.  This
# isolates the IDF, EnergyPlus outputs, and visualisations of every run so
# they never bleed into each other.
# ---------------------------------------------------------------------------


def _new_session_id() -> str:
    """Timestamp-style session id, e.g. ``20260617_143022``."""
    return time.strftime("%Y%m%d_%H%M%S")


def _session_dir(session_id: str) -> Path:
    """Directory holding every artefact for one session."""
    return SESSIONS_ROOT / session_id


def _session_meta_path(session_id: str) -> Path:
    return _session_dir(session_id) / "session_meta.json"


def _save_session_meta(
    session_id: str,
    user_input: str,
    summary: dict[str, Any] | None,
) -> None:
    """Persist a small meta file used to label the session dropdown."""
    meta = {
        "session_id": session_id,
        "user_input_preview": (user_input or "").strip()[:80],
        "zones_count": (summary or {}).get("zones_count", 0),
        "surfaces_count": (summary or {}).get("surfaces_count", 0),
        "fenestrations_count": (summary or {}).get("fenestrations_count", 0),
    }
    path = _session_meta_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_session_meta(session_id: str) -> dict[str, Any] | None:
    path = _session_meta_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _list_sessions() -> list[tuple[str, str]]:
    """Return ``[(label, session_id), ...]`` sorted newest-first.

    Label embeds the meta preview when available, e.g.
    ``20260617_143022 - office building (2 zones)``.
    """
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    for d in SESSIONS_ROOT.iterdir():
        if not d.is_dir():
            continue
        sid = d.name
        meta = _load_session_meta(sid) or {}
        zones = meta.get("zones_count")
        suffix = f" ({zones} zones)" if isinstance(zones, int) else ""
        label = f"{sid}{suffix}"
        items.append((label, sid))
    # Sort by session_id (= timestamp) descending
    items.sort(key=lambda x: x[1], reverse=True)
    return items


def _create_session() -> str:
    """Create a fresh session directory and return its id.

    Guards against same-second collisions (two sessions created within one
    second would otherwise share a timestamp id) by appending a numeric
    suffix when the directory already exists.
    """
    session_id = _new_session_id()
    target = _session_dir(session_id)
    # Resolve same-second collisions: 20260617_143022 -> 20260617_143022_2 -> ...
    if target.exists():
        n = 2
        while _session_dir(f"{session_id}_{n}").exists():
            n += 1
        session_id = f"{session_id}_{n}"
        target = _session_dir(session_id)
    target.mkdir(parents=True, exist_ok=True)
    return session_id


def _current_session_or_create(session_id: str | None) -> str:
    """Return a valid session id, creating one if none exists yet.

    Used at page load to guarantee the dropdown always has a current value.
    """
    if session_id and _session_dir(session_id).exists():
        return session_id
    # Reuse the newest existing session if there is one
    sessions = _list_sessions()
    if sessions:
        return sessions[0][1]
    return _create_session()


def _chat_history_path(session_id: str) -> Path:
    return _session_dir(session_id) / "chat_history.json"


def _save_chat_history(session_id: str, history: list[dict]) -> None:
    """Persist the chatbot conversation for a session so it survives reloads
    and session switching."""
    path = _chat_history_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def _load_chat_history(session_id: str) -> list[dict]:
    """Return the saved chatbot history, or an empty list if none."""
    path = _chat_history_path(session_id)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _format_history_for_revision(history: list[dict]) -> str:
    """Render recent chatbot turns into a compact text block for the revise LLM.

    Only user instructions and assistant summaries are kept (progress ticks
    and tool chatter are dropped) so the revision LLM sees what the user
    asked for in prior turns without context bloat.
    """
    if not history:
        return ""
    lines = ["## Previous conversation in this session:"]
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = msg.get("content", "")
        # content may be a list of parts (gradio format) - flatten to text
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict)
            )
        content = str(content).strip()
        if not content:
            continue
        # Skip one-line progress ticks like "Completed: ..."
        if content.startswith("Completed:"):
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            # Truncate long analysis reports
            preview = content[:300] + ("..." if len(content) > 300 else "")
            lines.append(f"Assistant: {preview}")
    return "\n".join(lines) if len(lines) > 1 else ""


NODE_LABELS: dict[str, str] = {
    "intake": "Parse Building Description",
    "zone": "Create Thermal Zones",
    "material": "Define Material Properties",
    "schedule": "Configure Schedules",
    "cross_ref_foundations": "Validate Foundation References",
    "construction": "Build Envelope Constructions",
    "surface": "Define Building Surfaces",
    "fenestration": "Configure Fenestration",
    "hvac": "Design HVAC System",
    "people": "Configure Occupant Loads",
    "lights": "Configure Lighting System",
    "cross_ref_complete": "Validate All References",
    "validate": "Validate Building Model",
    "simulate": "Run EnergyPlus",
    "analyze": "Analyze Simulation Results",
}

_AGENT_PROGRESS_STEPS: list[tuple[str, tuple[str, ...]]] = [
    ("Parse brief", ("Parse Building Description",)),
    ("Zones", ("Create Thermal Zones",)),
    (
        "Materials",
        (
            "Define Material Properties",
            "Validate Foundation References",
            "Build Envelope Constructions",
            "Define Building Surfaces",
            "Configure Fenestration",
        ),
    ),
    ("Schedules", ("Configure Schedules",)),
    (
        "Loads and HVAC",
        (
            "Configure Occupant Loads",
            "Configure Lighting System",
            "Design HVAC System",
            "Validate All References",
        ),
    ),
    (
        "Simulation",
        (
            "Validate Building Model",
            "Run EnergyPlus",
            "Analyze Simulation Results",
        ),
    ),
]

_graph = None
_graph_lock = threading.Lock()


def _get_graph():
    global _graph
    with _graph_lock:
        if _graph is None:
            _graph = build_graph()
    return _graph


def _fmt_summary(summary: dict[str, Any], errors: list[str]) -> str:
    lines = [
        "Model Validation Summary",
        f"- Thermal zones: {summary.get('zones_count', 0)}",
        f"- Materials: {summary.get('materials_count', 0)}",
        f"- Building surfaces: {summary.get('surfaces_count', 0)}",
        f"- Fenestration openings: {summary.get('fenestrations_count', 0)}",
    ]
    if errors:
        lines.append("\nCross-reference errors:")
        lines.extend(f"  * {e}" for e in errors)
    return "\n".join(lines)


def _collect_output_files(output_dir: Path) -> list[str]:
    exts = ("*.idf", "*.csv", "*.eso", "*.err", "*.htm", "*.html")
    files: list[str] = []
    for pat in exts:
        files.extend(str(p) for p in sorted(output_dir.glob(pat)))
    return files


def _latest_idf(output_dir: Path) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = sorted(
        output_dir.glob("*.idf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


History = list[dict]


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def run_agent(
    user_input: str,
    epw_file: Any,
    image_files: list[Any] | None,
    session_id: str,
) -> Generator[tuple[History, list[str]], None, None]:
    """Streaming generator: yields (chat_history, output_files) updates.

    All outputs (IDF, EnergyPlus results) are written into the session
    directory ``output/ui/sessions/<session_id>/``.
    """

    if not user_input.strip():
        yield (
            [_msg("assistant", "Please enter a building description before running.")],
            [],
        )
        return

    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    # Detect revision turn: a prior IDF exists in this session
    is_revision_turn = bool(_latest_idf(session_dir))

    event_q: queue.Queue = queue.Queue()
    thread_id = f"ui_{id(event_q)}_{session_id}"

    def on_event(node_name: str, update: dict) -> None:
        label = NODE_LABELS.get(node_name, node_name)
        event_q.put(("node", label))

    def on_interrupt(payload: dict) -> dict:
        event_q.put(("interrupt", payload))
        errors = payload.get("errors", [])
        if errors:
            return {
                "approved": False,
                "feedback": "Please fix the following errors: " + "; ".join(errors),
            }
        return {"approved": True}

    def worker() -> None:
        try:
            epw_path = Path(epw_file.name) if epw_file else DEFAULT_EPW
            images: list[str] = [f.name for f in image_files] if image_files else []

            # Multi-turn revision: if a previous IDF exists in this session,
            # rebuild config_state from it and flag is_revision so the graph
            # enters revise_node (incremental update) instead of intake.
            latest_idf = _latest_idf(session_dir)
            if latest_idf and latest_idf.exists():
                cs = ConfigState()
                cs.load_idf(latest_idf)
                # Carry the seed IDF as text in a declared ConfigState field
                # so it survives LangGraph's START-boundary input coercion
                # (which strips the PrivateAttr _idf). merge_config_state
                # rebuilds _idf from it on every channel write.
                cs.seed_idf_text = latest_idf.read_text(encoding="utf-8")
                history = _load_chat_history(session_id)
                history_text = _format_history_for_revision(history[-6:])
                effective_input = (
                    f"{history_text}\n\n## New instruction:\n{user_input}"
                    if history_text
                    else user_input
                )
                initial = AgentState(
                    user_input=effective_input,
                    image_paths=images,
                    config_state=cs,
                    is_revision=True,
                )
            else:
                initial = AgentState(user_input=user_input, image_paths=images)
            context = SimContext(epw_path=epw_path, output_dir=session_dir)
            config: dict = {"configurable": {"thread_id": thread_id}}

            state = run_session(
                _get_graph(),
                initial,
                context,
                RunnableConfig(configurable=config),
                on_interrupt=on_interrupt,
                on_event=on_event,
            )
            event_q.put(("done", state))
        except Exception:
            err = traceback.format_exc()
            print(f"\n{'=' * 60}\nUI WORKER ERROR:\n{err}\n{'=' * 60}", flush=True)
            event_q.put(("error", err))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    opening = (
        f"Modifying the existing model in session `{session_id}`..."
        if is_revision_turn
        else f"Starting Agent in session `{session_id}`..."
    )
    history: History = [
        _msg("user", user_input),
        _msg("assistant", opening),
    ]
    yield history, []

    while True:
        try:
            event = event_q.get(timeout=180)
        except queue.Empty:
            history.append(
                _msg(
                    "assistant",
                    "Wait timed out after 3 minutes. Please check the network "
                    "or model service.",
                )
            )
            _save_chat_history(session_id, history)
            yield history, []
            break

        kind = event[0]

        if kind == "node":
            history.append(_msg("assistant", f"Completed: {event[1]}"))
            yield history, []

        elif kind == "interrupt":
            payload = event[1]
            summary = payload.get("summary", {})
            errors = payload.get("errors", [])
            msg = _fmt_summary(summary, errors)
            status = (
                "Errors found; sending feedback to the model for correction..."
                if errors
                else "Validation passed; starting simulation..."
            )
            history.append(_msg("assistant", f"{msg}\n\n{status}"))
            yield history, []

        elif kind == "done":
            state = event[1]
            # Persist session meta for the dropdown label
            try:
                summary = state.get("config_state", None)
                summary_dict = (
                    summary.get_summary().model_dump()
                    if summary and hasattr(summary, "get_summary")
                    else {}
                )
            except Exception:
                summary_dict = {}
            _save_session_meta(session_id, user_input, summary_dict)
            # Show the analysis report from the final [analyze] message if present
            analyze_msg = next(
                (
                    str(getattr(m, "content", ""))
                    for m in reversed(state.get("messages", []))
                    if "[analyze]" in str(getattr(m, "content", ""))
                ),
                None,
            )
            files = _collect_output_files(session_dir)
            if analyze_msg:
                # Strip the "[analyze] " prefix for display
                report = analyze_msg.replace("[analyze] ", "", 1)
                history.append(_msg("assistant", report))
            elif files:
                file_list = "\n".join(f"  {f}" for f in files)
                history.append(
                    _msg(
                        "assistant",
                        f"Simulation completed. Output files:\n{file_list}",
                    )
                )
            else:
                history.append(
                    _msg(
                        "assistant",
                        "Simulation completed, but the output directory is empty. "
                        "The simulation step may not have run.",
                    )
                )
            # Hint the user that the visualization panel has been populated
            history.append(
                _msg(
                    "assistant",
                    "The **3D model** and **energy analysis charts** are now "
                    "available in the right-hand panel.",
                )
            )
            _save_chat_history(session_id, history)
            yield history, files
            break

        elif kind == "error":
            err = event[1]
            history.append(_msg("assistant", f"Run failed:\n{err}"))
            _save_chat_history(session_id, history)
            yield history, []
            break

    t.join(timeout=5)


def _schedule_zone_choices(output_dir: Path) -> list[tuple[str, str]]:
    try:
        ts = load_results(output_dir).timeseries
        keys = result_charts.list_schedule_zone_keys(ts)
        return [("All zones (max)", ZONE_ALL)] + [
            (result_charts.format_zone_label(k), k) for k in keys
        ]
    except FileNotFoundError:
        return [("All zones (max)", ZONE_ALL)]


def _load_visualizations(
    output_dir: Path,
    metric_label: str,
    zone_key: str,
) -> tuple:
    """Load simulation results and return all Plotly figures (10 plots)."""
    metric_key = _METRIC_MAP.get(metric_label, "cooling")
    empty = (None,) * 10

    try:
        result = load_results(output_dir)
    except FileNotFoundError:
        return empty

    ts = result.timeseries
    tabular = result.tabular
    zone_meta = tabular.get("zone_summary", {})

    fig_3d = None
    fig_solar_3d = None
    if result.idf_path and result.idf_path.exists():
        try:
            zones = parse_idf_geometry(result.idf_path)
            fig_3d = result_charts.zone_energy_3d(
                zones,
                ts,
                metric=metric_key,
                zone_metadata=zone_meta,
            )
            solar_vals, unit, note = resolve_surface_solar(
                zones,
                result.run_dir,
                result.idf_path,
            )
            fig_solar_3d = result_charts.exterior_solar_irradiation_3d(
                zones,
                solar_vals,
                unit=unit,
                source_note=note,
            )
        except Exception as exc:
            print(f"[viz] 3D chart error: {exc}", flush=True)

    try:
        fig_schedule_people, fig_schedule_equipment = (
            result_charts.operation_schedule_pair(
                ts,
                zone_key=zone_key,
            )
        )
        figs_2d = result_charts.all_2d_charts(ts, tabular)
    except Exception as exc:
        print(f"[viz] 2D charts error: {exc}", flush=True)
        fig_schedule_people = fig_schedule_equipment = None
        figs_2d = {}

    return (
        fig_3d,
        fig_solar_3d,
        fig_schedule_people,
        fig_schedule_equipment,
        figs_2d.get("end_use"),
        figs_2d.get("comfort"),
        figs_2d.get("monthly_hvac"),
        figs_2d.get("temp_heatmap"),
        figs_2d.get("hvac_demand"),
        figs_2d.get("temp_rh_scatter"),
    )


def _update_schedules_only(output_dir: Path, zone_key: str) -> tuple:
    try:
        result = load_results(output_dir)
        return result_charts.operation_schedule_pair(
            result.timeseries, zone_key=zone_key
        )
    except FileNotFoundError:
        return None, None


def _update_3d_only(output_dir: Path, metric_label: str):
    """Recompute only the zone 3-D chart when the metric dropdown changes."""
    metric_key = _METRIC_MAP.get(metric_label, "cooling")
    try:
        result = load_results(output_dir)
    except FileNotFoundError:
        return None

    if result.idf_path and result.idf_path.exists():
        try:
            zones = parse_idf_geometry(result.idf_path)
            return result_charts.zone_energy_3d(
                zones,
                result.timeseries,
                metric=metric_key,
                zone_metadata=result.tabular.get("zone_summary", {}),
            )
        except Exception as exc:
            print(f"[viz] 3D chart error: {exc}", flush=True)
    return None


def _model_status_markdown(idf_path: Path | None) -> str:
    """Build the status string shown above the 3-D viewer."""
    if idf_path is None:
        return (
            "**No IDF loaded.** Run a simulation first, upload an `.idf` "
            "file, or click *Load Latest IDF*."
        )
    try:
        from src.results import parse_fenestrations

        zones = parse_idf_geometry(idf_path)
        fens = parse_fenestrations(idf_path)
        n_surfaces = sum(len(z.surfaces) for z in zones.values())
        return (
            f"**Loaded:** `{idf_path.name}`\n\n"
            f"- Thermal zones: **{len(zones)}**\n"
            f"- Surfaces: **{n_surfaces}**\n"
            f"- Window/door openings: **{len(fens)}**"
        )
    except Exception as exc:
        return f"**Loaded:** `{idf_path.name}` (stats unavailable: {exc})"


def on_load_3d_model(session_dir_value, idf_file):
    """Render the 3-D model from an uploaded IDF, or fall back to the
    most recent IDF in the current session directory.  Never raises.
    """
    idf_path: Path | None = None
    session_dir = Path(session_dir_value) if session_dir_value else None
    try:
        if idf_file is not None:
            idf_path = Path(idf_file)
        elif session_dir is not None:
            idf_path = _latest_idf(session_dir)
    except Exception:
        idf_path = None

    if idf_path is None or not idf_path.exists():
        from ui.idf_viewer import _empty_figure

        return (
            _empty_figure("No IDF found. Run a simulation or upload an .idf file."),
            "**No IDF loaded.** Run a simulation or upload an `.idf` file.",
        )

    try:
        fig = build_idf_3d_model(idf_path)
        return fig, _model_status_markdown(idf_path)
    except Exception as exc:
        from ui.idf_viewer import _empty_figure

        print(f"[viz] 3D model viewer error: {exc}", flush=True)
        return _empty_figure(
            f"Failed to build 3-D model: {exc}"
        ), _model_status_markdown(idf_path)


def _session_info_markdown(session_id: str) -> str:
    """Status string describing the current session directory + artefacts."""
    if not session_id:
        return "**No active session.** Click *New Session* to create one."
    sdir = _session_dir(session_id)
    if not sdir.exists():
        return f"**Session `{session_id}`** - directory missing."
    meta = _load_session_meta(session_id) or {}
    has_idf = any(sdir.glob("temp_*.idf"))
    has_csv = (sdir / "eplusout.csv").exists()
    parts = [
        f"**Session:** `{session_id}`",
        f"- Directory: `{sdir}`",
        f"- IDF: {'generated' if has_idf else 'not yet'}",
        f"- Simulation results: {'available' if has_csv else 'not yet'}",
    ]
    preview = meta.get("user_input_preview")
    if preview:
        parts.append(f"- Description: _{preview}_")
    return "\n".join(parts)


def _refresh_all_for_session(session_id: str, metric_label: str, zone_key: str):
    """Recompute every visualisation for a session.

    Returns a tuple aligned to the ``_all_plots + model_3d + model_status +
    zone_dd`` output bundle used by the session-change / run-complete handlers.
    """
    sdir = _session_dir(session_id) if session_id else None
    empty_10 = (None,) * 10
    if sdir is None or not sdir.exists():
        return (*empty_10, None, "**Session directory missing.**", gr.update())

    figs = _load_visualizations(sdir, metric_label, zone_key)
    model_fig, model_status_text = on_load_3d_model(str(sdir), None)
    zone_choices = _schedule_zone_choices(sdir)
    # Keep current zone_key if still valid, else reset to ZONE_ALL
    valid_keys = {c[1] for c in zone_choices}
    new_zone = zone_key if zone_key in valid_keys else ZONE_ALL
    return (
        *figs,
        model_fig,
        model_status_text,
        gr.update(choices=zone_choices, value=new_zone),
    )


# ---------------------------------------------------------------------------
# Claude-style workbench CSS (injected via launch(css=...))
# ---------------------------------------------------------------------------

_CHATGPT_CSS = """
:root {
    --bg: #f4f4f2;
    --surface: #ffffff;
    --surface-2: #eeeeec;
    --surface-3: #deded9;
    --fg: #10100f;
    --fg-2: #444440;
    --muted: #76766f;
    --border: #d5d5cf;
    --border-strong: #9a9a92;
    --accent-on: #ffffff;
    --viz-cooling: #4f4f49;
    --viz-lighting: #8f8f86;
    --viz-equipment: #696961;
    --viz-comfort: #77776f;
    --viz-solar: #a6a69c;
    --viz-alert: #2f2f2c;
    --font-display: ui-serif, Georgia, "Times New Roman", serif;
    --font-body: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    --font-mono: "JetBrains Mono", "SF Mono", Consolas, ui-monospace, monospace;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-pill: 999px;
    --shadow-soft: 0 18px 48px rgba(16, 16, 15, 0.08);
    --focus-ring: 0 0 0 4px rgba(16, 16, 15, 0.12);
}

html, body, body.dark, .dark, .gradio-container {
    color-scheme: light !important;
    background: var(--bg) !important;
    color: var(--fg) !important;
    font-family: var(--font-body) !important;
    --body-background-fill: var(--bg) !important;
    --background-fill-primary: var(--surface) !important;
    --background-fill-secondary: var(--surface-2) !important;
    --block-background-fill: var(--surface) !important;
    --block-border-color: var(--border) !important;
    --border-color-primary: var(--border) !important;
    --border-color-accent: var(--fg) !important;
    --input-background-fill: var(--surface) !important;
    --input-border-color: var(--border-strong) !important;
    --button-primary-background-fill: var(--fg) !important;
    --button-primary-background-fill-hover: #2f2f2c !important;
    --button-primary-text-color: var(--accent-on) !important;
    --button-secondary-background-fill: var(--surface) !important;
    --button-secondary-background-fill-hover: var(--surface-2) !important;
    --button-secondary-text-color: var(--fg) !important;
    --checkbox-label-background-fill-selected: var(--surface) !important;
    --checkbox-label-text-color-selected: var(--fg) !important;
    --color-accent: var(--fg) !important;
    --color-accent-soft: var(--surface-2) !important;
}

.gradio-container {
    max-width: 100% !important;
    min-height: 100vh !important;
    padding: 0 !important;
}

.gradio-container * {
    letter-spacing: 0 !important;
}

.main, .wrap, .contain, .app {
    background: var(--bg) !important;
}

.wrap.sidebar-parent,
.wrap.svelte-zxu34v.sidebar-parent {
    min-height: 0 !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
}

.wrap.sidebar-parent > .contain,
.wrap.svelte-zxu34v.sidebar-parent > .contain,
.wrap.sidebar-parent > .contain > .column,
.wrap.svelte-zxu34v.sidebar-parent > .contain > .column {
    display: contents !important;
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    height: 0 !important;
    min-height: 0 !important;
    max-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    overflow: visible !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    outline: 0 !important;
}

#chat-sidebar, #viz-panel {
    height: 100vh !important;
    max-height: 100vh !important;
    overflow: auto !important;
    background: var(--surface-2) !important;
    border-color: var(--border) !important;
}

#chat-sidebar > button,
#viz-panel > button {
    display: none !important;
}

#chat-sidebar {
    left: 0 !important;
    right: auto !important;
    transform: none !important;
    padding: 16px !important;
    border-right: 1px solid var(--border) !important;
    z-index: 20 !important;
}

#viz-panel {
    left: auto !important;
    right: 0 !important;
    transform: none !important;
    padding: 0 !important;
    border-left: 1px solid var(--border) !important;
    z-index: 20 !important;
}

#viz-panel .sidebar-content {
    padding: 0 !important;
}

#viz-panel .sidebar-content > .column,
#viz-panel .sidebar-content > div,
#viz-panel .html-container,
#viz-panel .gradio-html,
#viz-panel .prose {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

#viz-panel .html-container {
    padding: 0 !important;
}

#chat-pane {
    position: fixed !important;
    inset: 0 500px 0 248px !important;
    height: 100vh !important;
    min-width: 0 !important;
    width: auto !important;
    max-width: none !important;
    margin: 0 !important;
    display: grid !important;
    grid-template-rows: auto minmax(0, 1fr) auto !important;
    background: var(--surface) !important;
}

#chat-pane > * {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

#top-bar, #chat-scroll, #composer-wrap {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

#brand-lockup {
    display: grid;
    grid-template-columns: 36px minmax(0, 1fr);
    gap: 12px;
    align-items: center;
    padding: 8px;
    margin-bottom: 16px;
}

#brand-lockup .brand-mark {
    width: 36px;
    height: 36px;
    display: grid;
    place-items: center;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    background: var(--fg);
    color: var(--accent-on);
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 800;
}

#brand-lockup .brand-name {
    font-weight: 750;
    color: var(--fg);
}

#brand-lockup .brand-sub {
    margin-top: 1px;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 12px;
}

.section-label p {
    margin: 18px 8px 8px !important;
    color: var(--muted) !important;
    font-family: var(--font-mono) !important;
    font-size: 12px !important;
    text-transform: uppercase !important;
}

.sidebar-footer {
    margin-top: 18px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--surface);
}

.sidebar-footer strong {
    color: var(--fg);
}

.sidebar-footer p {
    margin: 0 !important;
    color: var(--muted) !important;
    font-size: 13px !important;
}

.new-run, .send-button, .btn-primary {
    min-height: 40px !important;
    border: 1px solid var(--fg) !important;
    border-radius: var(--radius-md) !important;
    background: var(--fg) !important;
    color: var(--accent-on) !important;
    font-weight: 750 !important;
    box-shadow: none !important;
}

.new-run {
    width: 100% !important;
    min-height: 44px !important;
}

.new-run:hover, .send-button:hover, .btn-primary:hover {
    transform: translateY(-1px);
}

.secondary-button, .btn-secondary, #viz-panel button, #chat-sidebar button:not(.new-run) {
    min-height: 40px !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    color: var(--fg) !important;
    font-weight: 650 !important;
    box-shadow: none !important;
}

.secondary-button:hover, .btn-secondary:hover, #viz-panel button:hover {
    border-color: var(--border-strong) !important;
    background: var(--surface-2) !important;
}

#session-list {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#session-list label {
    width: 100%;
    min-height: 40px;
    margin: 0 0 4px !important;
    padding: 8px 10px !important;
    border: 1px solid transparent !important;
    border-radius: var(--radius-md) !important;
    background: transparent !important;
    color: var(--fg-2) !important;
    font-size: 13px !important;
}

#session-list label:hover,
#session-list input[type="radio"]:checked + label {
    border-color: var(--border) !important;
    background: var(--surface) !important;
    color: var(--fg) !important;
}

#top-bar {
    display: grid !important;
    grid-template-columns: minmax(0, 1fr) auto !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 16px !important;
    padding: 16px 24px !important;
    border-bottom: 1px solid var(--border) !important;
    background: color-mix(in oklab, var(--surface) 88%, transparent) !important;
    backdrop-filter: blur(10px);
}

#top-title {
    min-width: 0 !important;
    overflow: hidden !important;
    width: 100% !important;
    max-width: 100% !important;
}

#top-title h1, #top-title h2, #top-title h3 {
    margin: 0 !important;
    color: var(--fg) !important;
    font-family: var(--font-display) !important;
    font-size: 30px !important;
    line-height: 1.08 !important;
}

#top-title p {
    margin: 4px 0 0 !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#chat-area {
    min-height: 0 !important;
    height: auto !important;
    border: 0 !important;
    background: var(--surface) !important;
    box-shadow: none !important;
    overflow: visible !important;
}

#chat-area .prose,
#chat-area .html-container,
#chat-area .gradio-html {
    width: 100% !important;
    max-width: none !important;
    background: transparent !important;
    border: 0 !important;
    box-shadow: none !important;
    padding: 0 !important;
}

.conversation-stream {
    display: grid;
    gap: 14px;
    width: min(880px, 100%);
    margin: 0 auto;
}

.workbench-message {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 12px;
    width: 100%;
    align-items: start;
}

#chat-area .message, #chat-area .bubble, #chat-area [data-testid="bot"], #chat-area [data-testid="user"] {
    max-width: 880px !important;
}

#chat-area .bubble-wrap, #chat-area .message-wrap {
    gap: 12px !important;
}

#chat-area [data-testid="bot"] > div,
#chat-area [data-testid="user"] > div,
#chat-area .message > div {
    border-radius: var(--radius-lg) !important;
}

.workbench-message.user {
    grid-template-columns: minmax(0, 1fr) 34px;
}

.workbench-message.user .avatar {
    grid-column: 2;
    grid-row: 1;
    background: var(--fg);
    color: var(--accent-on);
}

.workbench-message.user .bubble {
    grid-column: 1;
    grid-row: 1;
    justify-items: end;
}

.workbench-message.user .bubble-card {
    width: min(680px, 100%);
    background: var(--fg);
    border-color: var(--fg);
}

.workbench-message.user .bubble-card p {
    color: var(--accent-on);
}

#composer-wrap {
    display: block !important;
    width: 100% !important;
    padding: 16px 24px 20px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--surface) !important;
}

#composer {
    display: block !important;
    width: min(880px, 100%) !important;
    max-width: 880px !important;
    margin-inline: auto !important;
    overflow: hidden !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
    box-shadow: var(--shadow-soft) !important;
}

#msg-input, #msg-input > div {
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

#msg-input textarea {
    min-height: 96px !important;
    max-height: 180px !important;
    resize: vertical !important;
    border: 0 !important;
    border-radius: 0 !important;
    padding: 16px !important;
    color: var(--fg) !important;
    background: transparent !important;
    line-height: 1.5 !important;
}

#msg-input textarea:focus {
    box-shadow: none !important;
}

#composer-footer {
    display: flex !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 12px !important;
    padding: 12px !important;
    border-top: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
}

.composer-file {
    min-width: 120px !important;
    max-width: 180px !important;
    flex: 0 0 auto !important;
}

.composer-file label span {
    color: var(--fg-2) !important;
    font-size: 12px !important;
}

.composer-file .wrap,
.composer-file .container,
.composer-file [data-testid="file"],
.composer-file .file-preview-holder,
.composer-file .upload-container {
    min-height: 38px !important;
    max-height: 48px !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    overflow: hidden !important;
}

.composer-file .file-preview,
.composer-file .file-name,
.composer-file .file-size,
.composer-file svg {
    display: none !important;
}

#inspector-header {
    display: grid;
    gap: 12px;
    padding: 16px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-2);
}

#inspector-header h2 {
    margin: 0;
    color: var(--fg);
    font-family: var(--font-display);
    font-size: 22px;
    line-height: 1.08;
}

#inspector-header p {
    margin: 4px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.status-pill {
    display: inline-flex;
    min-height: 28px;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 4px 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    background: var(--surface);
    color: var(--fg-2);
    font-size: 12px;
    white-space: nowrap;
}

.status-pill::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: var(--radius-pill);
    background: var(--fg);
}

#inspector-tabs {
    padding: 16px !important;
}

#inspector-tabs .tab-nav {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    gap: 8px !important;
    margin: 0 0 16px !important;
    border: 0 !important;
}

#inspector-tabs button[role="tab"] {
    min-height: 38px !important;
    justify-content: center !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
    color: var(--fg-2) !important;
    font-weight: 700 !important;
}

#inspector-tabs button[role="tab"][aria-selected="true"] {
    border-color: var(--fg) !important;
    background: var(--fg) !important;
    color: var(--accent-on) !important;
}

.inspector-panel, .chart-card {
    overflow: hidden !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
}

.panel-title h2, .panel-title h3, .panel-title p {
    margin: 0 !important;
}

.panel-title h2, .panel-title h3 {
    color: var(--fg) !important;
    font-family: var(--font-display) !important;
    font-size: 18px !important;
    line-height: 1.08 !important;
}

.panel-title p, .model-note p {
    color: var(--muted) !important;
    font-size: 13px !important;
}

.model-stage {
    overflow: hidden !important;
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--border) !important;
    background:
        linear-gradient(0deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        linear-gradient(90deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        var(--surface-2) !important;
    background-size: 26px 26px !important;
}

.model-stage .plot-container, .model-stage .js-plotly-plot {
    background: transparent !important;
}

.metric-row {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 8px !important;
}

.metric-card {
    min-height: 72px !important;
    padding: 12px !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface-2) !important;
}

.metric-card p {
    margin: 0 !important;
    color: var(--muted) !important;
    font-size: 12px !important;
}

.metric-card strong {
    display: block;
    margin-top: 4px;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: 18px;
}

.chart-card {
    padding: 12px !important;
}

.chart-card label, .chart-card .label-wrap {
    color: var(--fg) !important;
    font-family: var(--font-display) !important;
    font-size: 16px !important;
}

.chart-card .plot-container {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface-2) !important;
}

#output-files-panel {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-lg) !important;
    background: var(--surface) !important;
}

.gradio-dropdown, .gradio-textbox, .gradio-file, .gradio-plot {
    border-color: var(--border) !important;
    border-radius: var(--radius-md) !important;
}

.nav-section {
    display: grid;
    gap: 8px;
    margin: 18px 0 0;
}

.section-label-inline {
    padding-inline: 8px;
    color: var(--muted);
    font-family: var(--font-mono);
    font-size: 12px;
    text-transform: uppercase;
}

.nav-list, .metric-list {
    display: grid;
    gap: 4px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.nav-button {
    width: 100%;
    min-height: 40px;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    border: 1px solid transparent;
    border-radius: var(--radius-md);
    background: transparent;
    color: var(--fg-2);
    text-align: left;
    transition: background 140ms cubic-bezier(0.2, 0, 0, 1), border-color 140ms cubic-bezier(0.2, 0, 0, 1);
}

.nav-button svg {
    width: 17px;
    height: 17px;
    flex: 0 0 auto;
}

.nav-button:hover, .nav-button.active {
    border-color: var(--border);
    background: var(--surface);
    color: var(--fg);
}

#chat-scroll {
    min-height: 0 !important;
    height: 100% !important;
    overflow: auto !important;
    padding: 24px !important;
    background: var(--surface) !important;
}

#chat-scroll > .form {
    border: 0 !important;
    background: transparent !important;
}

.reference-progress-card {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 12px;
    width: min(880px, 100%);
    margin: 0 auto 8px;
}

.avatar {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface-2);
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: 12px;
    font-weight: 700;
}

.bubble {
    display: grid;
    gap: 12px;
    min-width: 0;
}

.bubble-card {
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--surface);
}

.bubble-card.assistant {
    background: var(--surface-2);
}

.bubble-card p {
    margin: 0;
    color: var(--fg-2);
}

.tool-card {
    display: grid;
    gap: 12px;
    padding: 16px;
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    background: var(--surface);
}

.tool-head, .chart-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

.tool-head strong, .chart-head h3 {
    color: var(--fg);
    font-family: var(--font-display);
    font-size: 18px;
    line-height: 1.08;
}

.progress-track {
    height: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    background: var(--surface-2);
}

.progress-track span {
    display: block;
    width: var(--progress, 68%);
    height: 100%;
    border-radius: inherit;
    background: var(--fg);
}

.node-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
}

.node {
    display: grid;
    gap: 4px;
    min-height: 74px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
}

.node.current {
    border-color: var(--fg);
    box-shadow: inset 0 0 0 1px var(--fg);
}

.node.done {
    background: var(--surface-2);
    border-color: var(--border-strong);
}

.node span:first-child {
    color: var(--fg);
    font-size: 13px;
    font-weight: 700;
}

.node span:last-child {
    color: var(--muted);
    font-size: 12px;
}

.header-actions {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    flex-wrap: nowrap !important;
    gap: 8px !important;
    min-width: max-content !important;
    justify-self: end !important;
}

.header-actions > button {
    flex: 0 0 auto !important;
    min-width: 136px !important;
    max-width: 160px !important;
}

#top-bar {
    flex-wrap: nowrap !important;
    min-height: 82px !important;
}

#top-bar > div:has(#top-title) {
    min-width: 0 !important;
    max-width: 100% !important;
    overflow: hidden !important;
}

#top-bar > div:has(.header-actions) {
    width: auto !important;
    min-width: max-content !important;
    justify-self: end !important;
}

#top-bar > #top-title {
    flex: 1 1 auto !important;
    width: auto !important;
    max-width: none !important;
}

#top-bar > .header-actions {
    flex: 0 0 auto !important;
    width: auto !important;
}

#top-title h1 {
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    white-space: nowrap !important;
}

#composer-tools {
    display: flex !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    gap: 8px !important;
    flex: 1 0 100% !important;
    width: 100% !important;
    min-width: 100% !important;
}

#composer-tools > button {
    flex: 0 0 auto !important;
}

#composer-footer {
    flex-wrap: wrap !important;
}

.prompt-chip {
    min-height: 28px !important;
    padding: 4px 12px !important;
    border-radius: var(--radius-pill) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
}

.prompt-chip.active::before {
    content: "";
    width: 7px;
    height: 7px;
    border-radius: var(--radius-pill);
    background: var(--fg);
}

.composer-file {
    width: 96px !important;
    min-width: 96px !important;
    max-width: 112px !important;
}

.composer-file button {
    min-height: 38px !important;
    max-height: 38px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    overflow: hidden !important;
    font-size: 0 !important;
}

.composer-file button > * {
    display: none !important;
}

.composer-file button::after {
    content: "Upload";
    color: var(--fg-2);
    font-size: 12px;
}

#composer-footer > .composer-file:nth-of-type(2) button::after {
    content: "EPW";
}

#composer-footer > .composer-file:nth-of-type(3) button::after {
    content: "Images";
}

#chat-scroll,
#chat-scroll > .form,
#chat-scroll .block,
#chat-scroll .wrap,
#chat-scroll .html-container,
#chat-scroll .gradio-html {
    background: var(--surface) !important;
    color: var(--fg) !important;
}

#chat-area,
#chat-area .wrap,
#chat-area .empty,
#chat-area .placeholder,
#chat-area .chatbot,
#chat-area .messages,
#chat-area .bubble-wrap,
#chat-area .message-wrap,
#chat-area .message,
#chat-area [data-testid="bot"],
#chat-area [data-testid="user"] {
    background: var(--surface) !important;
    color: var(--fg) !important;
}

#composer,
#composer .form,
#composer .block,
#composer .wrap,
#msg-input,
#msg-input .wrap,
#msg-input textarea {
    background: var(--surface) !important;
    color: var(--fg) !important;
}

#composer-footer,
#composer-footer .form,
#composer-footer .row,
#composer-footer .block {
    background: var(--surface-2) !important;
}

.composer-file,
.composer-file .wrap,
.composer-file .container,
.composer-file .upload-container,
.composer-file [data-testid="file"],
.composer-file button,
.composer-file label {
    border-color: var(--border) !important;
    background: var(--surface) !important;
    color: var(--fg) !important;
    box-shadow: none !important;
}

.composer-file label,
.composer-file label span,
.composer-file .or {
    color: var(--fg-2) !important;
}

.composer-file label {
    display: none !important;
}

#inspector-tabs,
#inspector-tabs > div,
#inspector-tabs .tabitem,
#inspector-tabs [role="tabpanel"] {
    background: var(--surface-2) !important;
    color: var(--fg) !important;
}

.inspector-panel,
.inspector-panel > .form,
.inspector-panel > .block,
.inspector-panel .block,
.inspector-panel .form,
#model-preview-panel,
#model-dossier-panel,
#results-summary-panel,
#output-files-panel {
    background: var(--surface) !important;
    color: var(--fg) !important;
}

.reference-model-stage,
.model-stage,
.metric-card,
.dossier-card,
.kpi-card,
.chart-card,
.file-row {
    color: var(--fg) !important;
}

.reference-model-stage,
.model-stage {
    background:
        linear-gradient(0deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        linear-gradient(90deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        var(--surface-2) !important;
}

#inspector-scroll {
    min-height: 0 !important;
    overflow: auto !important;
    padding: 0 !important;
}

#model-preview-panel, #model-dossier-panel, #results-summary-panel {
    padding: 0 !important;
}

.reference-model-stage {
    position: relative;
    min-height: 288px;
    display: grid;
    place-items: center;
    overflow: hidden;
    margin-bottom: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background:
        linear-gradient(0deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        linear-gradient(90deg, color-mix(in oklab, var(--fg) 4%, transparent) 1px, transparent 1px),
        var(--surface-2);
    background-size: 26px 26px;
}

.reference-model-stage svg {
    width: 96%;
    height: auto;
}

.model-layer {
    position: absolute;
    left: 12px;
    bottom: 12px;
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
}

.metric-list {
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin-top: 12px;
}

.dossier-grid, .result-kpis {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 8px;
}

.dossier-card, .kpi-card {
    min-width: 0;
    display: grid;
    gap: 4px;
    padding: 12px;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface-2);
}

.dossier-card strong, .kpi-card strong {
    overflow: hidden;
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: 16px;
    text-overflow: ellipsis;
    white-space: nowrap;
}

.result-card {
    position: relative;
    overflow: hidden;
}

.result-card::before {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
        linear-gradient(90deg, color-mix(in oklab, var(--viz-cooling) 12%, transparent), transparent 45%),
        linear-gradient(180deg, color-mix(in oklab, var(--viz-comfort) 8%, transparent), transparent 56%);
    opacity: 0.72;
}

.result-card > * {
    position: relative;
    z-index: 1;
}

.bar-list {
    display: grid;
    gap: 8px;
}

.bar-row {
    display: grid;
    grid-template-columns: 78px minmax(0, 1fr) 40px;
    gap: 12px;
    align-items: center;
    color: var(--fg-2);
    font-size: 13px;
}

.bar-track {
    height: 10px;
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: var(--radius-pill);
    background: var(--surface-2);
}

.bar-track span {
    display: block;
    width: var(--bar);
    height: 100%;
    border-radius: inherit;
}

.bar-track span.cooling { background: var(--viz-cooling); }
.bar-track span.lighting { background: var(--viz-lighting); }
.bar-track span.equipment { background: var(--viz-equipment); }
.bar-track span.fans { background: var(--viz-comfort); }

.bar-row strong {
    color: var(--fg);
    font-family: var(--font-mono);
    font-size: 12px;
    text-align: right;
}

.annotation-row {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    align-items: center;
}

.annotation-tag {
    display: inline-flex;
    min-height: 26px;
    align-items: center;
    gap: 8px;
    padding: 0 8px;
    border: 1px solid color-mix(in oklab, var(--mark) 44%, var(--border));
    border-radius: var(--radius-pill);
    background: color-mix(in oklab, var(--mark) 14%, var(--surface));
    color: var(--fg);
    font-size: 12px;
    white-space: nowrap;
}

.annotation-tag::before {
    content: "";
    width: 8px;
    height: 8px;
    border-radius: var(--radius-pill);
    background: var(--mark);
}

.annotation-tag.cooling { --mark: var(--viz-cooling); }
.annotation-tag.solar { --mark: var(--viz-solar); }
.annotation-tag.comfort { --mark: var(--viz-comfort); }
.annotation-tag.alert { --mark: var(--viz-alert); }
.annotation-tag.equipment { --mark: var(--viz-equipment); }

#workbench-toast {
    position: fixed;
    right: 20px;
    bottom: 20px;
    z-index: 200;
    max-width: 320px;
    padding: 12px 16px;
    border: 1px solid var(--border-strong);
    border-radius: var(--radius-md);
    background: var(--fg);
    color: var(--accent-on);
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 220ms cubic-bezier(0.2, 0, 0, 1), transform 220ms cubic-bezier(0.2, 0, 0, 1);
    pointer-events: none;
}

#workbench-toast.show {
    opacity: 1;
    transform: translateY(0);
}

button:focus-visible, textarea:focus-visible, select:focus-visible {
    outline: none !important;
    box-shadow: var(--focus-ring) !important;
}

@media (max-width: 1180px) {
    #chat-sidebar {
        width: 84px !important;
        min-width: 84px !important;
        padding: 12px !important;
    }
    #brand-lockup {
        grid-template-columns: 1fr;
        justify-items: center;
    }
    #brand-lockup .brand-name,
    #brand-lockup .brand-sub,
    #chat-sidebar .section-label,
    #chat-sidebar .sidebar-footer,
    #session-list,
    #chat-sidebar .new-run span {
        display: none !important;
    }
    #chat-pane {
        height: auto !important;
        min-height: 100vh !important;
    }
    #viz-panel {
        width: 100% !important;
        max-height: none !important;
        height: auto !important;
        border-left: 0 !important;
        border-top: 1px solid var(--border) !important;
    }
}

@media (max-width: 760px) {
    #chat-sidebar {
        position: sticky !important;
        top: 0 !important;
        z-index: 10 !important;
        width: 100% !important;
        min-width: 0 !important;
        height: auto !important;
        max-height: none !important;
        border-right: 0 !important;
        border-bottom: 1px solid var(--border) !important;
    }
    #top-bar {
        align-items: flex-start !important;
        padding: 16px 12px !important;
    }
    #top-title h1, #top-title h2, #top-title h3 {
        font-size: 22px !important;
    }
    #composer-wrap {
        padding: 12px !important;
    }
    #composer-footer {
        align-items: stretch !important;
        flex-direction: column !important;
    }
    .composer-file {
        max-width: none !important;
        width: 100% !important;
    }
    .send-button {
        width: 100% !important;
    }
    .metric-row {
        grid-template-columns: 1fr !important;
    }
}
"""


def _session_title(session_id: str) -> str:
    """One-line title for the top bar, derived from session meta."""
    meta = _load_session_meta(session_id) or {}
    preview = meta.get("user_input_preview", "").strip()
    if preview and not _has_cjk(preview):
        return f"{preview[:50]}"
    return f"Session {session_id}"


def _top_title_markdown(session_id: str) -> str:
    title = _session_title(session_id)
    return (
        f"# {title}\n"
        "Describe the building, review the generated IDF, and run EnergyPlus "
        "from one conversation."
    )


def _workspace_nav_html() -> str:
    """Static navigation rail copied from the reference workbench."""
    return """
    <nav class="nav-section" aria-label="Workspace">
      <div class="section-label-inline">Workspace</div>
      <ul class="nav-list">
        <li><button class="nav-button active" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 5h16v14H4zM8 9h8M8 13h5"/></svg><span>Chat workbench</span></button></li>
        <li><button class="nav-button" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M4 17V7l8-4 8 4v10l-8 4zM12 3v18M4 7l8 4 8-4"/></svg><span>Model library</span></button></li>
        <li><button class="nav-button" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M5 19V5M5 19h14M9 16v-5M13 16V8M17 16v-7"/></svg><span>Result reports</span></button></li>
        <li><button class="nav-button" type="button"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><path d="M7 4h10M7 20h10M8 4v4l4 4 4-4V4M8 20v-4l4-4 4 4v4"/></svg><span>Agent traces</span></button></li>
      </ul>
    </nav>
    """


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content if isinstance(part, dict)
        )
    return str(content or "")


def _message_body_html(content: Any) -> str:
    text = _message_text(content).strip()
    if not text:
        return ""
    paragraphs = [
        escape(part).replace("\n", "<br>")
        for part in text.split("\n\n")
        if part.strip()
    ]
    return "".join(f"<p>{part}</p>" for part in paragraphs)


def _completed_agent_labels(history: History | None) -> set[str]:
    labels: set[str] = set()
    for msg in history or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = _message_text(msg.get("content")).strip()
        if content.startswith("Completed:"):
            labels.add(content.removeprefix("Completed:").strip())
    return labels


def _history_has_terminal_message(history: History | None) -> bool:
    terminal_markers = (
        "Run failed:",
        "Wait timed out",
        "Please enter a building description",
        "The **3D model** and **energy analysis charts**",
        "Simulation completed",
    )
    for msg in history or []:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = _message_text(msg.get("content")).strip()
        if any(marker in content for marker in terminal_markers):
            return True
    return False


def _agent_progress_html(history: History | None = None) -> str:
    history = history or []
    completed = _completed_agent_labels(history)
    started = any(
        isinstance(msg, dict) and msg.get("role") == "user" for msg in history
    )
    terminal = _history_has_terminal_message(history)
    failed = any(
        isinstance(msg, dict)
        and msg.get("role") == "assistant"
        and "Run failed:" in _message_text(msg.get("content"))
        for msg in history
    )

    completed_steps = 0
    current_assigned = False
    node_html: list[str] = []
    for title, labels in _AGENT_PROGRESS_STEPS:
        is_done = any(label in completed for label in labels)
        if terminal and started and not failed:
            is_done = True
        if is_done:
            completed_steps += 1
            node_class = "node done"
            status = "Done"
        elif started and not terminal and not current_assigned:
            current_assigned = True
            node_class = "node current"
            status = "Running"
        else:
            node_class = "node"
            status = (
                "Queued"
                if started
                else ("Ready" if title == "Parse brief" else "Queued")
            )
        node_html.append(
            f'<div class="{node_class}"><span>{escape(title)}</span>'
            f"<span>{escape(status)}</span></div>"
        )

    if terminal and started and not failed:
        progress = 100
        summary = "Agent run complete. Model assets and result views are synced to the right workspace."
    elif failed:
        progress = max(12, int((completed_steps / len(_AGENT_PROGRESS_STEPS)) * 100))
        summary = "Agent run stopped with an error. The last successful step remains visible below."
    elif started:
        progress = min(
            95,
            int(
                (
                    (completed_steps + (0.35 if current_assigned else 0))
                    / len(_AGENT_PROGRESS_STEPS)
                )
                * 100
            ),
        )
        summary = "Agents are working through the building brief. Completed steps update here as the graph runs."
    else:
        progress = 8
        summary = "The live model window is open on the right. Geometry, validation status, generated files, and simulation charts stay in sync with this session."

    return f"""
    <article class="workbench-message assistant reference-progress-card" aria-label="Agent graph progress">
      <div class="avatar">EA</div>
      <div class="bubble">
        <div class="bubble-card assistant">
          <p>{escape(summary)}</p>
        </div>
        <div class="tool-card">
          <div class="tool-head">
            <strong>Agent graph progress</strong>
            <span class="mono small">Live pipeline</span>
          </div>
          <div class="progress-track" aria-label="Agent graph progress"><span style="--progress: {progress}%;"></span></div>
          <div class="node-grid">
            {"".join(node_html)}
          </div>
        </div>
      </div>
    </article>
    """


def _conversation_message_html(role: str, content: Any) -> str:
    body = _message_body_html(content)
    if not body:
        return ""
    safe_role = "user" if role == "user" else "assistant"
    avatar = "You" if safe_role == "user" else "EA"
    return f"""
    <article class="workbench-message {safe_role}">
      <div class="avatar">{avatar}</div>
      <div class="bubble">
        <div class="bubble-card {safe_role}">
          {body}
        </div>
      </div>
    </article>
    """


def _conversation_html(history: History | None) -> str:
    history = history or []
    parts = ['<div class="conversation-stream" aria-live="polite">']

    if not history:
        parts.append(_agent_progress_html([]))
        parts.append("</div>")
        return "".join(parts)

    inserted_progress = False
    has_user_message = any(
        isinstance(msg, dict) and msg.get("role") == "user" for msg in history
    )
    for msg in history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "assistant")
        content = _message_text(msg.get("content")).strip()
        if not content:
            continue
        if role == "assistant" and content.startswith("Completed:"):
            continue
        parts.append(_conversation_message_html(role, content))
        if (
            has_user_message
            and role == "assistant"
            and (
                content.startswith("Starting Agent")
                or content.startswith("Modifying the existing model")
            )
        ):
            parts.append(_agent_progress_html(history))
            inserted_progress = True

    if has_user_message and _completed_agent_labels(history) and not inserted_progress:
        parts.append(_agent_progress_html(history))

    parts.append("</div>")
    return "".join(parts)


def _model_preview_html() -> str:
    return """
    <div class="reference-model-stage" aria-label="Five-floor office model preview">
      <svg viewBox="0 0 640 360" role="img" aria-label="Five-floor office building with central atrium and circulation core">
        <g fill="none" stroke="currentColor" stroke-width="1.4" opacity="0.42">
          <path d="M60 270h440l78-58H140z"/>
          <path d="M140 212V92L60 150v120"/>
          <path d="M500 270V150L578 92v120"/>
          <path d="M140 92h438l-78 58H60z"/>
        </g>
        <g fill="var(--surface)" stroke="var(--fg)" stroke-width="1.8">
          <path d="M118 254h362l52-38H170z"/>
          <path d="M118 220h362l52-38H170z" opacity="0.92"/>
          <path d="M118 186h362l52-38H170z" opacity="0.84"/>
          <path d="M118 152h362l52-38H170z" opacity="0.76"/>
          <path d="M118 118h362l52-38H170z" opacity="0.68"/>
        </g>
        <g fill="var(--surface-3)" stroke="var(--fg)" stroke-width="1.6">
          <path d="M284 253h84l24-18h-84z"/>
          <path d="M284 219h84l24-18h-84z"/>
          <path d="M284 185h84l24-18h-84z"/>
          <path d="M284 151h84l24-18h-84z"/>
          <path d="M284 117h84l24-18h-84z"/>
        </g>
        <g stroke="var(--fg)" stroke-width="1.1" opacity="0.38">
          <path d="M202 112v136M242 107v136M414 98v136M454 94v136"/>
          <path d="M132 243h362M132 209h362M132 175h362M132 141h362"/>
        </g>
        <g fill="var(--fg)">
          <circle cx="326" cy="175" r="4"/>
          <circle cx="436" cy="124" r="4"/>
          <circle cx="208" cy="218" r="4"/>
        </g>
      </svg>
      <div class="model-layer">
        <span class="chip">Atrium zones</span>
        <span class="chip">Core stack</span>
        <span class="chip">Envelope surfaces</span>
      </div>
    </div>
    """


def _model_counts(session_id: str) -> dict[str, str]:
    meta = _load_session_meta(session_id) or {}
    zones = meta.get("zones_count")
    surfaces = meta.get("surfaces_count")
    windows = meta.get("fenestrations_count")
    if not all(isinstance(v, int) and v > 0 for v in (zones, surfaces, windows)):
        idf_path = _latest_idf(_session_dir(session_id))
        if idf_path is not None:
            try:
                from src.results import parse_fenestrations

                parsed_zones = parse_idf_geometry(idf_path)
                parsed_fens = parse_fenestrations(idf_path)
                zones = len(parsed_zones)
                surfaces = sum(len(z.surfaces) for z in parsed_zones.values())
                windows = len(parsed_fens)
            except Exception:
                pass
    return {
        "zones": str(zones) if isinstance(zones, int) and zones > 0 else "--",
        "surfaces": str(surfaces)
        if isinstance(surfaces, int) and surfaces > 0
        else "--",
        "windows": str(windows) if isinstance(windows, int) and windows > 0 else "--",
    }


def _model_metric_cards(session_id: str) -> str:
    counts = _model_counts(session_id)
    return f"""
    <ul class="metric-list" aria-label="Model metrics">
      <li class="metric-card"><span class="small">Thermal zones</span><strong>{counts["zones"]}</strong></li>
      <li class="metric-card"><span class="small">Surfaces</span><strong>{counts["surfaces"]}</strong></li>
      <li class="metric-card"><span class="small">Windows</span><strong>{counts["windows"]}</strong></li>
    </ul>
    """


def _model_dossier_html(session_id: str) -> str:
    safe_sid = escape(session_id or "new-session")
    meta = _load_session_meta(session_id) or {}
    preview_text = str(meta.get("user_input_preview") or "Waiting for building brief")
    if _has_cjk(preview_text):
        preview_text = "Conversation brief saved for this session."
    preview = escape(preview_text)
    return f"""
    <div class="dossier-grid">
      <article class="dossier-card"><span class="small">Site</span><strong>Shenzhen</strong></article>
      <article class="dossier-card"><span class="small">Session</span><strong>{safe_sid[-8:]}</strong></article>
      <article class="dossier-card"><span class="small">Weather</span><strong>{escape(DEFAULT_EPW.name)}</strong></article>
      <article class="dossier-card"><span class="small">Output</span><strong>output/ui</strong></article>
    </div>
    <p class="model-note">Parsed brief: {preview}</p>
    """


def _result_summary_html(session_id: str) -> str:
    has_results = (_session_dir(session_id) / "eplusout.csv").exists()
    state = "Loaded from output/ui" if has_results else "Waiting for run"
    return f"""
    <div class="result-kpis" aria-label="Simulation result highlights">
      <article class="kpi-card"><span class="annotation-tag cooling">Cooling driver</span><strong>{"Ready" if has_results else "--"}</strong><span class="small">Highest end-use share</span></article>
      <article class="kpi-card"><span class="annotation-tag solar">Solar exposure</span><strong>{"Mapped" if has_results else "--"}</strong><span class="small">Exterior surface view</span></article>
      <article class="kpi-card"><span class="annotation-tag comfort">Comfort band</span><strong>{"Charted" if has_results else "--"}</strong><span class="small">Occupied hours review</span></article>
      <article class="kpi-card"><span class="annotation-tag alert">Run state</span><strong>{state}</strong><span class="small">EnergyPlus status</span></article>
    </div>
    """


def _result_preview_html() -> str:
    return """
    <article class="chart-card result-card">
      <div class="chart-head">
        <h3>Annotated end-use preview</h3>
        <span class="small">Color marks show result categories</span>
      </div>
      <div class="bar-list">
        <div class="bar-row"><span>Cooling</span><div class="bar-track"><span class="cooling" style="--bar: 72%;"></span></div><strong>72</strong></div>
        <div class="bar-row"><span>Lighting</span><div class="bar-track"><span class="lighting" style="--bar: 48%;"></span></div><strong>48</strong></div>
        <div class="bar-row"><span>Equipment</span><div class="bar-track"><span class="equipment" style="--bar: 34%;"></span></div><strong>34</strong></div>
        <div class="bar-row"><span>Fans</span><div class="bar-track"><span class="fans" style="--bar: 26%;"></span></div><strong>26</strong></div>
      </div>
      <div class="annotation-row">
        <span class="annotation-tag cooling">Cooling load dominates</span>
        <span class="annotation-tag equipment">Plug load secondary</span>
      </div>
    </article>
    """


def _radio_label_for(sid: str) -> str:
    """Return the Radio label that represents *sid* (or a fallback title)."""
    for lbl, s in _list_sessions():
        if s == sid:
            return lbl
    return _session_title(sid)


def _radio_choices() -> list[str]:
    return [lbl for lbl, _s in _list_sessions()]


def build_ui() -> gr.Blocks:
    """Build the Claude-style three-column UI.

    Layout:
      - Left  ``gr.Sidebar``  : dark session list + New session button
      - Center main column    : Chatbot (bubble layout) + capsule input box
      - Right ``gr.Sidebar``  : collapsible visualization panel (3D + charts)
    """
    with gr.Blocks(title="EnergyPlus Agent", fill_height=True, fill_width=True) as demo:
        _initial_session = _current_session_or_create(None)
        _initial_idf = _latest_idf(_session_dir(_initial_session))
        _initial_history = _load_chat_history(_initial_session)

        # ------------------------------------------------------------------ #
        # LEFT SIDEBAR - dark session list                                     #
        # ------------------------------------------------------------------ #
        with gr.Sidebar(
            position="left",
            width=248,
            open=True,
            elem_id="chat-sidebar",
        ):
            gr.HTML(
                """
                <div id="brand-lockup">
                  <div class="brand-mark" aria-hidden="true">EA</div>
                  <div>
                    <div class="brand-name">EnergyPlus Agent</div>
                    <div class="brand-sub">MCP / LangGraph / IDF</div>
                  </div>
                </div>
                """
            )
            new_btn = gr.Button("New simulation", elem_classes="new-run")
            gr.HTML(_workspace_nav_html())
            gr.Markdown("Recent", elem_classes="section-label")
            session_radio = gr.Radio(
                choices=_radio_choices(),
                value=_radio_label_for(_initial_session),
                label="",
                container=False,
                elem_id="session-list",
            )
            gr.HTML(
                """
                <div class="sidebar-footer">
                  <div class="status-row">
                    <strong>Local session</strong>
                    <span class="status-pill">Ready</span>
                  </div>
                  <p>Weather defaults to Shenzhen EPW. Output files are collected from <span class="mono">output/ui</span>.</p>
                </div>
                """
            )

        # ------------------------------------------------------------------ #
        # CENTER - conversation area                                           #
        # ------------------------------------------------------------------ #
        with gr.Column(scale=1, elem_id="chat-pane"):
            with gr.Row(elem_id="top-bar"):
                top_title = gr.Markdown(
                    _top_title_markdown(_initial_session), elem_id="top-title"
                )
                with gr.Row(elem_classes="header-actions"):
                    copy_summary_btn = gr.Button(
                        "Copy summary",
                        elem_id="copy-summary",
                        elem_classes="secondary-button",
                    )
                    export_btn = gr.Button("Export", elem_classes="secondary-button")

            with gr.Group(elem_id="chat-scroll"):
                chatbot = gr.HTML(
                    _conversation_html(_initial_history),
                    elem_id="chat-area",
                )

            output_files = gr.File(
                label="Output files",
                file_count="multiple",
                visible=False,
                interactive=False,
            )

            with gr.Group(elem_id="composer-wrap"), gr.Group(elem_id="composer"):
                input_text = gr.Textbox(
                    placeholder="Ask for a model revision, simulation run, validation pass, or result explanation.",
                    lines=4,
                    max_lines=8,
                    container=False,
                    show_label=False,
                    elem_id="msg-input",
                )
                with gr.Row(elem_id="composer-footer"):
                    with gr.Row(elem_id="composer-tools"):
                        quick_validate = gr.Button(
                            "Validate references",
                            elem_classes=["prompt-chip"],
                            min_width=120,
                        )
                        quick_run = gr.Button(
                            "Run pipeline",
                            elem_classes=["prompt-chip", "active"],
                            min_width=104,
                        )
                        quick_explain = gr.Button(
                            "Explain load drivers",
                            elem_classes=["prompt-chip"],
                            min_width=130,
                        )
                    epw_file = gr.File(
                        label="EPW",
                        file_types=[".epw"],
                        scale=0,
                        min_width=120,
                        elem_classes="composer-file",
                    )
                    image_files = gr.File(
                        label="Images",
                        file_types=["image"],
                        file_count="multiple",
                        scale=0,
                        min_width=140,
                        elem_classes="composer-file",
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=0,
                        min_width=86,
                        elem_classes="send-button",
                    )

        # ------------------------------------------------------------------ #
        # RIGHT SIDEBAR - model and results inspector                          #
        # ------------------------------------------------------------------ #
        with gr.Sidebar(
            position="right",
            width=500,
            open=True,
            elem_id="viz-panel",
        ) as viz_panel:
            gr.HTML(
                """
                <div id="inspector-header">
                  <div>
                    <h2>Right workspace</h2>
                    <p>Switch between live geometry assembly and annotated simulation outputs.</p>
                  </div>
                  <span class="status-pill">Ready</span>
                </div>
                """
            )
            with (
                gr.Group(elem_id="inspector-scroll"),
                gr.Tabs(elem_id="inspector-tabs"),
            ):
                with gr.Tab("Model"):
                    with gr.Group(
                        elem_classes="inspector-panel",
                        elem_id="model-preview-panel",
                    ):
                        gr.Markdown(
                            "## 3D model workspace\nLive IDF assembly",
                            elem_classes="panel-title",
                        )
                        gr.HTML(_model_preview_html(), elem_id="model-preview")
                        load_3d_btn = gr.Button(
                            "Reload model", elem_classes="secondary-button"
                        )
                        model_3d = gr.Plot(
                            label="",
                            elem_classes="model-stage",
                            visible=_initial_idf is not None,
                        )
                        model_metrics = gr.HTML(
                            _model_metric_cards(_initial_session),
                            elem_id="model-metrics",
                        )
                        model_status = gr.Markdown(
                            _model_status_markdown(_initial_idf),
                            elem_classes="model-note",
                        )
                    with gr.Group(
                        elem_classes="inspector-panel",
                        elem_id="model-dossier-panel",
                    ):
                        gr.Markdown(
                            "## Building dossier\nParsed from conversation",
                            elem_classes="panel-title",
                        )
                        model_dossier = gr.HTML(
                            _model_dossier_html(_initial_session),
                            elem_id="model-dossier",
                        )
                    with gr.Group(
                        elem_classes="inspector-panel", elem_id="output-files-panel"
                    ):
                        gr.Markdown(
                            "## IDF assets\nGenerated files",
                            elem_classes="panel-title",
                        )
                        output_files_panel = gr.File(
                            label="Download",
                            file_count="multiple",
                            interactive=False,
                        )

                with gr.Tab("Results"):
                    with gr.Group(
                        elem_classes="inspector-panel",
                        elem_id="results-summary-panel",
                    ):
                        gr.Markdown(
                            "## Annotated results\nLoaded after EnergyPlus",
                            elem_classes="panel-title",
                        )
                        result_summary = gr.HTML(
                            _result_summary_html(_initial_session),
                            elem_id="result-summary",
                        )
                        with gr.Row():
                            metric_dd = gr.Dropdown(
                                choices=_METRIC_OPTIONS,
                                value=_METRIC_OPTIONS[0],
                                label="Zone coloring metric",
                                scale=2,
                            )
                            zone_dd = gr.Dropdown(
                                choices=_schedule_zone_choices(
                                    _session_dir(_initial_session)
                                ),
                                value=ZONE_ALL,
                                label="Schedule zone",
                                scale=2,
                            )
                        load_btn = gr.Button(
                            "Load charts", elem_classes="secondary-button"
                        )
                    gr.HTML(_result_preview_html(), elem_id="result-preview")
                    plot_3d_energy = gr.Plot(
                        label="3D Zone Energy Use", elem_classes="chart-card"
                    )
                    plot_solar_3d = gr.Plot(
                        label="3D Exterior Solar Irradiation",
                        elem_classes="chart-card",
                    )
                    plot_sched_people = gr.Plot(
                        label="Occupancy Schedule", elem_classes="chart-card"
                    )
                    plot_sched_equip = gr.Plot(
                        label="Equipment Schedule", elem_classes="chart-card"
                    )
                    plot_enduse = gr.Plot(
                        label="Annual End-Use Energy", elem_classes="chart-card"
                    )
                    plot_comfort = gr.Plot(
                        label="Thermal Comfort", elem_classes="chart-card"
                    )
                    plot_monthly = gr.Plot(
                        label="Monthly HVAC Energy", elem_classes="chart-card"
                    )
                    plot_heatmap = gr.Plot(
                        label="Zone Temperature Heatmap", elem_classes="chart-card"
                    )
                    plot_demand = gr.Plot(
                        label="HVAC Electric Demand", elem_classes="chart-card"
                    )
                    plot_scatter = gr.Plot(
                        label="Temperature-Humidity Scatter",
                        elem_classes="chart-card",
                    )

        # ------------------------------------------------------------------ #
        # Hidden state carrying the active session_id across events           #
        # ------------------------------------------------------------------ #
        session_state = gr.Textbox(
            value=_initial_session,
            visible=False,
            elem_id="session-state",
        )

        _all_viz_plots = [
            plot_3d_energy,
            plot_solar_3d,
            plot_sched_people,
            plot_sched_equip,
            plot_enduse,
            plot_comfort,
            plot_monthly,
            plot_heatmap,
            plot_demand,
            plot_scatter,
        ]

        # ------------------------------------------------------------------ #
        # Helpers                                                             #
        # ------------------------------------------------------------------ #
        def _refresh_viz_panel(sid: str, metric_label: str, zone_key: str):
            """Return all right-panel dynamic values for a session."""
            sdir = _session_dir(sid) if sid else None
            if not sdir or not sdir.exists():
                empty10 = (None,) * 10
                return (
                    *empty10,
                    gr.update(value=None, visible=False),
                    _model_status_markdown(None),
                    _model_metric_cards(sid or ""),
                    _model_dossier_html(sid or ""),
                    _result_summary_html(sid or ""),
                    gr.update(),
                )
            figs = _load_visualizations(sdir, metric_label, zone_key)
            has_idf = _latest_idf(sdir) is not None
            model_fig, model_status_text = on_load_3d_model(str(sdir), None)
            model_update = (
                gr.update(value=model_fig, visible=True)
                if has_idf
                else gr.update(value=None, visible=False)
            )
            zone_choices = _schedule_zone_choices(sdir)
            valid_keys = {c[1] for c in zone_choices}
            new_zone = zone_key if zone_key in valid_keys else ZONE_ALL
            return (
                *figs,
                model_update,
                model_status_text,
                _model_metric_cards(sid),
                _model_dossier_html(sid),
                _result_summary_html(sid),
                gr.update(choices=zone_choices, value=new_zone),
            )

        # ------------------------------------------------------------------ #
        # Event: New session                                                  #
        # ------------------------------------------------------------------ #
        def on_new_session():
            sid = _create_session()
            return (
                gr.update(
                    choices=_radio_choices(), value=_radio_label_for(sid)
                ),  # radio
                sid,  # session_state
                _top_title_markdown(sid),  # top_title
                gr.update(value=_conversation_html([])),  # chatbot
                gr.update(value=None, visible=False),  # output_files
                gr.update(open=True),  # viz_panel
                *([None] * 10),  # _all_viz_plots
                gr.update(value=None, visible=False),  # model_3d
                _model_status_markdown(None),  # model_status
                _model_metric_cards(sid),  # model_metrics
                _model_dossier_html(sid),  # model_dossier
                _result_summary_html(sid),  # result_summary
                gr.update(choices=[("All zones (max)", ZONE_ALL)], value=ZONE_ALL),
                gr.update(value=None),  # output_files_panel
                gr.update(value=""),  # input_text
            )

        new_btn.click(
            fn=on_new_session,
            inputs=None,
            outputs=[
                session_radio,
                session_state,
                top_title,
                chatbot,
                output_files,
                viz_panel,
                *_all_viz_plots,
                model_3d,
                model_status,
                model_metrics,
                model_dossier,
                result_summary,
                zone_dd,
                output_files_panel,
                input_text,
            ],
        )

        # ------------------------------------------------------------------ #
        # Event: Switch session (Radio .select)                               #
        # ------------------------------------------------------------------ #
        def on_session_select(radio_label, metric_label, zone_key):
            sid = None
            for lbl, s in _list_sessions():
                if lbl == radio_label:
                    sid = s
                    break
            if not sid:
                sid = _current_session_or_create(None)
            history = _load_chat_history(sid)
            refreshed = _refresh_viz_panel(sid, metric_label, zone_key)
            (_session_dir(sid) / "eplusout.csv").exists()
            files = _collect_output_files(_session_dir(sid))
            return (
                sid,  # session_state
                _top_title_markdown(sid),  # top_title
                gr.update(value=_conversation_html(history)),  # chatbot
                gr.update(open=True),  # viz_panel
                *refreshed,  # right-panel values
                gr.update(value=files if files else None),  # output_files_panel
            )

        session_radio.select(
            fn=on_session_select,
            inputs=[session_radio, metric_dd, zone_dd],
            outputs=[
                session_state,
                top_title,
                chatbot,
                viz_panel,
                *_all_viz_plots,
                model_3d,
                model_status,
                model_metrics,
                model_dossier,
                result_summary,
                zone_dd,
                output_files_panel,
            ],
        )

        # ------------------------------------------------------------------ #
        # Event: Run simulation (send button / textbox submit)                #
        # ------------------------------------------------------------------ #
        def on_run(text, epw, images, sid, metric_label, zone_key):
            if not sid:
                sid = _current_session_or_create(None)
            last_files: list[str] = []
            last_history = []
            for history, files in run_agent(text, epw, images, sid):
                last_files = files
                last_history = history
                yield (
                    gr.update(value=_conversation_html(last_history)),
                    gr.update(value=files if files else None, visible=bool(files)),
                    gr.update(open=True),
                    *([None] * 10),  # _all_viz_plots
                    gr.update(value=None, visible=False),  # model_3d
                    gr.update(),  # model_status
                    gr.update(),  # model_metrics
                    gr.update(),  # model_dossier
                    gr.update(),  # result_summary
                    gr.update(),  # zone_dd
                    gr.update(),  # output_files_panel
                    gr.update(value=""),  # input_text
                )
            refreshed = _refresh_viz_panel(sid, metric_label, zone_key)
            yield (
                gr.update(value=_conversation_html(last_history)),
                gr.update(
                    value=last_files if last_files else None, visible=bool(last_files)
                ),
                gr.update(open=True),
                *refreshed,
                gr.update(value=last_files if last_files else None),
                gr.update(value=""),
            )

        _run_outputs = [
            chatbot,
            output_files,
            viz_panel,
            *_all_viz_plots,
            model_3d,
            model_status,
            model_metrics,
            model_dossier,
            result_summary,
            zone_dd,
            output_files_panel,
            input_text,
        ]
        send_btn.click(
            fn=on_run,
            inputs=[
                input_text,
                epw_file,
                image_files,
                session_state,
                metric_dd,
                zone_dd,
            ],
            outputs=_run_outputs,
        )
        input_text.submit(
            fn=on_run,
            inputs=[
                input_text,
                epw_file,
                image_files,
                session_state,
                metric_dd,
                zone_dd,
            ],
            outputs=_run_outputs,
        )

        # ------------------------------------------------------------------ #
        # Event: Load charts button                                           #
        # ------------------------------------------------------------------ #
        def on_load_charts(metric_label, zone_key, sid):
            if not sid:
                return (None,) * 10
            return _load_visualizations(_session_dir(sid), metric_label, zone_key)

        load_btn.click(
            fn=on_load_charts,
            inputs=[metric_dd, zone_dd, session_state],
            outputs=_all_viz_plots,
        )

        # ------------------------------------------------------------------ #
        # Event: Metric dropdown -> 3D zone-energy chart                       #
        # ------------------------------------------------------------------ #
        def on_metric_change(metric_label, sid):
            if not sid:
                return None
            return _update_3d_only(_session_dir(sid), metric_label)

        metric_dd.change(
            fn=on_metric_change,
            inputs=[metric_dd, session_state],
            outputs=[plot_3d_energy],
        )

        # ------------------------------------------------------------------ #
        # Event: Zone dropdown -> schedule charts                              #
        # ------------------------------------------------------------------ #
        def on_zone_change(zone_key, sid):
            if not sid:
                return None, None
            return _update_schedules_only(_session_dir(sid), zone_key)

        zone_dd.change(
            fn=on_zone_change,
            inputs=[zone_dd, session_state],
            outputs=[plot_sched_people, plot_sched_equip],
        )

        # ------------------------------------------------------------------ #
        # Event: 3D reload                                                    #
        # ------------------------------------------------------------------ #
        def on_load_3d(sid):
            sdir = _session_dir(sid) if sid else None
            has_idf = _latest_idf(sdir) is not None if sdir else False
            fig, status = on_load_3d_model(str(sdir) if sdir else None, None)
            fig_update = (
                gr.update(value=fig, visible=True)
                if has_idf
                else gr.update(value=None, visible=False)
            )
            return fig_update, status, _model_metric_cards(sid or "")

        load_3d_btn.click(
            fn=on_load_3d,
            inputs=[session_state],
            outputs=[model_3d, model_status, model_metrics],
        )

        quick_validate.click(
            fn=lambda: (
                "Validate all construction, schedule, zone, surface, people, lights, and HVAC references before exporting IDF."
            ),
            inputs=None,
            outputs=[input_text],
            queue=False,
        )
        quick_run.click(
            fn=lambda: (
                "Run the full simulation pipeline, then load the 3D model and annotated result charts in the right workspace."
            ),
            inputs=None,
            outputs=[input_text],
            queue=False,
        )
        quick_explain.click(
            fn=lambda: (
                "Compare cooling load distribution by floor and identify zones that need shading or schedule changes."
            ),
            inputs=None,
            outputs=[input_text],
            queue=False,
        )

        copy_summary_btn.click(
            fn=None,
            inputs=[session_state],
            outputs=None,
            queue=False,
            js="""
            (sid) => {
              const title = document.querySelector("#top-title")?.innerText?.trim() || "EnergyPlus Agent";
              const metrics = document.querySelector("#model-metrics")?.innerText?.trim() || "No model metrics loaded";
              const summary = `${title}\\n\\n${metrics}\\n\\nSession: ${sid || "unknown"}`;
              navigator.clipboard?.writeText(summary);
              if (typeof flashWorkbenchToast === "function") flashWorkbenchToast("Summary copied");
              return [];
            }
            """,
        )

        # ------------------------------------------------------------------ #
        # Event: Export button                                                #
        # ------------------------------------------------------------------ #
        def on_export(sid):
            if not sid:
                return gr.update(value=None)
            files = _collect_output_files(_session_dir(sid))
            return gr.update(value=files if files else None, visible=True)

        export_btn.click(
            fn=on_export,
            inputs=[session_state],
            outputs=[output_files_panel],
        )

        gr.HTML(
            '<div id="workbench-toast" role="status" aria-live="polite">Summary copied</div>'
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=False,
        theme=gr.themes.Soft(),
        css=_CHATGPT_CSS,
        js=_ENGLISH_UI_JS,
        footer_links=[],
    )
