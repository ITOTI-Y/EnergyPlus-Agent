"""Gradio web UI for testing the EnergyPlus Agent."""

from __future__ import annotations

import json
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Generator

import gradio as gr

from scripts._share import SIMPLE_USER_INPUT
from src.agent import AgentState, SimContext, build_graph
from src.agent.runner import run_session
from src.utils.logging import setup_logger
from src.results import load_results, parse_idf_geometry
from src.results import charts as result_charts
from src.results.solar import resolve_surface_solar
from src.results.charts import ZONE_ALL
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


DEFAULT_EPW = Path("data/weather/Shenzhen.epw")
OUTPUT_DIR = Path("output/ui")
SESSIONS_ROOT = OUTPUT_DIR / "sessions"


# ---------------------------------------------------------------------------
# Session management — each simulation run writes to / reads from its own
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


def _list_sessions() -> list[tuple[str, str]]:
    """Return ``[(label, session_id), ...]`` sorted newest-first.

    Label embeds the meta preview when available, e.g.
    ``20260617_143022 — office building (2 zones)``.
    """
    SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
    items: list[tuple[str, str]] = []
    for d in SESSIONS_ROOT.iterdir():
        if not d.is_dir():
            continue
        sid = d.name
        meta = _load_session_meta(sid) or {}
        preview = meta.get("user_input_preview", "")
        zones = meta.get("zones_count")
        suffix = f" ({zones} zones)" if isinstance(zones, int) else ""
        label = f"{sid} — {preview}{suffix}" if preview else f"{sid}{suffix}"
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
    try:
        path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass


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
        yield [_msg("assistant", "Please enter a building description before running.")], []
        return

    session_dir = _session_dir(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

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

            initial = AgentState(user_input=user_input, image_paths=images)
            context = SimContext(epw_path=epw_path, output_dir=session_dir)
            config: dict = {"configurable": {"thread_id": thread_id}}

            state = run_session(
                _get_graph(),
                initial,
                context,
                config,
                on_interrupt=on_interrupt,
                on_event=on_event,
            )
            event_q.put(("done", state))
        except Exception:
            err = traceback.format_exc()
            print(f"\n{'='*60}\nUI WORKER ERROR:\n{err}\n{'='*60}", flush=True)
            event_q.put(("error", err))

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    history: History = [
        _msg("user", user_input),
        _msg("assistant", f"Starting Agent in session `{session_id}`..."),
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
                summary_dict = summary.get_summary().model_dump() if summary and hasattr(summary, "get_summary") else {}
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
                    "📊 The **3D model** and **energy analysis charts** are now "
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
                zones, ts, metric=metric_key, zone_metadata=zone_meta,
            )
            solar_vals, unit, note = resolve_surface_solar(
                zones, result.run_dir, result.idf_path,
            )
            fig_solar_3d = result_charts.exterior_solar_irradiation_3d(
                zones, solar_vals, unit=unit, source_note=note,
            )
        except Exception as exc:
            print(f"[viz] 3D chart error: {exc}", flush=True)

    try:
        fig_schedule_people, fig_schedule_equipment = result_charts.operation_schedule_pair(
            ts, zone_key=zone_key,
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
        return result_charts.operation_schedule_pair(result.timeseries, zone_key=zone_key)
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


def _latest_idf(output_dir: Path) -> Path | None:
    """Return the most recently modified ``temp_*.idf`` under *output_dir*."""
    candidates = sorted(
        output_dir.glob("temp_*.idf"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


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
        return _empty_figure(f"Failed to build 3-D model: {exc}"), _model_status_markdown(idf_path)


def _session_info_markdown(session_id: str) -> str:
    """Status string describing the current session directory + artefacts."""
    if not session_id:
        return "**No active session.** Click *New Session* to create one."
    sdir = _session_dir(session_id)
    if not sdir.exists():
        return f"**Session `{session_id}`** — directory missing."
    meta = _load_session_meta(session_id) or {}
    has_idf = any(sdir.glob("temp_*.idf"))
    has_csv = (sdir / "eplusout.csv").exists()
    parts = [
        f"**Session:** `{session_id}`",
        f"- Directory: `{sdir}`",
        f"- IDF: {'✅ generated' if has_idf else '— (not yet)'}",
        f"- Simulation results: {'✅ available' if has_csv else '— (not yet)'}",
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
        return empty_10 + (None, "**Session directory missing.**", gr.update())

    figs = _load_visualizations(sdir, metric_label, zone_key)
    model_fig, model_status_text = on_load_3d_model(str(sdir), None)
    zone_choices = _schedule_zone_choices(sdir)
    # Keep current zone_key if still valid, else reset to ZONE_ALL
    valid_keys = {c[1] for c in zone_choices}
    new_zone = zone_key if zone_key in valid_keys else ZONE_ALL
    return figs + (model_fig, model_status_text, gr.update(choices=zone_choices, value=new_zone))




# ---------------------------------------------------------------------------
# ChatGPT-style custom CSS (injected via launch(css=...))
# ---------------------------------------------------------------------------

_CHATGPT_CSS = """
/* ===== Layout: full-bleed three-column ===== */
.gradio-container { max-width: 100% !important; padding: 0 !important; }

/* ===== Left sidebar: dark ChatGPT-style ===== */
#chat-sidebar { background: #171717 !important; border: none !important; }
#chat-sidebar * { color: #ECECF1 !important; }
#chat-sidebar .gr-button {
    background: transparent !important; border: 1px solid #4D4D4F !important;
    color: #ECECF1 !important; border-radius: 8px !important;
    font-size: 13px !important; justify-content: flex-start !important;
}
#chat-sidebar .gr-button:hover { background: #212121 !important; }
#chat-sidebar .gr-radio label {
    color: #B4B4B4 !important; font-size: 13px !important;
    padding: 8px 10px !important; border-radius: 8px !important; width: 100%;
}
#chat-sidebar .gr-radio label:hover { background: #212121 !important; }
#chat-sidebar input[type="radio"]:checked + label {
    background: #2A2A2A !important; color: #ECECF1 !important;
    border-left: 3px solid #10A37F !important;
}
#chat-sidebar .gradio-markdown {
    color: #8E8EA0 !important; font-size: 11px !important;
    text-transform: uppercase; letter-spacing: 0.6px; font-weight: 600;
}

/* ===== Top bar ===== */
#top-bar { border-bottom: 1px solid #ECECF1; padding: 10px 20px !important; }
#top-title p { font-size: 15px !important; font-weight: 600 !important; }

/* ===== Chatbot ===== */
#chat-area { border: none !important; }

/* ===== Input box: capsule ===== */
#msg-input textarea {
    border: 1px solid #E5E5E5 !important; border-radius: 24px !important;
    padding: 12px 16px !important; font-size: 14px !important;
}
#msg-input textarea:focus {
    border-color: #10A37F !important;
    box-shadow: 0 0 0 2px rgba(16,163,127,0.12) !important;
}

/* ===== Accent buttons ===== */
.btn-primary { background: #10A37F !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 600 !important; }
.btn-primary:hover { background: #0D8A6A !important; }
.btn-secondary { background: transparent !important; color: #6E6E80 !important;
    border: 1px solid #E5E5E5 !important; border-radius: 8px !important; }
.btn-secondary:hover { background: #F7F7F8 !important; }

/* ===== Right viz panel ===== */
#viz-panel { border-left: 1px solid #E5E5E5; }
"""


def _session_title(session_id: str) -> str:
    """One-line title for the top bar, derived from session meta."""
    meta = _load_session_meta(session_id) or {}
    preview = meta.get("user_input_preview", "").strip()
    return f"◉ {preview[:50]}" if preview else f"◉ Session {session_id}"


def _empty_chat_placeholder() -> str:
    """HTML shown when the chatbot has no messages yet."""
    return (
        '<div style="text-align:center; padding:60px 20px; color:#6E6E80;">'
        '<div style="font-size:42px; margin-bottom:12px;">⚡</div>'
        '<div style="font-size:22px; font-weight:600; color:#0D0D0D; margin-bottom:8px;">'
        "EnergyPlus Agent</div>"
        '<div style="font-size:14px;">Describe a building to get a full energy '
        "simulation with a 3D model and analysis.</div>"
        "</div>"
    )


def _radio_label_for(sid: str) -> str:
    """Return the Radio label that represents *sid* (or a fallback title)."""
    for lbl, s in _list_sessions():
        if s == sid:
            return lbl
    return _session_title(sid)


def _radio_choices() -> list[str]:
    return [lbl for lbl, _s in _list_sessions()]


def build_ui() -> gr.Blocks:
    """Build the ChatGPT-style three-column UI.

    Layout:
      - Left  ``gr.Sidebar``  : dark session list + New session button
      - Center main column    : Chatbot (bubble layout) + capsule input box
      - Right ``gr.Sidebar``  : collapsible visualization panel (3D + charts)
    """
    with gr.Blocks(title="EnergyPlus Agent", fill_height=True, fill_width=True) as demo:
        _initial_session = _current_session_or_create(None)

        # ------------------------------------------------------------------ #
        # LEFT SIDEBAR — dark session list                                     #
        # ------------------------------------------------------------------ #
        with gr.Sidebar(
            position="left", width=260, open=True, elem_id="chat-sidebar",
        ) as left_panel:
            gr.Markdown("### ⚡ EnergyPlus")
            new_btn = gr.Button("＋  New session", elem_classes="btn-secondary")
            gr.Markdown("RECENT")
            session_radio = gr.Radio(
                choices=_radio_choices(),
                value=_radio_label_for(_initial_session),
                label="", container=False, elem_id="session-list",
            )
            gr.Markdown("---\nguest@energyplus  ·  Free plan")

        # ------------------------------------------------------------------ #
        # CENTER — conversation area                                           #
        # ------------------------------------------------------------------ #
        with gr.Column(scale=1):
            with gr.Row(elem_id="top-bar"):
                top_title = gr.Markdown(
                    _session_title(_initial_session), elem_id="top-title"
                )
                export_btn = gr.Button("⤓ Export", elem_classes="btn-secondary")

            chatbot = gr.Chatbot(
                label="", layout="bubble",
                value=_load_chat_history(_initial_session),
                placeholder=_empty_chat_placeholder(),
                height=540, elem_id="chat-area",
            )

            output_files = gr.File(
                label="Output files", file_count="multiple",
                visible=False, interactive=False,
            )

            with gr.Row(elem_id="input-row"):
                epw_file = gr.File(
                    label="EPW", file_types=[".epw"], scale=0, min_width=80,
                )
                image_files = gr.File(
                    label="Images", file_types=["image"], file_count="multiple",
                    scale=0, min_width=90,
                )
                input_text = gr.Textbox(
                    placeholder="Describe a building to simulate… (Enter to send)",
                    lines=1, max_lines=6, scale=4, container=False,
                    elem_id="msg-input",
                )
                send_btn = gr.Button("➤", variant="primary", scale=0, min_width=50,
                                     elem_classes="btn-primary")

        # ------------------------------------------------------------------ #
        # RIGHT SIDEBAR — visualization panel (collapsed until run completes)  #
        # ------------------------------------------------------------------ #
        with gr.Sidebar(
            position="right", width=540, open=False, elem_id="viz-panel",
        ) as viz_panel:
            gr.Markdown("### 🏢 Building 3D Model")
            with gr.Row():
                metric_dd = gr.Dropdown(
                    choices=_METRIC_OPTIONS, value=_METRIC_OPTIONS[0],
                    label="Zone coloring metric", scale=2,
                )
                load_3d_btn = gr.Button("⟳ Reload", scale=1)
            model_3d = gr.Plot(label="")
            model_status = gr.Markdown(
                _model_status_markdown(_latest_idf(_session_dir(_initial_session)))
            )

            with gr.Accordion("📊 Energy Analysis (10 charts)", open=False):
                with gr.Row():
                    zone_dd = gr.Dropdown(
                        choices=_schedule_zone_choices(_session_dir(_initial_session)),
                        value=ZONE_ALL, label="Schedule zone", scale=2,
                    )
                    load_btn = gr.Button("Load charts", scale=1)
                plot_3d_energy = gr.Plot(label="3D Zone Energy Use")
                plot_solar_3d = gr.Plot(label="3D Exterior Solar Irradiation")
                with gr.Row():
                    plot_sched_people = gr.Plot(label="Occupancy Schedule")
                    plot_sched_equip = gr.Plot(label="Equipment Schedule")
                with gr.Row():
                    plot_enduse = gr.Plot(label="Annual End-Use Energy")
                    plot_comfort = gr.Plot(label="Thermal Comfort")
                with gr.Row():
                    plot_monthly = gr.Plot(label="Monthly HVAC Energy")
                    plot_heatmap = gr.Plot(label="Zone Temperature Heatmap")
                with gr.Row():
                    plot_demand = gr.Plot(label="HVAC Electric Demand")
                    plot_scatter = gr.Plot(label="Temperature-Humidity Scatter")

            with gr.Accordion("📋 Output Files", open=False):
                output_files_panel = gr.File(
                    label="Download", file_count="multiple", interactive=False,
                )

        # ------------------------------------------------------------------ #
        # Hidden state carrying the active session_id across events           #
        # ------------------------------------------------------------------ #
        session_state = gr.Textbox(
            value=_initial_session, visible=False, elem_id="session-state",
        )

        _all_viz_plots = [
            plot_3d_energy, plot_solar_3d,
            plot_sched_people, plot_sched_equip,
            plot_enduse, plot_comfort, plot_monthly, plot_heatmap,
            plot_demand, plot_scatter,
        ]

        # ------------------------------------------------------------------ #
        # Helpers                                                             #
        # ------------------------------------------------------------------ #
        def _refresh_viz_panel(sid: str, metric_label: str, zone_key: str):
            """Return (10 figs, model_fig, model_status, zone_dd update)."""
            sdir = _session_dir(sid) if sid else None
            if not sdir or not sdir.exists():
                empty10 = (None,) * 10
                return empty10 + (None, _model_status_markdown(None), gr.update())
            figs = _load_visualizations(sdir, metric_label, zone_key)
            model_fig, model_status_text = on_load_3d_model(str(sdir), None)
            zone_choices = _schedule_zone_choices(sdir)
            valid_keys = {c[1] for c in zone_choices}
            new_zone = zone_key if zone_key in valid_keys else ZONE_ALL
            return figs + (model_fig, model_status_text,
                           gr.update(choices=zone_choices, value=new_zone))

        # ------------------------------------------------------------------ #
        # Event: New session                                                  #
        # ------------------------------------------------------------------ #
        def on_new_session():
            sid = _create_session()
            return (
                gr.update(choices=_radio_choices(), value=_radio_label_for(sid)),  # radio
                sid,                                            # session_state
                _session_title(sid),                            # top_title
                [],                                             # chatbot
                gr.update(value=None, visible=False),          # output_files
                gr.update(open=False),                          # viz_panel closed
                *([None] * 10),                                 # _all_viz_plots
                None,                                           # model_3d
                _model_status_markdown(None),                   # model_status
                gr.update(choices=[("All zones (max)", ZONE_ALL)], value=ZONE_ALL),
                gr.update(value=None),                          # output_files_panel
            )

        new_btn.click(
            fn=on_new_session,
            inputs=None,
            outputs=[
                session_radio, session_state, top_title, chatbot, output_files,
                viz_panel, *_all_viz_plots, model_3d, model_status, zone_dd,
                output_files_panel,
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
            has_results = (_session_dir(sid) / "eplusout.csv").exists()
            files = _collect_output_files(_session_dir(sid))
            return (
                sid,                                  # session_state
                _session_title(sid),                  # top_title
                history,                              # chatbot
                gr.update(open=has_results),          # viz_panel
                *refreshed,                           # 10 figs + model + status + zone_dd
                gr.update(value=files if files else None),  # output_files_panel
            )

        session_radio.select(
            fn=on_session_select,
            inputs=[session_radio, metric_dd, zone_dd],
            outputs=[
                session_state, top_title, chatbot, viz_panel,
                *_all_viz_plots, model_3d, model_status, zone_dd,
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
                    last_history,
                    gr.update(value=files if files else None, visible=bool(files)),
                    gr.update(open=False),
                    *([None] * 10),
                    None,
                )
            refreshed = _refresh_viz_panel(sid, metric_label, zone_key)
            yield (
                last_history,
                gr.update(value=last_files if last_files else None,
                          visible=bool(last_files)),
                gr.update(open=True),
                *refreshed,
                gr.update(value=last_files if last_files else None),
            )

        _run_outputs = [
            chatbot, output_files, viz_panel,
            *_all_viz_plots, model_3d, model_status, zone_dd,
            output_files_panel,
        ]
        send_btn.click(
            fn=on_run,
            inputs=[input_text, epw_file, image_files, session_state,
                    metric_dd, zone_dd],
            outputs=_run_outputs,
        )
        input_text.submit(
            fn=on_run,
            inputs=[input_text, epw_file, image_files, session_state,
                    metric_dd, zone_dd],
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
        # Event: Metric dropdown → 3D zone-energy chart                        #
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
        # Event: Zone dropdown → schedule charts                               #
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
            return on_load_3d_model(str(sdir) if sdir else None, None)

        load_3d_btn.click(
            fn=on_load_3d,
            inputs=[session_state],
            outputs=[model_3d, model_status],
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

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=_CHATGPT_CSS,
    )
