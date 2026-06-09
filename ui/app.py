"""Gradio web UI for testing the EnergyPlus Agent."""

from __future__ import annotations

import queue
import threading
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

setup_logger(level="WARNING")

# Dropdown label → internal metric key used by charts.zone_energy_3d
_METRIC_MAP: dict[str, str] = {
    "供冷负荷 (kWh)": "cooling",
    "供热负荷 (kWh)": "heating",
    "年均温度 (°C)": "temperature",
    "照明能耗 (kWh)": "lighting",
}
_METRIC_OPTIONS = list(_METRIC_MAP.keys())


DEFAULT_EPW = Path("data/weather/Shenzhen.epw")
OUTPUT_DIR = Path("output/ui")

NODE_LABELS: dict[str, str] = {
    "intake": "解析建筑描述",
    "zone": "创建空间分区",
    "material": "定义材料属性",
    "schedule": "设置时间表",
    "cross_ref_foundations": "基础交叉验证",
    "construction": "构建围护结构",
    "surface": "定义建筑表面",
    "fenestration": "配置门窗",
    "hvac": "设计暖通系统",
    "people": "设置人员荷载",
    "lights": "配置照明系统",
    "cross_ref_complete": "完整交叉验证",
    "validate": "验证建筑模型",
    "simulate": "运行 EnergyPlus",
    "analyze": "分析仿真结果",
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
        "模型验证摘要",
        f"- 空间分区：{summary.get('zones_count', 0)} 个",
        f"- 材料：{summary.get('materials_count', 0)} 种",
        f"- 建筑表面：{summary.get('surfaces_count', 0)} 个",
        f"- 门窗开口：{summary.get('fenestrations_count', 0)} 个",
    ]
    if errors:
        lines.append("\n交叉引用错误：")
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
) -> Generator[tuple[History, list[str]], None, None]:
    """Streaming generator: yields (chat_history, output_files) updates."""

    if not user_input.strip():
        yield [_msg("assistant", "请输入建筑描述后再运行。")], []
        return

    event_q: queue.Queue = queue.Queue()
    thread_id = f"ui_{id(event_q)}"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def on_event(node_name: str, update: dict) -> None:
        label = NODE_LABELS.get(node_name, node_name)
        event_q.put(("node", label))

    def on_interrupt(payload: dict) -> dict:
        event_q.put(("interrupt", payload))
        errors = payload.get("errors", [])
        if errors:
            return {"approved": False, "feedback": "请修正以下错误：" + "; ".join(errors)}
        return {"approved": True}

    def worker() -> None:
        try:
            epw_path = Path(epw_file.name) if epw_file else DEFAULT_EPW
            images: list[str] = [f.name for f in image_files] if image_files else []

            initial = AgentState(user_input=user_input, image_paths=images)
            context = SimContext(epw_path=epw_path, output_dir=OUTPUT_DIR)
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
        _msg("assistant", "正在启动 Agent..."),
    ]
    yield history, []

    while True:
        try:
            event = event_q.get(timeout=180)
        except queue.Empty:
            history.append(_msg("assistant", "等待超时（3 分钟），请检查网络或模型服务。"))
            yield history, []
            break

        kind = event[0]

        if kind == "node":
            history.append(_msg("assistant", f"✓ {event[1]}"))
            yield history, []

        elif kind == "interrupt":
            payload = event[1]
            summary = payload.get("summary", {})
            errors = payload.get("errors", [])
            msg = _fmt_summary(summary, errors)
            status = "发现错误，正在反馈给模型重新修正..." if errors else "验证通过，正在提交仿真..."
            history.append(_msg("assistant", f"{msg}\n\n{status}"))
            yield history, []

        elif kind == "done":
            state = event[1]
            # Show the analysis report from the final [analyze] message if present
            analyze_msg = next(
                (
                    str(getattr(m, "content", ""))
                    for m in reversed(state.get("messages", []))
                    if "[analyze]" in str(getattr(m, "content", ""))
                ),
                None,
            )
            files = _collect_output_files(OUTPUT_DIR)
            if analyze_msg:
                # Strip the "[analyze] " prefix for display
                report = analyze_msg.replace("[analyze] ", "", 1)
                history.append(_msg("assistant", report))
            elif files:
                file_list = "\n".join(f"  {f}" for f in files)
                history.append(_msg("assistant", f"仿真完成！输出文件：\n{file_list}"))
            else:
                history.append(_msg("assistant", "仿真完成，输出目录为空（可能仿真步骤未运行）。"))
            yield history, files
            break

        elif kind == "error":
            err = event[1]
            history.append(_msg("assistant", f"运行出错：\n{err}"))
            yield history, []
            break

    t.join(timeout=5)


def _schedule_zone_choices(output_dir: Path) -> list[tuple[str, str]]:
    try:
        ts = load_results(output_dir).timeseries
        keys = result_charts.list_schedule_zone_keys(ts)
        return [("全部区域（取最大）", ZONE_ALL)] + [
            (result_charts.format_zone_label(k), k) for k in keys
        ]
    except FileNotFoundError:
        return [("全部区域（取最大）", ZONE_ALL)]


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


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="EnergyPlus Agent") as demo:
        gr.Markdown("# EnergyPlus Agent 测试界面")

        # ------------------------------------------------------------------ #
        # Tab 1: simulation                                                    #
        # ------------------------------------------------------------------ #
        with gr.Tab("仿真控制"):
            gr.Markdown("输入建筑描述，上传天气文件（可选），点击 **开始仿真**。")

            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="建筑描述",
                        placeholder="描述你的建筑（楼层数、尺寸、材料、HVAC、人员、照明等）...",
                        lines=12,
                        value=SIMPLE_USER_INPUT,
                    )
                    with gr.Row():
                        epw_file = gr.File(
                            label="天气文件 (.epw)  [不上传则使用默认深圳气象]",
                            file_types=[".epw"],
                        )
                        image_files = gr.File(
                            label="建筑图片（可选，支持多张）",
                            file_types=["image"],
                            file_count="multiple",
                        )
                    run_btn = gr.Button("开始仿真", variant="primary")

                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Agent 运行日志",
                        height=500,
                    )
                    output_files = gr.File(
                        label="输出文件（仿真完成后可下载）",
                        file_count="multiple",
                        visible=False,
                        interactive=False,
                    )

        # ------------------------------------------------------------------ #
        # Tab 2: visualization                                                 #
        # ------------------------------------------------------------------ #
        with gr.Tab("仿真结果可视化"):
            with gr.Row():
                load_btn = gr.Button("加载 output/ui 结果", variant="secondary", scale=1)
                metric_dd = gr.Dropdown(
                    choices=_METRIC_OPTIONS,
                    value=_METRIC_OPTIONS[0],
                    label="3D 热区着色指标",
                    scale=1,
                )
                zone_dd = gr.Dropdown(
                    choices=_schedule_zone_choices(OUTPUT_DIR),
                    value=ZONE_ALL,
                    label="运行时间表分区",
                    scale=1,
                )
            plot_3d = gr.Plot(label="3D 热区能耗（悬停显示面积与层数）")
            plot_solar_3d = gr.Plot(label="3D 外表面太阳辐照")

            with gr.Row():
                plot_schedule_people = gr.Plot(label="人员运行时间（日期×小时）")
                plot_schedule_equipment = gr.Plot(label="设备运行时间（日期×小时）")

            with gr.Row():
                plot_enduse  = gr.Plot(label="年度终端用途能耗")
                plot_comfort = gr.Plot(label="热舒适度分析")
            with gr.Row():
                plot_monthly = gr.Plot(label="分区月度 HVAC 能耗")
                plot_heatmap = gr.Plot(label="区域温度热力图")
            with gr.Row():
                plot_demand  = gr.Plot(label="全楼 HVAC 电需求曲线")
                plot_scatter = gr.Plot(label="温度–湿度散点分布")

        # ------------------------------------------------------------------ #
        # Event wiring                                                         #
        # ------------------------------------------------------------------ #

        _all_plots = [
            plot_3d, plot_solar_3d,
            plot_schedule_people, plot_schedule_equipment,
            plot_enduse, plot_comfort, plot_monthly, plot_heatmap, plot_demand, plot_scatter,
        ]

        def on_run(text, epw, images, metric_label, zone_key):
            """Run simulation then auto-refresh all charts when done."""
            last_files: list[str] = []
            last_history: History = []

            for history, files in run_agent(text, epw, images):
                last_files = files
                last_history = history
                visible = len(files) > 0
                yield (
                    last_history,
                    gr.update(value=files if files else None, visible=visible),
                    *([None] * 10),
                )

            figs = _load_visualizations(OUTPUT_DIR, metric_label, zone_key)
            yield (
                last_history,
                gr.update(value=last_files if last_files else None, visible=bool(last_files)),
                *figs,
            )

        run_btn.click(
            fn=on_run,
            inputs=[input_text, epw_file, image_files, metric_dd, zone_dd],
            outputs=[chatbot, output_files, *_all_plots],
        )

        load_btn.click(
            fn=lambda m, z: _load_visualizations(OUTPUT_DIR, m, z),
            inputs=[metric_dd, zone_dd],
            outputs=_all_plots,
        )

        metric_dd.change(
            fn=lambda m: _update_3d_only(OUTPUT_DIR, m),
            inputs=[metric_dd],
            outputs=[plot_3d],
        )

        zone_dd.change(
            fn=lambda z: _update_schedules_only(OUTPUT_DIR, z),
            inputs=[zone_dd],
            outputs=[plot_schedule_people, plot_schedule_equipment],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
