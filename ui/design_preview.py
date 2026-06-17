"""Render the UI design mockup as a Gradio page so it can be viewed in a browser.

Run:  .venv/bin/python ui/design_preview.py
Open: http://localhost:7860
"""
from pathlib import Path

import gradio as gr

MOCKUP_HTML = (Path(__file__).resolve().parent.parent / "output" / "ui_design" / "mockup.html").read_text(encoding="utf-8")

# Wrap the mockup so it fills the viewport inside Gradio's HTML component.
WRAPPER = f"""
<div style="width:100%; min-height:840px; background:#ECECEC; padding:0; margin:0;">
  {MOCKUP_HTML}
</div>
"""

with gr.Blocks(
    title="EnergyPlus Agent — UI Design",
    css="""
    body, .gradio-container { background: #ECECEC !important; padding: 0 !important; max-width: 100% !important; }
    .html-component { width: 100% !important; }
    """,
) as demo:
    gr.HTML(WRAPPER)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
