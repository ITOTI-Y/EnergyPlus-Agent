"""SAM 3 instance segmentation server for SEGMENT_URL.

Runs on a GPU host (about 4 GB of GPU memory) with transformers 5 and the
SAM 3 weights in transformers format, e.g. ModelScope's `facebook/sam3`
without its `sam3.pt`:

    python scripts/sam3_segment_server.py --model /path/to/sam3 --port 8082

`POST /segment` takes an image (base64) and a text prompt, e.g.
'building', and returns every matching instance: its score and its mask as
a base64 PNG (white inside). SAM 3 weights are under Meta's SAM License.
"""

import argparse
import base64
import io
import threading

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel


class SegmentRequest(BaseModel):
    image: str
    prompt: str
    threshold: float = 0.4


class Instance(BaseModel):
    score: float
    mask: str


def make_app(model_path: str) -> FastAPI:
    from transformers import (
        Sam3Model,
        Sam3Processor,
    )

    model = Sam3Model.from_pretrained(model_path, device_map="cuda")
    processor = Sam3Processor.from_pretrained(model_path)
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/segment")
    def segment(request: SegmentRequest) -> list[Instance]:
        image = Image.open(io.BytesIO(base64.b64decode(request.image))).convert("RGB")
        inputs = processor(images=image, text=request.prompt, return_tensors="pt").to(
            model.device
        )
        with lock, torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs,
            threshold=request.threshold,
            mask_threshold=0.5,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        instances = []
        for mask, score in zip(result["masks"], result["scores"], strict=True):
            buffer = io.BytesIO()
            pixels = mask.cpu().numpy().astype(np.uint8) * 255
            Image.fromarray(pixels).save(buffer, "PNG")
            instances.append(
                Instance(
                    score=float(score),
                    mask=base64.b64encode(buffer.getvalue()).decode("ascii"),
                )
            )
        return instances

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True, help="SAM 3 in transformers format")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8082)
    args = parser.parse_args()
    uvicorn.run(make_app(args.model), host=args.host, port=args.port)
