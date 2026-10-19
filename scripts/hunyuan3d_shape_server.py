"""Shape-only Hunyuan3D-2.1 server for IMAGE_TO_3D_URL.

Runs on a GPU host (about 10 GB of GPU memory) with a clone of
https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 and its shape
requirements (transformers 4.46: later releases rename the image
encoder's weights), not in this project's environment:

    python scripts/hunyuan3d_shape_server.py --repo /path/to/Hunyuan3D-2.1 --port 8081

`POST /generate` takes the request of the repository's own `api_server.py`
(`image` as base64, `remove_background`, `texture`, `seed`) and returns the
mesh as GLB. That server also loads the texture pipeline, which needs its
compiled extensions and more GPU memory; the agent uses the shape only.
"""

import argparse
import base64
import io
import sys
import threading

import torch  # ty: ignore[unresolved-import]
import uvicorn
from fastapi import FastAPI, HTTPException, Response  # ty: ignore[unresolved-import]
from PIL import Image
from pydantic import BaseModel


class GenerationRequest(BaseModel):
    image: str
    remove_background: bool = True
    texture: bool = False
    seed: int = 1234
    # None keeps the pipeline's own default.
    octree_resolution: int | None = None
    num_inference_steps: int | None = None
    guidance_scale: float | None = None


def make_app(repo: str, model: str) -> FastAPI:
    # hy3dshape lives in the Hunyuan3D-2.1 clone on the GPU host only.
    sys.path.insert(0, f"{repo}/hy3dshape")
    from hy3dshape.pipelines import (  # ty: ignore[unresolved-import]
        Hunyuan3DDiTFlowMatchingPipeline,
    )
    from hy3dshape.rembg import BackgroundRemover  # ty: ignore[unresolved-import]

    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(model)
    remover = BackgroundRemover()
    # The pipeline's scheduler keeps per-run state: concurrent requests
    # would step each other's runs.
    lock = threading.Lock()
    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "healthy"}

    @app.post("/generate")
    def generate(request: GenerationRequest) -> Response:
        if request.texture:
            raise HTTPException(
                400, "this server generates shapes only; send texture: false"
            )
        image = Image.open(io.BytesIO(base64.b64decode(request.image))).convert("RGBA")
        if request.remove_background:
            image = remover(image.convert("RGB"))
        options = request.model_dump(
            include={"num_inference_steps", "guidance_scale", "octree_resolution"},
            exclude_none=True,
        )
        with lock:
            mesh = pipeline(
                image=image,
                generator=torch.Generator(device="cuda").manual_seed(request.seed),
                **options,
            )[0]
        return Response(mesh.export(file_type="glb"), media_type="model/gltf-binary")

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", required=True, help="Clone of Hunyuan3D-2.1")
    parser.add_argument("--model", default="tencent/Hunyuan3D-2.1")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    uvicorn.run(make_app(args.repo, args.model), host=args.host, port=args.port)
