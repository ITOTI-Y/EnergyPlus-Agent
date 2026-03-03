from typing import Any

from pydantic import BaseModel, ConfigDict
import numpy as np
from typing import Any
from typing import Literal

class ToolInput(BaseModel):
    model_config = ConfigDict(
        validate_by_name=True,
        validate_by_alias=True,
        extra="forbid",
    )


def to_payload(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(by_alias=True, exclude_none=True)

class VertexValidationError(BaseModel):
    """Vertex validation error model"""
    error_type: Literal["not_coplanar", "not_horizontal", "not_clockwise", "insufficient_vertices", "duplicate_vertices"]
    message: str
    details: dict[str, Any] | None = None

def validate_floor_vertices(
    vertices: list[dict], 
    tolerance: float = 1e-6
) -> tuple[bool, VertexValidationError | None]:
    if not vertices or len(vertices) < 3:
        return False, VertexValidationError(
            error_type="insufficient_vertices",
            message=f"The base requires at least 3 vertices, but currently there are only {len(vertices) if vertices else 0} vertices"
        )
    pts = np.array([[v["X"], v["Y"], v["Z"]] for v in vertices], dtype=float)
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    if np.any(distances < tolerance):
        duplicate_indices = np.argwhere(distances < tolerance)
        return False, VertexValidationError(
            error_type="duplicate_vertices",
            message="There are duplicate or overly close vertices",
            details={"duplicate_pairs": duplicate_indices.tolist()}
        )
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    normal_length = np.linalg.norm(normal)
    
    if normal_length < tolerance:
        return False, VertexValidationError(
            error_type="not_coplanar",
            message="The first three vertices are collinear, so it is impossible to determine the plane"
        )
    
    normal = normal / normal_length
    for i in range(3, n):
        v = pts[i] - pts[0]
        distance_to_plane = abs(np.dot(v, normal))
        if distance_to_plane > tolerance:
            return False, VertexValidationError(
                error_type="not_coplanar",
                message=f"Vertex {i} is not on the plane, with a distance of {distance_to_plane:.6f} to the plane"
            )
    if abs(abs(normal[2]) - 1.0) > tolerance:
        return False, VertexValidationError(
            error_type="not_horizontal",
            message=f"The bottom surface is not horizontal, and the Z component of the normal vector is {normal[2]:.6f}, which is expected to be close to ±1"
        )
    z_values = pts[:, 2]
    z_range = np.max(z_values) - np.min(z_values)
    if z_range > tolerance:
        return False, VertexValidationError(
            error_type="not_horizontal",
            message=f"The vertex Z values are inconsistent, with a range of {z_range:.6f}"
        )
    signed_area = 0.0
    for i in range(n):
        j = (i + 1) % n
        signed_area += (pts[j, 0] - pts[i, 0]) * (pts[j, 1] + pts[i, 1])
    if signed_area > 0:
        return False, VertexValidationError(
            error_type="not_clockwise",
            message=f"The vertex order is counterclockwise (signed area = {signed_area:.2f}), which should be clockwise",
            details={"signed_area": signed_area}
        )
    
    return True, None

def convert_vertices_to_mcp_format(vertices: list[dict]) -> list[dict]:
    result = []
    for v in vertices:
        if isinstance(v, dict):
            result.append({
                "X": float(v.get("X", v.get("x", 0))),
                "Y": float(v.get("Y", v.get("y", 0))),
                "Z": float(v.get("Z", v.get("z", 0)))
            })
        elif isinstance(v, (list, tuple)) and len(v) >= 3:
            result.append({"X": float(v[0]), "Y": float(v[1]), "Z": float(v[2])})
    return result

