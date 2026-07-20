"""Render a bird's-eye (3-D isometric) PNG of an EnergyPlus IDF.

Used by the robustness benchmark to capture a quick visual snapshot of the
building the agent produced.

Loads ``src/results/idf_geometry.py`` *directly by file path* (via
importlib) rather than through ``src.results``: importing that package
executes its ``__init__.py``, which pulls in the results parser and,
transitively, ``idfpy`` — and ``import idfpy`` is slow / can hang in this
environment. The geometry parser itself is pure-text (re + dataclasses)
with no idfpy dependency, so loading the module file directly is both safe
and fast.

Public API: ``render_top_view(idf_path, out_png, dpi=150) -> Path``.
"""

import contextlib
import importlib.util
import sys
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")  # headless: must be set before pyplot import

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO_ROOT: Final = Path(__file__).resolve().parents[2]


def _load_geometry_module():
    """Load src/results/idf_geometry.py as an isolated module (no package init)."""
    mod_path = REPO_ROOT / "src" / "results" / "idf_geometry.py"
    mod_name = "_idf_geometry_isolated"
    spec = importlib.util.spec_from_file_location(mod_name, mod_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spec for {mod_path}")
    module = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec: @dataclass looks up
    # cls.__module__ in sys.modules during class processing, and a missing
    # entry raises AttributeError('NoneType' has no __dict__).
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


_geo = _load_geometry_module()
parse_idf_geometry = _geo.parse_idf_geometry
parse_fenestrations = _geo.parse_fenestrations

# Surface-type -> (face color, alpha). Floors nearly transparent so walls
# and roof read clearly in the isometric view.
_TYPE_STYLE: Final[dict[str, tuple[str, float]]] = {
    "wall": ("#9ecae1", 0.55),
    "roof": ("#fdd0a2", 0.70),
    "ceiling": ("#e7e7e7", 0.35),
    "floor": ("#3182bd", 0.12),
}

# Fenestration styling, keyed by surface_type: (face color, alpha, edge
# linewidth). Drawn ON TOP of the host wall as high-contrast opaque patches
# so window positions are visible at a glance; a plain Door stays muted (it
# is not glazing).
_FENESTRATION_STYLE: Final[dict[str, tuple[str, float, float]]] = {
    "window": ("#e41a1c", 0.92, 0.7),
    "glassdoor": ("#ff7f00", 0.92, 0.7),
    "door": ("#984ea3", 0.55, 0.5),
    "tubulardaylightdome": ("#ffff33", 0.92, 0.6),
    "tubulardaylightdiffuser": ("#ffff33", 0.92, 0.6),
}
_FENESTRATION_DEFAULT_STYLE: Final[tuple[str, float, float]] = ("#e41a1c", 0.92, 0.7)


def _zone_color(zone_index: int) -> tuple[float, float, float]:
    """Pick a stable color per zone from the tab20 colormap."""
    cmap = matplotlib.colormaps["tab20"]
    return cmap(zone_index % 20)[:3]


def render_top_view(idf_path: Path | str, out_png: Path | str, dpi: int = 150) -> Path:
    """Render an isometric PNG of the building in ``idf_path`` to ``out_png``.

    Faces are colored by thermal zone (so zones are visually separable),
    with edge lines drawn on every polygon. Axes are equal-scaled in x/y/z.

    Args:
        idf_path: IDF file to render.
        out_png: Output PNG path (parent directories are created).
        dpi: Output resolution.

    Returns:
        The output path.

    Raises:
        ValueError: If the IDF has no parseable zones or drawable surfaces.
    """
    idf_path = Path(idf_path)
    out_png = Path(out_png)
    zones = parse_idf_geometry(idf_path)
    if not zones:
        raise ValueError(f"no zones/surfaces parsed from {idf_path}")

    # Fenestrations (windows / doors) live on the plane of their host wall
    # and are rendered on top of it. Vertices are already in world coords.
    fenestrations = parse_fenestrations(idf_path)

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    all_xs: list[float] = []
    all_ys: list[float] = []
    all_zs: list[float] = []

    # Sort zone names for stable coloring.
    for zi, zone_name in enumerate(sorted(zones)):
        zone_color = _zone_color(zi)
        for surf in zones[zone_name].surfaces:
            verts = surf.vertices
            if len(verts) < 3:
                continue
            base_alpha = _TYPE_STYLE.get(surf.surface_type.lower(), ("#cccccc", 0.5))[1]
            rgba = to_rgba(zone_color, alpha=base_alpha)
            ax.add_collection3d(
                Poly3DCollection(
                    [verts],
                    facecolors=[rgba],
                    edgecolors=("#333333",),
                    linewidths=0.4,
                )
            )
            for x, y, z in verts:
                all_xs.append(x)
                all_ys.append(y)
                all_zs.append(z)

    for fen in fenestrations:
        verts = fen.vertices
        if len(verts) < 3:
            continue
        face_color, alpha, lw = _FENESTRATION_STYLE.get(
            fen.surface_type.lower(), _FENESTRATION_DEFAULT_STYLE
        )
        rgba = to_rgba(face_color, alpha=alpha)
        ax.add_collection3d(
            Poly3DCollection(
                [verts],
                facecolors=[rgba],
                edgecolors=("#000000",),
                linewidths=lw,
            )
        )
        for x, y, z in verts:
            all_xs.append(x)
            all_ys.append(y)
            all_zs.append(z)

    if not all_xs:
        raise ValueError(f"no drawable surfaces in {idf_path}")

    xs = np.array(all_xs)
    ys = np.array(all_ys)
    zs = np.array(all_zs)
    pad_xy = max(1.0, max(xs.max() - xs.min(), ys.max() - ys.min()) * 0.05)
    pad_z = max(1.0, (zs.max() - zs.min()) * 0.10)

    ax.set_xlim(xs.min() - pad_xy, xs.max() + pad_xy)
    ax.set_ylim(ys.min() - pad_xy, ys.max() + pad_xy)
    ax.set_zlim(zs.min() - pad_z, zs.max() + pad_z)
    # Older matplotlib has no set_box_aspect on 3D; equal axis is the fallback.
    with contextlib.suppress(AttributeError, TypeError):
        ax.set_box_aspect(
            (
                xs.max() - xs.min() + 2 * pad_xy,
                ys.max() - ys.min() + 2 * pad_xy,
                zs.max() - zs.min() + 2 * pad_z,
            )
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    n_fen = len(fenestrations)
    title = f"Bird's-eye view — {idf_path.name}"
    if n_fen:
        title += f"  ({n_fen} window{'s' if n_fen != 1 else ''} in red)"
    ax.set_title(title)
    ax.view_init(elev=30, azim=-60)  # isometric

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
