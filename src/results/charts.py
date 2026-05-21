"""Plotly chart generators for EnergyPlus simulation results.

All functions accept parsed data objects from parser.py / idf_geometry.py
and return plotly.graph_objects.Figure instances, compatible with gr.Plot.

Chart catalogue
---------------
2-D charts
  end_use_bar            – annual end-use energy breakdown (from tabular)
  monthly_hvac_energy    – monthly zone heating + cooling (stacked bar)
  zone_temperature_heatmap – zones × months mean temperature
  thermal_comfort_bars   – comfort / hot / cold hours per zone
  hvac_demand_profile    – hourly facility HVAC demand (line + peak)
  temp_humidity_scatter  – zone mean temp vs RH (coloured by zone)

3-D chart
  zone_energy_3d         – 3-D building with zones coloured by energy metric
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from src.results.idf_geometry import ZoneGeometry

# ---------------------------------------------------------------------------
# Colour palette shared across 2-D charts
# ---------------------------------------------------------------------------

_ZONE_COLOURS = [
    "#4C78A8",  # blue
    "#F58518",  # orange
    "#E45756",  # red
    "#72B7B2",  # teal
    "#54A24B",  # green
    "#B279A2",  # purple
    "#FF9DA6",  # pink
    "#9D755D",  # brown
]

_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# EnergyPlus CSV variable names (must match exactly)
_VAR_COOL = "Zone Ideal Loads Supply Air Total Cooling Energy"
_VAR_HEAT = "Zone Ideal Loads Supply Air Total Heating Energy"
_VAR_TEMP = "Zone Mean Air Temperature"
_VAR_RH   = "Zone Air Relative Humidity"
_VAR_LIGHT = "Zone Lights Electricity Energy"
_VAR_HVAC  = "Facility Total HVAC Electricity Demand Rate"

# Layout defaults
_LAYOUT = dict(
    font=dict(family="Inter, Arial, sans-serif", size=12),
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(l=60, r=30, t=50, b=50),
)


# ---------------------------------------------------------------------------
# Helper: extract monthly aggregates from timeseries
# ---------------------------------------------------------------------------


def _monthly_sum(ts: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Return a DataFrame (12 rows × N zones) of monthly summed values.

    Rows are months 1-12.  Columns are zone keys that have this variable.
    """
    cols = [c for c in ts.columns if c[1] == variable]
    if not cols:
        return pd.DataFrame()

    # Extract the selected columns as a flat DataFrame to avoid MultiIndex groupby warning
    sub = pd.DataFrame(
        {c[0]: ts[c].values for c in cols},
        index=ts.index,
    )
    months = sub.index.str.strip().str.split("/").str[0].astype(int)
    sub = sub.copy()
    sub["_month"] = months
    monthly = sub.groupby("_month").sum(numeric_only=True)
    monthly.index.name = "month"
    return monthly


def _monthly_mean(ts: pd.DataFrame, variable: str) -> pd.DataFrame:
    """Same as _monthly_sum but uses mean aggregation."""
    cols = [c for c in ts.columns if c[1] == variable]
    if not cols:
        return pd.DataFrame()

    sub = pd.DataFrame(
        {c[0]: ts[c].values for c in cols},
        index=ts.index,
    )
    months = sub.index.str.strip().str.split("/").str[0].astype(int)
    sub = sub.copy()
    sub["_month"] = months
    monthly = sub.groupby("_month").mean(numeric_only=True)
    monthly.index.name = "month"
    return monthly


def _annual_sum(ts: pd.DataFrame, variable: str) -> dict[str, float]:
    """Return {zone_key: annual_total} for *variable*."""
    cols = [c for c in ts.columns if c[1] == variable]
    return {c[0]: float(ts[c].sum(skipna=True)) for c in cols}


def _annual_mean(ts: pd.DataFrame, variable: str) -> dict[str, float]:
    """Return {zone_key: annual_mean} for *variable*."""
    cols = [c for c in ts.columns if c[1] == variable]
    return {c[0]: float(ts[c].mean(skipna=True)) for c in cols}


def _comfort_hours(
    ts: pd.DataFrame,
    tmin: float = 20.0,
    tmax: float = 26.0,
) -> pd.DataFrame:
    """Return DataFrame with columns [comfort, hot, cold] indexed by zone key."""
    cols = [c for c in ts.columns if c[1] == _VAR_TEMP]
    rows = []
    for col in cols:
        vals = ts[col].dropna()
        n = len(vals)
        if n == 0:
            rows.append({"zone": col[0], "comfort": 0, "hot": 0, "cold": 0})
            continue
        comfort = int(((vals >= tmin) & (vals <= tmax)).sum())
        hot     = int((vals > tmax).sum())
        cold    = int((vals < tmin).sum())
        rows.append({"zone": col[0], "comfort": comfort, "hot": hot, "cold": cold})
    return pd.DataFrame(rows).set_index("zone") if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Chart 1: end_use_bar
# ---------------------------------------------------------------------------


def end_use_bar(tabular: dict) -> go.Figure:
    """Horizontal bar chart of annual end-use energy breakdown.

    Uses data from eplustbl.csv.  Displays electricity, district cooling and
    district heating in kWh side-by-side for each end-use category.
    """
    end_uses: list[dict] = tabular.get("end_uses", [])
    if not end_uses:
        fig = go.Figure()
        fig.update_layout(title="No end-use data available", **_LAYOUT)
        return fig

    # Filter to rows with any non-zero value; exclude "Total" row
    fuel_keys = [k for k in end_uses[0] if k.endswith("_kwh")]
    useful_fuel_keys = []
    useful_rows = []
    for row in end_uses:
        if row["use"].strip().lower() == "total end uses":
            continue
        if any((row.get(k) or 0) != 0 for k in fuel_keys):
            useful_rows.append(row)

    # Identify fuel columns that have any non-zero value
    for fk in fuel_keys:
        if any((r.get(fk) or 0) != 0 for r in useful_rows):
            useful_fuel_keys.append(fk)

    if not useful_rows or not useful_fuel_keys:
        fig = go.Figure()
        fig.update_layout(title="All end-use values are zero", **_LAYOUT)
        return fig

    use_labels = [r["use"] for r in useful_rows]

    fuel_display = {
        "electricity_kwh": "Electricity",
        "district_cooling_kwh": "District Cooling",
        "district_heating_water_kwh": "District Heating Water",
        "district_heating_steam_kwh": "District Heating Steam",
        "natural_gas_kwh": "Natural Gas",
    }

    fig = go.Figure()
    colours = ["#4C78A8", "#72B7B2", "#F58518", "#E45756", "#54A24B"]
    for ci, fk in enumerate(useful_fuel_keys):
        values = [r.get(fk) or 0 for r in useful_rows]
        label = fuel_display.get(fk, fk.replace("_kwh", "").replace("_", " ").title())
        fig.add_trace(go.Bar(
            name=label,
            y=use_labels,
            x=values,
            orientation="h",
            marker_color=colours[ci % len(colours)],
        ))

    building = tabular.get("building_name", "")
    eui = tabular.get("eui_mj_per_m2")
    eui_str = f" | EUI: {eui:.1f} MJ/m²" if eui else ""

    fig.update_layout(
        title=f"Annual End-Use Energy{eui_str}",
        xaxis_title="Energy (kWh)",
        yaxis_title="",
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 2: monthly_hvac_energy
# ---------------------------------------------------------------------------


def monthly_hvac_energy(ts: pd.DataFrame) -> go.Figure:
    """Stacked bar chart of monthly HVAC heating + cooling energy per zone."""
    cool = _monthly_sum(ts, _VAR_COOL)
    heat = _monthly_sum(ts, _VAR_HEAT)

    if cool.empty and heat.empty:
        fig = go.Figure()
        fig.update_layout(title="No HVAC energy data available", **_LAYOUT)
        return fig

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Monthly Cooling Energy (kWh)", "Monthly Heating Energy (kWh)"),
        shared_yaxes=False,
    )

    zones = list(cool.columns) if not cool.empty else list(heat.columns)

    for zi, zone in enumerate(zones):
        colour = _ZONE_COLOURS[zi % len(_ZONE_COLOURS)]
        zone_label = zone.replace("_", " ").title()

        if not cool.empty and zone in cool.columns:
            fig.add_trace(go.Bar(
                name=zone_label,
                x=[_MONTH_LABELS[m - 1] for m in cool.index],
                y=cool[zone].tolist(),
                marker_color=colour,
                legendgroup=zone,
                showlegend=True,
            ), row=1, col=1)

        if not heat.empty and zone in heat.columns:
            fig.add_trace(go.Bar(
                name=zone_label,
                x=[_MONTH_LABELS[m - 1] for m in heat.index],
                y=heat[zone].tolist(),
                marker_color=colour,
                legendgroup=zone,
                showlegend=False,
            ), row=1, col=2)

    fig.update_layout(
        title="Monthly HVAC Energy by Zone",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 3: zone_temperature_heatmap
# ---------------------------------------------------------------------------


def zone_temperature_heatmap(ts: pd.DataFrame) -> go.Figure:
    """Heatmap of monthly mean air temperature for each zone."""
    monthly = _monthly_mean(ts, _VAR_TEMP)

    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(title="No temperature data available", **_LAYOUT)
        return fig

    zones = [z.replace("_", " ").title() for z in monthly.columns]
    month_labels = [_MONTH_LABELS[m - 1] for m in monthly.index]

    fig = go.Figure(data=go.Heatmap(
        z=monthly.values.T.tolist(),
        x=month_labels,
        y=zones,
        colorscale="RdYlBu_r",
        colorbar=dict(title="°C"),
        hovertemplate="Month: %{x}<br>Zone: %{y}<br>Temp: %{z:.1f} °C<extra></extra>",
    ))

    fig.update_layout(
        title="Monthly Mean Zone Air Temperature (°C)",
        xaxis_title="Month",
        yaxis_title="Zone",
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 4: thermal_comfort_bars
# ---------------------------------------------------------------------------


def thermal_comfort_bars(
    ts: pd.DataFrame,
    tmin: float = 20.0,
    tmax: float = 26.0,
) -> go.Figure:
    """Stacked horizontal bar: comfort / hot / cold hours per zone."""
    df = _comfort_hours(ts, tmin, tmax)

    if df.empty:
        fig = go.Figure()
        fig.update_layout(title="No temperature data for comfort analysis", **_LAYOUT)
        return fig

    zones = [z.replace("_", " ").title() for z in df.index]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"Comfort ({tmin}–{tmax} °C)",
        y=zones,
        x=df["comfort"].tolist(),
        orientation="h",
        marker_color="#54A24B",
    ))
    fig.add_trace(go.Bar(
        name=f"Too Hot (>{tmax} °C)",
        y=zones,
        x=df["hot"].tolist(),
        orientation="h",
        marker_color="#E45756",
    ))
    fig.add_trace(go.Bar(
        name=f"Too Cold (<{tmin} °C)",
        y=zones,
        x=df["cold"].tolist(),
        orientation="h",
        marker_color="#4C78A8",
    ))

    # Comfort percentage annotation
    total_hours = df["comfort"] + df["hot"] + df["cold"]
    comfort_pct = (df["comfort"] / total_hours * 100).round(1)
    for i, (zone, pct) in enumerate(zip(zones, comfort_pct)):
        fig.add_annotation(
            x=total_hours.iloc[i] + 20,
            y=zone,
            text=f"{pct}%",
            showarrow=False,
            font=dict(size=11, color="#333"),
        )

    fig.update_layout(
        title=f"Thermal Comfort Hours per Zone (comfort band {tmin}–{tmax} °C)",
        xaxis_title="Hours",
        barmode="stack",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 5: hvac_demand_profile
# ---------------------------------------------------------------------------


def hvac_demand_profile(ts: pd.DataFrame) -> go.Figure:
    """Line chart of hourly facility HVAC electricity demand (W)."""
    cols = [c for c in ts.columns if c[1] == _VAR_HVAC]

    if not cols:
        fig = go.Figure()
        fig.update_layout(title="No HVAC demand data available", **_LAYOUT)
        return fig

    col = cols[0]
    values = ts[col].fillna(0)

    peak_idx = int(values.to_numpy().argmax()) if not values.empty else 0
    peak_val = float(values.max())

    # Downsample to daily max for readability (8760 → 365 points)
    daily_max = values.values.reshape(-1, 24).max(axis=1)
    day_index = list(range(1, len(daily_max) + 1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_index,
        y=daily_max.tolist(),
        mode="lines",
        name="Daily Peak Demand (W)",
        line=dict(color="#4C78A8", width=1.5),
        fill="tozeroy",
        fillcolor="rgba(76,120,168,0.15)",
    ))

    # Peak annotation
    peak_day = peak_idx // 24 + 1
    fig.add_annotation(
        x=peak_day,
        y=peak_val,
        text=f"Peak: {peak_val/1000:.1f} kW",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#E45756",
        font=dict(color="#E45756", size=11),
        ax=30,
        ay=-30,
    )

    fig.update_layout(
        title="Facility HVAC Electricity Demand Profile (Daily Peak)",
        xaxis_title="Day of Year",
        yaxis_title="Demand (W)",
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 6: temp_humidity_scatter
# ---------------------------------------------------------------------------


def temp_humidity_scatter(ts: pd.DataFrame) -> go.Figure:
    """Scatter plot of zone mean temperature vs relative humidity."""
    temp_cols = [c for c in ts.columns if c[1] == _VAR_TEMP]
    rh_cols   = [c for c in ts.columns if c[1] == _VAR_RH]

    if not temp_cols or not rh_cols:
        fig = go.Figure()
        fig.update_layout(title="No temperature or humidity data available", **_LAYOUT)
        return fig

    # Match zones
    temp_zones = {c[0]: c for c in temp_cols}
    rh_zones   = {c[0]: c for c in rh_cols}
    common_zones = sorted(set(temp_zones) & set(rh_zones))

    if not common_zones:
        fig = go.Figure()
        fig.update_layout(title="No matching zones for temp/humidity scatter", **_LAYOUT)
        return fig

    # Downsample: take every 6th hour for readability
    fig = go.Figure()
    for zi, zone in enumerate(common_zones):
        t_vals = ts[temp_zones[zone]].iloc[::6].dropna()
        rh_vals = ts[rh_zones[zone]].iloc[::6].dropna()
        common_idx = t_vals.index.intersection(rh_vals.index)

        fig.add_trace(go.Scatter(
            x=t_vals.loc[common_idx].tolist(),
            y=rh_vals.loc[common_idx].tolist(),
            mode="markers",
            name=zone.replace("_", " ").title(),
            marker=dict(
                color=_ZONE_COLOURS[zi % len(_ZONE_COLOURS)],
                size=4,
                opacity=0.5,
            ),
        ))

    # Comfort zone reference box
    fig.add_shape(type="rect", x0=20, x1=26, y0=30, y1=70,
                  line=dict(color="#54A24B", dash="dot"),
                  fillcolor="rgba(84,162,75,0.06)")
    fig.add_annotation(x=23, y=72, text="Comfort Zone",
                       showarrow=False, font=dict(color="#54A24B", size=10))

    fig.update_layout(
        title="Zone Temperature vs Relative Humidity",
        xaxis_title="Mean Air Temperature (°C)",
        yaxis_title="Relative Humidity (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **_LAYOUT,
    )
    return fig


# ---------------------------------------------------------------------------
# Chart 7: zone_energy_3d
# ---------------------------------------------------------------------------

# Metric display configuration
_METRIC_CONFIG: dict[str, dict] = {
    "cooling": {
        "variable": _VAR_COOL,
        "agg": "sum",
        "unit": "kWh",
        "label": "Annual Cooling Energy",
        "colorscale": "RdYlBu_r",
    },
    "heating": {
        "variable": _VAR_HEAT,
        "agg": "sum",
        "unit": "kWh",
        "label": "Annual Heating Energy",
        "colorscale": "Blues",
    },
    "temperature": {
        "variable": _VAR_TEMP,
        "agg": "mean",
        "unit": "°C",
        "label": "Annual Mean Temperature",
        "colorscale": "RdYlBu_r",
    },
    "lighting": {
        "variable": _VAR_LIGHT,
        "agg": "sum",
        "unit": "kWh",
        "label": "Annual Lighting Energy",
        "colorscale": "YlOrRd",
    },
}


def _triangulate_polygon(vertices: list[tuple[float, float, float]]) -> list[tuple[int, int, int]]:
    """Fan triangulation from the first vertex of a convex polygon.

    Returns list of (i, j, k) index triples into *vertices*.
    """
    n = len(vertices)
    if n < 3:
        return []
    return [(0, i, i + 1) for i in range(1, n - 1)]


def _build_zone_mesh(
    zone: "ZoneGeometry",
) -> tuple[list[float], list[float], list[float], list[int], list[int], list[int]]:
    """Return (x, y, z, i, j, k) arrays for a Plotly Mesh3d of this zone."""
    all_verts: list[tuple[float, float, float]] = []
    vert_index: dict[tuple, int] = {}
    tri_i: list[int] = []
    tri_j: list[int] = []
    tri_k: list[int] = []

    def _vert_idx(v: tuple[float, float, float]) -> int:
        key = (round(v[0], 6), round(v[1], 6), round(v[2], 6))
        if key not in vert_index:
            vert_index[key] = len(all_verts)
            all_verts.append(key)
        return vert_index[key]

    for surface in zone.surfaces:
        verts = surface.vertices
        if len(verts) < 3:
            continue
        local_indices = [_vert_idx(v) for v in verts]
        for (li, lj, lk) in _triangulate_polygon(verts):
            tri_i.append(local_indices[li])
            tri_j.append(local_indices[lj])
            tri_k.append(local_indices[lk])

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    zs = [v[2] for v in all_verts]
    return xs, ys, zs, tri_i, tri_j, tri_k


def zone_energy_3d(
    zones: dict[str, "ZoneGeometry"],
    ts: pd.DataFrame,
    metric: str = "cooling",
) -> go.Figure:
    """3-D building scene with zones coloured by an energy/thermal metric.

    Each zone is rendered as a closed Mesh3d polyhedron; colour intensity
    reflects the chosen metric value for that zone.

    Parameters
    ----------
    zones:
        Output of :func:`idf_geometry.parse_idf_geometry`.
    ts:
        Parsed timeseries DataFrame from :func:`parser.parse_timeseries`.
    metric:
        One of ``"cooling"``, ``"heating"``, ``"temperature"``, ``"lighting"``.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    cfg = _METRIC_CONFIG.get(metric, _METRIC_CONFIG["cooling"])
    variable = cfg["variable"]
    agg = cfg["agg"]
    unit = cfg["unit"]
    label = cfg["label"]
    colorscale = cfg["colorscale"]

    # Compute per-zone metric values
    if agg == "sum":
        zone_values = _annual_sum(ts, variable)
    else:
        zone_values = _annual_mean(ts, variable)

    # Build CSV-key → IDF-zone-name mapping
    # CSV keys are like "ZONE_CORE", IDF names are like "Zone_Core"
    from src.results.idf_geometry import idf_zone_to_csv_key  # local import to avoid circulars

    idf_to_csv: dict[str, str] = {z: idf_zone_to_csv_key(z) for z in zones}

    all_vals = [zone_values.get(csv_key, 0.0) for csv_key in idf_to_csv.values()]
    # Also check variant without "IDEAL LOADS AIR SYSTEM" suffix in CSV key
    # CSV key for HVAC is "ZONE_CORE IDEAL LOADS AIR SYSTEM" but zone name is "ZONE_CORE"
    # For temperature/lighting the key IS the zone name
    for idf_name, csv_key in list(idf_to_csv.items()):
        if zone_values.get(csv_key) is None:
            # Try with " IDEAL LOADS AIR SYSTEM" suffix (for HVAC variables)
            alt_key = csv_key + " IDEAL LOADS AIR SYSTEM"
            if alt_key in zone_values:
                idf_to_csv[idf_name] = alt_key

    all_vals = [zone_values.get(idf_to_csv[z], 0.0) for z in zones]
    vmin = min(all_vals) if all_vals else 0.0
    vmax = max(all_vals) if all_vals else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    fig = go.Figure()

    # One Mesh3d trace per zone
    zone_list = sorted(zones.keys())
    for zi, zone_name in enumerate(zone_list):
        zone_geom = zones[zone_name]
        csv_key = idf_to_csv[zone_name]
        metric_val = zone_values.get(csv_key, 0.0)

        xs, ys, zs, ti, tj, tk = _build_zone_mesh(zone_geom)
        if not xs:
            continue

        # Normalised intensity [0, 1] for each vertex
        norm_val = (metric_val - vmin) / (vmax - vmin)
        intensity = [norm_val] * len(xs)

        cx, cy, cz = zone_geom.centroid()
        hover_text = (
            f"<b>{zone_name}</b><br>"
            f"{label}: {metric_val:.2f} {unit}"
        )

        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=ti, j=tj, k=tk,
            intensity=intensity,
            cmin=0.0,
            cmax=1.0,
            colorscale=colorscale,
            showscale=(zi == 0),  # only first trace shows colorbar
            colorbar=dict(
                title=f"{label}<br>({unit})",
                tickvals=[0, 0.5, 1],
                ticktext=[
                    f"{vmin:.1f}",
                    f"{(vmin+vmax)/2:.1f}",
                    f"{vmax:.1f}",
                ],
            ) if zi == 0 else None,
            opacity=0.85,
            name=zone_name,
            hoverinfo="text",
            hovertext=hover_text,
            flatshading=True,
            lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1),
        ))

        # Zone label annotation (as 3-D scatter text)
        fig.add_trace(go.Scatter3d(
            x=[cx], y=[cy], z=[cz],
            mode="text",
            text=[f"{zone_name.replace('Zone_', '')}<br>{metric_val:.0f} {unit}"],
            textfont=dict(size=10, color="black"),
            showlegend=False,
            hoverinfo="skip",
        ))

    fig.update_layout(
        title=f"3D Zone Energy Map – {label}",
        scene=dict(
            xaxis_title="X (m)",
            yaxis_title="Y (m)",
            zaxis_title="Z (m)",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=1.2)),
            xaxis=dict(backgroundcolor="rgba(240,240,240,0.5)", gridcolor="white"),
            yaxis=dict(backgroundcolor="rgba(230,230,230,0.5)", gridcolor="white"),
            zaxis=dict(backgroundcolor="rgba(220,220,220,0.5)", gridcolor="white"),
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="white",
        font=dict(family="Inter, Arial, sans-serif", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Convenience: generate all 2-D charts at once
# ---------------------------------------------------------------------------


def all_2d_charts(
    ts: pd.DataFrame,
    tabular: dict,
) -> dict[str, go.Figure]:
    """Return all six 2-D figures keyed by short name."""
    return {
        "end_use": end_use_bar(tabular),
        "monthly_hvac": monthly_hvac_energy(ts),
        "temp_heatmap": zone_temperature_heatmap(ts),
        "comfort": thermal_comfort_bars(ts),
        "hvac_demand": hvac_demand_profile(ts),
        "temp_rh_scatter": temp_humidity_scatter(ts),
    }
