"""Read-only analysis tools for EnergyPlus simulation output files.

Factory: make_analysis_tools(output_dir: Path) -> list[BaseTool]

Six tools are produced:
  get_simulation_status      – reads .end and .err
  get_available_variables    – reads CSV header only
  get_variable_statistics    – aggregates timeseries data (J → kWh auto)
  get_peak_hours             – top-N max/min timesteps for one variable/key
  get_comfort_statistics     – thermal-comfort hours per zone
  get_energy_summary         – parses eplustbl.htm End Uses table
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.tools import BaseTool, tool

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_J_TO_KWH = 1.0 / 3_600_000.0

# EnergyPlus CSV column format: KEY:VariableName [Unit](Freq)
_COL_PATTERN = re.compile(r"^(.+?):(.+?)\s+\[(.+?)\]\((.+?)\)$")


def _ok(msg: str, data: Any = None) -> str:
    return json.dumps({"success": True, "message": msg, "data": data})


def _err(msg: str, data: Any = None) -> str:
    return json.dumps({"success": False, "message": msg, "data": data})


def _find_run_dir(output_dir: Path) -> Path | None:
    """Locate the directory that contains EnergyPlus output files.

    Accepts a directory as a valid run dir if it contains eplusout.csv
    (successful run) OR eplusout.end (any run, including failed ones).

    Search order:
    1. output_dir itself (post-fix: output_directory passed to runner)
    2. output_dir/energyplus_runs_* subdirs, newest first (legacy fallback)
    3. Returns None when nothing is found.
    """
    def _is_run_dir(d: Path) -> bool:
        return (d / "eplusout.csv").exists() or (d / "eplusout.end").exists()

    if _is_run_dir(output_dir):
        return output_dir
    subdirs = sorted(output_dir.glob("energyplus_runs_*"), reverse=True)
    for sd in subdirs:
        if _is_run_dir(sd):
            return sd
    return None


def _load_csv(csv_path: Path) -> tuple[list[str], list[str], np.ndarray]:
    """Parse EnergyPlus hourly CSV into arrays.

    Returns
    -------
    headers    : list[str]   – raw column names (column 0 is Date/Time)
    date_times : list[str]   – N datetime strings from column 0
    data       : np.ndarray  – shape (N, len(headers)-1) float64; NaN for blanks
    """
    rows: list[list[str]] = []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        headers = next(reader)
        for row in reader:
            rows.append(row)

    if not rows:
        return headers, [], np.empty((0, max(len(headers) - 1, 0)), dtype=np.float64)

    date_times = [r[0] for r in rows]
    n_cols = len(headers) - 1
    data = np.full((len(rows), n_cols), np.nan, dtype=np.float64)
    for i, row in enumerate(rows):
        for j in range(n_cols):
            idx = j + 1
            if idx < len(row) and row[idx].strip():
                try:
                    data[i, j] = float(row[idx])
                except ValueError:
                    pass
    return headers, date_times, data


def _monthly_stats(values: np.ndarray, date_times: list[str]) -> list[dict]:
    """Return per-month aggregates.  date_times strings contain 'MM/DD HH:MM:SS'."""
    months: list[dict] = []
    for month in range(1, 13):
        prefix = f"{month:02d}/"
        idxs = [i for i, dt in enumerate(date_times) if dt.strip().startswith(prefix)]
        if not idxs:
            months.append({"month": month, "mean": None, "min": None, "max": None, "total": None})
            continue
        v = values[idxs]
        v_clean = v[~np.isnan(v)]
        if v_clean.size == 0:
            months.append({"month": month, "mean": None, "min": None, "max": None, "total": None})
        else:
            months.append({
                "month": month,
                "mean": round(float(np.mean(v_clean)), 4),
                "min": round(float(np.min(v_clean)), 4),
                "max": round(float(np.max(v_clean)), 4),
                "total": round(float(np.sum(v_clean)), 4),
            })
    return months


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_analysis_tools(output_dir: Path) -> list[BaseTool]:
    """Create 6 read-only analysis tools scoped to *output_dir*.

    Args:
        output_dir: Path to the directory containing EnergyPlus output files.
                    May be the run directory directly, or a parent that contains
                    ``energyplus_runs_*`` subdirectories.
    """
    # CSV cache: avoid re-reading 2.5 MB file for each tool call in one session
    _csv_cache: dict[Path, tuple[list[str], list[str], np.ndarray]] = {}

    def _get_csv() -> tuple[Path | None, list[str], list[str], np.ndarray]:
        run_dir = _find_run_dir(output_dir)
        if run_dir is None:
            return None, [], [], np.empty((0, 0))
        csv_path = run_dir / "eplusout.csv"
        if csv_path not in _csv_cache:
            _csv_cache[csv_path] = _load_csv(csv_path)
        headers, date_times, data = _csv_cache[csv_path]
        return run_dir, headers, date_times, data

    # -- Tool 1 ---------------------------------------------------------------

    @tool
    def get_simulation_status() -> str:
        """Check whether the EnergyPlus simulation completed successfully.

        Reads ``eplusout.end`` (one-line completion summary) and
        ``eplusout.err`` (warnings/severe errors).  Call this FIRST before
        any other analysis tool.

        Returns a JSON object with keys:
          - completed_successfully (bool)
          - warnings (int)
          - severe_errors (int)
          - elapsed_time (str | null)
          - warning_messages (list[str])  – up to 10 samples
          - severe_messages (list[str])   – up to 10 samples
        """
        run_dir = _find_run_dir(output_dir)
        if run_dir is None:
            return _err(
                "No EnergyPlus output directory found. "
                "Simulation may not have run yet.",
                {"output_dir": str(output_dir)},
            )

        # --- .end file -------------------------------------------------------
        end_path = run_dir / "eplusout.end"
        completed = False
        elapsed_time: str | None = None
        warnings_count = 0
        severe_count = 0

        if end_path.exists():
            end_text = end_path.read_text(encoding="utf-8", errors="replace").strip()
            completed = "EnergyPlus Completed Successfully" in end_text
            # Extract warning/error counts from .end line
            m = re.search(r"(\d+)\s+Warning", end_text, re.IGNORECASE)
            if m:
                warnings_count = int(m.group(1))
            m = re.search(r"(\d+)\s+Severe", end_text, re.IGNORECASE)
            if m:
                severe_count = int(m.group(1))
            m = re.search(r"Time=([^\s]+)", end_text)
            if m:
                elapsed_time = m.group(1)

        # --- .err file -------------------------------------------------------
        err_path = run_dir / "eplusout.err"
        warning_messages: list[str] = []
        severe_messages: list[str] = []

        if err_path.exists():
            for line in err_path.read_text(encoding="utf-8", errors="replace").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                low = stripped.lower()
                if re.search(r"\*\*\s*warning\s*\*\*", low):
                    if len(warning_messages) < 10:
                        warning_messages.append(stripped)
                elif re.search(r"\*\*\s*(severe|fatal)\s*\*\*", low):
                    if len(severe_messages) < 10:
                        severe_messages.append(stripped)

        status = "completed successfully" if completed else "did not complete successfully"
        return _ok(
            f"Simulation {status}. {warnings_count} warning(s), {severe_count} severe error(s).",
            {
                "completed_successfully": completed,
                "warnings": warnings_count,
                "severe_errors": severe_count,
                "elapsed_time": elapsed_time,
                "warning_messages": warning_messages,
                "severe_messages": severe_messages,
                "run_dir": str(run_dir),
            },
        )

    # -- Tool 2 ---------------------------------------------------------------

    @tool
    def get_available_variables() -> str:
        """List all output variables recorded in the simulation CSV.

        Reads only the header row (fast, no data loading).  Returns a mapping
        of variable_name → {unit, freq, keys} where ``keys`` is the list of
        zone/surface/system names that reported this variable.

        Call this SECOND to discover what variable names and keys exist before
        calling get_variable_statistics or get_peak_hours.
        """
        run_dir = _find_run_dir(output_dir)
        if run_dir is None:
            return _err("No simulation output found. Run the simulation first.")

        csv_path = run_dir / "eplusout.csv"
        with csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            headers = next(reader)

        variables: dict[str, dict] = {}
        for col in headers[1:]:  # skip Date/Time
            m = _COL_PATTERN.match(col.strip())
            if not m:
                continue
            key, var_name, unit, freq = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
            if var_name not in variables:
                variables[var_name] = {"unit": unit, "freq": freq, "keys": []}
            if key not in variables[var_name]["keys"]:
                variables[var_name]["keys"].append(key)

        return _ok(
            f"Found {len(variables)} distinct output variables across {len(headers)-1} columns.",
            variables,
        )

    # -- Tool 3 ---------------------------------------------------------------

    @tool
    def get_variable_statistics(variable_name: str, include_monthly: bool = True) -> str:
        """Compute annual and monthly statistics for an output variable.

        Loads the full CSV (results are cached for subsequent calls).
        Energy values in Joules are automatically converted to kWh.

        Args:
            variable_name: Exact variable name as returned by get_available_variables
                           (e.g. "Zone Mean Air Temperature").
            include_monthly: When True, include month-by-month aggregates (default True).

        Returns per-key statistics: mean, min, max, total, unit, and optionally
        monthly breakdown.
        """
        run_dir, headers, date_times, data = _get_csv()
        if run_dir is None:
            return _err("No simulation output found. Run the simulation first.")

        # Find all columns matching the variable name
        matched: list[tuple[str, int]] = []  # (key, col_index)
        for col_idx, col in enumerate(headers[1:]):
            m = _COL_PATTERN.match(col.strip())
            if m and m.group(2).strip() == variable_name:
                matched.append((m.group(1), col_idx))

        if not matched:
            return _err(
                f"Variable '{variable_name}' not found in simulation output. "
                "Use get_available_variables to see what is available."
            )

        # Determine unit from first match; auto-convert J → kWh
        first_col_header = headers[matched[0][1] + 1]
        unit_m = _COL_PATTERN.match(first_col_header.strip())
        unit = unit_m.group(3) if unit_m else "unknown"
        convert_to_kwh = unit == "J"
        display_unit = "kWh" if convert_to_kwh else unit

        results: list[dict] = []
        for key, col_idx in matched:
            values = data[:, col_idx].copy()
            if convert_to_kwh:
                values = values * _J_TO_KWH
            clean = values[~np.isnan(values)]
            if clean.size == 0:
                rec: dict = {
                    "key": key,
                    "unit": display_unit,
                    "annual": {"mean": None, "min": None, "max": None, "total": None},
                }
            else:
                rec = {
                    "key": key,
                    "unit": display_unit,
                    "annual": {
                        "mean": round(float(np.mean(clean)), 4),
                        "min": round(float(np.min(clean)), 4),
                        "max": round(float(np.max(clean)), 4),
                        "total": round(float(np.sum(clean)), 4),
                    },
                }
            if include_monthly:
                rec["monthly"] = _monthly_stats(values, date_times)
            results.append(rec)

        return _ok(
            f"Statistics for '{variable_name}' ({display_unit}) across {len(results)} key(s).",
            {"variable_name": variable_name, "unit": display_unit, "results": results},
        )

    # -- Tool 4 ---------------------------------------------------------------

    @tool
    def get_peak_hours(
        variable_name: str,
        key: str,
        n_top: int = 10,
        mode: str = "max",
    ) -> str:
        """Return the top-N peak timesteps for a specific variable and zone/key.

        Args:
            variable_name: Variable name (e.g. "Zone Mean Air Temperature").
            key: Zone/surface/system key (e.g. "ZONE1" or "*" for facility-level).
            n_top: Number of timesteps to return (1–50, default 10).
            mode: "max" for highest values, "min" for lowest values.
        """
        if mode not in ("max", "min"):
            return _err("mode must be 'max' or 'min'.")
        n_top = max(1, min(50, n_top))

        run_dir, headers, date_times, data = _get_csv()
        if run_dir is None:
            return _err("No simulation output found. Run the simulation first.")

        # Find exact column
        col_idx: int | None = None
        display_unit = "unknown"
        convert_to_kwh = False
        for ci, col in enumerate(headers[1:]):
            m = _COL_PATTERN.match(col.strip())
            if m and m.group(2).strip() == variable_name and m.group(1) == key:
                col_idx = ci
                display_unit = m.group(3)
                convert_to_kwh = display_unit == "J"
                if convert_to_kwh:
                    display_unit = "kWh"
                break

        if col_idx is None:
            return _err(
                f"Column '{key}:{variable_name}' not found. "
                "Use get_available_variables to check exact variable names and keys."
            )

        values = data[:, col_idx].copy()
        if convert_to_kwh:
            values = values * _J_TO_KWH

        valid_mask = ~np.isnan(values)
        valid_indices = np.where(valid_mask)[0]
        if valid_indices.size == 0:
            return _ok(f"No valid data for '{variable_name}' key='{key}'.", [])

        valid_values = values[valid_indices]
        if mode == "max":
            order = np.argsort(valid_values)[::-1]
        else:
            order = np.argsort(valid_values)

        top_order = order[:n_top]
        top_indices = valid_indices[top_order]

        peak_list = [
            {
                "datetime": date_times[int(i)].strip(),
                "value": round(float(values[int(i)]), 4),
            }
            for i in top_indices
        ]

        return _ok(
            f"Top {len(peak_list)} {mode} values for '{variable_name}' (key='{key}', unit={display_unit}).",
            {"variable_name": variable_name, "key": key, "unit": display_unit, "mode": mode, "peaks": peak_list},
        )

    # -- Tool 5 ---------------------------------------------------------------

    @tool
    def get_comfort_statistics(
        temp_variable_name: str = "Zone Mean Air Temperature",
        comfort_min_c: float = 20.0,
        comfort_max_c: float = 26.0,
    ) -> str:
        """Compute thermal comfort hours per zone based on air temperature.

        For each zone, counts how many of the 8760 hours fall within the
        comfort band [comfort_min_c, comfort_max_c].  Also provides the
        building-level weighted average.

        Args:
            temp_variable_name: Temperature variable name (default
                "Zone Mean Air Temperature").
            comfort_min_c: Lower comfort threshold in °C (default 20.0).
            comfort_max_c: Upper comfort threshold in °C (default 26.0).
        """
        run_dir, headers, date_times, data = _get_csv()
        if run_dir is None:
            return _err("No simulation output found. Run the simulation first.")

        # Gather all zone temperature columns
        zone_cols: list[tuple[str, int]] = []
        for ci, col in enumerate(headers[1:]):
            m = _COL_PATTERN.match(col.strip())
            if m and m.group(2).strip() == temp_variable_name:
                zone_cols.append((m.group(1), ci))

        if not zone_cols:
            return _err(
                f"Variable '{temp_variable_name}' not found. "
                "Use get_available_variables to check temperature variable names."
            )

        zone_results: list[dict] = []
        total_comfort = 0
        total_hot = 0
        total_cold = 0
        total_hours = 0

        for key, ci in zone_cols:
            temps = data[:, ci]
            valid = ~np.isnan(temps)
            n = int(np.sum(valid))
            if n == 0:
                zone_results.append({"key": key, "total_hours": 0,
                                     "comfort_hours": 0, "hot_hours": 0, "cold_hours": 0,
                                     "comfort_pct": None, "hot_pct": None, "cold_pct": None})
                continue
            v = temps[valid]
            comfort = int(np.sum((v >= comfort_min_c) & (v <= comfort_max_c)))
            hot = int(np.sum(v > comfort_max_c))
            cold = int(np.sum(v < comfort_min_c))
            zone_results.append({
                "key": key,
                "total_hours": n,
                "comfort_hours": comfort,
                "hot_hours": hot,
                "cold_hours": cold,
                "comfort_pct": round(comfort / n * 100, 2),
                "hot_pct": round(hot / n * 100, 2),
                "cold_pct": round(cold / n * 100, 2),
            })
            total_comfort += comfort
            total_hot += hot
            total_cold += cold
            total_hours += n

        building_summary: dict = {}
        if total_hours > 0:
            building_summary = {
                "total_hours": total_hours,
                "comfort_hours": total_comfort,
                "hot_hours": total_hot,
                "cold_hours": total_cold,
                "comfort_pct": round(total_comfort / total_hours * 100, 2),
                "hot_pct": round(total_hot / total_hours * 100, 2),
                "cold_pct": round(total_cold / total_hours * 100, 2),
            }

        return _ok(
            f"Comfort statistics for {len(zone_results)} zone(s). "
            f"Comfort band: [{comfort_min_c}, {comfort_max_c}] °C.",
            {
                "temp_variable": temp_variable_name,
                "comfort_band_c": [comfort_min_c, comfort_max_c],
                "zones": zone_results,
                "building": building_summary,
            },
        )

    # -- Tool 6 ---------------------------------------------------------------

    @tool
    def get_energy_summary() -> str:
        """Parse the EnergyPlus tabular summary for end-use energy breakdown.

        Tries ``eplustbl.htm`` first (HTML format), then falls back to
        ``eplustbl.csv`` (comma format) which EnergyPlus 25.x produces by
        default.  Extracts the **End Uses** table (GJ → kWh) and EUI.

        Returns a dict with keys:
          - end_uses: list of {use, electricity_kwh, natural_gas_kwh, ...}
          - eui_mj_per_m2 (float | null)
          - total_electricity_kwh (float)
          - total_natural_gas_kwh (float)
          - source (str): "htm" or "csv"
        """
        run_dir = _find_run_dir(output_dir)
        if run_dir is None:
            return _err("No simulation output found. Run the simulation first.")

        htm_path = run_dir / "eplustbl.htm"
        csv_path = run_dir / "eplustbl.csv"

        # ---- Try HTML path first -------------------------------------------
        if htm_path.exists():
            html = htm_path.read_text(encoding="utf-8", errors="replace")

            eui: float | None = None
            eui_match = re.search(
                r"Energy Use Intensity.*?<td[^>]*>\s*([\d.]+)\s*</td>",
                html,
                re.IGNORECASE | re.DOTALL,
            )
            if eui_match:
                try:
                    eui = round(float(eui_match.group(1)), 4)
                except ValueError:
                    pass

            table_match = re.search(
                r"End Uses.*?(<table[^>]*>.*?</table>)",
                html,
                re.IGNORECASE | re.DOTALL,
            )

            end_uses: list[dict] = []
            total_elec = 0.0
            total_gas = 0.0

            if table_match:
                table_html = table_match.group(1)
                header_row = re.search(r"<tr[^>]*>(.*?)</tr>", table_html, re.IGNORECASE | re.DOTALL)
                col_names: list[str] = []
                if header_row:
                    col_names = [
                        re.sub(r"<[^>]+>", "", c).strip()
                        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", header_row.group(1), re.IGNORECASE | re.DOTALL)
                    ]
                rows = re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, re.IGNORECASE | re.DOTALL)
                for row_html in rows[1:]:
                    cells = [
                        re.sub(r"<[^>]+>", "", c).strip()
                        for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row_html, re.IGNORECASE | re.DOTALL)
                    ]
                    if not cells or not cells[0]:
                        continue
                    use_name = cells[0]
                    is_total_row = use_name.strip().lower() == "total"
                    row_dict: dict = {"use": use_name}
                    for i, col in enumerate(col_names[1:], start=1):
                        if i < len(cells):
                            try:
                                gj_val = float(cells[i].replace(",", ""))
                                kwh_val = round(gj_val * 277.778, 2)
                            except ValueError:
                                kwh_val = None
                            col_key = re.sub(r"\s+", "_", col.lower())
                            row_dict[f"{col_key}_kwh"] = kwh_val
                            if kwh_val is not None and is_total_row:
                                if "electricity" in col.lower():
                                    total_elec = kwh_val
                                if "gas" in col.lower():
                                    total_gas = kwh_val
                    end_uses.append(row_dict)

            return _ok(
                f"Energy summary parsed from HTM. {len(end_uses)} end-use categories found.",
                {
                    "eui_mj_per_m2": eui,
                    "total_electricity_kwh": round(total_elec, 2),
                    "total_natural_gas_kwh": round(total_gas, 2),
                    "end_uses": end_uses,
                    "source": "htm",
                },
            )

        # ---- Fall back to eplustbl.csv --------------------------------------
        if not csv_path.exists():
            return _err(
                "Neither eplustbl.htm nor eplustbl.csv found. "
                "Energy summary is unavailable; use get_variable_statistics instead.",
                {"run_dir": str(run_dir)},
            )

        try:
            from src.results.parser import parse_tabular as _parse_tabular
            tabular = _parse_tabular(csv_path)
        except Exception as exc:
            return _err(f"Failed to parse eplustbl.csv: {exc}")

        end_uses = tabular.get("end_uses", [])
        total_elec = 0.0
        total_gas = 0.0
        for row in end_uses:
            if row.get("use", "").strip().lower() == "total end uses":
                total_elec = row.get("electricity_kwh") or 0.0
                total_gas = row.get("natural_gas_kwh") or 0.0
                break

        return _ok(
            f"Energy summary parsed from CSV. {len(end_uses)} end-use categories found.",
            {
                "eui_mj_per_m2": tabular.get("eui_mj_per_m2"),
                "total_electricity_kwh": round(total_elec, 2),
                "total_natural_gas_kwh": round(total_gas, 2),
                "end_uses": end_uses,
                "source": "csv",
            },
        )

    return [
        get_simulation_status,
        get_available_variables,
        get_variable_statistics,
        get_peak_hours,
        get_comfort_statistics,
        get_energy_summary,
    ]
