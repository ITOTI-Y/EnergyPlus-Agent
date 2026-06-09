"""EnergyPlus simulation result parser.

Parses eplusout.csv (hourly timeseries) and eplustbl.csv (annual tabular
summary) into structured Python objects suitable for visualization.

Public API
----------
load_results(output_dir)  -> SimulationResult
parse_timeseries(csv_path) -> pd.DataFrame
parse_tabular(csv_path)   -> dict
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# EnergyPlus CSV column format:  KEY:VariableName [Unit](Freq)
_COL_PATTERN = re.compile(r"^(.+?):(.+?)\s+\[(.+?)\]\((.+?)\)$")
_J_TO_KWH = 1.0 / 3_600_000.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SimulationResult:
    """Container for a completed EnergyPlus simulation run."""

    timeseries: pd.DataFrame
    """8760-row DataFrame.  MultiIndex columns: (zone_key, variable, unit).
    Energy columns originally in Joules are converted to kWh."""

    tabular: dict
    """Parsed eplustbl.csv data:
      building_name, building_area_m2, eui_mj_per_m2,
      end_uses: list[dict], site_energy: dict"""

    run_dir: Path
    """Directory that contains the EnergyPlus output files."""

    idf_path: Path | None = field(default=None)
    """Path to the IDF file used for this run (may be None)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_run_dir(output_dir: Path) -> Path | None:
    """Return the subdirectory (or output_dir itself) that holds eplusout.csv."""
    def _is_run_dir(d: Path) -> bool:
        return (d / "eplusout.csv").exists() or (d / "eplusout.end").exists()

    if _is_run_dir(output_dir):
        return output_dir
    for sd in sorted(output_dir.glob("energyplus_runs_*"), reverse=True):
        if _is_run_dir(sd):
            return sd
    return None


def _find_idf(run_dir: Path) -> Path | None:
    """Return the most recent temp_*.idf in the run directory."""
    idfs = sorted(run_dir.glob("temp_*.idf"), reverse=True)
    return idfs[0] if idfs else None


# ---------------------------------------------------------------------------
# timeseries parser
# ---------------------------------------------------------------------------


def parse_timeseries(csv_path: Path) -> pd.DataFrame:
    """Parse eplusout.csv into a tidy DataFrame.

    Columns become a 3-level MultiIndex: (zone_key, variable_name, unit).
    Energy values in Joules are automatically converted to kWh (unit becomes
    'kWh').  The datetime strings from column 0 are used as the index.

    Parameters
    ----------
    csv_path:
        Path to ``eplusout.csv``.

    Returns
    -------
    pd.DataFrame with shape (N_timesteps, N_variables).
    """
    raw = pd.read_csv(csv_path, index_col=0, low_memory=False)
    raw.index.name = "datetime"

    tuples: list[tuple[str, str, str]] = []
    rename: dict[str, tuple] = {}

    for col in raw.columns:
        m = _COL_PATTERN.match(col.strip())
        if m:
            key, var, unit, _freq = m.group(1), m.group(2).strip(), m.group(3), m.group(4)
            if unit == "J":
                unit = "kWh"
            tuples.append((key, var, unit))
            rename[col] = (key, var, unit)
        else:
            tuples.append((col, col, ""))
            rename[col] = (col, col, "")

    raw = raw.rename(columns=rename)
    raw.columns = pd.MultiIndex.from_tuples(tuples, names=["zone_key", "variable", "unit"])

    # Convert J → kWh
    for col in raw.columns:
        _key, _var, unit = col
        if unit == "kWh":
            raw[col] = pd.to_numeric(raw[col], errors="coerce") * _J_TO_KWH
        else:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    return raw


# ---------------------------------------------------------------------------
# tabular summary parser
# ---------------------------------------------------------------------------


def parse_tabular(csv_path: Path) -> dict:
    """Parse eplustbl.csv into a structured dict.

    Returns a dict with keys:
      building_name   (str)
      building_area_m2 (float | None)
      eui_mj_per_m2   (float | None)
      site_energy     (dict: total/net site/source GJ)
      end_uses        (list[dict]: use → fuel_kwh values)
    """
    text = csv_path.read_text(encoding="utf-8", errors="replace")
    lines = [ln.rstrip("\n") for ln in text.splitlines()]

    result: dict = {
        "building_name": None,
        "building_area_m2": None,
        "eui_mj_per_m2": None,
        "site_energy": {},
        "end_uses": [],
        "zone_summary": {},  # ZONE_NAME -> {area_m2, multiplier, volume_m3, ...}
    }

    # ---- building name -------------------------------------------------------
    for ln in lines:
        if ln.startswith("Building:,"):
            result["building_name"] = ln.split(",", 2)[1].strip()
            break

    # ---- scan sections -------------------------------------------------------
    i = 0
    while i < len(lines):
        ln = lines[i]

        # Zone Summary table (PERFORMANCE section)
        if ln.strip() == "Zone Summary":
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            header_parts: list[str] = []
            if j < len(lines):
                header_parts = [p.strip() for p in lines[j].split(",")]
            j += 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("REPORT"):
                parts = [p.strip() for p in lines[j].split(",")]
                if len(parts) >= 3 and parts[1] and not parts[1].startswith("Total"):
                    zname = parts[1].upper().replace(" ", "_")
                    row: dict = {"name": parts[1]}
                    # Typical columns: Area, Conditioned, PartOfTotal, Volume, Multipliers, ...
                    if len(parts) > 2:
                        try:
                            row["area_m2"] = float(parts[2])
                        except ValueError:
                            pass
                    if len(parts) > 6:
                        try:
                            row["multiplier"] = float(parts[6])
                        except ValueError:
                            pass
                    if len(parts) > 5:
                        try:
                            row["volume_m3"] = float(parts[5])
                        except ValueError:
                            pass
                    result["zone_summary"][zname] = row
                j += 1

        # Building Area table
        if ln.strip() == "Building Area":
            # skip blanks to find header, then advance to data rows
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            j += 1  # skip header row
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("REPORT"):
                parts = [p.strip() for p in lines[j].split(",")]
                if len(parts) >= 3 and parts[1] == "Total Building Area":
                    try:
                        result["building_area_m2"] = float(parts[2])
                    except ValueError:
                        pass
                j += 1

        # Site and Source Energy table
        elif ln.strip() == "Site and Source Energy":
            # Skip blank lines to find the header row
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            header_parts: list[str] = []
            if j < len(lines):
                header_parts = [p.strip() for p in lines[j].split(",")]
            # Find which column index holds "Energy Per Total Building Area"
            eui_col: int | None = None
            for ci, hp in enumerate(header_parts):
                if "per total building area" in hp.lower():
                    eui_col = ci
                    break
            j += 1  # move past header to first data row
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("REPORT"):
                parts = [p.strip() for p in lines[j].split(",")]
                if len(parts) >= 3 and parts[1]:
                    try:
                        result["site_energy"][parts[1]] = float(parts[2])
                    except ValueError:
                        pass
                    # Extract EUI from "Total Site Energy" row
                    if (
                        result["eui_mj_per_m2"] is None
                        and parts[1].strip().lower() == "total site energy"
                        and eui_col is not None
                        and eui_col < len(parts)
                    ):
                        try:
                            result["eui_mj_per_m2"] = float(parts[eui_col])
                        except ValueError:
                            pass
                j += 1

        # End Uses table — only the first occurrence (before "End Uses By Subcategory")
        elif ln.strip() == "End Uses" and "subcategory" not in lines[i + 1].lower() if i + 1 < len(lines) else True:
            # next non-empty line is header
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                i += 1
                continue
            header_parts = [p.strip() for p in lines[j].split(",")]
            # header_parts[0] and [1] are empty; fuel columns start at [2]
            fuel_cols = header_parts[2:]
            # strip units from fuel column names, e.g. "Electricity [GJ]" → "electricity"
            fuel_names = [
                re.sub(r"\s*\[.*?\]", "", fc).strip().lower().replace(" ", "_")
                for fc in fuel_cols
            ]
            j += 1
            end_uses: list[dict] = []
            while j < len(lines) and lines[j].strip() and not lines[j].startswith("REPORT"):
                parts = [p.strip() for p in lines[j].split(",")]
                # parts[0] empty, parts[1] = use name, parts[2:] = values
                if len(parts) >= 3 and parts[1]:
                    use_name = parts[1]
                    row: dict = {"use": use_name}
                    for fi, fname in enumerate(fuel_names):
                        val_idx = fi + 2
                        if val_idx < len(parts):
                            try:
                                gj = float(parts[val_idx])
                                row[f"{fname}_kwh"] = round(gj * 277.778, 2)
                            except ValueError:
                                row[f"{fname}_kwh"] = None
                        else:
                            row[f"{fname}_kwh"] = None
                    end_uses.append(row)
                j += 1
            if end_uses:
                result["end_uses"] = end_uses

        i += 1

    return result


# ---------------------------------------------------------------------------
# Top-level loader
# ---------------------------------------------------------------------------


def load_results(output_dir: Path) -> SimulationResult:
    """Load all available simulation results from *output_dir*.

    Searches for eplusout.csv, eplustbl.csv, and the latest temp_*.idf.
    Returns a :class:`SimulationResult` with all available data populated.

    Raises
    ------
    FileNotFoundError
        If no EnergyPlus output directory is found under *output_dir*.
    """
    run_dir = _find_run_dir(output_dir)
    if run_dir is None:
        raise FileNotFoundError(
            f"No EnergyPlus output found in {output_dir}. "
            "Run the simulation first."
        )

    csv_path = run_dir / "eplusout.csv"
    tbl_path = run_dir / "eplustbl.csv"
    idf_path = _find_idf(run_dir)

    ts = parse_timeseries(csv_path) if csv_path.exists() else pd.DataFrame()
    tabular = parse_tabular(tbl_path) if tbl_path.exists() else {}

    return SimulationResult(
        timeseries=ts,
        tabular=tabular,
        run_dir=run_dir,
        idf_path=idf_path,
    )
