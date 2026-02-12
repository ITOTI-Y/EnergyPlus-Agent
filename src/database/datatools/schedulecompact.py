from pathlib import Path
import sqlite3
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
from src.database.datatools.datadescription import update_description_schedule_compact

def create_schedule_compact(db_path: str,
                             name: str,
                             latitude: float,
                             longitude: float,
                             architecture_type: str,
                             schedule_type_limit_name: str,
                             compact_values: list[str]) -> None:
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        field_cols = ", ".join([f"field_{i}" for i in range(1, 201)])
        placeholders = ", ".join(["?"] * 200)
    
        sql = f"""
                INSERT INTO schedule_compact (
                    name, latitude, longitude, architecture_type,
                    schedule_type_limit_name,
                    {field_cols},
                    datetime
                ) VALUES (?, ?, ?, ?, ?, {placeholders}, ?)
            """
        full_compact_values = (compact_values + [None] * 200)[:200]
        des_data = [name, latitude, longitude, architecture_type, schedule_type_limit_name] + full_compact_values

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)

        dt = des_data.copy()
        dt.append(timestamp_int)

        cursor.execute(sql, dt)
        new_id = cursor.lastrowid
        des_data.insert(0, new_id)
    
        conn.commit()
    finally:
        conn.close()

    update_description_schedule_compact(db_path, des_data)

def update_schedule_compact(db_path: str,
                             schedule_compact_id: int,
                             name: Optional[str] = None,
                             latitude: Optional[float] = None,
                             longitude: Optional[float] = None,
                             architecture_type: Optional[str] = None,
                             schedule_type_limit_name: Optional[str] = None,
                             compact_values: Optional[list[str]] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM schedule_compact WHERE id = ?", (schedule_compact_id,))
    row = cursor.fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Schedule Compact with ID {schedule_compact_id} not found.")

    sql = """
        UPDATE schedule_compact 
        SET name = ?, latitude = ?, longitude = ?, architecture_type = ?, 
            schedule_type_limit_name = ?, 
            """ + ", ".join([f"field_{i} = ?" for i in range(1, 201)]) + """,
            datetime = ?
        WHERE id = ?
    """
    dt = [
        name if name is not None else row['name'],
        latitude if latitude is not None else row['latitude'],
        longitude if longitude is not None else row['longitude'],
        architecture_type if architecture_type is not None else row['architecture_type'],
        schedule_type_limit_name if schedule_type_limit_name is not None else row['schedule_type_limit_name'],
    ]
    if compact_values is not None:
        dt.extend((compact_values + [None] * 200)[:200])
    else:
        dt.extend([row[f'field_{i}'] for i in range(1, 201)])
    
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M")
    timestamp_int = int(timestamp)
    dt.append(timestamp_int)
    dt.append(schedule_compact_id)

    cursor.execute(sql, dt)
    conn.commit()
    conn.close()

    update_description_schedule_compact(db_path, [schedule_compact_id] + dt[:-2])

def delete_schedulecompact(db_path: str, schedule_compact_id: int) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_name = "schedule_compact"
    sql = f"DELETE FROM {table_name} WHERE id = ?"

    cursor.execute(sql, (schedule_compact_id,))

    conn.commit()
    conn.close()

def list_schedule_compact(db_path: str) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    table_name = "schedule_compact"

    sql = f"SELECT * FROM {table_name}"

    cursor.execute(sql)
    rows = cursor.fetchall()

    conn.close()
    return rows
