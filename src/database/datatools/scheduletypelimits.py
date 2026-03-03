from pathlib import Path
import sqlite3
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
from src.database.datatools.datadescription import update_description_schedule_type_limits

def create_schedule_type_limits(db_path: str,
                                name: str,
                                latitude: float,
                                longitude: float,
                                architecture_type: str,
                                lower_limit_value: float,
                                upper_limit_value: float,
                                numeric_type: str,
                                unit_type: str) -> None:
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        table_name = "schedule_type_limits"

        sql = f"INSERT INTO {table_name} (name, latitude, longitude, architecture_type, lower_limit_value, upper_limit_value, numeric_type, unit_type, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"

        des_data = [name, latitude, longitude, architecture_type, lower_limit_value, upper_limit_value, numeric_type, unit_type]

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

    update_description_schedule_type_limits(db_path, des_data)

def update_schedule_type_limits(db_path: str,
                                schedule_type_limits_id: int,
                                name: Optional[str] = None,
                                latitude: Optional[float] = None,
                                longitude: Optional[float] = None,
                                architecture_type: Optional[str] = None,
                                lower_limit_value: Optional[float] = None,
                                upper_limit_value: Optional[float] = None,
                                numeric_type: Optional[str] = None,
                                unit_type: Optional[str] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schedule_type_limits WHERE id = ?", (schedule_type_limits_id,))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"Schedule Type Limits with id {schedule_type_limits_id} does not exist.")
        updated_name = name if name is not None else row['name']
        updated_latitude = latitude if latitude is not None else row['latitude']
        updated_longitude = longitude if longitude is not None else row['longitude']
        updated_architecture_type = architecture_type if architecture_type is not None else row['architecture_type']
        updated_lower_limit_value = lower_limit_value if lower_limit_value is not None else row['lower_limit_value']
        updated_upper_limit_value = upper_limit_value if upper_limit_value is not None else row['upper_limit_value']
        updated_numeric_type = numeric_type if numeric_type is not None else row['numeric_type']
        updated_unit_type = unit_type if unit_type is not None else row['unit_type']

        sql = """
            UPDATE schedule_type_limits 
            SET name = ?, latitude = ?, longitude = ?, architecture_type = ?, 
                lower_limit_value = ?, upper_limit_value = ?, numeric_type = ?, unit_type = ?, datetime = ?
            WHERE id = ?
        """

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)

        dt = [
            updated_name, updated_latitude, updated_longitude, updated_architecture_type,
            updated_lower_limit_value, updated_upper_limit_value, updated_numeric_type, updated_unit_type,
            timestamp_int,
            schedule_type_limits_id
        ]

        cursor.execute(sql, dt)
    
        conn.commit()
    finally:
        conn.close()
    des_data = [
        schedule_type_limits_id, 
        updated_name, 
        updated_latitude, 
        updated_longitude, 
        updated_architecture_type,
        updated_lower_limit_value, 
        updated_upper_limit_value, 
        updated_numeric_type, 
        updated_unit_type
    ]
    update_description_schedule_type_limits(db_path, des_data) 

def delete_scheduletypelimits(db_path: str, scheduletypelimits_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM schedule_type_limits WHERE id = ?", (scheduletypelimits_id,))
        conn.commit()
    finally:
        conn.close()

def list_schedule_type_limits(db_path: str) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM schedule_type_limits")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()
