from pathlib import Path
import sqlite3
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
from src.database.datatools.datadescription import update_description_construction

def _fetch_id(cursor, mat_name):
    if mat_name is None:
        return None
    cursor.execute("SELECT id FROM all_materials WHERE name=?", (mat_name,))
    res = cursor.fetchone()
    if res:
        return res['id']
    raise ValueError(f"Material '{mat_name}' not found in all_materials.")

def create_construction(db_path: str,
                        name: str,
                        latitude: float,
                        longitude: float,
                        architecture_type: str,
                        layer1: str,
                        layer2: Optional[str] = None,
                        layer3: Optional[str] = None,
                        layer4: Optional[str] = None,
                        layer5: Optional[str] = None,
                        layer6: Optional[str] = None,
                        layer7: Optional[str] = None,
                        layer8: Optional[str] = None,
                        layer9: Optional[str] = None,
                        layer10: Optional[str] = None,
                        layer11: Optional[str] = None,
                        layer12: Optional[str] = None,
                        layer13: Optional[str] = None,
                        layer14: Optional[str] = None,
                        layer15: Optional[str] = None,
                        layer16: Optional[str] = None,
                        layer17: Optional[str] = None,
                        layer18: Optional[str] = None,
                        layer19: Optional[str] = None,
                        layer20: Optional[str] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
  
    table_name = "constructions"

    sql = f"INSERT INTO {table_name} (name, latitude, longitude, architecture_type, layer_1, layer_2, layer_3, layer_4, layer_5, layer_6, layer_7, layer_8, layer_9, layer_10, layer_11, layer_12, layer_13, layer_14, layer_15, layer_16, layer_17, layer_18, layer_19, layer_20, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
    try:
        lay1 = _fetch_id(cursor, layer1)
        lay2 = _fetch_id(cursor, layer2) if layer2 else None
        lay3 = _fetch_id(cursor, layer3) if layer3 else None
        lay4 = _fetch_id(cursor, layer4) if layer4 else None
        lay5 = _fetch_id(cursor, layer5) if layer5 else None
        lay6 = _fetch_id(cursor, layer6) if layer6 else None
        lay7 = _fetch_id(cursor, layer7) if layer7 else None
        lay8 = _fetch_id(cursor, layer8) if layer8 else None
        lay9 = _fetch_id(cursor, layer9) if layer9 else None
        lay10 = _fetch_id(cursor, layer10) if layer10 else None
        lay11 = _fetch_id(cursor, layer11) if layer11 else None
        lay12 = _fetch_id(cursor, layer12) if layer12 else None
        lay13 = _fetch_id(cursor, layer13) if layer13 else None
        lay14 = _fetch_id(cursor, layer14) if layer14 else None
        lay15 = _fetch_id(cursor, layer15) if layer15 else None
        lay16 = _fetch_id(cursor, layer16) if layer16 else None
        lay17 = _fetch_id(cursor, layer17) if layer17 else None
        lay18 = _fetch_id(cursor, layer18) if layer18 else None
        lay19 = _fetch_id(cursor, layer19) if layer19 else None
        lay20 = _fetch_id(cursor, layer20) if layer20 else None

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)
        des_data = [name, latitude, longitude, architecture_type, lay1, lay2, lay3, lay4, lay5, lay6, lay7, lay8, lay9, lay10, lay11, lay12, lay13, lay14, lay15, lay16, lay17, lay18, lay19, lay20, timestamp_int]
        cursor.execute(sql, des_data)
        new_id = cursor.lastrowid
        des_data.insert(0, new_id)
        update_description_construction(des_data[:-1], cur=cursor)

        conn.commit()
    finally:
        conn.close()


def update_construction(db_path: str,
                        construction_id: int,
                        name: Optional[str] = None,
                        latitude: Optional[float] = None,
                        longitude: Optional[float] = None,
                        architecture_type: Optional[str] = None,
                        layer1: Optional[str] = None,
                        layer2: Optional[str] = None,
                        layer3: Optional[str] = None,
                        layer4: Optional[str] = None,
                        layer5: Optional[str] = None,
                        layer6: Optional[str] = None,
                        layer7: Optional[str] = None,
                        layer8: Optional[str] = None,
                        layer9: Optional[str] = None,
                        layer10: Optional[str] = None,
                        layer11: Optional[str] = None,
                        layer12: Optional[str] = None,
                        layer13: Optional[str] = None,
                        layer14: Optional[str] = None,
                        layer15: Optional[str] = None,
                        layer16: Optional[str] = None,
                        layer17: Optional[str] = None,
                        layer18: Optional[str] = None,
                        layer19: Optional[str] = None,
                        layer20: Optional[str] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM constructions WHERE id=?", (construction_id,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"No construction found with id {construction_id}")
    
        def fetch_id(mat_name):
            if mat_name is None: return None
            cursor.execute("SELECT id FROM all_materials WHERE name=?", (mat_name,))
            res = cursor.fetchone()
            if res: return res['id']
            raise ValueError(f"Material {mat_name} not found.")

        updated_name = name if name is not None else row['name']
        updated_lat = latitude if latitude is not None else row['latitude']
        updated_lon = longitude if longitude is not None else row['longitude']
        updated_arch = architecture_type if architecture_type is not None else row['architecture_type']
        updated_ly1 = fetch_id(layer1) if layer1 is not None else row['layer_1']
        updated_ly2 = fetch_id(layer2) if layer2 is not None else row['layer_2']
        updated_ly3 = fetch_id(layer3) if layer3 is not None else row['layer_3']
        updated_ly4 = fetch_id(layer4) if layer4 is not None else row['layer_4']
        updated_ly5 = fetch_id(layer5) if layer5 is not None else row['layer_5']
        updated_ly6 = fetch_id(layer6) if layer6 is not None else row['layer_6']
        updated_ly7 = fetch_id(layer7) if layer7 is not None else row['layer_7']
        updated_ly8 = fetch_id(layer8) if layer8 is not None else row['layer_8']
        updated_ly9 = fetch_id(layer9) if layer9 is not None else row['layer_9']
        updated_ly10 = fetch_id(layer10) if layer10 is not None else row['layer_10']
        updated_ly11 = fetch_id(layer11) if layer11 is not None else row['layer_11']
        updated_ly12 = fetch_id(layer12) if layer12 is not None else row['layer_12']
        updated_ly13 = fetch_id(layer13) if layer13 is not None else row['layer_13']
        updated_ly14 = fetch_id(layer14) if layer14 is not None else row['layer_14']
        updated_ly15 = fetch_id(layer15) if layer15 is not None else row['layer_15']
        updated_ly16 = fetch_id(layer16) if layer16 is not None else row['layer_16']
        updated_ly17 = fetch_id(layer17) if layer17 is not None else row['layer_17']
        updated_ly18 = fetch_id(layer18) if layer18 is not None else row['layer_18']
        updated_ly19 = fetch_id(layer19) if layer19 is not None else row['layer_19']
        updated_ly20 = fetch_id(layer20) if layer20 is not None else row['layer_20'] 
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)
   
        sql = """
            UPDATE constructions 
            SET name = ?, latitude = ?, longitude = ?, architecture_type = ?, 
                layer_1 = ?, layer_2 = ?, layer_3 = ?, layer_4 = ?, layer_5 = ?, 
                layer_6 = ?, layer_7 = ?, layer_8 = ?, layer_9 = ?, layer_10 = ?, 
                layer_11 = ?, layer_12 = ?, layer_13 = ?, layer_14 = ?, layer_15 = ?, 
                layer_16 = ?, layer_17 = ?, layer_18 = ?, layer_19 = ?, layer_20 = ?, datetime = ?
            WHERE id = ?
        """
        dt = [
            updated_name, updated_lat, updated_lon, updated_arch,
            updated_ly1, updated_ly2, updated_ly3, updated_ly4, updated_ly5,
            updated_ly6, updated_ly7, updated_ly8, updated_ly9, updated_ly10,
            updated_ly11, updated_ly12, updated_ly13, updated_ly14, updated_ly15,
            updated_ly16, updated_ly17, updated_ly18, updated_ly19, updated_ly20,
            timestamp_int,
            construction_id
        ]
        cursor.execute(sql, dt)
        des_data = [construction_id, updated_name, updated_lat, updated_lon, updated_arch,
                    updated_ly1, updated_ly2, updated_ly3, updated_ly4, updated_ly5,
                    updated_ly6, updated_ly7, updated_ly8, updated_ly9, updated_ly10,
                    updated_ly11, updated_ly12, updated_ly13, updated_ly14, updated_ly15,
                    updated_ly16, updated_ly17, updated_ly18, updated_ly19, updated_ly20]
        
        update_description_construction(des_data, cur=cursor)
        conn.commit()
    finally:
        conn.close()
    
def delete_construction(db_path: str, construction_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM constructions WHERE id = ?", (construction_id,))
        conn.commit()
    finally:
        conn.close()

def list_constructions(db_path: str) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM constructions")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()
