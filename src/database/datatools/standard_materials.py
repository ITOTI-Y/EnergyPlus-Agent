from pathlib import Path
import sqlite3
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
from src.database.datatools.datadescription import update_description_material, update_description_all_materials
  
def create_standard_materials(db_path: str,
                              name: str,
                              latitude: float,
                              longitude: float,
                              architecture_type: str,
                              roughness: str,
                              thickness: float,
                              conductivity: float,
                              density: float,
                              specific_heat: float,
                              thermal_absorptance: Optional[float] = None,
                              solar_absorptance: Optional[float] = None,
                              visible_absorptance: Optional[float] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        table_name = "standard_materials"
        sql = f"INSERT INTO {table_name} (name, latitude, longitude, architecture_type, roughness, thickness, conductivity, density, specific_heat, thermal_absorptance, solar_absorptance, visible_absorptance, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        des_data = [name, latitude, longitude, architecture_type, roughness, thickness, conductivity, density, specific_heat, thermal_absorptance, solar_absorptance, visible_absorptance]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)

        dt = des_data.copy()
        dt.append(timestamp_int)

        cursor.execute(sql, dt)
        new_id = cursor.lastrowid
        des_data.insert(0, new_id)

        cursor.execute("""
            INSERT INTO all_materials (
                name, material_type, standard_material_id, no_mass_material_id
            ) VALUES (?, 'Mass', ?, NULL)
        """, (
            name,
            new_id
        ))

        new_am_id = cursor.lastrowid
    
        conn.commit()
    finally:
        conn.close()

    update_description_material(db_path, des_data)
    update_description_all_materials(db_path, [new_am_id, name, 'Mass', new_id, None])
    

def update_standard_material(db_path: str,
                             material_id: int,
                             name: Optional[str] = None,
                             latitude: Optional[float] = None,
                             longitude: Optional[float] = None,
                             architecture_type: Optional[str] = None,
                             roughness: Optional[str] = None,
                             thickness: Optional[float] = None,
                             conductivity: Optional[float] = None,
                             density: Optional[float] = None,
                             specific_heat: Optional[float] = None,
                             thermal_absorptance: Optional[float] = None,
                             solar_absorptance: Optional[float] = None,
                             visible_absorptance: Optional[float] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM standard_materials WHERE id = ?", (material_id,))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"Material with id {material_id} does not exist.")
        updated_name = name if name is not None else row['name']
        updated_lat = latitude if latitude is not None else row['latitude']
        updated_lon = longitude if longitude is not None else row['longitude']
        updated_arch = architecture_type if architecture_type is not None else row['architecture_type']
        updated_rough = roughness if roughness is not None else row['roughness']
        updated_thick = thickness if thickness is not None else row['thickness']
        updated_cond = conductivity if conductivity is not None else row['conductivity']
        updated_dens = density if density is not None else row['density']
        updated_spec = specific_heat if specific_heat is not None else row['specific_heat']
        updated_ther = thermal_absorptance if thermal_absorptance is not None else row['thermal_absorptance']
        updated_solr = solar_absorptance if solar_absorptance is not None else row['solar_absorptance']
        updated_visb = visible_absorptance if visible_absorptance is not None else row['visible_absorptance']

        sql = """
            UPDATE standard_materials 
            SET name = ?, latitude = ?, longitude = ?, architecture_type = ?, 
                roughness = ?, thickness = ?, conductivity = ?, density = ?, 
                specific_heat = ?, thermal_absorptance = ?, solar_absorptance = ?, 
                visible_absorptance = ?, datetime = ?
            WHERE id = ?
        """

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)

        dt = [
            updated_name, updated_lat, updated_lon, updated_arch,
            updated_rough, updated_thick, updated_cond, updated_dens,
            updated_spec, updated_ther, updated_solr, updated_visb,
            timestamp_int,
            material_id
        ]
        cursor.execute(sql, dt)

        if name is not None:
            cursor.execute("SELECT id FROM all_materials WHERE standard_material_id = ?", (material_id,))
            am_row = cursor.fetchone()

            if am_row is None:
                conn.close()
                raise ValueError(f"No all_materials entry found for standard_material_id {material_id}")
            am_id = am_row['id']
            cursor.execute("UPDATE all_materials SET name = ? WHERE id = ?", (name, am_id))
            cursor.execute("UPDATE all_materials SET datetime = ? WHERE id = ?", (timestamp_int, am_id))

        conn.commit()
    finally:
        conn.close()
    des_data = [material_id] + dt[:-2] 
    update_description_material(db_path, des_data)
    if name is not None:
        update_description_all_materials(db_path, [am_id, name, 'Mass', material_id, None])


def delete_standard_material(db_path: str, material_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM standard_materials WHERE id = ?", (material_id,))
        cursor.execute("DELETE FROM all_materials WHERE standard_material_id = ?", (material_id,))
        conn.commit()
    finally:
        conn.close()

def list_standard_materials(db_path: str) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM standard_materials")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()
