from pathlib import Path
import sqlite3
import csv
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
from src.database.datatools.datadescription import update_description_sizingperiod_designday

def create_sizingperiod_designday(db_path: str,
                                  name: str,
                                  latitude: float,
                                  longitude: float,
                                  architecture_type: str,
                                  month: int,
                                  day_of_month: int,
                                  day_type: str,
                                  wind_speed: float,
                                  wind_direction: float,
                                  max_dry_bulb_temp: Optional[float] = None,
                                  daily_dry_bulb_range: Optional[float] = None,
                                  dry_bulb_temp_range_modifier_type: Optional[str] = None,
                                  dry_bulb_temp_range_modifier_day_schedule_name: Optional[str] = None,
                                  humidity_condition_type: Optional[str] = None,
                                  wetbulb_or_dewpoint_at_maximum_drybulb: Optional[str] = None,
                                  humidity_condition_day_schedule_name: Optional[str] = None,
                                  humidity_ratio_at_maximum_drybulb: Optional[float] = None,
                                  enthalpy_at_maximum_drybulb: Optional[float] = None,
                                  daily_wetbulb_temperature_range: Optional[float] = None,
                                  barometric_pressure: Optional[float] = None,
                                  rain_indicator: Optional[str] = None,
                                  snow_indicator: Optional[str] = None,
                                  daylight_saving_time_indicator: Optional[str] = None,
                                  solar_model_indicator: Optional[str] = None,
                                  beam_solar_day_schedule_name: Optional[str] = None,
                                  diffuse_solar_day_schedule_name: Optional[str] = None,
                                  ashrae_clear_sky_optical_depth_for_beam_irradiance_taub: Optional[float] = None,
                                  ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud: Optional[float] = None,
                                  sky_clearness: Optional[float] = None,
                                  maximum_number_warmup_days: Optional[int] = None,
                                  begin_environment_reset_mode: Optional[str] = None) -> None:
    
    
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()

        table_name = "sizingperiod_designday"

        sql = f"INSERT INTO {table_name} (name, latitude, longitude, architecture_type, month, day_of_month, day_type, max_drybulb_temperature, daily_drybulb_temperature_range, drybulb_temperature_range_modifier_type, drybulb_temperature_range_modifier_day_schedule_name, humidity_condition_type, wetbulb_or_dewpoint_at_maximum_drybulb, humidity_condition_day_schedule_name, humidity_ratio_at_maximum_drybulb, enthalpy_at_maximum_drybulb, daily_wetbulb_temperature_range, barometric_pressure, wind_speed, wind_direction, rain_indicator, snow_indicator, daylight_saving_time_indicator, solar_model_indicator, beam_solar_day_schedule_name, diffuse_solar_day_schedule_name, ashrae_clear_sky_optical_depth_for_beam_irradiance_taub, ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud, sky_clearness, maximum_number_warmup_days, begin_environment_reset_mode, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        des_data = [name,
                    latitude,
                    longitude,
                    architecture_type,
                    month,
                    day_of_month,
                    day_type,
                    max_dry_bulb_temp,
                    daily_dry_bulb_range,
                    dry_bulb_temp_range_modifier_type,
                    dry_bulb_temp_range_modifier_day_schedule_name,
                    humidity_condition_type,
                    wetbulb_or_dewpoint_at_maximum_drybulb,
                    humidity_condition_day_schedule_name,
                    humidity_ratio_at_maximum_drybulb,
                    enthalpy_at_maximum_drybulb,
                    daily_wetbulb_temperature_range,
                    barometric_pressure,
                    wind_speed,
                    wind_direction,
                    rain_indicator,
                    snow_indicator,
                    daylight_saving_time_indicator,
                    solar_model_indicator,
                    beam_solar_day_schedule_name,
                    diffuse_solar_day_schedule_name,
                    ashrae_clear_sky_optical_depth_for_beam_irradiance_taub,
                    ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud,
                    sky_clearness,
                    maximum_number_warmup_days,
                    begin_environment_reset_mode]

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)
        des_data.append(timestamp_int)
        cursor.execute(sql, des_data)
        new_id = cursor.lastrowid
        des_data.insert(0, new_id)

        update_description_sizingperiod_designday(des_data[:-1], cur=cursor)
        conn.commit()
    finally:
        conn.close()


def update_sizingperiod_designday(db_path: str,
                                  designday_id: int,
                                  name: Optional[str] = None,
                                  latitude: Optional[float] = None,
                                  longitude: Optional[float] = None,
                                  architecture_type: Optional[str] = None,
                                  month: Optional[int] = None,
                                  day_of_month: Optional[int] = None,
                                  day_type: Optional[str] = None,
                                  wind_speed: Optional[float] = None,
                                  wind_direction: Optional[float] = None,
                                  max_dry_bulb_temp: Optional[float] = None,
                                  daily_dry_bulb_range: Optional[float] = None,
                                  dry_bulb_temp_range_modifier_type: Optional[str] = None,
                                  dry_bulb_temp_range_modifier_day_schedule_name: Optional[str] = None,
                                  humidity_condition_type: Optional[str] = None,
                                  wetbulb_or_dewpoint_at_maximum_drybulb: Optional[str] = None,
                                  humidity_condition_day_schedule_name: Optional[str] = None,
                                  humidity_ratio_at_maximum_drybulb: Optional[float] = None,
                                  enthalpy_at_maximum_drybulb: Optional[float] = None,
                                  daily_wetbulb_temperature_range: Optional[float] = None,
                                  barometric_pressure: Optional[float] = None,
                                  rain_indicator: Optional[str] = None,
                                  snow_indicator: Optional[str] = None,
                                  daylight_saving_time_indicator: Optional[str] = None,
                                  solar_model_indicator: Optional[str] = None,
                                  beam_solar_day_schedule_name: Optional[str] = None,
                                  diffuse_solar_day_schedule_name: Optional[str] = None,
                                  ashrae_clear_sky_optical_depth_for_beam_irradiance_taub: Optional[float] = None,
                                  ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud: Optional[float] = None,
                                  sky_clearness: Optional[float] = None,
                                  maximum_number_warmup_days: Optional[int] = None,
                                  begin_environment_reset_mode: Optional[str] = None) -> None:
    
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sizingperiod_designday WHERE id = ?", (designday_id,))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"SizingPeriod:DesignDay with id {designday_id} does not exist.")
        updated_name = name if name is not None else row['name']
        updated_latitude = latitude if latitude is not None else row['latitude']
        updated_longitude = longitude if longitude is not None else row['longitude']
        updated_architecture_type = architecture_type if architecture_type is not None else row['architecture_type']
        updated_month = month if month is not None else row['month']
        updated_day_of_month = day_of_month if day_of_month is not None else row['day_of_month']
        updated_day_type = day_type if day_type is not None else row['day_type']
        updated_wind_speed = wind_speed if wind_speed is not None else row['wind_speed']
        updated_wind_direction = wind_direction if wind_direction is not None else row['wind_direction']
        updated_max_dry_bulb_temp = max_dry_bulb_temp if max_dry_bulb_temp is not None else row['max_drybulb_temperature']
        updated_daily_dry_bulb_range = daily_dry_bulb_range if daily_dry_bulb_range is not None else row['daily_drybulb_temperature_range']
        updated_dry_bulb_temp_range_modifier_type = dry_bulb_temp_range_modifier_type if dry_bulb_temp_range_modifier_type is not None else row['drybulb_temperature_range_modifier_type']
        updated_dry_bulb_temp_range_modifier_day_schedule_name = dry_bulb_temp_range_modifier_day_schedule_name if dry_bulb_temp_range_modifier_day_schedule_name is not None else row['drybulb_temperature_range_modifier_day_schedule_name']
        updated_humidity_condition_type = humidity_condition_type if humidity_condition_type is not None else row['humidity_condition_type']
        updated_wetbulb_or_dewpoint_at_maximum_drybulb = wetbulb_or_dewpoint_at_maximum_drybulb if wetbulb_or_dewpoint_at_maximum_drybulb is not None else row['wetbulb_or_dewpoint_at_maximum_drybulb']
        updated_humidity_condition_day_schedule_name = humidity_condition_day_schedule_name if humidity_condition_day_schedule_name is not None else row['humidity_condition_day_schedule_name']
        updated_humidity_ratio_at_maximum_drybulb = humidity_ratio_at_maximum_drybulb if humidity_ratio_at_maximum_drybulb is not None else row['humidity_ratio_at_maximum_drybulb']
        updated_enthalpy_at_maximum_drybulb = enthalpy_at_maximum_drybulb if enthalpy_at_maximum_drybulb is not None else row['enthalpy_at_maximum_drybulb']
        updated_daily_wetbulb_temperature_range = daily_wetbulb_temperature_range if daily_wetbulb_temperature_range is not None else row['daily_wetbulb_temperature_range']
        updated_barometric_pressure = barometric_pressure if barometric_pressure is not None else row['barometric_pressure']
        updated_rain_indicator = rain_indicator if rain_indicator is not None else row['rain_indicator']
        updated_snow_indicator = snow_indicator if snow_indicator is not None else row['snow_indicator']
        updated_daylight_saving_time_indicator = daylight_saving_time_indicator if daylight_saving_time_indicator is not None else row['daylight_saving_time_indicator']
        updated_solar_model_indicator = solar_model_indicator if solar_model_indicator is not None else row['solar_model_indicator']
        updated_beam_solar_day_schedule_name = beam_solar_day_schedule_name if beam_solar_day_schedule_name is not None else row['beam_solar_day_schedule_name']
        updated_diffuse_solar_day_schedule_name = diffuse_solar_day_schedule_name if diffuse_solar_day_schedule_name is not None else row['diffuse_solar_day_schedule_name']
        updated_ashrae_clear_sky_optical_depth_for_beam_irradiance_taub = ashrae_clear_sky_optical_depth_for_beam_irradiance_taub if ashrae_clear_sky_optical_depth_for_beam_irradiance_taub is not None else row['ashrae_clear_sky_optical_depth_for_beam_irradiance_taub']
        updated_ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud = ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud if ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud is not None else row['ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud']
        updated_sky_clearness = sky_clearness if sky_clearness is not None else row['sky_clearness']
        updated_maximum_number_warmup_days = maximum_number_warmup_days if maximum_number_warmup_days is not None else row['maximum_number_warmup_days']
        updated_begin_environment_reset_mode = begin_environment_reset_mode if begin_environment_reset_mode is not None else row['begin_environment_reset_mode']
        sql = """
            UPDATE sizingperiod_designday 
            SET name = ?, latitude = ?, longitude = ?, architecture_type = ?, 
                month = ?, day_of_month = ?, day_type = ?, max_drybulb_temperature = ?, 
                daily_drybulb_temperature_range = ?, drybulb_temperature_range_modifier_type = ?, 
                drybulb_temperature_range_modifier_day_schedule_name = ?, humidity_condition_type = ?, 
                wetbulb_or_dewpoint_at_maximum_drybulb = ?, humidity_condition_day_schedule_name = ?, 
                humidity_ratio_at_maximum_drybulb = ?, enthalpy_at_maximum_drybulb = ?, 
                daily_wetbulb_temperature_range = ?, barometric_pressure = ?, wind_speed = ?, 
                wind_direction = ?, rain_indicator = ?, snow_indicator = ?, 
                daylight_saving_time_indicator = ?, solar_model_indicator = ?, 
                beam_solar_day_schedule_name = ?, diffuse_solar_day_schedule_name = ?, 
                ashrae_clear_sky_optical_depth_for_beam_irradiance_taub = ?, 
                ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud = ?, 
                sky_clearness = ?, maximum_number_warmup_days = ?, 
                begin_environment_reset_mode = ?, datetime = ?
            WHERE id = ?
        """
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M")
        timestamp_int = int(timestamp)
        dt = [
            updated_name, updated_latitude, updated_longitude, updated_architecture_type,
            updated_month, updated_day_of_month, updated_day_type, updated_max_dry_bulb_temp,
            updated_daily_dry_bulb_range, updated_dry_bulb_temp_range_modifier_type,
            updated_dry_bulb_temp_range_modifier_day_schedule_name, updated_humidity_condition_type,
            updated_wetbulb_or_dewpoint_at_maximum_drybulb, updated_humidity_condition_day_schedule_name,
            updated_humidity_ratio_at_maximum_drybulb, updated_enthalpy_at_maximum_drybulb,
            updated_daily_wetbulb_temperature_range, updated_barometric_pressure, updated_wind_speed,
            updated_wind_direction, updated_rain_indicator, updated_snow_indicator,
            updated_daylight_saving_time_indicator, updated_solar_model_indicator,
            updated_beam_solar_day_schedule_name, updated_diffuse_solar_day_schedule_name,
            updated_ashrae_clear_sky_optical_depth_for_beam_irradiance_taub,
            updated_ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud,
            updated_sky_clearness, updated_maximum_number_warmup_days,
            updated_begin_environment_reset_mode,
            timestamp_int,
            designday_id
        ]
        cursor.execute(sql, dt)
        des_data = [designday_id] + dt[:-2]
        update_description_sizingperiod_designday(des_data, cur=cursor)
        conn.commit()
    finally:
        conn.close()

def delete_sizingperiod_designday(db_path: str, designday_id: int) -> None:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM sizingperiod_designday WHERE id = ?", (designday_id,))
        conn.commit()
    finally:
        conn.close()

def list_sizingperiod_designday(db_path: str) -> List[Tuple]:
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sizingperiod_designday")
        rows = cursor.fetchall()
        return rows
    finally:
        conn.close()
