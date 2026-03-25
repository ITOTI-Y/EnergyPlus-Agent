import sqlite3
from datetime import datetime

from src.database.datatools._share import TIMESTAMP
from src.database.datatools.datadescription import (
    update_description_sizingperiod_designday,
)


def create_sizingperiod_designday(
    db_path: str,
    name: str,
    latitude: float,
    longitude: float,
    architecture_type: str,
    month: int,
    day_of_month: int,
    day_type: str,
    wind_speed: float,
    wind_direction: float,
    max_dry_bulb_temp: float | None = None,
    daily_dry_bulb_range: float | None = None,
    dry_bulb_temp_range_modifier_type: str | None = None,
    dry_bulb_temp_range_modifier_day_schedule_name: str | None = None,
    humidity_condition_type: str | None = None,
    wetbulb_or_dewpoint_at_maximum_drybulb: float | None = None,
    humidity_condition_day_schedule_name: str | None = None,
    humidity_ratio_at_maximum_drybulb: float | None = None,
    enthalpy_at_maximum_drybulb: float | None = None,
    daily_wetbulb_temperature_range: float | None = None,
    barometric_pressure: float | None = None,
    rain_indicator: str | None = None,
    snow_indicator: str | None = None,
    daylight_saving_time_indicator: str | None = None,
    solar_model_indicator: str | None = None,
    beam_solar_day_schedule_name: str | None = None,
    diffuse_solar_day_schedule_name: str | None = None,
    ashrae_clear_sky_optical_depth_for_beam_irradiance_taub: float | None = None,
    ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud: float | None = None,
    sky_clearness: float | None = None,
    maximum_number_warmup_days: int | None = None,
    begin_environment_reset_mode: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        sql = "INSERT INTO sizingperiod_designday (name, latitude, longitude, architecture_type, month, day_of_month, day_type, max_drybulb_temperature, daily_drybulb_temperature_range, drybulb_temperature_range_modifier_type, drybulb_temperature_range_modifier_day_schedule_name, humidity_condition_type, wetbulb_or_dewpoint_at_maximum_drybulb, humidity_condition_day_schedule_name, humidity_ratio_at_maximum_drybulb, enthalpy_at_maximum_drybulb, daily_wetbulb_temperature_range, barometric_pressure, wind_speed, wind_direction, rain_indicator, snow_indicator, daylight_saving_time_indicator, solar_model_indicator, beam_solar_day_schedule_name, diffuse_solar_day_schedule_name, ashrae_clear_sky_optical_depth_for_beam_irradiance_taub, ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud, sky_clearness, maximum_number_warmup_days, begin_environment_reset_mode, datetime) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        des_data = [
            name,
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
            begin_environment_reset_mode,
        ]

        timestamp_int = int(datetime.now().strftime(TIMESTAMP))
        des_data.append(timestamp_int)
        cursor.execute(sql, des_data)
        new_id = cursor.lastrowid
        des_data.insert(0, new_id)

        update_description_sizingperiod_designday(des_data[:-1], cur=cursor)
        conn.commit()


# Column names in DB order (for update merge logic)
_DB_COLUMNS = [
    "name",
    "latitude",
    "longitude",
    "architecture_type",
    "month",
    "day_of_month",
    "day_type",
    "max_drybulb_temperature",
    "daily_drybulb_temperature_range",
    "drybulb_temperature_range_modifier_type",
    "drybulb_temperature_range_modifier_day_schedule_name",
    "humidity_condition_type",
    "wetbulb_or_dewpoint_at_maximum_drybulb",
    "humidity_condition_day_schedule_name",
    "humidity_ratio_at_maximum_drybulb",
    "enthalpy_at_maximum_drybulb",
    "daily_wetbulb_temperature_range",
    "barometric_pressure",
    "wind_speed",
    "wind_direction",
    "rain_indicator",
    "snow_indicator",
    "daylight_saving_time_indicator",
    "solar_model_indicator",
    "beam_solar_day_schedule_name",
    "diffuse_solar_day_schedule_name",
    "ashrae_clear_sky_optical_depth_for_beam_irradiance_taub",
    "ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud",
    "sky_clearness",
    "maximum_number_warmup_days",
    "begin_environment_reset_mode",
]

# Map: function param name -> DB column name (only where they differ)
_PARAM_TO_COL = {
    "max_dry_bulb_temp": "max_drybulb_temperature",
    "daily_dry_bulb_range": "daily_drybulb_temperature_range",
    "dry_bulb_temp_range_modifier_type": "drybulb_temperature_range_modifier_type",
    "dry_bulb_temp_range_modifier_day_schedule_name": "drybulb_temperature_range_modifier_day_schedule_name",
}


def update_sizingperiod_designday(
    db_path: str,
    designday_id: int,
    name: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    architecture_type: str | None = None,
    month: int | None = None,
    day_of_month: int | None = None,
    day_type: str | None = None,
    wind_speed: float | None = None,
    wind_direction: float | None = None,
    max_dry_bulb_temp: float | None = None,
    daily_dry_bulb_range: float | None = None,
    dry_bulb_temp_range_modifier_type: str | None = None,
    dry_bulb_temp_range_modifier_day_schedule_name: str | None = None,
    humidity_condition_type: str | None = None,
    wetbulb_or_dewpoint_at_maximum_drybulb: float | None = None,
    humidity_condition_day_schedule_name: str | None = None,
    humidity_ratio_at_maximum_drybulb: float | None = None,
    enthalpy_at_maximum_drybulb: float | None = None,
    daily_wetbulb_temperature_range: float | None = None,
    barometric_pressure: float | None = None,
    rain_indicator: str | None = None,
    snow_indicator: str | None = None,
    daylight_saving_time_indicator: str | None = None,
    solar_model_indicator: str | None = None,
    beam_solar_day_schedule_name: str | None = None,
    diffuse_solar_day_schedule_name: str | None = None,
    ashrae_clear_sky_optical_depth_for_beam_irradiance_taub: float | None = None,
    ashrae_clear_sky_optical_depth_for_diffuse_irradiance_taud: float | None = None,
    sky_clearness: float | None = None,
    maximum_number_warmup_days: int | None = None,
    begin_environment_reset_mode: str | None = None,
) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM sizingperiod_designday WHERE id = ?", (designday_id,)
        )
        row = cursor.fetchone()
        if row is None:
            raise ValueError(
                f"SizingPeriod:DesignDay with id {designday_id} does not exist."
            )

        provided = locals()
        updated_values = []
        for col in _DB_COLUMNS:
            param = next((p for p, c in _PARAM_TO_COL.items() if c == col), col)
            val = provided.get(param)
            updated_values.append(val if val is not None else row[col])

        timestamp_int = int(datetime.now().strftime(TIMESTAMP))
        set_clause = ", ".join(f"{col} = ?" for col in _DB_COLUMNS)
        sql = (
            f"UPDATE sizingperiod_designday SET {set_clause}, datetime = ? WHERE id = ?"
        )
        cursor.execute(sql, [*updated_values, timestamp_int, designday_id])

        des_data = [designday_id, *updated_values]
        update_description_sizingperiod_designday(des_data, cur=cursor)
        conn.commit()


def delete_sizingperiod_designday(db_path: str, designday_id: int) -> None:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM sizingperiod_designday WHERE id = ?", (designday_id,)
        )
        conn.commit()


def list_sizingperiod_designday(db_path: str) -> list[tuple]:
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sizingperiod_designday")
        return cursor.fetchall()
