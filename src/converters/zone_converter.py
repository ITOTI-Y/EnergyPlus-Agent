from eppy.modeleditor import IDF
from typing import Dict, Any, List

from src.converters.base_converter import BaseConverter
from src.validator.data_model import ZoneSchema

class ZoneConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        self.logger.info("Converting zone data...")
        for zd in data.get('Zone', []):
            try:
                val_data = self.validate(zd)
                self._add_to_idf(val_data)
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(f"Error processing Zone: {e}", exc_info=True)
                continue

    def _add_to_idf(self, data:Any) -> None:
        if self.idf.getobject("Zone", name=data.name):
            self.logger.warning(f"Zone with name {data.name} already exists in IDF. Skipping addition.")
            self.state['skipped'] += 1
            return
        try:
            self.idf.newidfobject(
                "Zone",
                Name=data.name,
                Direction_of_Relative_North=data.direction_of_relative_north,
                X_Origin=data.x_origin,
                Y_Origin=data.y_origin,
                Z_Origin=data.z_origin,
                Type=data.type,
                Multiplier=data.multiplier,
                Ceiling_Height=data.ceiling_height,
                Volume=data.volume,
                Floor_Area=data.floor_area,
                Zone_Inside_Convection_Algorithm=data.zone_inside_convection_algorithm,
                Zone_Outside_Convection_Algorithm=data.zone_outside_convection_algorithm,
                Part_of_Total_Floor_Area=data.part_of_total_floor_area
            )
            self.state['success'] += 1
            self.logger.info(f"Zone with name {data.name} added to IDF.")
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error Adding Zone Data to IDF: {e}", exc_info=True)

    def validate(self, data: Dict) -> Any:
        val_data = ZoneSchema.model_validate(data)
        return val_data
