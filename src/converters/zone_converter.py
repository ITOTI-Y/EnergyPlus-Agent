from eppy.modeleditor import IDF
from typing import Dict, TypedDict

from src.converters.base_converter import BaseConverter
from src.validator.data_model import ZoneSchema

class ZoneConverter(BaseConverter):

    def __init__(self, idf: IDF, yaml_data: dict):
        # 调用父类BaseConverter的构造方法
        super().__init__(idf)
        self.yaml_data = yaml_data

    def convert(self) -> None:
        self.logger.info("Converting zone data...")
        # 完成该部分的转换逻辑
        #self.convert_data = self._load_yaml("./schemas/building_schema.yaml")
        self.convert_data = self.yaml_data

        self.add_to_idf()

    def add_to_idf(self) -> None:
        
        # 完成将数据添加到 IDF 的逻辑
        self.convert_data = self.validate(self.convert_data)
        self.logger.info("Adding zone data to IDF...")

        for name_object, object in self.convert_data.items():
            #print(name_object)
            #print(object)

            #修改：先验证再循环！！！！（已修改）
            
            if name_object == "zones":
                for zone in object:
                    #self.logger.info(zone)
                    zone_name = zone.get("name", "Unnamed_Zone")
                    self.logger.info(f"Adding zone: {zone_name}")

                    zone_input = self.idf.newidfobject("ZONE")
                    zone_input.Name = zone_name
                    zone_input.Direction_of_Relative_North = zone.get("direction_of_relative_north", 0.0)

                    #print(zone.get("direction_of_relative_north", 0.0))
                    zone_input.X_Origin = zone.get("x_origin", 0.0)
                    zone_input.Y_Origin = zone.get("y_origin", 0.0)
                    zone_input.Z_Origin = zone.get("z_origin", 0.0)
                    zone_input.Type = zone.get("zone_type", 1)
                    zone_input.Multiplier = zone.get("multiplier", 1)
                    zone_input.Ceiling_Height = zone.get("ceiling_height", "autocalculate")
                    zone_input.Volume = zone.get("volume", "autocalculate")
                    zone_input.Floor_Area = zone.get("floor_area", "autocalculate")
                    zone_input.Part_of_Total_Floor_Area = zone.get("part_of_total_floor_area", "Yes")                    
                pass

    def validate(self, data) -> Dict:
        self.logger.info("validating zone data to IDF...")
        data_result = []
        for name_object, object in data.items():
            if name_object == "zones":
                for zone in object:
                    # 验证整个zone对象，而不是只验证zone_name（已修改）
                    
                    data_validated = ZoneSchema.model_validate(zone)
                    
                    zone["name"] = data_validated.name
                    data_result.append(zone)
                    
        data["zones"] = data_result
        #self.logger.info(data)
        return data
