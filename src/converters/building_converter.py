from eppy.modeleditor import IDF
from typing import Dict
from logging import getLogger
from src.converters.base_converter import BaseConverter
from schemas.data_models import UserInput as BuildingSchema
from pydantic import BaseModel, ValidationError
from src.utils.logging import get_logger

logger = getLogger(__name__)

class BuildingConverter(BaseConverter):
    def __init__(self, idf: IDF, data: Dict):
        super().__init__(idf, data)
        self._validate_building_data()  # 确保数据验证
    
    def _validate_building_data(self):
        """完整的数据验证方法"""
        logger.info("正在验证建筑数据完整性...")
        user_input = self.data.get('user_input', {})
        
        # 基础字段验证
        required_fields = ['project_name', 'building_type', 'city']
        for field in required_fields:
            if field not in user_input:
                raise ValueError(f"建筑数据缺少必要字段: {field}")
        
        # 高级验证（示例）
        try:
            BuildingSchema.model_validate(user_input)
            logger.info("建筑数据验证通过")
        except ValidationError as e:
            logger.error(f"建筑数据验证失败: {e}")
            raise
        except Exception as e:
            logger.error(f"未知验证错误: {e}")
            raise
    
    def convert(self) -> None:
        logger.info("  > 开始建筑数据转换...")
        
        user_input = self.data.get('user_input', {})
        if not user_input:
            logger.warning("未找到建筑数据输入")
            self.state['failed'] += 1
            return
        
        self._validate_building_data()  # 再次验证
        
        try:
            self.add_to_idf(user_input)
            logger.info("    - 建筑数据转换完成")
            self.state['success'] += 1
        except Exception as e:
            logger.error(f"建筑数据转换失败: {e}")
            self.state['failed'] += 1
    
    def add_to_idf(self, building_data: Dict) -> None:
        """将验证过的building_data添加到IDF"""
        self.logger.info("    - Adding Building and Location objects to IDF...")
        try:
            # ... (添加Version等样板代码) ...

            # --- 创建Building对象 ---
            # 【核心修改点1】: 确保这里面没有City, building_type等不相关的字段
            building_obj = self.idf.newidfobject(
                'BUILDING',
                Name=building_data.get('project_name', 'Default Building'),
                North_Axis=0,
                Terrain='City' 
            )
            self.logger.info(f"    - Building '{building_obj.Name}' added.")

            # --- 【核心修改点2】: 创建一个独立的 Site:Location 对象来存储城市信息 ---
            #    从传入的数据中获取城市名
            city_name = building_data.get('city', 'DefaultCity')
            
            location_obj = self.idf.newidfobject(
                'SITE:LOCATION',
                Name=city_name,
                # 为了让IDF能运行，需要提供一个大概的经纬度
                Latitude=39.9,  # 示例：北京的纬度
                Longitude=116.4, # 示例：北京的经度
                Time_Zone=8,      # 示例：东八区
                Elevation=50      # 示例：海拔
            )
            self.logger.info(f"    - Site:Location '{location_obj.Name}' added.")
            
            self.state['success'] += 1

        except Exception as e:
            self.logger.error(f"    - 错误: 创建对象时失败: {e}", exc_info=True)
            self.state['failed'] += 1
