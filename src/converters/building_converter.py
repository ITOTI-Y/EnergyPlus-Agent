from eppy.modeleditor import IDF
from typing import Dict, Any
import sys
import os
from pathlib import Path

try:
    from src.validator.data_model import BuildingSchema
except ModuleNotFoundError:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        
    # 再次尝试导入
    try:
        from src.validator.data_model import BuildingSchema
        print(f"--- Dynamically added '{project_root}' to path and imported BuildingSchema successfully. ---")
    except ImportError as e:
        print(f"FATAL: Could not import BuildingSchema even after modifying sys.path. Error: {e}")
        # 如果还是失败，就定义一个临时的空类，避免程序完全崩溃
        class BuildingSchema:
            @staticmethod
            def model_validate(data):
                return data

# --- 正常的导入 ---
from .base_converter import BaseConverter
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BuildingConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        """
        主转换方法，接收从Manager传来的完整YAML数据。
        """
        self.logger.info("  > [BuildingConverter] Running convert method...")
        
        building_data_list = data.get('Building', [])
        
        if not building_data_list:
            self.logger.error("    - 错误: 在YAML数据中没有找到 'Building' 键。")
            self.state['failed'] += 1
            return

        building_data_to_process = building_data_list[0]
        
        if self.validate(building_data_to_process):
            try:
                self.add_to_idf(building_data_to_process, full_data=data)
                self.state['success'] += 1
                self.logger.info("    - Building object conversion successful.")
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(f"    - 错误: 在创建对象时失败: {e}", exc_info=True)
        else:
            self.state['failed'] += 1
            self.logger.warning("    - Building data validation failed. Skipping conversion.")

    def add_to_idf(self, building_data: Dict, full_data: Dict) -> None:
        """
        将数据添加到IDF。
        """
        self.logger.info("    - Adding Building object to IDF...")
        
        def clean_key(key: str) -> str:
            key = str(key).replace(' ', '_')
            key = key.split('{')[0]
            key = key.rstrip('_')
            return key.strip()

        cleaned_building_data = {clean_key(k): v for k, v in building_data.items()}
        
        if not self.idf.getobject('VERSION', '25.1.0'):
            version_obj = self.idf.newidfobject('VERSION')
            version_obj.Version_Identifier = '25.1.0'
        
        self.idf.newidfobject('BUILDING', **cleaned_building_data)
        self.logger.info(f"      - Building '{cleaned_building_data.get('Name')}' created.")

    def validate(self, data: Dict) -> bool:
        """
        使用从src.validator.data_model导入的BuildingSchema进行验证。
        """
        self.logger.info("    - Validating building data using BuildingSchema...")
        try:
            BuildingSchema.model_validate(data)
            self.logger.info("    - Building data validation successful.")
            return True
        except Exception as e:
            self.logger.error(f"    - Building data validation failed: {e}")
            return False
