from eppy.modeleditor import IDF
from typing import Dict
from .base_converter import BaseConverter
from src.utils.logging import get_logger

logger = get_logger(__name__)

class BuildingConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        """
        这是转换的唯一入口。
        """
        self.logger.info("  > [BuildingConverter] Running convert method...")
        
        # 从总数据中提取 'Building' 部分
        building_data_list = data.get('Building', [])
        
        if not building_data_list:
            self.logger.error("    - 错误: 在YAML数据中没有找到 'Building' 键。")
            self.state['failed'] += 1
            return

        building_data_to_process = building_data_list[0]

        # 调用 validate (虽然它暂时什么都不做)
        if self.validate(building_data_to_process):
            try:
                self.add_to_idf(building_data_to_process)
                self.state['success'] += 1
                self.logger.info("    - Building object conversion successful.")
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(f"    - 错误: 在创建Building对象时失败: {e}", exc_info=True)
        else:
            self.state['failed'] += 1
            self.logger.warning("    - Building data validation failed. Skipping conversion.")

    def add_to_idf(self, building_data: Dict) -> None:
        """
        将building数据添加到IDF。在添加前，会手动修正字段名。
        """
        self.logger.info("    - Adding Building object to IDF...")

        def clean_key(key: str) -> str:
            # 1. 替换空格为下划线
            key = key.replace(' ', '_')
            # 2. 移除花括号内容
            key = key.split('{')[0]
            # 3. 【新增】移除可能出现在末尾的下划线
            key = key.rstrip('_')
            # 4. 移除首尾空格（作为保险）
            key = key.strip()
            return key

        # 使用字典推导式，遍历原始数据，生成一个键名干净的新字典
        # 例如： "North Axis {deg}" -> "North_Axis"
        cleaned_building_data = {clean_key(key): value for key, value in building_data.items()}
        
        self.logger.info(f"      - Cleaned data keys: {list(cleaned_building_data.keys())}")


        # --- 现在，使用这个“干净”的字典来创建IDF对象 ---
        try:
            # 添加Version等样板对象
            if not self.idf.getobject('VERSION', '9.5'):
                version_obj = self.idf.newidfobject('VERSION')
                version_obj.Version_Identifier = '9.5'
                self.logger.info("      - Default Version object created.")
            
            # 使用修正后的 cleaned_building_data 进行解包
            building_obj = self.idf.newidfobject(
                'BUILDING',
                **cleaned_building_data
            )
            self.logger.info(f"      - Building '{building_obj.Name}' created.")

        except Exception as e:
            # 如果这里的eppy报错，可以直接抛出，让外层的convert方法捕获
            self.logger.error(f"      - Eppy failed to create object with cleaned data. Error: {e}")
            raise # 抛出异常，让convert知道失败了

    # ===============================================================
    # 【核心修复3】: 实现父类要求的 validate 方法
    # ===============================================================
    def validate(self, data: Dict) -> bool:
        """
        一个临时的、总是返回True的验证方法，以满足抽象基类的要求。
        """
        self.logger.info("    - Running placeholder validate method (always returns True).")
        # 在这个阶段，我们不进行复杂的Pydantic验证，只是确保方法存在
        if 'Name' in data:
            return True
        else:
            self.logger.error("    - Validation failed: 'Name' field is missing in Building data.")
            return False