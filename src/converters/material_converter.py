# src/converters/material_converter.py

from eppy.modeleditor import IDF
from typing import Dict, Any, List
from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
# 导入我们刚刚创建的 MaterialSchema
from src.validator.data_model import MaterialSchema

class MaterialConverter(BaseConverter):
    """
    负责将 YAML 中定义的 Material 对象列表转换为 IDF 中的 Material 对象。
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: Dict[str, Any]) -> None:
        """
        从输入数据中提取 'Material' 列表，并对每一项进行转换。
        """
        self.logger.info("Converting Material data...")
        material_list = data.get('Material', [])
        
        # 遍历 YAML 中定义的所有材料
        for material_data in material_list:
            try:
                # 1. 验证单个材料数据
                validated_material = self.validate(material_data)
                # 2. 将验证后的数据添加到 IDF
                self._add_to_idf(validated_material)
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(
                    f"Failed to convert Material '{material_data.get('Name', 'N/A')}': {e}", 
                    exc_info=True
                )
                continue # 跳过失败的项，继续处理下一个

    def _add_to_idf(self, data: MaterialSchema) -> None:
        """
        将单个验证后的 MaterialSchema 实例添加到 IDF 对象中。
        """
        # 安全检查：如果同名材料已存在，则跳过
        if self.idf.getobject("MATERIAL", data.Name):
            self.logger.warning(
                f"Material with name '{data.Name}' already exists. Skipping addition."
            )
            self.state['skipped'] += 1
            return
            
        try:
            self.logger.info(f"Adding Material '{data.Name}' to IDF.")
            
            # 使用 model_dump 将 Pydantic 模型转换为 eppy 需要的字典
            material_dict = data.model_dump(exclude_none=True)
            # 创建新的 'Material' 对象
            self.idf.newidfobject("MATERIAL", **material_dict)
            
            self.state['success'] += 1
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error adding Material '{data.Name}' to IDF: {e}", exc_info=True)

    def validate(self, data: Dict) -> MaterialSchema:
        """
        使用 MaterialSchema 验证单个材料的数据。
        """
        return MaterialSchema.model_validate(data)
