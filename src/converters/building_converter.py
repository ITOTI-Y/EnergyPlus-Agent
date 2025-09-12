from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from validator.data_model import BuildingSchema

class BuildingConverter(BaseConverter):
    
    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self) -> None:
        self.logger.info("Converting building data...")
        # 完成该部分的转换逻辑
        self.add_to_idf()

    def add_to_idf(self) -> None:
        self.logger.info("Adding building data to IDF...")
        # 完成将数据添加到 IDF 的逻辑
        pass

    def validate(self, data) -> bool:
        if data := BuildingSchema.model_validate(data):
            return True
        return False