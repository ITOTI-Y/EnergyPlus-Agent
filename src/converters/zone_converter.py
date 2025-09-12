from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from validator.data_model import ZoneSchema

class ZoneConverter(BaseConverter):

    def __init__(self, idf: IDF):
        """
        初始化 ZoneConverter 实例。
        
        在调用父类初始化后，创建一个用于记录转换结果的状态字典 self.state：
        - "success": 成功转换的条目数（初始值 0）
        - "failed": 转换失败的条目数（初始值 0）
        """
        super().__init__(idf)

        self.state = {
            "success" : 0,
            "failed" : 0
        }

    def convert(self) -> None:
        """
        执行建筑数据转换并将结果加入 IDF。
        
        该方法记录转换开始，执行转换逻辑（占位）并调用 add_to_idf 将转换结果写入 IDF。转换成功/失败计数由实例属性 self.state（包含 "success" 和 "failed"）维护。
        """
        self.logger.info("Converting building data...")
        # 完成该部分的转换逻辑
        self.add_to_idf()

    def add_to_idf(self) -> None:
        self.logger.info("Adding building data to IDF...")
        # 完成将数据添加到 IDF 的逻辑
        pass

    def validate(self, data) -> bool:
        if data := ZoneSchema.model_validate(data):
            return True
        return False