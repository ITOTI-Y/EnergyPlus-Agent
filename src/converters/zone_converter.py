from eppy.modeleditor import IDF
from typing import Dict

from src.converters.base_converter import BaseConverter
from src.validator.data_model import ZoneSchema

class ZoneConverter(BaseConverter):

    def __init__(self, idf: IDF):
        super().__init__(idf)

    def convert(self, data: Dict) -> None:
        zone_datas: Dict = data.get('Zone', {})
        self.logger.info("Converting zone data...")
        # 完成该部分的转换逻辑
        val_data = [self.validate(data) for data in zone_datas]
        for vd in val_data:
            try:
                self._add_to_idf(vd)
                self.state['success'] += 1
            except Exception as e:
                self.state['failed'] += 1
                self.logger.error(f"Error Convert Zone Data: {e}", exc_info=True)

    def _add_to_idf(self, data: Dict) -> None:
        self.logger.info("Adding zone data to IDF...")
        # 完成将数据添加到 IDF 的逻辑
        pass

    def validate(self, data) -> Dict:
        if data := ZoneSchema.model_validate(data):
            return {"zone_data": data}
        return {}