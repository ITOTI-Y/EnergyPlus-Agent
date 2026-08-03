from abc import ABC, abstractmethod
from typing import Any, TypedDict

from idfpy import IDF
from idfpy.models import IDFBaseModel
from pydantic import BaseModel

from src._share import UNSET
from src.utils.logging import get_logger


class ConvertState(TypedDict):
    success: int
    skipped: int
    failed: int


class BaseConverter(ABC):
    def __init__(self, idf: IDF):
        self.idf = idf
        self.logger = get_logger(__name__)
        self.state: ConvertState = {"success": 0, "skipped": 0, "failed": 0}

    @staticmethod
    def _to_idf_model[T: IDFBaseModel](
        model_type: type[T], source: BaseModel, **overrides: Any
    ) -> T:
        source_data = source.model_dump()
        model_data = {
            field_name: source_data[field_name]
            for field_name in model_type.model_fields.keys() & source_data.keys()
        }
        for field_name, value in overrides.items():
            if value is UNSET:
                model_data.pop(field_name, None)
            else:
                model_data[field_name] = value
        return model_type.model_validate(model_data)

    @abstractmethod
    def convert(self, data: dict) -> None: ...

    @abstractmethod
    def _add_to_idf(self, val_data: Any) -> None: ...

    @abstractmethod
    def validate(self, data: dict) -> Any: ...
