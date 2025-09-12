from eppy.modeleditor import IDF
from abc import ABC, abstractmethod
from typing import Dict, TypedDict

from src.utils.logging import get_logger

class ConvertState(TypedDict):
    success: int
    failed: int

class BaseConverter(ABC):

    def __init__(self, idf: IDF):
        self.idf = idf
        self.logger = get_logger(__name__)
        self.state: ConvertState = {
            "success": 0,
            "failed": 0
        }

    @abstractmethod
    def convert(self) -> IDF:
        pass

    @abstractmethod
    def add_to_idf(self) -> None:
        pass
    
    @abstractmethod
    def validate(self, data:Dict) -> bool:
        pass