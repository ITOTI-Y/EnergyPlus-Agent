from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import ConstructionSchema


class ConstructionConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("Converting Construction data...")
        construction_list = data.get("Construction", [])

        for construction_data in construction_list:
            try:
                validated_construction = self.validate(construction_data)
                self._add_to_idf(validated_construction)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error(
                    f"Failed to convert Construction '{construction_data.get('Name', 'N/A')}': {e}",
                    exc_info=True,
                )
                continue

    def _add_to_idf(self, data: ConstructionSchema) -> None:
        if self.idf.getobject("CONSTRUCTION", data.name):
            self.logger.warning(
                f"Construction with name '{data.name}' already exists. Skipping addition."
            )
            self.state["skipped"] += 1
            return

        try:
            self.logger.debug(f"Adding Construction '{data.name}' to IDF.")

            for layer_name in data.layers:
                if not (
                    self.idf.getobject("MATERIAL", layer_name)
                    or self.idf.getobject("MATERIAL:NOMASS", layer_name)
                    or self.idf.getobject("MATERIAL:AIRGAP", layer_name)
                    or self.idf.getobject(
                        "WINDOWMATERIAL:SIMPLEGLAZINGSYSTEM", layer_name
                    )
                ):
                    raise ValueError(
                        f"Material '{layer_name}' referenced in Construction '{data.name}' "
                        f"does not exist in IDF. Please add the material first."
                    )

            construction_obj = self.idf.newidfobject("CONSTRUCTION", Name=data.name)

            for i, layer_name in enumerate(data.layers):
                field_name = "Outside_Layer" if i == 0 else f"Layer_{i + 1}"

                setattr(construction_obj, field_name, layer_name)
                self.logger.debug(
                    f"  - Set {field_name} to '{layer_name}' for '{data.name}'."
                )
            self.state["success"] += 1
            self.logger.success(
                f"Construction '{data.name}' with {len(data.layers)} layers added successfully."
            )

        except ValueError:
            self.state["failed"] += 1
            self.logger.exception(f"Failed to add Construction '{data.name}'")

        except AttributeError:
            self.state["failed"] += 1
            self.logger.exception(
                f"Error adding Construction '{data.name}'. A specified field name was not found.",
            )
        except Exception:
            self.state["failed"] += 1
            self.logger.exception(
                f"An unexpected error occurred while adding Construction '{data.name}'"
            )

    def validate(self, data: dict) -> ConstructionSchema:
        return ConstructionSchema.model_validate(data)
