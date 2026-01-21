from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import PeopleSchema


class PeopleConverter(BaseConverter):
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("Converting People data...")
        people_list = data.get("People") or data.get("people", [])

        for people_data in people_list:
            try:
                validated_people = self.validate(people_data)
                self._add_to_idf(validated_people)
            except Exception as e:
                self.state["failed"] += 1
                self.logger.error(
                    f"Failed to convert People '{people_data.get('Name', 'N/A')}': {e}",
                    exc_info=True,
                )
                continue

    def validate(self, data: dict) -> PeopleSchema:
        return PeopleSchema.model_validate(data)

    def _add_to_idf(self, val_data: PeopleSchema) -> None:
        if self.idf.getobject("PEOPLE", val_data.name):
            self.logger.warning(
                f"People object with name '{val_data.name}' already exists. Skipping addition."
            )
            self.state["skipped"] += 1
            return

        try:
            self.logger.debug(f"Adding People '{val_data.name}' to IDF.")

            if not self.idf.getobject("ZONE", val_data.zone_name):
                raise ValueError(
                    f"Zone '{val_data.zone_name}' referenced in People '{val_data.name}' "
                    f"does not exist in IDF."
                )

            def check_schedule(sched_name, field_desc):
                exists = False
                for sched_type in [
                    "SCHEDULE:COMPACT",
                    "SCHEDULE:YEAR",
                    "SCHEDULE:CONSTANT",
                    "SCHEDULE:FILE",
                ]:
                    if self.idf.getobject(sched_type, sched_name):
                        exists = True
                        break
                if not exists:
                    raise ValueError(
                        f"{field_desc} '{sched_name}' referenced in People '{val_data.name}' "
                        f"does not exist in IDF."
                    )

            check_schedule(
                val_data.number_of_people_schedule_name, "Number of People Schedule"
            )
            check_schedule(
                val_data.activity_level_schedule_name, "Activity Level Schedule"
            )

            people_obj = self.idf.newidfobject("PEOPLE")

            people_obj.Name = val_data.name

            zone_field = self._get_idd_field_name(people_obj, ["Zone", "Name"])
            if zone_field:
                setattr(people_obj, zone_field, val_data.zone_name)
            else:
                raise AttributeError(
                    "Could not find 'Zone Name' field in IDD definition for People."
                )

            people_obj.Number_of_People_Schedule_Name = (
                val_data.number_of_people_schedule_name
            )
            people_obj.Activity_Level_Schedule_Name = (
                val_data.activity_level_schedule_name
            )

            people_obj.Number_of_People_Calculation_Method = (
                val_data.number_of_people_calc_method
            )

            if val_data.number_of_people_calc_method == "People":
                num_field = self._get_idd_field_name(people_obj, ["Number", "People"])
                if (
                    num_field
                    and "Schedule" not in num_field
                    and "Method" not in num_field
                ):
                    setattr(people_obj, num_field, val_data.number_of_people)
                else:
                    people_obj.Number_of_People = val_data.number_of_people

            elif val_data.number_of_people_calc_method == "People/Area":
                people_obj.People_per_Zone_Floor_Area = (
                    val_data.people_per_zone_floor_area
                )

            elif val_data.number_of_people_calc_method == "Area/Person":
                people_obj.Zone_Floor_Area_per_Person = (
                    val_data.zone_floor_area_per_person
                )

            people_obj.Fraction_Radiant = val_data.fraction_radiant
            people_obj.Sensible_Heat_Fraction = val_data.sensible_heat_fraction
            people_obj.Carbon_Dioxide_Generation_Rate = (
                val_data.carbon_dioxide_generation_rate
            )

            self.state["success"] += 1
            self.logger.success(
                f"People '{val_data.name}' added successfully to Zone '{val_data.zone_name}'."
            )

        except ValueError as e:
            self.state["failed"] += 1
            self.logger.error(f"Validation Error adding People '{val_data.name}': {e}")
        except AttributeError as e:
            self.state["failed"] += 1
            self.logger.error(f"IDD/Schema Error adding People '{val_data.name}': {e}")
        except Exception:
            self.state["failed"] += 1
            self.logger.exception(
                f"An unexpected error occurred while adding People '{val_data.name}'"
            )

    def _get_idd_field_name(self, idf_obj, keywords: list[str]) -> str | None:
        """
        Support Function: Resolves IDD Version Field Name Inconsistency Issue
        """
        for field in idf_obj.fieldnames:
            if all(k.lower() in field.lower() for k in keywords):
                return field
        return None
