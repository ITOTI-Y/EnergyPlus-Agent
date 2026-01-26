from typing import Any

from eppy.modeleditor import IDF

from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import PeopleSchema

class PeopleConverter(BaseConverter):
    """
    Converts People definitions from YAML data into IDF objects.
    Strictly follows the EnergyPlus IDD field names provided in the screenshot.
    """
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: dict[str, Any]) -> None:
        self.logger.info("People Converter Starting...")
        people_list = data.get("People") or data.get("people", [])
        if not people_list:
            self.logger.info("No People data found in YAML.")
            return

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

    def validate(self, data: dict[str, Any]) -> PeopleSchema:
        return PeopleSchema.model_validate(data)

    def _add_to_idf(self, val_data: PeopleSchema) -> None:
        if self.idf.getobject("PEOPLE", val_data.name):
            self.logger.warning(
                f"People object with name '{val_data.name}' already exists. Skipping."
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

            self._check_schedule_exists(val_data.number_of_people_schedule_name, "Number of People Schedule")
            self._check_schedule_exists(val_data.activity_level_schedule_name, "Activity Level Schedule")

            self.idf.newidfobject(
                "PEOPLE",
                Name=val_data.name,
                Zone_or_ZoneList_or_Space_or_SpaceList_Name=val_data.zone_name,
                Number_of_People_Schedule_Name=val_data.number_of_people_schedule_name,
                Number_of_People_Calculation_Method=val_data.number_of_people_calc_method,
                Number_of_People=val_data.number_of_people,
                People_per_Floor_Area=val_data.people_per_floor_area,
                Floor_Area_per_Person=val_data.floor_area_per_person,
                Fraction_Radiant=val_data.fraction_radiant,
                Sensible_Heat_Fraction=val_data.sensible_heat_fraction,
                Activity_Level_Schedule_Name=val_data.activity_level_schedule_name,
                Carbon_Dioxide_Generation_Rate=val_data.carbon_dioxide_generation_rate
            )
            self.state["success"] += 1
            self.logger.success(f"People '{val_data.name}' added successfully to Zone '{val_data.zone_name}'.")

        except Exception as e:
            self.state["failed"] += 1
            self.logger.error(f"Failed to add People object: {e}")

    def _check_schedule_exists(self, sched_name: str, field_desc: str):
        """
        Robustly checks if a schedule exists, ignoring case and whitespace.
        If not found, prints ALL available schedules to help debug.
        """
        target_clean = sched_name.strip().upper()
        found = False
        schedule_types = ["SCHEDULE:COMPACT", "SCHEDULE:YEAR", "SCHEDULE:CONSTANT", "SCHEDULE:FILE"]
        for sched_type in schedule_types:
            objects = self.idf.idfobjects[sched_type]
            for obj in objects:
                if obj.Name.strip().upper() == target_clean:
                    found = True
                    break
            if found:
                break
        if not found:
            existing_schedules = []
            for st in schedule_types:
                existing_schedules.extend([o.Name for o in self.idf.idfobjects[st]])
            error_msg = (
                f"{field_desc} '{sched_name}' referenced in People object NOT found in IDF.\n"
                f"   ---> Current Schedules in IDF: {existing_schedules}\n"
                f"   ---> Please check if ScheduleConverter ran successfully."
            )
            raise ValueError(error_msg)
