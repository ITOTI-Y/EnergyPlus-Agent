from eppy.modeleditor import IDF
from typing import Dict, Any
from src.converters.base_converter import BaseConverter
from src.utils.logging import get_logger
from src.validator.data_model import (
    MaterialSchema, 
    StandardMaterialProperties, 
    NoMassMaterialProperties, 
    AirGapMaterialProperties, 
    GlazingMaterialProperties
)

class MaterialConverter(BaseConverter):
    """
    Converts material definitions from YAML data into appropriate IDF objects.
    """
    
    def __init__(self, idf: IDF):
        super().__init__(idf)
        self.logger = get_logger(__name__)

    def convert(self, data: Dict[str, Any]) -> None:
        """
        Processes a list of material definitions from the YAML data.
        """
        self.logger.info("Converting Material data...")
        material_list = data.get('Material', [])
        
        if not material_list:
            self.logger.info("No materials found in YAML data.")
            return
        
        for material_data in material_list:
            try:
                material_name = material_data.get('Name', 'Unknown Material')
                self.logger.debug(f"Processing material: {material_name}")
                
                validated_material = self.validate(material_data)
                self._add_to_idf(validated_material)
                
            except Exception as e:
                self.state['failed'] += 1
                material_name = material_data.get('Name', 'Unknown Material')
                self.logger.error(
                    "Failed to convert Material '{}': {}".format(material_name, str(e))
                )
                continue

    def _add_to_idf(self, material: MaterialSchema) -> None:
 
        try:
            # 确定 IDF 键
            idf_key = self._get_idf_key(material.Type)
            
            if not idf_key:
                self.logger.error(f"Unknown material type '{material.Type}' for material '{material.Name}'")
                self.state['failed'] += 1
                return

            # 检查是否已存在
            if self.idf.getobject(idf_key, material.Name):
                self.logger.warning(
                    f"{idf_key} with name '{material.Name}' already exists. Skipping addition."
                )
                self.state['skipped'] += 1
                return
            
            self.logger.info(f"Adding {idf_key} '{material.Name}' to IDF.")
            
            # 准备数据
            material_dict = self._prepare_material_dict(material)
            
            # 创建 IDF 对象
            self.idf.newidfobject(idf_key, **material_dict)
            self.state['success'] += 1
            self.logger.debug(f"Successfully added {idf_key} '{material.Name}'")
            
        except Exception as e:
            self.state['failed'] += 1
            self.logger.error(f"Error adding material '{material.Name}' to IDF: {str(e)}")

    def _get_idf_key(self, material_type: str) -> str:
        """
        根据材料类型返回对应的 IDF 键。
        """
        type_to_key = {
            "Standard": "Material",
            "NoMass": "Material:NoMass", 
            "AirGap": "Material:AirGap",
            "Glazing": "WindowMaterial:SimpleGlazingSystem"
        }
        return type_to_key.get(material_type)

    def _prepare_material_dict(self, material: MaterialSchema) -> Dict[str, Any]:
        """
        根据材料类型准备数据字典。
        """
        material_dict = {'Name': material.Name}
        
        # 根据类型处理属性
        
        if material.Type == "Standard":
            props = material.Properties
            if isinstance(props, StandardMaterialProperties):
                material_dict.update({
                    'Roughness': props.Roughness,
                    'Thickness': props.Thickness,
                    'Conductivity': props.Conductivity,
                    'Density': props.Density,
                    'Specific_Heat': props.Specific_Heat
                })
                
        elif material.Type == "NoMass":
            props = material.Properties
            if isinstance(props, NoMassMaterialProperties):
                material_dict.update({
                    'Roughness': props.Roughness,
                    'Thermal_Resistance': props.Thermal_Resistance
                })
                
        elif material.Type == "AirGap":
            props = material.Properties
            if isinstance(props, AirGapMaterialProperties):
                material_dict['Thermal_Resistance'] = props.Thermal_Resistance
                
        elif material.Type == "Glazing":
            props = material.Properties
            if isinstance(props, GlazingMaterialProperties):
                material_dict.update({
                    # 'Optical_Data_Type' 字段在 WindowMaterial:SimpleGlazingSystem 中不存在，需要移除
                    # 'Optical_Data_Type': props.Optical_Data_Type,
                    
                    # 确保所有字段名都使用下划线
                    'U_Factor': props.U_Factor,
                    'Solar_Heat_Gain_Coefficient': props.Solar_Heat_Gain_Coefficient
                })
                # 'Visible_Transmittance' 是可选字段，命名正确
                if props.Visible_Transmittance is not None:
                    material_dict['Visible_Transmittance'] = props.Visible_Transmittance
                    
        return material_dict

    def validate(self, data: Dict) -> MaterialSchema:
        """
        Validates material data using the MaterialSchema.
        """
        return MaterialSchema.model_validate(data)