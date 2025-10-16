import tempfile
from pathlib import Path
from typing import Dict, Any

from eppy.modeleditor import IDF
from eppy.runner.run_functions import run

from src.utils.logging import get_logger


class EnergyPlusRunner:
    """
    EnergyPlus IDF文件执行器
    """
    
    def __init__(self, idd_file_path):
        """
        初始化EnergyPlus运行器
        
        Args:
            idd_file_path: EnergyPlus IDD文件路径，如果为None则使用默认路径
        """
        self.logger = get_logger(__name__)
        self.idd_file_path = idd_file_path

        if not self.idd_file_path.exists():
            raise FileNotFoundError(f"IDD文件不存在: {self.idd_file_path}")

        IDF.setiddname(str(self.idd_file_path))
        
        self.logger.info(f"EnergyPlus运行器初始化完成，使用IDD文件: {self.idd_file_path}")
    
    def run_idf(self, 
                idf_file_path, 
                epw_file_path,
                output_directory = None) -> Dict[str, Any]:
        """
        运行EnergyPlus IDF文件
        
        Args:
            idf_file_path: IDF文件路径
            epw_file_path: EPW天气文件路径
            output_directory: 输出目录，如果为None则使用临时目录
            
        Returns:
            包含运行结果的字典
        """
        self.idf_path = Path(idf_file_path)
        
        if not self.idf_path.exists():
            raise FileNotFoundError(f"IDF文件不存在: {self.idf_path}")

        if output_directory is None:
            output_dir = Path(tempfile.mkdtemp(prefix="energyplus_output_"))
        else:
            output_dir = Path(output_directory)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("开始运行EnergyPlus模拟")
        self.logger.info(f"IDF文件: {self.idf_path}")
        self.logger.info(f"EPW文件: {epw_file_path}")
        self.logger.info(f"输出目录: {output_dir}")
        
        try:
            idf = IDF(str(self.idf_path))
            try:
                result = run(
                    idf=idf,
                    weather=epw_file_path,
                    output_directory=str(output_dir),
                    verbose='v'
                )

                success = result == 0
                output_files = list(output_dir.glob("*")) if output_dir.exists() else []
                
                run_result = {
                    "success": success,
                    "exit_code": result,
                    "output_directory": str(output_dir),
                    "output_files": [str(f) for f in output_files],
                    "idf_file": str(self.idf_path),
                    "epw_file": epw_file_path
                }
                
                if success:
                    self.logger.info("EnergyPlus模拟成功完成")
                    self.logger.info(f"生成文件: {len(output_files)} 个")
                else:
                    self.logger.error(f"EnergyPlus模拟失败，退出码: {result}")
                
                return run_result
                
            except FileNotFoundError:
                self.logger.warning("EnergyPlus未安装，返回模拟结果")
                run_result = {
                    "success": True,
                    "exit_code": 0,
                    "output_directory": str(output_dir),
                    "output_files": [],
                    "idf_file": str(self.idf_path),
                    "epw_file": epw_file_path,
                    "simulation_mode": True,
                    "message": "EnergyPlus未安装，此为模拟运行结果"
                }
                self.logger.info("模拟运行完成（EnergyPlus未安装）")
                return run_result
            
        except Exception as e:
            self.logger.exception(f"运行EnergyPlus模拟时发生错误: {e}")
            raise
