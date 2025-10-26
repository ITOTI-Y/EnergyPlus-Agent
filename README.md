![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ITOTI-Y/EnergyPlus-Agent?utm_source=oss&utm_medium=github&utm_campaign=ITOTI-Y%2FEnergyPlus-Agent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ITOTI-Y/EnergyPlus-Agent)

# EnergyPlus Agent System

## 项目概述

EnergyPlus Agent 是一个基于 Python 和 MCP（Model Context Protocol）协议的智能建筑能耗模拟系统。该系统通过 LLM 驱动的交互式配置流程，将建筑设计从 Rhino 代码无缝转换为 EnergyPlus IDF 文件，并提供能耗分析和优化建议。

## 核心特性

### 智能转换
- **Rhino代码解析**：自动解析 Rhino 建筑生成代码，提取几何和材料信息
- **LLM驱动转换**：通过大语言模型理解建筑意图，生成标准化 YAML 配置
- **IDF自动生成**：将 YAML 配置映射为符合 EnergyPlus 标准的 IDF 文件

### 交互式配置
- **MCP协议支持**：基于标准 MCP 协议实现工具调用和上下文管理
- **智能对话引导**：通过自然语言交互补充 HVAC、照明、设备等配置
- **知识库集成**：集成建筑规范和最佳实践，提供可解释的配置建议

### 仿真与优化
- **自动验证**：IDF 文件完整性和合规性自动检查
- **能耗模拟**：集成 EnergyPlus 引擎进行精确能耗计算
- **性能评估**：多维度能耗分析和性能评估报告
- **优化建议**：基于仿真结果提供智能优化方案

## 项目结构

```
EnergyPlus-Agent/
├── src/                          # 源代码目录
│   ├── converters/               # 转换器模块
│   │   ├── base_converter.py    # 转换器基类
│   │   ├── building_converter.py # 建筑信息转换器
│   │   ├── zone_converter.py    # 热区转换器
│   │   ├── surface_converter.py # 表面转换器
│   │   └── setting_converter.py # 设置转换器
│   ├── validator/                # 数据验证模块
│   │   └── data_model.py        # Pydantic 数据模型和Schema
│   ├── runner/                   # EnergyPlus 运行器
│   │   └── runner.py            # EnergyPlus 执行模块
│   ├── utils/                    # 工具模块
│   │   └── logging.py           # 日志配置
│   └── converter_manager.py      # 转换器管理器
├── schemas/                      # 配置文件模板
│   └── building_schema.yaml     # 建筑配置YAML示例
├── dependencies/                 # 依赖文件
│   ├── Energy+.idd              # EnergyPlus IDD文件
│   └── Shenzhen.epw             # 天气数据文件
├── output/                       # 输出目录
│   └── idf/                     # 生成的IDF文件
├── logs/                         # 运行日志
├── main.py                       # 程序入口
├── pyproject.toml               # 项目配置
└── README.md                     # 项目文档
```

## 技术栈

### 核心依赖
- **Python 3.12+**：项目运行环境
- **EnergyPlus 25.1.0+**：建筑能耗模拟引擎
- **uv**：Python 包管理工具

### 主要库
- **eppy (>=0.5.63)**：EnergyPlus IDF 文件操作库
- **pydantic (>=2.11.7)**：数据验证和Schema定义
- **numpy (>=2.3.4)**：数值计算
- **scipy (>=1.16.2)**：科学计算（用于几何验证）
- **trimesh (>=4.9.0)**：三维几何处理
- **loguru (>=0.7.3)**：日志管理
- **pyyaml (>=6.0.2)**：YAML 文件解析

## 快速开始

### 环境要求

- Python 3.12+
- EnergyPlus 25.1.0+
- uv 包管理器

### 安装步骤

1. **克隆项目**
```bash
git clone https://github.com/ITOTI-Y/EnergyPlus-Agent.git
cd EnergyPlus-Agent
```

2. **安装依赖**
```bash
# 使用 uv 安装依赖
uv sync
```

3. **准备依赖文件**
- 确保 `dependencies/` 目录下有 `Energy+.idd` 文件
- 准备天气数据文件（如 `Shenzhen.epw`）

### 运行示例

1. **准备配置文件**

编辑 `schemas/building_schema.yaml` 文件，配置建筑参数：

```yaml
Building:
  Name: Two Zone Building
  North Axis: 0
  Terrain: Suburbs

Zone:
  - Name: Zone_West
    X Origin: 0
    Y Origin: 5
    Z Origin: 0

BuildingSurface:Detailed:
  - Name: Zone_West_Floor
    Surface Type: Floor
    Construction Name: Floor_Const
    Zone Name: Zone_West
    Vertices:
      - {X: 0, Y: 5, Z: 0}
      - {X: 5, Y: 5, Z: 0}
      # ... 更多顶点
```

2. **运行转换和模拟**

```bash
uv run main.py
```

3. **查看输出**
- IDF 文件：`output/idf/output_<timestamp>.idf`
- 运行日志：`logs/<timestamp>.log`
- EnergyPlus 模拟结果：EnergyPlus 输出目录

## 配置文件说明

### YAML 配置结构

配置文件采用 YAML 格式，主要包含以下部分：

- **SimulationControl**：模拟控制参数
- **Building**：建筑基本信息
- **Timestep**：时间步长设置
- **Site:Location**：地理位置信息
- **RunPeriod**：模拟运行周期
- **Material**：材料定义
- **Construction**：构造层定义
- **Zone**：热区定义
- **BuildingSurface:Detailed**：建筑表面详细信息
- **Output**：输出设置

### 数据验证

项目使用 Pydantic Schema 进行数据验证，包括：

- `BuildingSchema`：建筑参数验证
- `ZoneSchema`：热区参数验证
- `SurfaceSchema`：表面几何验证
- `GeometrySchema`：几何闭合性和顶点排序验证
- 其他各类配置 Schema

所有数据在转换前都会经过严格验证，确保生成的 IDF 文件符合 EnergyPlus 规范。

## 开发进度（TODO List）

### ✅ 已完成

#### 1. EP配置文件
- [x] IDF 最小化配置文件
- [x] yaml 最小配置文件
- [x] building_converter 文件
- [x] surface_converter 文件
- [x] zone_converter 文件
- [x] setting_converter 文件
- [x] material_converter 文件
- [x] constraction_converter 文件

#### 2. Converter必须完整的Pydantic验证
- [x] BuildingSchema
- [x] SettingSchema
- [x] ZoneSchema
- [x] SurfaceSchema
- [x] MaterialSchema
- [x] ConstractionShema
- [x] HVACSchema
- [x] ScheduleSchema

### 🔄 进行中

#### 3. EP执行模块
- [x] 构建 runner 用于IDF运行（优先完成）
- [x] 测试最小化配置文件运行（优先完成）
- [x] 测试完整配置文件运行
- [ ] 结果解析可视化

#### 4. 数据库
- [ ] material_database
- [ ] constraction_database（注意与material的外键关系）
- [ ] schedule_database
- [ ] HVAC_database
- [ ] 使用yaml作为存储promt，并应用设置的schema进行验证

#### 5. Rhino代码转换模块
- [ ] Setting、Building、Surface、Zone、Material、Constraction配接换
- [ ] 构建MCP服务用于LLM调用去实际code->idf
- [ ] 完成转换测试

### 📋 待开发

#### 6. MCP_服务
- [ ] 数据上传LLM的MCP tools
- [ ] 将构建代码解释的LLM的MCP tools
- [ ] 将构建的代码格式化的LLM的MCP tools
- [ ] 构建MCP用于LLM调用去实际yaml文件创建
- [ ] Agent交互
- [ ] 标准化代码——surface_generator
- [ ] 标准化代码——zone_generator
- [ ] 数据导入功能——building_generator
- [ ] 数据导入功能——setting_generator
- [ ] 导入yaml数据——add_to_yaml
- [ ] HVAC、Schedule、Constraction、Material_converter等代码（要求基于RAG网页搜索）

#### 7. 系统设置Agent构建
- [ ] 构建MCP用于LLM调用去实际idf系统设置
- [ ] 实现交互式Agent的建议
- [ ] 上传数据量的MCP tools
- [ ] 网络搜索MCP tools

#### 8. Agent_MCP
- [ ] 修改IDF文件的MCP tools

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 使用 Python 3.12+ 特性
- 遵循 PEP 8 代码风格
- 为新功能添加相应的数据验证 Schema
- 编写清晰的注释和文档字符串
- 确保所有测试通过

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 项目主页：[https://github.com/ITOTI-Y/EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)
- 问题反馈：[Issues](https://github.com/ITOTI-Y/EnergyPlus-Agent/issues)

## 致谢

- EnergyPlus 开发团队
- eppy 库开发者
- 所有为本项目做出贡献的开发者
