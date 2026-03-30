![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ITOTI-Y/EnergyPlus-Agent?utm_source=oss&utm_medium=github&utm_campaign=ITOTI-Y%2FEnergyPlus-Agent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ITOTI-Y/EnergyPlus-Agent)

# EnergyPlus Agent System

## 项目概述

EnergyPlus Agent 是一个基于 Python 和 MCP（Model Context Protocol）协议的智能建筑能耗模拟系统。该系统通过 LLM 驱动的交互式配置流程，将建筑设计从 YAML 配置无缝转换为 EnergyPlus IDF 文件，并集成 RAG 知识库提供能耗分析和优化建议。未来计划通过 LangGraph 实现多模态（图片+文本）输入，结合 MCP 工具自动构建 IDF 文件。

## 核心特性

### 智能转换
- **YAML 配置解析**：自动解析 YAML 建筑配置，提取几何、材料、HVAC、负荷等信息
- **LLM 驱动转换**：通过大语言模型理解建筑意图，生成标准化 YAML 配置
- **IDF 自动生成**：13 个专用转换器将 YAML 配置映射为符合 EnergyPlus 标准的 IDF 文件
- **严格数据验证**：基于 Pydantic Schema 的完整数据验证，覆盖所有 EnergyPlus 对象

### MCP 服务器
- **FastMCP 框架**：基于 FastMCP 实现的高性能 MCP 服务器
- **多传输协议**：支持 stdio、HTTP、SSE、streamable-http 多种传输方式
- **完整 CRUD 工具集**：覆盖 Building、Zone、Surface、Material、Construction、Fenestration、Schedule、HVAC、People、Light 等所有组件
- **工作流工具**：配置导出、加载、验证和模拟运行

### RAG 知识库
- **异步向量化管道**：基于 Gemini Embedding 和 Qdrant 向量数据库的异步 RAG 系统
- **速率限制与重试**：内置速率限制、并发控制和 429/RESOURCE_EXHAUSTED 自动重试
- **增量同步**：支持增量同步和过期数据自动清理
- **类型化结果**：使用 dataclass 类型化的搜索结果（QdrantData、RowRecord、VectorizedResult）

### 数据库工具
- **EnergyPlus 数据管理**：标准材料、无质量材料、构造、日程、设计日等数据管理
- **SQLite 索引**：基于 SQLite 的数据索引和检索

### 仿真与优化
- **自动验证**：IDF 文件完整性和跨引用合规性自动检查
- **能耗模拟**：集成 EnergyPlus 引擎进行精确能耗计算

## 项目结构

```
EnergyPlus-Agent/
├── src/                              # 源代码目录
│   ├── converters/                   # IDF 转换器模块（13个转换器）
│   │   ├── base_converter.py         # 转换器基类
│   │   ├── building_converter.py     # 建筑信息转换器
│   │   ├── construction_converter.py # 构造层转换器
│   │   ├── fenestration_converter.py # 窗户/开口转换器
│   │   ├── hvac_converter.py         # HVAC 系统转换器
│   │   ├── light_converter.py        # 照明负荷转换器
│   │   ├── material_converter.py     # 材料转换器
│   │   ├── people_converter.py       # 人员负荷转换器
│   │   ├── schedule_converter.py     # 时间表转换器
│   │   ├── setting_converter.py      # 模拟设置转换器
│   │   ├── surface_converter.py      # 表面转换器
│   │   └── zone_converter.py         # 热区转换器
│   ├── mcp/                          # MCP 服务器模块
│   │   ├── server.py                 # FastMCP 服务器入口
│   │   ├── state.py                  # 配置状态管理（ConfigState）
│   │   ├── interface.py              # 接口和数据模型定义
│   │   ├── api/                      # MCP 工具注册（按功能分组）
│   │   │   ├── core.py               # 核心工具（Building, Location, Zone, Surface）
│   │   │   ├── envelope.py           # 围护工具（Material, Construction, Fenestration）
│   │   │   ├── schedule.py           # 日程工具（ScheduleTypeLimits, ScheduleCompact）
│   │   │   ├── hvac.py               # HVAC 工具（Thermostat, IdealLoadsSystem）
│   │   │   ├── loads.py              # 负荷工具（People, Light）
│   │   │   ├── workflow.py           # 工作流工具（export, load, validate, simulate）
│   │   │   ├── resources.py          # 资源端点
│   │   │   └── common.py             # 通用工具函数
│   │   └── tools/                    # MCP 工具实现（14个工具类）
│   │       ├── base.py               # CRUD 工具基类
│   │       ├── zone.py               # 热区工具
│   │       ├── building.py           # 建筑工具
│   │       ├── location.py           # 位置工具
│   │       ├── material.py           # 材料工具
│   │       ├── construction.py       # 构造工具
│   │       ├── surface.py            # 表面工具
│   │       ├── fenestration.py       # 窗户/开口工具
│   │       ├── schedule_type_limits.py  # 日程类型限制工具
│   │       ├── schedule_compact.py   # 紧凑日程工具
│   │       ├── thermostat.py         # 恒温器工具
│   │       ├── ideal_loads_system.py # 理想负荷系统工具
│   │       ├── people.py             # 人员工具
│   │       ├── light.py              # 照明工具
│   │       └── workflow.py           # 工作流工具
│   ├── rag/                          # RAG 检索增强生成模块
│   │   ├── rag.py                    # RAG 系统主类（异步同步管道）
│   │   ├── embedding.py             # Gemini 嵌入模型
│   │   ├── vector.py                # Qdrant 向量存储（同步/异步）
│   │   └── chunk.py                 # 文本块处理和 SQLite 处理器
│   ├── database/                    # 数据库模块
│   │   └── datatools/               # 数据工具集
│   │       ├── standard_materials.py # 标准材料数据
│   │       ├── nomass_materials.py   # 无质量材料数据
│   │       ├── constructions.py      # 构造数据
│   │       ├── schedulecompact.py    # 紧凑日程数据
│   │       ├── scheduletypelimits.py # 日程类型限制数据
│   │       ├── designday.py          # 设计日数据
│   │       └── datadescription.py    # 数据描述
│   ├── validator/                   # 数据验证模块
│   │   └── data_model.py           # Pydantic 数据模型和 Schema（33+ 类）
│   ├── runner/                      # EnergyPlus 运行器
│   │   └── runner.py               # EnergyPlus 执行模块
│   ├── configs/                     # 配置管理
│   │   ├── config.py               # EmbeddingConfig（Pydantic + YAML）
│   │   └── embedding.yaml          # 嵌入模型配置
│   ├── utils/                       # 工具模块
│   │   └── logging.py              # Loguru 日志配置
│   └── converter_manager.py         # 转换器管理器
├── data/                            # 数据目录
│   ├── schemas/                     # YAML 配置文件
│   │   ├── building_schema.yaml     # 建筑配置示例
│   │   └── example/                 # 更多示例文件
│   ├── dependencies/                # 依赖文件
│   │   └── Energy+.idd             # EnergyPlus IDD 数据字典
│   ├── weather/                     # 天气数据
│   │   └── Shenzhen.epw            # 深圳天气文件
│   └── examples/                    # 示例数据库
│       └── EP_Agent_data.db        # SQLite 示例数据
├── docker/                          # Docker 配置
│   ├── Dockerfile                   # 基于 nrel/energyplus:25.1.0
│   └── docker-compose.yml          # Docker Compose 配置
├── output/                          # 输出目录（IDF、日志、YAML）
├── main.py                          # 程序入口（CLI）
└── pyproject.toml                   # 项目配置
```

## 技术栈

### 核心依赖
- **Python 3.12+**：项目运行环境
- **EnergyPlus 25.1.0+**：建筑能耗模拟引擎
- **uv**：Python 包管理工具

### 主要库
| 库 | 版本 | 用途 |
|---|---|---|
| **fastmcp** | >=2.14.1 | MCP 协议服务器框架 |
| **eppy** | >=0.5.63 | EnergyPlus IDF 文件操作 |
| **pydantic** | >=2.11.7 | 数据验证和 Schema 定义 |
| **google-genai** | >=1.68.0 | Gemini Embedding API |
| **qdrant-client** | >=1.17.1 | Qdrant 向量数据库客户端 |
| **omegaconf** | >=2.3.0 | 配置管理 |
| **typer** | >=0.20.1 | CLI 框架 |
| **numpy** | >=2.3.4 | 数值计算 |
| **scipy** | >=1.16.2 | 科学计算（几何验证） |
| **trimesh** | >=4.9.0 | 三维几何处理 |
| **loguru** | >=0.7.3 | 日志管理 |
| **tqdm** | >=4.67.3 | 进度条显示 |
| **pyyaml** | >=6.0.2 | YAML 文件解析 |

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
uv sync
```

3. **准备依赖文件**
- 确保 `data/dependencies/` 目录下有 `Energy+.idd` 文件
- 准备天气数据文件（如 `data/weather/Shenzhen.epw`）

4. **配置环境变量**

复制 `.env.example` 为 `.env` 并填写：
```bash
cp .env.example .env
```

```env
# Qdrant 向量数据库配置
QDRANT_ENDPOINT=http://localhost:6333
QDRANT_API_KEY=                        # 本地 Docker 部署可留空

# Gemini API 配置（RAG 嵌入所需）
GEMINI_API_KEY=your_gemini_api_key
```

### 运行方式

#### 1. IDF 转换和模拟

```bash
# 将 YAML 配置转换为 IDF 并运行模拟
uv run main.py convert-idf
```

#### 2. MCP 服务器

```bash
# 启动 MCP 服务器（stdio 模式，用于 Claude Desktop 等）
uv run main.py mcp-server

# 启动 HTTP 模式服务器
uv run main.py mcp-server --transport http --host 0.0.0.0 --port 8000

# 支持的传输协议：stdio, http, sse, streamable-http
```

#### 3. RAG 数据库构建

```bash
# 启动 Qdrant 向量数据库（Docker）
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# 构建 RAG 嵌入索引
uv run main.py embedding --collection energyplus_database --db-path data/examples/EP_Agent_data.db
```

#### 4. Docker 部署

```bash
cd docker

# 使用 docker-compose 构建并启动 MCP HTTP 服务
docker-compose up -d
```

### 配置 Claude Desktop

在 Claude Desktop 的配置文件中添加：

```json
{
  "mcpServers": {
    "energyplus-agent": {
      "command": "uv",
      "args": ["--directory", "/path/to/EnergyPlus-Agent", "run", "main.py", "mcp-server"]
    }
  }
}
```

## MCP 服务器

### 可用工具

#### Core 核心工具
| 工具 | 描述 |
|------|------|
| `create_building` / `get_building` / `update_building` / `delete_building` / `list_buildings` | 建筑信息 CRUD |
| `create_location` / `get_location` / `update_location` / `delete_location` / `list_locations` | 站点位置 CRUD |
| `create_zone` / `get_zone` / `update_zone` / `delete_zone` / `list_zones` | 热区 CRUD |

#### Envelope 围护工具
| 工具 | 描述 |
|------|------|
| `create_standard_material` / `create_no_mass_material` / `create_air_gap_material` / `create_glazing_material` | 创建不同类型材料 |
| `get_material` / `update_*_material` / `delete_material` / `list_materials` | 材料查询/更新/删除 |
| `create_construction` / `get_construction` / `update_construction` / `delete_construction` / `list_constructions` | 构造层 CRUD |
| `create_surface` / `get_surface` / `update_surface` / `delete_surface` / `list_surfaces` | 建筑表面 CRUD |
| `create_fenestration_surface` / `get_fenestration_surface` / `update_fenestration_surface` / `delete_fenestration_surface` / `list_fenestration_surfaces` | 窗户/开口 CRUD |

#### Schedule 日程工具
| 工具 | 描述 |
|------|------|
| `create_schedule_type_limits` / `get_schedule_type_limits` / `update_schedule_type_limits` / `delete_schedule_type_limits` / `list_schedule_type_limits` | 日程类型限制 CRUD |
| `create_schedule_compact` / `get_schedule_compact` / `update_schedule_compact` / `delete_schedule_compact` / `list_schedule_compacts` | 紧凑日程 CRUD |

#### HVAC 暖通工具
| 工具 | 描述 |
|------|------|
| `create_hvac_thermostat` / `get_hvac_thermostat` / `update_hvac_thermostat` / `delete_hvac_thermostat` / `list_hvac_thermostats` | 恒温器 CRUD |
| `create_hvac_ideal_loads_system` / `get_hvac_ideal_loads_system` / `update_hvac_ideal_loads_system` / `delete_hvac_ideal_loads_system` / `list_hvac_ideal_loads_systems` | 理想负荷系统 CRUD |

#### Loads 负荷工具
| 工具 | 描述 |
|------|------|
| `create_people` / `get_people` / `update_people` / `delete_people` / `list_people` | 人员负荷 CRUD |
| `create_light` / `get_light` / `update_light` / `delete_light` / `list_lights` | 照明负荷 CRUD |

#### Workflow 工作流
| 工具 | 描述 |
|------|------|
| `export_yaml` | 导出当前配置为 YAML 文件 |
| `load_yaml` | 加载 YAML 配置文件 |
| `validate_config` | 验证所有跨引用配置 |
| `run_simulation` | 运行 EnergyPlus 模拟 |
| `get_summary` | 获取配置摘要 |
| `clear_all` | 清空所有配置 |

### 资源端点
| 资源 | 描述 |
|------|------|
| `config://current` | 获取当前完整配置（YAML 格式） |
| `config://summary` | 获取配置摘要 |

## 配置文件说明

### YAML 配置结构

配置文件采用 YAML 格式，主要包含以下部分：

- **SimulationControl**：模拟控制参数
- **Building**：建筑基本信息（名称、北轴、地形）
- **Timestep**：时间步长设置
- **Site:Location**：地理位置信息
- **RunPeriod**：模拟运行周期
- **GlobalGeometryRules**：全局几何规则
- **Material**：材料定义（标准、无质量、玻璃、空气间隙）
- **Construction**：构造层定义
- **Zone**：热区定义
- **BuildingSurface:Detailed**：建筑表面详细信息
- **FenestrationSurface:Detailed**：窗户/开口详细信息
- **ScheduleTypeLimits / Schedule:Compact**：日程定义
- **HVACTemplate:Thermostat / HVACTemplate:Zone:IdealLoadsAirSystem**：HVAC 系统
- **People / Lights**：人员和照明负荷
- **Output:Variable / Output:Meter**：输出设置

### 数据验证

项目使用 Pydantic Schema 进行数据验证（33+ Schema 类），包括：

- **建筑组件**：`BuildingSchema`、`SiteLocationSchema`、`ZoneSchema`
- **材料**：`StandardMaterialSchema`、`NoMassMaterialSchema`、`GlazingMaterialSchema`、`AirGapMaterialSchema`
- **构造**：`ConstructionSchema`
- **表面**：`SurfaceSchema`、`FenestrationSurfaceSchema`
- **几何**：`GeometrySchema`（验证顶点闭合性和排序）
- **日程**：`ScheduleTypeLimitsSchema`、`ScheduleCompactSchema`
- **HVAC**：`HVACTemplateThermostatSchema`、`HVACTemplateZoneIdealLoadsAirSystemSchema`
- **负荷**：`PeopleSchema`、`LightSchema`
- **模拟控制**：`SimulationControlSchema`、`RunPeriodSchema`、`GlobalGeometryRulesSchema`

所有数据在转换前都会经过严格验证，确保生成的 IDF 文件符合 EnergyPlus 规范。

## CLI 命令

| 命令 | 描述 |
|------|------|
| `uv run main.py convert-idf` | 将 YAML 配置转换为 IDF 并运行模拟 |
| `uv run main.py mcp-server [--transport] [--host] [--port]` | 启动 MCP 服务器 |
| `uv run main.py embedding --collection <name> --db-path <path>` | 构建 RAG 嵌入索引 |
| `energyplus-mcp` | 直接运行 MCP 服务器（通过 pyproject.toml scripts） |

## 开发进度（TODO List）

### 已完成

#### 1. EP 配置文件与转换器
- [x] IDF 最小化配置文件
- [x] YAML 最小配置文件
- [x] 13 个转换器（Building、Zone、Surface、Setting、Material、Construction、HVAC、Schedule、Fenestration、Light、People）

#### 2. Pydantic 数据验证
- [x] 33+ Schema 类覆盖所有 EnergyPlus 对象
- [x] 几何闭合性和顶点排序验证
- [x] 跨引用验证

#### 3. EP 执行模块
- [x] 构建 runner 用于 IDF 运行
- [x] 测试最小化和完整配置文件运行

#### 4. MCP 服务器
- [x] FastMCP 服务器框架搭建
- [x] 配置状态管理（ConfigState）
- [x] 完整 CRUD 工具集（Building、Location、Zone、Surface、Material、Construction、Fenestration、Schedule、HVAC、People、Light）
- [x] 工作流工具（load/export/validate/run/summary/clear）
- [x] 资源端点（config://current、config://summary）
- [x] 多传输协议支持（stdio/http/sse/streamable-http）
- [x] CLI 入口（Typer）
- [x] Docker 支持

#### 5. RAG 知识库
- [x] 异步嵌入管道（Gemini Embedding + Qdrant）
- [x] 速率限制、并发控制和重试机制
- [x] 增量同步和过期数据清理
- [x] 类型化搜索结果

#### 6. 数据库工具
- [x] 标准材料、无质量材料、构造、日程、设计日数据管理
- [x] SQLite 索引和数据描述

### 待开发

#### 7. 结果解析与可视化
- [ ] 模拟结果解析
- [ ] 结果可视化

#### 8. 多模态 IDF 构建（LangGraph）
- [ ] 通过 LLM 读取图片+文本输入，理解建筑设计意图
- [ ] 结合 MCP 工具自动构建 IDF 文件
- [ ] 基于 LangGraph 实现多步骤 Agent 编排

#### 9. MCP 工具扩展
- [ ] 数据上传 LLM 的 MCP tools
- [ ] 将构建代码解释的 LLM 的 MCP tools
- [ ] 将构建的代码格式化的 LLM 的 MCP tools
- [ ] 修改 IDF 文件的 MCP tools

#### 10. 系统设置 Agent 构建
- [ ] 构建 MCP 用于 LLM 调用去实际 idf 系统设置
- [ ] 实现交互式 Agent 的建议
- [ ] 网络搜索 MCP tools

## 贡献指南

欢迎贡献代码和建议！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

### 代码规范

- 使用 Python 3.12+ 特性
- 使用 Ruff 进行代码检查和格式化（配置见 `pyproject.toml`）
- 为新功能添加相应的 Pydantic 数据验证 Schema
- 编写清晰的注释和文档字符串
- 确保所有测试通过

## 联系方式

- 项目主页：[https://github.com/ITOTI-Y/EnergyPlus-Agent](https://github.com/ITOTI-Y/EnergyPlus-Agent)
- 问题反馈：[Issues](https://github.com/ITOTI-Y/EnergyPlus-Agent/issues)
