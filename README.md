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

### MCP 服务器
- **FastMCP 框架**：基于 FastMCP 实现的高性能 MCP 服务器
- **多传输协议**：支持 stdio、HTTP、SSE、streamable-http 多种传输方式
- **Zone 管理**：完整的热区 CRUD 操作接口
- **工作流工具**：配置导出、加载、验证和模拟运行

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
│   │   ├── base_converter.py     # 转换器基类
│   │   ├── building_converter.py # 建筑信息转换器
│   │   ├── construction_converter.py # 构造层转换器
│   │   ├── fenestration_converter.py # 窗户/开口转换器
│   │   ├── hvac_converter.py     # HVAC系统转换器
│   │   ├── material_converter.py # 材料转换器
│   │   ├── schedule_converter.py # 时间表转换器
│   │   ├── setting_converter.py  # 设置转换器
│   │   ├── surface_converter.py  # 表面转换器
│   │   └── zone_converter.py     # 热区转换器
│   ├── mcp/                      # MCP服务器模块
│   │   ├── server.py             # FastMCP服务器入口
│   │   ├── state.py              # 配置状态管理
│   │   ├── interface.py          # 接口定义
│   │   └── tools/                # MCP工具集
│   │       ├── base.py           # 工具基类
│   │       ├── workflow.py       # 工作流工具
│   │       └── zone.py           # 热区管理工具
│   ├── validator/                # 数据验证模块
│   │   └── data_model.py         # Pydantic 数据模型和Schema
│   ├── runner/                   # EnergyPlus 运行器
│   │   └── runner.py             # EnergyPlus 执行模块
│   ├── utils/                    # 工具模块
│   │   └── logging.py            # 日志配置
│   └── converter_manager.py      # 转换器管理器
├── schemas/                      # 配置文件模板
│   ├── building_schema.yaml      # 建筑配置YAML示例
│   └── example/                  # 示例文件
├── dependencies/                 # 依赖文件
│   ├── Energy+.idd               # EnergyPlus IDD文件
│   └── Shenzhen.epw              # 天气数据文件
├── docker/                       # Docker配置
│   ├── Dockerfile                # Docker镜像配置
│   └── docker-compose.yml        # Docker Compose配置
├── docs/                         # 文档目录
├── output/                       # 输出目录
│   └── idf/                      # 生成的IDF文件
├── logs/                         # 运行日志
├── main.py                       # 程序入口（CLI）
├── pyproject.toml                # 项目配置
└── README.md                     # 项目文档
```

## 技术栈

### 核心依赖
- **Python 3.12+**：项目运行环境
- **EnergyPlus 25.1.0+**：建筑能耗模拟引擎
- **uv**：Python 包管理工具

### 主要库
- **fastmcp (>=2.14.1)**：MCP协议服务器框架
- **eppy (>=0.5.63)**：EnergyPlus IDF 文件操作库
- **pydantic (>=2.11.7)**：数据验证和Schema定义
- **omegaconf (>=2.3.0)**：配置管理
- **typer (>=0.20.1)**：CLI框架
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

#### 3. Docker 部署

```bash
# 进入 docker 目录
cd docker

# 构建并启动服务
# 方式 A：使用 docker-compose 启动
docker-compose up -d

# 方式 B：仅使用 docker run （不要与方式 A 同时运行）
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

#### 4. .env 环境配置

```.env
# RAG data configuration
# Gemini API Configuration
GEMINI_API_KEY=Your Gemini api key

# Qdrant Configuration
QDRANT_API_KEY=  # 本地 Docker 部署可留空；云端部署需配置
QDRANT_ENDPOINT=http://localhost:6333
QDRANT_COLLECTION_NAME=energyplus_database

# Database Index Building Configuration

INDEX_DB_PATH=data/database/EP_Agent_data.db
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

#### Zone 管理
| 工具 | 描述 |
|------|------|
| `create_zone` | 创建新的热区 |
| `get_zone` | 获取热区信息 |
| `update_zone` | 更新热区配置 |
| `delete_zone` | 删除热区 |
| `list_zones` | 列出所有热区 |

#### 工作流
| 工具 | 描述 |
|------|------|
| `load_yaml` | 加载 YAML 配置文件 |
| `export_yaml` | 导出当前配置为 YAML |
| `validate_config` | 验证当前配置 |
| `run_simulation` | 运行 EnergyPlus 模拟 |
| `get_summary` | 获取配置摘要 |
| `clear_all` | 清空所有配置 |

### 资源端点
| 资源 | 描述 |
|------|------|
| `config://current` | 获取当前完整配置 |
| `config://summary` | 获取配置摘要 |

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

### 已完成

#### 1. EP配置文件
- [x] IDF 最小化配置文件
- [x] yaml 最小配置文件
- [x] building_converter 文件
- [x] surface_converter 文件
- [x] zone_converter 文件
- [x] setting_converter 文件
- [x] material_converter 文件
- [x] construction_converter 文件
- [x] hvac_converter 文件
- [x] schedule_converter 文件
- [x] fenestration_converter 文件

#### 2. Converter Pydantic验证
- [x] BuildingSchema
- [x] SettingSchema
- [x] ZoneSchema
- [x] SurfaceSchema
- [x] MaterialSchema
- [x] ConstructionSchema
- [x] HVACSchema
- [x] ScheduleSchema

#### 3. EP执行模块
- [x] 构建 runner 用于IDF运行
- [x] 测试最小化配置文件运行
- [x] 测试完整配置文件运行

#### 4. MCP服务器基础功能
- [x] FastMCP 服务器框架搭建
- [x] 配置状态管理（ConfigState）
- [x] Zone CRUD 工具实现
- [x] Workflow 工具实现（load/export/validate/run）
- [x] 多传输协议支持（stdio/http/sse/streamable-http）
- [x] CLI 入口（Typer）
- [x] Docker 支持

### 进行中

#### 5. 结果解析与可视化
- [ ] 模拟结果解析
- [ ] 结果可视化

#### 6. 数据库
- [ ] material_database
- [ ] construction_database（注意与material的外键关系）
- [ ] schedule_database
- [ ] HVAC_database
- [ ] 使用yaml作为存储prompt，并应用设置的schema进行验证

#### 7. Rhino代码转换模块
- [ ] Setting、Building、Surface、Zone、Material、Construction配置转换
- [ ] 构建MCP服务用于LLM调用去实际code → idf
- [ ] 完成转换测试

### 待开发

#### 8. MCP工具扩展
- [ ] Surface CRUD 工具
- [ ] Material CRUD 工具
- [ ] Construction CRUD 工具
- [ ] HVAC配置工具
- [ ] Schedule配置工具
- [ ] 数据上传LLM的MCP tools
- [ ] 将构建代码解释的LLM的MCP tools
- [ ] 将构建的代码格式化的LLM的MCP tools
- [ ] 修改IDF文件的MCP tools

#### 9. 系统设置Agent构建
- [ ] 构建MCP用于LLM调用去实际idf系统设置
- [ ] 实现交互式Agent的建议
- [ ] 上传数据量的MCP tools
- [ ] 网络搜索MCP tools

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
- 使用 Ruff 进行代码检查
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
- FastMCP 开发团队
- 所有为本项目做出贡献的开发者
