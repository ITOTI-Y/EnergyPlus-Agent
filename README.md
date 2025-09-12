![CodeRabbit Pull Request Reviews](https://img.shields.io/coderabbit/prs/github/ITOTI-Y/EnergyPlus-Agent?utm_source=oss&utm_medium=github&utm_campaign=ITOTI-Y%2FEnergyPlus-Agent&labelColor=171717&color=FF570A&link=https%3A%2F%2Fcoderabbit.ai&label=CodeRabbit+Reviews)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/ITOTI-Y/EnergyPlus-Agent)

# EnergyPlus Agent System

## 项目概述

EnergyPlus Agent是一个基于Python和MCP（Model Context Protocol）协议的智能建筑能耗模拟系统。该系统通过LLM驱动的交互式配置流程，将建筑设计从Rhino代码无缝转换为EnergyPlus IDF文件，并提供能耗分析和优化建议。

## 核心特性

### 智能转换
- **Rhino代码解析**：自动解析Rhino建筑生成代码，提取几何和材料信息
- **LLM驱动转换**：通过大语言模型理解建筑意图，生成标准化YAML配置
- **IDF自动生成**：将YAML配置映射为符合EnergyPlus标准的IDF文件

### 交互式配置
- **MCP协议支持**：基于标准MCP协议实现工具调用和上下文管理
- **智能对话引导**：通过自然语言交互补充HVAC、照明、设备等配置
- **知识库集成**：集成建筑规范和最佳实践，提供可解释的配置建议

### 仿真与优化
- **自动验证**：IDF文件完整性和合规性自动检查
- **能耗模拟**：集成EnergyPlus引擎进行精确能耗计算
- **性能评估**：多维度能耗分析和性能评估报告
- **优化建议**：基于仿真结果提供智能优化方案

## 快速开始

### 环境要求

- Python 3.11+
- EnergyPlus 25.1.0+
- uv