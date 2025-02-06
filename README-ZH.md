
[English Version](./README.md)

# Janus-Pro ComfyUI 插件

本插件将 **Janus-Pro** 多模态模型集成至 ComfyUI，支持图像理解与文生图生成功能。可实现端到端的图像分析和创意图像生成工作流。

> **Janus-Pro 模型主页**: [https://github.com/deepseek-ai/Janus](https://github.com/deepseek-ai/Janus)

---

## 功能特性

- **多模态理解**: 解析图像并生成详细描述
- **文生图生成**: 根据文本提示生成高质量图像
- **灵活配置**: 支持 1B/7B 模型尺寸及多种精度（BF16/INT8/INT4）
- **ComfyUI 集成**: 无缝对接节点式工作流
- **批量生成**: 并行生成多张图像提升效率

---

## 安装指南

### 1. 安装依赖

- ComfyUI 的 `custom_nodes` 目录下安装插件
- 确保已安装所需依赖库：

```bash
git clone https://github.com/greengerong/ComfyUI-JanusPro-PL
pip install -r requirements.txt
```

### 2. 模型管理

插件支持以下模型：
- `deepseek-ai/Janus-Pro-1B`
- `deepseek-ai/Janus-Pro-7B`

#### 自动下载
首次使用时将自动下载模型到 `models/Janus-Pro` 目录。

#### 手动部署
将模型文件放置于以下目录结构：
```
models/Janus-Pro/
├── Janus-Pro-7B/
│   ├── config.json
│   ├── pytorch_model.bin
│   └── ...
└── Janus-Pro-1B/
    └── ...
```

---

## 使用说明

### 节点概览

1. **JanusProModelLoader（模型加载器）**
   - 加载 Janus-Pro 模型，可配置精度和尺寸
   - 输出: 模型、处理器

2. **JanusProImageUnderstanding（图像理解）**
   - 分析输入图像并生成描述
   - 输入: 图像、问题
   - 输出: 文本描述

3. **JanusProImageGenerator（图像生成器）**
   - 根据文本提示生成图像
   - 输入: 提示词、温度系数、CFG 权重、图像尺寸、随机种子
   - 输出: 生成图像

---

### 示例工作流

以下工作流展示图像理解与生成的端到端流程：

![](./workflows/workflow.png)

---

### 参数详解

#### JanusProModelLoader
- **模型名称**: 选择 `Janus-Pro-1B` 或 `Janus-Pro-7B`
- **计算精度**: 可选 BF16/INT8/INT4
- **本地目录**: 模型存储路径（默认: `models/Janus-Pro`）

#### JanusProImageUnderstanding
- **图像输入**: 待分析的图像
- **分析问题**: 提示问题（如："详细描述此图像"）
- **最大长度**: 生成描述的最大 token 数（默认: 512）

#### JanusProImageGenerator
- **提示词**: 图像生成提示文本
- **温度系数**: 控制随机性（默认: 0.8，范围: 0.0-1.0）
- **CFG 权重**: 控制提示词遵循度（默认: 5.0，无上限）
- **并行数量**: 同时生成图像数（默认: 16）
- **图像尺寸**: 输出分辨率（当前仅支持 384）
- **随机种子**: 确保结果可复现

---

## 性能优化

- **低显存配置**：
  - 使用 `Janus-Pro-1B` 模型
  - 启用 `INT4` 量化
  - 减少 `并行数量`

- **加速生成**：
  - 使用 `BF16` 精度
  - 增加 `并行数量`（建议 16-32）

- **高质量输出**：
  - 使用 `Janus-Pro-7B` + `BF16` 精度
  - 设置 `温度系数` 0.6-0.8
  - 设置 `CFG 权重` 6.0-8.0

---

## 常见问题

1. **显存不足**：
   - 降低 `并行数量`
   - 使用小模型 (`Janus-Pro-1B`)
   - 启用量化 (`INT8/INT4`)

2. **模型加载失败**：
   - 检查网络连接
   - 验证 `models/Janus-Pro` 目录权限

3. **输出质量不佳**：
   - 提高 `CFG 权重`
   - 使用更详细的提示词
   - 调整 `温度系数`（低值更稳定）

---

## 致谢

- [DeepSeek-AI](https://github.com/deepseek-ai/Janus) 提供 Janus-Pro 模型
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) 优秀的工作流框架

---

## 联系我们

如有问题或建议，请在 GitHub 提交 issue 或联系维护者。
