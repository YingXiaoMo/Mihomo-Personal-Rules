# Mihomo-Personal-Rules

一个简单的个人规则配置项目，旨在提供高效的代理配置、规则集和智能节点权重优化。项目包括各种配置文件、规则列表、图标资源，以及一个Smart训练脚本用于优化节点选择。

## 目录结构

- `archives/` - 旧规则存档文件
- `configs/` - Mihomo 配置文件，包括 fakeip、redirhost 等模式配置。
  - `example/` - 示例配置文件，如代理集合。
- `icons/` - 图标资源。
- `rules/` - 规则集（仅自用规则）。 
- `smart_trainer/` - 智能节点权重优化训练脚本。
  - `requirements.txt` - Python 依赖。
  - `train_smart.py` - 训练脚本，用于生成智能模型。

## 相关文档

- [智能权重模型训练教程](./TUTORIAL.md) — 详细介绍了如何使用训练脚本、配置自动化 GitHub Actions 以及 Rclone 同步。