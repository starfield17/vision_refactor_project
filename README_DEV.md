# Developer README

本文件面向开发者，描述架构约束、开发方式和提交流程。

## 架构与约束

- 顶层模块：`train/`, `autolabel/`, `deploy/`, `scripts/`, `share/`, `work-dir/`
- 约束：`train/deploy/autolabel/scripts` 之间禁止互相 import，只能依赖 `share/`
- 业务逻辑尽量下沉到 `share/kernel/`
- 运行产物统一落在 `work-dir/`（本仓库默认不提交这些产物）

## 关键目录

- `share/config/`: 配置加载、schema 校验
- `share/types/`: 数据契约（Detection / Label / Stats / Errors）
- `share/kernel/`: 核心 pipeline 与适配器
  - `trainer/`, `infer/`, `autolabel/`, `deploy/`, `transport/`, `statistics/`
- `train/`, `autolabel/`, `deploy/`: CLI 薄入口
- `scripts/`: 运维脚本

## 配置策略

- 不提交真实配置：`work-dir/config.toml` 被 `.gitignore` 忽略
- 提交模板：`work-dir/config.example.toml`
- 相对路径解析以配置文件位置为基准，避免 cwd 变化导致不可复现

## 本地开发

```bash
python -m pip install -e .

# 语法检查（最小）
python -m py_compile share/config/config_loader.py
```

## 提交建议

- 先做小步提交（一个目标一个 commit）
- 提交前检查是否误加大文件：
  - 模型权重（`.pt`, `.onnx`）
  - 数据集、日志、数据库
  - `work-dir/` 下运行产物
- 若增加新配置项，务必同步：
  - `share/config/schema.py`
  - `work-dir/config.example.toml`
  - 相关 README 文档

## 可选下一步

- 增加 pytest smoke 回归（train/autolabel/deploy）
- 增加 CI（lint + py_compile + smoke）
