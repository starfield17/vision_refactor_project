# Vision Refactor Project

一个独立于旧仓库的模块化视觉工程（Train / AutoLabel / Deploy / Statistics）。

## 给使用者（User）

这份 README 侧重“怎么跑起来”。
开发细节请看 `README_DEV.md`。

### 功能概览

- 单入口训练：`python -m train.cli`
- 自动标注：`python -m autolabel.cli`（model / llm）
- 部署：
  - `deploy.edge.cli`（local / stream / llm）
  - `deploy.remote.cli`（远端推理服务）
- 统计系统：
  - `deploy.statistics.api`（7797）
  - `deploy.statistics.ui`（7796）

## 快速开始

### 1) 安装依赖

```bash
python -m pip install -e .
```

### 2) 准备本地配置（不会提交到 Git）

```bash
cp work-dir/config.example.toml work-dir/config.toml
```

然后按你的机器环境修改 `work-dir/config.toml` 里的路径与模型参数。

### 3) 常用命令

```bash
# 训练
python -m train.cli --workdir ./work-dir --config ./work-dir/config.toml

# 自动标注
python -m autolabel.cli --workdir ./work-dir --config ./work-dir/config.toml

# 启动统计 API/UI
python -m deploy.statistics.api --workdir ./work-dir --config ./work-dir/config.toml
python -m deploy.statistics.ui  --workdir ./work-dir --config ./work-dir/config.toml

# Edge 本地推理
python -m deploy.edge.cli --workdir ./work-dir --config ./work-dir/config.toml

# Remote 推理服务
python -m deploy.remote.cli --workdir ./work-dir --config ./work-dir/config.toml
```

## 仓库说明

- 代码目录：`train/` `autolabel/` `deploy/` `share/` `scripts/`
- 运行目录：`work-dir/`（默认仅保留模板，不提交运行产物）
- 交接记录 `codex.md` 与本地 `config.toml` 已默认被 `.gitignore` 排除
