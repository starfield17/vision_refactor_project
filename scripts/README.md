# scripts

这些脚本来自旧项目 `common/` 中的 shell 脚本迁移（仅 `.sh`）。

新增（重构项目运行辅助）：
- `start_stats.sh`：一键启动 Statistics API(7797) + UI(7796)，并做健康检查。
- `stop_stats.sh`：停止 Statistics API/UI（基于 pid 文件）。
- `status_stats.sh`：查看 Statistics API/UI 进程与健康状态。
- `restart_stats.sh`：重启 Statistics API/UI（stop + start）。

未迁移内容：
- `common/rk3588/*`（按重构手册要求移除 RK3588/NPU 相关）
- `common/uart-manager.sh`（按重构手册要求移除串口相关）
