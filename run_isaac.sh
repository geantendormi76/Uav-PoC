#!/bin/bash
# ================================================================= #
# run_isaac.sh - 工业级Isaac Sim for Windows 启动器
# 版本: 12.0 (The Final Path Fix)
#
# 功能:
#   基于对v11.0日志的最终分析，修正了向PowerShell传递错误路径
#   格式的根本性错误。确保在最终命令中使用的所有路径变量都
#   是经过wslpath转换的Windows路径。
#
# 变更日志 (v12.0):
#   - 决定性修正: 在powershell.exe -Command字符串中，将所有
#     路径变量都替换为经过wslpath转换后的Windows路径变量。
#     这直接解决了"无法加载模块"的路径解释错误。
# ================================================================= #

# --- 脚本行为设置 ---
set -euo pipefail

# --- 配置区 ---
readonly ISAAC_SIM_ROOT_PATH="/mnt/c/Users/zhz/Omniverse/Isaac-Sim"
readonly PROJECT_ROOT_PATH="/mnt/c/Users/zhz/uav"

# --- 定义权威的启动器路径 (WSL格式) ---
readonly ISAAC_LAUNCHER_WSL_PATH="${ISAAC_SIM_ROOT_PATH}/isaac-sim.bat"

# --- 执行区 ---

# 1. 检查输入参数
if [[ $# -eq 0 ]]; then
    echo "错误: 未提供要执行的Python脚本文件名。" >&2
    echo "用法: ./run_isaac.sh <your_script_name.py>" >&2
    exit 1
fi
readonly SCRIPT_NAME="$1"
readonly SCRIPT_WSL_PATH="$PROJECT_ROOT_PATH/$SCRIPT_NAME"

# 2. 验证路径
if [[ ! -f "$ISAAC_LAUNCHER_WSL_PATH" ]]; then
    echo "错误: 未找到Isaac Sim的官方启动器: '$ISAAC_LAUNCHER_WSL_PATH'" >&2
    exit 1
fi
if [[ ! -f "$SCRIPT_WSL_PATH" ]]; then
    echo "错误: 目标Python脚本不存在: '$SCRIPT_WSL_PATH'" >&2
    exit 1
fi

# 3. 将所有需要的WSL路径一次性转换为Windows路径
readonly ISAAC_LAUNCHER_WIN_PATH=$(wslpath -w "$ISAAC_LAUNCHER_WSL_PATH")
readonly SCRIPT_WIN_PATH=$(wslpath -w "$SCRIPT_WSL_PATH")

echo ">>> 正在通过最终版启动器(v12.0 - The Final Path Fix)启动 Isaac Sim..."
echo "    - 启动器 (Windows): $ISAAC_LAUNCHER_WIN_PATH"
echo "    - 脚本 (Windows): $SCRIPT_WIN_PATH"
echo "-----------------------------------------------------"

# 4. 使用最稳健的PowerShell执行官方启动器 (使用完全正确的Windows路径)
if ! powershell.exe -ExecutionPolicy Bypass -Command "& '${ISAAC_LAUNCHER_WIN_PATH}' --run '${SCRIPT_WIN_PATH}'"; then
    echo "-----------------------------------------------------" >&2
    echo ">>> 错误: Isaac Sim 执行过程中发生错误。" >&2
    exit 1
fi

echo "-----------------------------------------------------"
echo ">>> Isaac Sim 执行完成。"