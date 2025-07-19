# Project Chimera - 数字孪生无人机感知模块

## 愿景
构建一个能在边缘设备（无人机）上自主运行的、实时的、高能效的感知与决策系统。

## 当前阶段
**Project Apollo (PoC):** 在WSL2本地环境中，开发并验证一个基于ROS 2和YOLOv8的、模块化的视觉感知系统。

## 目录结构
- `/ros2_ws`: ROS 2工作空间，存放所有感知与决策节点。
- `/sim_scripts`: 存放与NVIDIA Isaac Sim交互的独立Python脚本。
- `/assets`: 存放项目所需的非代码资源（如测试视频）。
EOF