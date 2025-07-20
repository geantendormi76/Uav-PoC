# Project Chimera - 本地化无人机感知引擎 (PoC: Apollo)

[![ROS Version](https://img.shields.io/badge/ROS-Humble-blueviolet)](https://docs.ros.org/en/humble/index.html)
[![Python Version](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/downloads/release/python-3100/)
[![Build System](https://img.shields.io/badge/Build-colcon-cyan)](https://colcon.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Apache--2.0-green)](./LICENSE)

## 核心愿景 (Project Chimera)

构建一个能在边缘设备（如无人机）上自主运行的、实时的、高能效的感知与决策系统，最终实现自动化跟踪、拍摄与物流运送等高级功能。

## 项目背景与当前状态 (PoC: Apollo)

本项目是“奇美拉计划”的第一个、也是最关键的概念验证（PoC）阶段——“阿波罗计划”。

**核心痛点与解决方案：** 在复杂的机器人项目（尤其是涉及AI、多系统集成时）初期，开发者往往会陷入环境配置、依赖冲突和构建失败的“黑洞”中。为了规避这一风险，“阿波罗计划”严格遵循**“本地优先，增量验证”**的核心战略。我们首先在纯WSL2环境中，构建并验证一个完全解耦的、模块化的“本地感知引擎”。

**当前已完成的里程碑：**
我们已经成功地在Windows 11 + WSL2 (Ubuntu 22.04) 环境中，搭建了一个功能完整的、基于ROS 2 Humble的模块化视觉感知系统。该系统：
1.  **模拟数据源 (`uav_simulation`)**: 能从本地视频文件模拟无人机摄像头，并以`sensor_msgs/msg/Image`格式作为ROS 2话题循环发布。
2.  **高性能感知核心 (`uav_perception`)**: 利用业界先进的YOLOv12模型（ONNX格式）和高性能的ONNX Runtime（GPU加速），对视频流进行实时的物体检测。
3.  **标准化接口 (`uav_interfaces`)**: 将结构化的检测结果（类别、置信度、边界框）封装在自定义的ROS 2消息中发布。
4.  **实时可视化 (`uav_visualizer`)**: 将检测结果实时、直观地叠加在视频流上，并通过Rviz2进行可视化验证。

本项目不仅是一个功能原型，更重要的是，它沉淀了一套在复杂环境中（WSL2 + venv + ROS 2）进行Python节点开发的、经过反复验证的**“黄金标准工作流”**。

## “像素级”复刻指南

本指南旨在让任何具备基础ROS 2知识的开发者，都能在30分钟内，100%复现我们的成功结果。

### 第一步：环境准备

1.  **核心系统**: 确保您拥有一个标准的 **Windows 11 + WSL2 (Ubuntu 22.04)** 环境。
2.  **安装ROS 2 Humble**: 严格按照[ROS 2官方文档](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debians.html)的指引，通过`apt`安装`ros-humble-desktop`和`ros-dev-tools`。
3.  **克隆本项目**:
    ```bash
    git clone https://github.com/geantendormi76/isaac-sim-uav.git
    cd isaac-sim-uav
    ```
4.  **创建“混合”虚拟环境 (关键步骤)**:
    ```bash
    # 在项目根目录的ros2_ws下
    cd ros2_ws
    python3 -m venv .venv --system-site-packages --symlinks
    ```
    *   **原因**: `--system-site-packages`允许我们的虚拟环境访问系统级的ROS 2 Python包，同时又能隔离我们自己的项目依赖，这是解决编译时/运行时环境冲突的基石。

### 第二步：安装依赖

1.  **激活虚拟环境**:
    ```bash
    source .venv/bin/activate
    ```
2.  **安装Python依赖**:
    ```bash
    pip install -r requirements.in
    ```
    *   **注意**: `requirements.in`文件已精确锁定了`numpy<2.0`和`empy==3.3.4`等关键版本，以确保与ROS 2 Humble的兼容性。

### 第三步：准备资产

1.  **下载模型**: 下载`yolov12m.onnx`模型文件。您可以从[这里](<link_to_your_model_download>)下载，并将其放置在`assets/models/`目录下。
2.  **准备视频**: 准备一个您想要测试的视频文件（例如`city_traffic.mp4`），放置在`assets/videos/`目录下。
3.  **配置视频路径**: 打开`ros2_ws/src/uav_simulation/uav_simulation/video_publisher_node.py`文件，确保文件内的视频路径指向您刚刚放置的视频。

### 第四步：构建与运行 (黄金标准工作流)

这是整个项目的核心，请**严格遵循**以下顺序。建议在一个全新的终端中开始。

1.  **准备环境 (每次打开新终端都需执行)**:
    ```bash
    # 进入工作空间
    cd /path/to/your/isaac-sim-uav/ros2_ws

    # 激活虚拟环境
    source .venv/bin/activate

    # 加载ROS 2环境
    source /opt/ros/humble/setup.zsh
    ```
2.  **清理并构建 (首次构建或代码修改后执行)**:
    ```bash
    # 深度清理 (可选，但推荐在遇到问题时执行)
    rm -rf build install log
    find ./src -type d -name "*.egg-info" -exec rm -r {} +
    
    # 【核心】用venv的Python调用colcon进行构建
    python3 -m colcon build --symlink-install
    ```
3.  **加载工作空间并运行**:
    *   **加载本地环境 (必须)**:
        ```bash
        source install/setup.zsh
        ```
    *   **启动节点网络 (每个命令在一个单独的终端中运行)**:
        *   **终端1 (数据源)**: `ros2 run uav_simulation video_publisher_node`
        *   **终端2 (感知核心)**: `ros2 run uav_perception yolo_detector_node`
        *   **终端3 (可视化)**: `ros2 run uav_visualizer visualizer_node`
        *   **终端4 (Rviz2)**: `rviz2`

### 第五步：在Rviz2中验证

1.  在Rviz2中，将左上角的`Global Options -> Fixed Frame`设置为`uav_camera_link`。
2.  点击左下角的`Add` -> `By topic` -> 选择`/uav/vision/image_annotated`下的`Image`。
3.  您现在应该能看到带有实时检测框的视频流。

### (可选) 运行时参数调优

您可以在第五个终端中，实时调整检测的置信度阈值，以观察不同标准下的检测效果。

```bash
# 准备环境...
ros2 param set /yolo_detector_node confidence_threshold 0.2
```

