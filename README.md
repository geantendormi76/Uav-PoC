# Project Chimera - 数字孪生无人机感知模块

## 核心愿景
构建一个能在边缘设备（无人机）上自主运行的、实时的、高能效的感知与决策系统。

## 当前进度 (Project Apollo - PoC)
**里程碑三已完成。** 目前已在WSL2本地环境中，成功搭建了一个基于ROS 2的模块化视觉感知系统。该系统能够：
1.  从视频文件模拟无人机摄像头，并作为ROS 2图像话题发布。
2.  利用高性能的ONNX Runtime引擎，加载YOLOv12模型对图像进行实时物体检测。
3.  将结构化的检测结果（类别、置信度、边界框）作为自定义的ROS 2话题发布。

## 复刻步骤 (当前PoC)

1.  **环境准备:**
    *   确保已根据[开荒蓝图](<link_to_your_blueprint.md>)搭建好基础开发环境 (Windows 11 + WSL2 + ROS 2 Humble)。
    *   将项目克隆到本地。

2.  **安装依赖:**
    *   进入 `ros2_ws` 目录，`direnv`将自动创建并激活Python虚拟环境。
    *   在虚拟环境中安装Python依赖: `uv pip install opencv-python onnx onnxruntime-gpu "empy==3.3.4"`

3.  **准备资产:**
    *   下载 `yolov12m.onnx` 模型文件，放置于 `assets/models/` 目录下。
    *   准备一个测试视频文件，放置于 `assets/videos/` 目录下，并确保 `uav_simulation/video_publisher_node.py` 中的路径指向该文件。

4.  **编译与运行:**
    *   在 `ros2_ws` 目录下，执行 `colcon build`。
    *   `source install/setup.zsh` (或 `.bash`)。
    *   **终端1:** `ros2 run uav_simulation video_publisher_node` (建议循环播放模式)
    *   **终端2:** `ros2 run uav_perception yolo_detector_node`
    *   **终端3:** `ros2 topic echo /uav/vision/detections`

## 下一步
里程碑四：眼见为实 (Seeing is Believing) - 使用Rviz2将检测结果可视化。
