# 定义一帧图像中的所有检测结果

# 消息头，包含时间戳和坐标系信息
std_msgs/Header header

# 一个由单个检测结果组成的动态数组
Detection[] detections