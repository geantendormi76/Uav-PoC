# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os

class VideoPublisherNode(Node):
    """
    一个ROS 2节点，它会读取一个视频文件，并将其帧作为sensor_msgs/msg/Image消息发布。
    """
    def __init__(self):
        super().__init__('video_publisher_node')
        
        # --- 参数定义 ---
        # 声明一个参数'video_path'，用于指定视频文件的路径
        self.declare_parameter('video_path', '/mnt/c/Users/zhz/uav/assets/videos/city_traffic.mp4')
        
        # --- 发布者设置 ---
        # 创建一个发布者，用于发布图像消息。Topic名称为'uav/camera/image_raw'。
        # qos_profile=10 是服务质量设置，表示队列大小为10。
        self.publisher_ = self.create_publisher(Image, 'uav/camera/image_raw', 10)
        
        # --- 定时器设置 ---
        # 我们从视频中获取帧率，并以此为频率创建定时器
        video_path = self.get_parameter('video_path').get_parameter_value().string_value
        if not os.path.exists(video_path):
            self.get_logger().error(f"视频文件未找到: {video_path}")
            # 通过抛出异常来阻止节点继续运行
            raise FileNotFoundError(f"视频文件未找到: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        timer_period = 1.0 / fps  # 计算定时器周期（秒）
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # --- 工具初始化 ---
        # 创建一个CvBridge实例，用于在OpenCV图像和ROS图像消息之间进行转换
        self.bridge = CvBridge()
        
        self.get_logger().info(f"视频发布节点已启动，正在从 '{video_path}' 发布图像。")
        self.get_logger().info(f"视频帧率: {fps:.2f} FPS，发布周期: {timer_period:.4f} 秒。")

    def timer_callback(self):
        """
        定时器的回调函数，每次被调用时会读取一帧视频并发布。
        """
        # 从视频文件中读取一帧
        ret, frame = self.cap.read()
        
        if ret:
            # 如果成功读取到一帧 (ret is True)
            # 使用cv_bridge将OpenCV图像(BGR格式)转换为ROS Image消息
            image_message = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # 发布图像消息
            self.publisher_.publish(image_message)
            # self.get_logger().info('正在发布视频帧...') # 取消此行注释可以查看详细日志，但会刷屏
        else:
            # 如果视频播放完毕 (ret is False)
            self.get_logger().info('视频播放结束，正在循环播放...')
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # self.get_logger().info('视频播放结束，节点将关闭。')
            # self.cap.release() # 释放视频捕获对象
            # self.destroy_node() # 销毁节点
            # rclpy.shutdown() # 关闭rclpy

def main(args=None):
    rclpy.init(args=args)
    try:
        video_publisher_node = VideoPublisherNode()
        rclpy.spin(video_publisher_node)
    except FileNotFoundError as e:
        # 如果在节点初始化时找不到文件，这里会捕获到异常
        rclpy.logging.get_logger("video_publisher_main").error(str(e))
    except KeyboardInterrupt:
        # 捕获Ctrl+C中断
        pass
    finally:
        # 确保在退出时rclpy被正确关闭
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()