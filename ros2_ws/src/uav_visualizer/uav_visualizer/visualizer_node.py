# ros2_ws/src/uav_visualizer/uav_visualizer/visualizer_node.py
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from uav_interfaces.msg import Detections
from cv_bridge import CvBridge
import cv2
import message_filters

class VisualizerNode(Node):
    def __init__(self):
        super().__init__('visualizer_node')
        self.get_logger().info("UAV Visualizer 节点已启动.")

        self.bridge = CvBridge()
        self.annotated_image_publisher = self.create_publisher(Image, '/uav/vision/image_annotated', 10)

        image_sub = message_filters.Subscriber(self, Image, '/uav/camera/image_raw')
        detections_sub = message_filters.Subscriber(self, Detections, '/uav/vision/detections')

        self.time_synchronizer = message_filters.ApproximateTimeSynchronizer(
            [image_sub, detections_sub], 
            queue_size=10, 
            slop=0.1
        )
        self.time_synchronizer.registerCallback(self.synchronized_callback)
        self.get_logger().info("订阅者已设置，等待同步的图像和检测消息...")

    def synchronized_callback(self, image_msg, detections_msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
            
            for detection in detections_msg.detections:
                bbox = detection.bbox
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(cv_image, (x1, y1 - text_height - 10), (x1 + text_width, y1), (0, 255, 0), -1)
                cv2.putText(cv_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            annotated_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            annotated_image_msg.header = image_msg.header
            self.annotated_image_publisher.publish(annotated_image_msg)
        except Exception as e:
            self.get_logger().error(f"回调函数处理失败: {e}")

def main(args=None):
    rclpy.init(args=args)
    visualizer_node = VisualizerNode()
    try:
        rclpy.spin(visualizer_node)
    except KeyboardInterrupt:
        pass
    finally:
        visualizer_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()