# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort # 导入ONNX Runtime
import os

# 导入我们自定义的消息类型
from uav_interfaces.msg import Detection, Detections

# YOLOv12 COCO 数据集类别名称 (80类)
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')
        
        # --- 参数定义 ---
        default_model_path = '/mnt/c/Users/zhz/uav/assets/models/yolov12m.onnx'
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('confidence_threshold', 0.25) # 置信度阈值
        self.declare_parameter('iou_threshold', 0.4)      # IoU阈值 (用于NMS)
        
        # --- 加载ONNX模型 ---
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f"正在从本地路径加载ONNX模型: {model_path}...")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"模型文件未找到: {model_path}")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        # 使用ONNX Runtime创建推理会话，并指定使用CUDA
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.get_logger().info("ONNX模型加载成功，使用CUDA进行推理。")

        # 获取模型的输入和输出名称及形状
        self.input_name = self.session.get_inputs()[0].name
        model_input = self.session.get_inputs()[0]
        self.input_height = model_input.shape[2]
        self.input_width = model_input.shape[3]
        self.output_names = [output.name for output in self.session.get_outputs()]

        # --- 订阅者/发布者/工具 ---
        self.subscription = self.create_subscription(Image, 'uav/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Detections, 'uav/vision/detections', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("YOLO ONNX检测节点已启动，等待图像消息...")

    def preprocess(self, img):
        """对图像进行预处理以匹配模型输入"""
        img_height, img_width, _ = img.shape
        # 缩放图像并填充以保持纵横比
        scale = min(self.input_width / img_width, self.input_height / img_height)
        new_width, new_height = int(img_width * scale), int(img_height * scale)
        resized_img = cv2.resize(img, (new_width, new_height))
        
        # 创建一个用灰色填充的画布
        padded_img = np.full((self.input_height, self.input_width, 3), 114, dtype=np.uint8)
        # 将缩放后的图像粘贴到画布中央
        padded_img[(self.input_height - new_height) // 2:(self.input_height - new_height) // 2 + new_height,
                   (self.input_width - new_width) // 2:(self.input_width - new_width) // 2 + new_width, :] = resized_img
        
        # 转换维度 HWC -> CHW, BGR -> RGB
        padded_img = padded_img.transpose((2, 0, 1))[::-1]
        # 归一化到 0-1
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32) / 255.0
        # 增加一个batch维度
        padded_img = np.expand_dims(padded_img, axis=0)
        return padded_img, scale

    def postprocess(self, outputs, original_shape, scale):
        """对模型输出进行后处理 (v3 - 诊断版)"""
        # --- 日志1: 检查原始模型输出 ---
        self.get_logger().info(f"--- Frame Analysis Start ---")
        self.get_logger().info(f"Model raw output shape: {outputs[0].shape}")

        predictions = np.squeeze(outputs[0]).T

        scores = np.max(predictions[:, 4:], axis=1)
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        
        # --- 日志2: 检查置信度过滤前的检测数量 ---
        self.get_logger().info(f"Total proposals before confidence filtering: {len(scores)}")

        # 过滤掉低于置信度阈值的预测
        high_conf_indices = scores > conf_threshold
        predictions = predictions[high_conf_indices, :]
        scores = scores[high_conf_indices]

        # --- 日志3: 检查置信度过滤后的检测数量 ---
        self.get_logger().info(f"Proposals after confidence filtering (threshold={conf_threshold:.2f}): {len(scores)}")

        if predictions.shape[0] == 0:
            self.get_logger().warn("No proposals left after confidence filtering. Nothing to process.")
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes_cxcywh = predictions[:, :4]

        boxes_xywh = boxes_cxcywh.copy()
        boxes_xywh[:, 0] = boxes_cxcywh[:, 0] - (boxes_cxcywh[:, 2] / 2)
        boxes_xywh[:, 1] = boxes_cxcywh[:, 1] - (boxes_cxcywh[:, 3] / 2)
        
        iou_threshold = self.get_parameter('iou_threshold').get_parameter_value().double_value
        
        # --- 日志4: 检查送入NMS的数据 ---
        self.get_logger().info(f"Data sent to NMS: {len(boxes_xywh)} boxes, {len(scores)} scores.")

        indices = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), scores.tolist(), conf_threshold, iou_threshold)
        
        # --- 日志5: 检查NMS的输出结果 ---
        self.get_logger().info(f"NMS raw output: {indices}")

        if len(indices) > 0:
            indices = indices.flatten()
            self.get_logger().info(f"NMS result: {len(indices)} boxes left after suppression.")
        else:
            self.get_logger().warn("NMS suppressed all boxes. No detections to publish.")
            return [], [], []

        final_boxes_cxcywh = boxes_cxcywh[indices]
        final_scores = scores[indices]
        final_class_ids = class_ids[indices]
        
        # --- 日志6: 打印最终检测结果 ---
        final_class_names = [COCO_CLASSES[i] for i in final_class_ids]
        self.get_logger().info(f"Final Detections: {list(zip(final_class_names, final_scores))}")
        self.get_logger().info(f"--- Frame Analysis End ---")


        # 将坐标从模型输入尺寸转换回原始图像尺寸 (这部分代码保持不变)
        final_boxes_cxcywh[:, 0] -= ((self.input_width - original_shape[1] * scale) / 2)
        final_boxes_cxcywh[:, 1] -= ((self.input_height - original_shape[0] * scale) / 2)
        final_boxes_cxcywh /= scale
        
        return final_boxes_cxcywh, final_scores, final_class_ids


    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 预处理
        input_tensor, scale = self.preprocess(cv_image)
        
        # ONNX推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        boxes, scores, class_ids = self.postprocess(outputs, cv_image.shape, scale)

        # 封装成自定义消息
        detections_msg = Detections()
        # --- 关键修正：将接收到的图像消息头传递下去 ---
        detections_msg.header = msg.header

        for box, score, class_id in zip(boxes, scores, class_ids):
            detection = Detection()
            detection.class_name = COCO_CLASSES[class_id]
            detection.confidence = float(score)
            
            cx, cy, w, h = box
            x1 = int(cx - w / 2)
            y1 = int(cy - h / 2)
            x2 = int(cx + w / 2)
            y2 = int(cy + h / 2)
            
            detection.bbox = [x1, y1, x2, y2]
            detections_msg.detections.append(detection)
            
        self.publisher_.publish(detections_msg)


def main(args=None):
    rclpy.init(args=args)
    yolo_detector_node = YoloDetectorNode()
    rclpy.spin(yolo_detector_node)
    
    # 节点关闭时销毁
    yolo_detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()