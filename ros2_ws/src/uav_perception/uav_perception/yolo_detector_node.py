# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import onnxruntime as ort
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
        # --- 将默认置信度恢复为项目初始设定的0.25，但我们知道可以随时调整 ---
        self.declare_parameter('confidence_threshold', 0.25)
        self.declare_parameter('iou_threshold', 0.45)
        
        # --- 加载ONNX模型 ---
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_width = 640
        self.input_height = 640
        
        self.get_logger().info(f"正在从本地路径加载ONNX模型: {model_path}...")
        
        if not os.path.exists(model_path):
            self.get_logger().error(f"模型文件未找到: {model_path}")
            raise FileNotFoundError(f"模型文件未找到: {model_path}")
        
        try:
            self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.get_logger().info("ONNX模型加载成功，使用CUDA进行推理。")
        except Exception as e:
            self.get_logger().error(f"无法加载ONNX模型。请检查ONNX Runtime和CUDA/cuDNN环境。错误: {e}")
            rclpy.shutdown()
            return

        # 获取模型的输入和输出名称
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # --- 订阅者/发布者/工具 ---
        self.subscription = self.create_subscription(Image, 'uav/camera/image_raw', self.image_callback, 10)
        self.publisher_ = self.create_publisher(Detections, 'uav/vision/detections', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info("YOLO ONNX检测节点已启动并完成初始化，等待图像消息...")

    def preprocess_warpAffine(self, image, dst_width, dst_height):
        """
        [黄金标准] 使用仿射变换进行预处理
        """
        scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
        ox = (dst_width - scale * image.shape[1]) / 2
        oy = (dst_height - scale * image.shape[0]) / 2
        
        M = np.array([
            [scale, 0, ox],
            [0, scale, oy]
        ], dtype=np.float32)
        
        img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        
        IM = cv2.invertAffineTransform(M)

        img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
        img_pre = img_pre.transpose(2, 0, 1)[None]
        
        return img_pre, IM

    def postprocess(self, pred, IM):
        """
        [黄金标准] 后处理函数
        """
        conf_thres = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        iou_thres = self.get_parameter('iou_threshold').get_parameter_value().double_value

        boxes = []
        proposals = pred[0]
        
        for item in proposals:
            label = item[4:].argmax()
            confidence = item[4 + label]
            if confidence < conf_thres:
                continue

            cx, cy, w, h = item[:4]
            left = cx - w * 0.5
            top = cy - h * 0.5
            right = cx + w * 0.5
            bottom = cy + h * 0.5
            boxes.append([left, top, right, bottom, confidence, label])

        if not boxes:
            return np.array([])
        
        boxes = np.array(boxes)
        
        lr = boxes[:, [0, 2]]
        tb = boxes[:, [1, 3]]
        boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
        boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]
        
        confidences = boxes[:, 4]
        labels = boxes[:, 5]
        bboxes = boxes[:, :4]
        
        indices = cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), conf_thres, iou_thres)
        
        if len(indices) > 0:
            indices = indices.flatten()
            return boxes[indices]
        else:
            return np.array([])

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # 预处理
        input_tensor, inverse_matrix = self.preprocess_warpAffine(cv_image, self.input_width, self.input_height)
        
        # ONNX推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
        
        # 后处理
        # outputs[0] 形状 (1, 84, 8400), 转置为 (1, 8400, 84)
        predictions = np.squeeze(outputs[0]).T[None, :, :]
        final_boxes = self.postprocess(predictions, inverse_matrix)

        # 封装成自定义消息
        detections_msg = Detections()
        detections_msg.header = msg.header

        for box in final_boxes:
            left, top, right, bottom, confidence, label_id = box
            
            detection = Detection()
            detection.class_name = COCO_CLASSES[int(label_id)]
            detection.confidence = float(confidence)
            detection.bbox = [int(left), int(top), int(right), int(bottom)]
            
            detections_msg.detections.append(detection)
            
        self.publisher_.publish(detections_msg)
        if len(final_boxes) > 0:
            self.get_logger().info(f"成功发布了 {len(detections_msg.detections)} 个检测结果。")


def main(args=None):
    rclpy.init(args=args)
    yolo_detector_node = YoloDetectorNode()
    try:
        rclpy.spin(yolo_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        yolo_detector_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()