# -*- coding: utf-8 -*-
import cv2
import numpy as np
import onnxruntime

# --- 配置区 ---
MODEL_PATH = '/mnt/c/Users/zhz/uav/assets/models/yolov12m.onnx'
IMAGE_PATH = '/mnt/c/Users/zhz/uav/assets/videos/test_frame.jpg'
# --- 核心修正：根据上一轮的诊断输出，将阈值调低以进行验证 ---
CONF_THRESHOLD = 0.10  # 从 0.25 降低到 0.10
IOU_THRESHOLD = 0.45
INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# COCO 类别名称
COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
                'scissors', 'teddy bear', 'hair drier', 'toothbrush']

def preprocess_warpAffine(image, dst_width, dst_height):
    """
    使用仿射变换进行预处理 (来自CSDN黄金标准)
    返回: 预处理后的图像张量 和 逆仿射矩阵
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

    # BGR -> RGB, HWC -> CHW, 归一化
    img_pre = (img_pre[..., ::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None] # 增加batch维度
    
    return img_pre, IM

def postprocess(pred, IM, conf_thres, iou_thres):
    """
    后处理函数 (来自CSDN黄金标准) - 【诊断模式】
    """
    print(f"--- [诊断] 进入后处理，原始pred形状: {pred.shape} ---")
    proposals = pred[0]
    
    all_class_scores = proposals[:, 4:]
    max_scores_per_proposal = np.max(all_class_scores, axis=1)
    global_max_score = np.max(max_scores_per_proposal)
    print(f"--- [诊断] 所有8400个提议中的最高置信度分数是: {global_max_score:.6f} ---")
    top_5_scores = np.sort(max_scores_per_proposal)[-5:]
    print(f"--- [诊断] 最高的5个置信度分数是: {top_5_scores} ---")
    print(f"--- [诊断] 当前置信度阈值是: {conf_thres} ---")
    
    boxes = []
    # 找出所有置信度大于阈值的提议
    for i, item in enumerate(proposals):
        label = item[4:].argmax()
        confidence = item[4 + label]
        
        if confidence >= conf_thres:
            cx, cy, w, h = item[:4]
            left = cx - w * 0.5
            top = cy - h * 0.5
            right = cx + w * 0.5
            bottom = cy + h * 0.5
            boxes.append([left, top, right, bottom, confidence, label])

    if not boxes:
        print("--- [诊断] 即使降低阈值，仍然没有一个提议通过过滤。---")
        return np.array([])
    
    print(f"--- [诊断] 通过置信度过滤的提议数量: {len(boxes)} ---")
    boxes = np.array(boxes)
    
    # --- 使用逆矩阵精确映射坐标 ---
    lr = boxes[:, [0, 2]]
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]
    
    # --- 使用OpenCV的NMS ---
    confidences = boxes[:, 4]
    labels = boxes[:, 5]
    bboxes = boxes[:, :4]
    
    indices = cv2.dnn.NMSBoxes(bboxes.tolist(), confidences.tolist(), conf_thres, iou_thres)
    
    if len(indices) > 0:
        indices = indices.flatten()
        print(f"--- [诊断] NMS之后剩余的最终物体数量: {len(indices)} ---")
        return boxes[indices]
    else:
        print("--- [诊断] NMS抑制了所有提议。---")
        return np.array([])

def main():
    print("--- 开始YOLOv12 ONNX独立验证 (诊断模式 v2 - 低阈值) ---")
    try:
        session = onnxruntime.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"ONNX模型加载成功: {MODEL_PATH}")
    except Exception as e:
        print(f"错误：无法加载ONNX模型。 {e}")
        return

    original_img = cv2.imread(IMAGE_PATH)
    if original_img is None:
        print(f"错误：无法读取图像: {IMAGE_PATH}")
        return
    
    print("图像读取成功，正在进行预处理...")
    input_tensor, inverse_matrix = preprocess_warpAffine(original_img, INPUT_WIDTH, INPUT_HEIGHT)

    print("预处理完成，开始推理...")
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    outputs = session.run([output_name], {input_name: input_tensor})
    
    print("推理完成，开始后处理...")
    predictions = np.squeeze(outputs[0]).T[None, :, :]
    final_boxes = postprocess(predictions, inverse_matrix, CONF_THRESHOLD, IOU_THRESHOLD)

    print(f"检测到 {len(final_boxes)} 个物体。")
    if len(final_boxes) > 0:
        print("--- 最终检测结果详情 ---")
    for box in final_boxes:
        left, top, right, bottom, confidence, label_id = box
        label_id = int(label_id)
        label_text = f"{COCO_CLASSES[label_id]}: {confidence:.2f}"
        print(f"  - 类别: {COCO_CLASSES[label_id]}, 置信度: {confidence:.4f}, BBox: [{int(left)}, {int(top)}, {int(right)}, {int(bottom)}]")
        
        cv2.rectangle(original_img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
        cv2.putText(original_img, label_text, (int(left), int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    output_path = "inference_result.jpg"
    cv2.imwrite(output_path, original_img)
    print(f"--- 验证完成，结果已保存到: {output_path} ---")

if __name__ == '__main__':
    main()