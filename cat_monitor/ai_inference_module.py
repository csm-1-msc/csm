"""
AI 推理模块
AI Inference Module

基于 YOLOv8 实现猫咪和家具的目标检测。
支持电脑本地推理，预留树莓派 Hailo AI Kit 部署接口。

核心功能：
- 加载 YOLOv8 预训练模型
- 检测猫咪、沙发、椅子等目标
- 返回检测框和置信度
"""

import cv2
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# 尝试导入 ultralytics，如未安装则使用模拟模式
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("ultralytics 未安装，AI 推理模块将使用模拟模式")

logger = logging.getLogger(__name__)


class DetectionClass(Enum):
    """检测类别枚举"""
    CAT = "cat"           # 猫咪
    SOFA = "sofa"         # 沙发
    CHAIR = "chair"       # 椅子
    COUCH = "couch"       # 躺椅/长沙发
    BED = "bed"           # 床
    TABLE = "table"       # 桌子
    FURNITURE = "furniture"  # 家具（通用）


@dataclass
class Detection:
    """检测结果数据类"""
    class_id: str         # 类别 ID
    label: str            # 标签名称
    confidence: float     # 置信度 (0-1)
    bbox: Tuple[int, int, int, int]  # 边界框 (x1, y1, x2, y2)
    center: Tuple[int, int]  # 中心点 (x, y)
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]
    
    @property
    def area(self) -> int:
        return self.width * self.height


class AIInferenceModule:
    """
    AI 推理模块类
    
    封装 YOLOv8 目标检测模型，提供猫咪和家具检测功能。
    
    Attributes:
        model_path (str): 模型文件路径
        confidence_threshold (float): 置信度阈值
        iou_threshold (float): NMS IoU 阈值
        model: YOLO 模型对象
        is_loaded (bool): 模型是否已加载
    """
    
    # 默认检测的类别（YOLOv8 COCO 数据集类别 ID）
    # 0: person, 15: cat, 56: couch, 57: chair, 59: bed, 60: dining table
    COCO_CATEGORIES = {
        "cat": 15,
        "couch": 56,
        "chair": 57,
        "bed": 59,
        "dining table": 60,
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # nano 版本，速度快
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = "cpu"  # 或 "cuda"
    ):
        """
        初始化 AI 推理模块
        
        Args:
            model_path: YOLOv8 模型路径
                - "yolov8n.pt": nano 版本（推荐，速度快）
                - "yolov8s.pt": small 版本
                - "yolov8m.pt": medium 版本
                - 自定义训练的 .pt 文件
            confidence_threshold: 置信度阈值，低于此值的检测结果将被过滤
            iou_threshold: NMS 非极大值抑制的 IoU 阈值
            device: 推理设备 "cpu" 或 "cuda"
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.model = None
        self.is_loaded = False
        
        logger.info(f"AIInferenceModule 初始化：model={model_path}, device={device}")
    
    def load_model(self) -> bool:
        """
        加载 YOLOv8 模型
        
        Returns:
            bool: 是否成功加载
        """
        if not YOLO_AVAILABLE:
            logger.error("ultralytics 库未安装，无法加载 YOLO 模型")
            return False
        
        try:
            logger.info(f"正在加载 YOLOv8 模型：{self.model_path}")
            self.model = YOLO(self.model_path)
            
            # 设置推理参数
            self.model.overrides = {
                'conf': self.confidence_threshold,
                'iou': self.iou_threshold,
            }
            
            self.is_loaded = True
            logger.info("YOLOv8 模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载模型时发生错误：{e}")
            return False
    
    def detect(self, frame: cv2.Mat) -> List[Detection]:
        """
        执行目标检测
        
        Args:
            frame: 输入图像帧 (BGR 格式)
        
        Returns:
            List[Detection]: 检测结果列表
        """
        if not self.is_loaded or self.model is None:
            logger.warning("模型未加载，返回空检测结果")
            return []
        
        try:
            # 执行推理
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            detections = []
            
            for result in results:
                if result.boxes is None:
                    continue
                
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy()
                names = result.names
                
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    class_id = int(cls_ids[i])
                    confidence = float(confs[i])
                    label = names[class_id]
                    
                    # 只保留我们关心的类别
                    if label not in ["cat", "couch", "chair", "bed", "dining table"]:
                        continue
                    
                    detection = Detection(
                        class_id=str(class_id),
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=((x1 + x2) // 2, (y1 + y2) // 2)
                    )
                    detections.append(detection)
            
            logger.debug(f"检测到 {len(detections)} 个目标")
            return detections
            
        except Exception as e:
            logger.error(f"推理时发生错误：{e}")
            return []
    
    def detect_cats(self, frame: cv2.Mat) -> List[Detection]:
        """
        仅检测猫咪
        
        Args:
            frame: 输入图像帧
        
        Returns:
            List[Detection]: 猫咪检测结果
        """
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.label == "cat"]
    
    def detect_furniture(self, frame: cv2.Mat) -> List[Detection]:
        """
        仅检测家具
        
        Args:
            frame: 输入图像帧
        
        Returns:
            List[Detection]: 家具检测结果
        """
        all_detections = self.detect(frame)
        furniture_labels = ["couch", "chair", "bed", "dining table"]
        return [d for d in all_detections if d.label in furniture_labels]
    
    def draw_detections(
        self, 
        frame: cv2.Mat, 
        detections: List[Detection],
        show_label: bool = True,
        show_confidence: bool = True
    ) -> cv2.Mat:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入图像
            detections: 检测结果列表
            show_label: 是否显示标签
            show_confidence: 是否显示置信度
        
        Returns:
            cv2.Mat: 绘制后的图像
        """
        output = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # 根据类别选择颜色
            if det.label == "cat":
                color = (0, 255, 0)  # 绿色 - 猫咪
            elif det.label in ["couch", "sofa"]:
                color = (0, 0, 255)  # 红色 - 沙发
            elif det.label == "chair":
                color = (255, 0, 0)  # 蓝色 - 椅子
            else:
                color = (255, 255, 0)  # 青色 - 其他家具
            
            # 绘制边界框
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签和置信度
            if show_label or show_confidence:
                label_parts = []
                if show_label:
                    label_parts.append(det.label)
                if show_confidence:
                    label_parts.append(f"{det.confidence:.2f}")
                
                label = " ".join(label_parts)
                
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                
                # 绘制标签背景
                cv2.rectangle(
                    output,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # 绘制文本
                cv2.putText(
                    output,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1
                )
        
        return output
    
    # =========================================================================
    # 扩展接口：树莓派 Hailo AI Kit 部署
    # =========================================================================
    # 后续迁移到树莓派时，可使用 Hailo 模型转换工具将 YOLO 模型转换为
    # Hailo 高效格式 (.hef)，在此添加对应的推理接口
    # =========================================================================
    
    def load_hailo_model(self, hef_path: str):
        """
        加载 Hailo 模型（预留接口）
        
        Args:
            hef_path: Hailo 高效格式模型路径 (.hef)
        
        TODO: 迁移到树莓派 + Hailo 时实现
        """
        logger.info(f"[预留接口] 加载 Hailo 模型：{hef_path}")
        # 后续实现：
        # from hailo_platform import HEF, VDevice
        # hef = HEF(hef_path)
        # ...
        pass
    
    def unload_model(self):
        """卸载模型，释放资源"""
        self.model = None
        self.is_loaded = False
        logger.info("模型已卸载")