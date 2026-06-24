"""
行为判定模块
Behavior Analysis Module

基于目标检测结果，分析猫咪与家具的空间关系和时间序列行为，
判定是否存在破坏家具的异常行为。

核心功能：
- 检测猫咪与家具的距离/重叠关系
- 追踪行为持续时间
- 判定抓挠、啃咬等破坏行为
- 触发异常报警
"""

import cv2
import logging
import os
import csv
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import time

from .ai_inference_module import Detection

# 日志文件路径
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
BEHAVIOR_LOG_FILE = os.path.join(LOG_DIR, 'behavior_log.csv')

logger = logging.getLogger(__name__)


class BehaviorType(Enum):
    """行为类型枚举"""
    NORMAL = "normal"           # 正常行为
    APPROACHING = "approaching"  # 接近家具
    SCRATCHING = "scratching"    # 抓挠行为
    BITING = "biting"           # 啃咬行为
    CLIMBING = "climbing"       # 攀爬行为
    RESTING = "resting"         # 休息（在家具上）


class AlertLevel(Enum):
    """警报级别"""
    NONE = "none"       # 无警报
    LOW = "low"         # 低级别（接近家具）
    MEDIUM = "medium"   # 中级别（可疑行为）
    HIGH = "high"       # 高级别（破坏行为）


@dataclass
class BehaviorEvent:
    """行为事件数据类"""
    behavior_type: BehaviorType
    alert_level: AlertLevel
    timestamp: datetime
    cat_detection: Optional[Detection]
    furniture_detection: Optional[Detection]
    duration_seconds: float
    description: str
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "behavior_type": self.behavior_type.value,
            "alert_level": self.alert_level.value,
            "timestamp": self.timestamp.isoformat(),
            "cat_detected": self.cat_detection is not None,
            "furniture_type": self.furniture_detection.label if self.furniture_detection else None,
            "duration_seconds": self.duration_seconds,
            "description": self.description
        }


@dataclass
class CatFurnitureInteraction:
    """猫咪 - 家具交互状态"""
    cat_id: Optional[str] = None
    furniture_id: Optional[str] = None
    distance: float = float('inf')  # 猫咪与家具的距离
    overlap_ratio: float = 0.0      # 边界框重叠率
    is_touching: bool = False       # 是否接触
    start_time: Optional[float] = None
    last_update_time: Optional[float] = None
    behavior_type: BehaviorType = BehaviorType.NORMAL


class BehaviorAnalysisModule:
    """
    行为判定模块类
    
    分析猫咪与家具的交互行为，判定是否存在破坏行为。
    
    判定逻辑：
    1. 距离判定：猫咪与家具边界框距离 < 阈值 → 接近行为
    2. 重叠判定：边界框重叠率 > 阈值 → 接触/攀爬行为
    3. 时间判定：持续接触时间 > 阈值 → 可疑行为
    4. 动作判定：快速移动 + 接触 → 抓挠行为
    
    Attributes:
        distance_threshold (float): 距离阈值（像素）
        overlap_threshold (float): 重叠率阈值
        scratch_duration_threshold (float): 抓挠持续时间阈值（秒）
        bite_duration_threshold (float): 啃咬持续时间阈值（秒）
    """
    
    def __init__(
        self,
        distance_threshold: float = 50.0,
        overlap_threshold: float = 0.3,
        scratch_duration_threshold: float = 3.0,
        bite_duration_threshold: float = 5.0,
        approach_alert_delay: float = 2.0
    ):
        """
        初始化行为判定模块
        
        Args:
            distance_threshold: 距离阈值，猫咪与家具距离小于此值视为接近
            overlap_threshold: 重叠率阈值，超过此值视为接触
            scratch_duration_threshold: 抓挠行为持续时间阈值（秒）
            bite_duration_threshold: 啃咬行为持续时间阈值（秒）
            approach_alert_delay: 接近警报延迟时间（秒）
        """
        self.distance_threshold = distance_threshold
        self.overlap_threshold = overlap_threshold
        self.scratch_duration_threshold = scratch_duration_threshold
        self.bite_duration_threshold = bite_duration_threshold
        self.approach_alert_delay = approach_alert_delay
        
        # 交互状态追踪
        self.interactions: Dict[str, CatFurnitureInteraction] = {}
        
        # 行为事件历史
        self.event_history: List[BehaviorEvent] = []
        
        # 当前警报状态
        self.current_alert_level = AlertLevel.NONE
        self.last_alert_time: Optional[float] = None
        
        # 防抖状态追踪（用于 2 秒时间窗口防抖）
        self.debounce_states: Dict[str, Dict] = {}
        self.debounce_window = 2.0  # 2 秒防抖窗口
        
        # 确保日志目录存在
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            # 初始化 CSV 日志文件（如果不存在）
            self._init_csv_log()
        except Exception as e:
            logger.warning(f"初始化日志目录失败：{e}")
        
        logger.info("BehaviorAnalysisModule 初始化完成")
    
    def _init_csv_log(self):
        """初始化 CSV 日志文件"""
        if not os.path.exists(BEHAVIOR_LOG_FILE):
            with open(BEHAVIOR_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp',
                    'behavior_type',
                    'alert_level',
                    'furniture_type',
                    'duration_seconds',
                    'description'
                ])
            logger.info(f"CSV 日志文件已创建：{BEHAVIOR_LOG_FILE}")
    
    def log_behavior(self, event: BehaviorEvent):
        """
        记录行为事件到 CSV 日志文件
        
        Args:
            event: 行为事件对象
        """
        try:
            with open(BEHAVIOR_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    event.timestamp.isoformat(),
                    event.behavior_type.value,
                    event.alert_level.value,
                    event.furniture_detection.label if event.furniture_detection else None,
                    f"{event.duration_seconds:.2f}",
                    event.description
                ])
        except Exception as e:
            logger.error(f"记录行为日志失败：{e}")
    
    def analyze(
        self, 
        cats: List[Detection], 
        furniture: List[Detection],
        current_time: Optional[float] = None
    ) -> List[BehaviorEvent]:
        """
        分析猫咪与家具的交互行为
        
        Args:
            cats: 猫咪检测结果列表
            furniture: 家具检测结果列表
            current_time: 当前时间戳（用于计算持续时间）
        
        Returns:
            List[BehaviorEvent]: 检测到的行为事件列表
        """
        if current_time is None:
            current_time = time.time()
        
        events = []
        
        # 如果没有检测到猫咪，重置状态
        if not cats:
            self._reset_interactions()
            return events
        
        # 分析每只猫咪与家具的交互
        for cat in cats:
            cat_events = self._analyze_cat_interaction(
                cat, furniture, current_time
            )
            events.extend(cat_events)
        
        # 更新当前警报级别
        if events:
            self.current_alert_level = max(
                (e.alert_level for e in events),
                key=lambda x: list(AlertLevel).index(x)
            )
        else:
            self.current_alert_level = AlertLevel.NONE
        
        # 记录到历史
        self.event_history.extend(events)
        
        # 记录到 CSV 日志文件
        for event in events:
            self.log_behavior(event)
        
        # 限制历史记录大小
        if len(self.event_history) > 1000:
            self.event_history = self.event_history[-500:]
        
        return events
    
    def _analyze_cat_interaction(
        self,
        cat: Detection,
        furniture_list: List[Detection],
        current_time: float
    ) -> List[BehaviorEvent]:
        """
        分析单只猫咪与家具的交互
        
        Returns:
            List[BehaviorEvent]: 行为事件列表
        """
        events = []
        cat_key = f"cat_{cat.class_id}_{cat.center}"
        
        # 找到最近的家具
        closest_furniture = None
        min_distance = float('inf')
        
        for furn in furniture_list:
            distance = self._calculate_distance(cat, furn)
            if distance < min_distance:
                min_distance = distance
                closest_furniture = furn
        
        # 获取或创建交互状态
        if cat_key not in self.interactions:
            self.interactions[cat_key] = CatFurnitureInteraction(
                cat_id=cat.class_id,
                start_time=current_time,
                last_update_time=current_time
            )
        
        interaction = self.interactions[cat_key]
        interaction.last_update_time = current_time
        
        if closest_furniture:
            interaction.furniture_id = closest_furniture.class_id
            interaction.distance = min_distance
            interaction.overlap_ratio = self._calculate_overlap_ratio(cat, closest_furniture)
            interaction.is_touching = interaction.overlap_ratio > self.overlap_threshold
        else:
            interaction.distance = float('inf')
            interaction.overlap_ratio = 0.0
            interaction.is_touching = False
        
        # 判定行为类型
        behavior_event = self._determine_behavior(
            cat, closest_furniture, interaction, current_time
        )
        
        if behavior_event:
            events.append(behavior_event)
        
        return events
    
    def _determine_behavior(
        self,
        cat: Detection,
        furniture: Optional[Detection],
        interaction: CatFurnitureInteraction,
        current_time: float
    ) -> Optional[BehaviorEvent]:
        """
        根据交互状态判定行为类型
        
        Returns:
            Optional[BehaviorEvent]: 行为事件（如有）
        """
        behavior_type = BehaviorType.NORMAL
        alert_level = AlertLevel.NONE
        description = "猫咪正常活动"
        
        duration = 0.0
        if interaction.start_time:
            duration = current_time - interaction.start_time
        
        # 判定逻辑
        if furniture is None:
            # 没有家具在附近
            if interaction.behavior_type != BehaviorType.NORMAL:
                interaction.behavior_type = BehaviorType.NORMAL
                interaction.start_time = current_time
            return None
        
        if interaction.is_touching:
            # 猫咪与家具接触
            if duration > self.bite_duration_threshold:
                # 长时间接触 → 可能是啃咬
                behavior_type = BehaviorType.BITING
                alert_level = AlertLevel.HIGH
                description = f"检测到猫咪啃咬{furniture.label}，持续{duration:.1f}秒"
                
            elif duration > self.scratch_duration_threshold:
                # 中等时长接触 → 可能是抓挠
                behavior_type = BehaviorType.SCRATCHING
                alert_level = AlertLevel.HIGH
                description = f"检测到猫咪抓挠{furniture.label}，持续{duration:.1f}秒"
                
            else:
                # 短暂接触 → 可能是路过
                behavior_type = BehaviorType.APPROACHING
                alert_level = AlertLevel.LOW
                description = f"猫咪接触{furniture.label}"
                
        elif interaction.distance < self.distance_threshold:
            # 接近但未接触
            behavior_type = BehaviorType.APPROACHING
            alert_level = AlertLevel.LOW
            description = f"猫咪接近{furniture.label}，距离{interaction.distance:.0f}像素"
            
            # 延迟警报，避免频繁触发
            if duration < self.approach_alert_delay:
                return None
        else:
            # 距离较远
            if interaction.behavior_type != BehaviorType.NORMAL:
                interaction.behavior_type = BehaviorType.NORMAL
                interaction.start_time = current_time
            return None
        
        # 更新交互状态
        interaction.behavior_type = behavior_type
        
        # 创建行为事件
        event = BehaviorEvent(
            behavior_type=behavior_type,
            alert_level=alert_level,
            timestamp=datetime.now(),
            cat_detection=cat,
            furniture_detection=furniture,
            duration_seconds=duration,
            description=description
        )
        
        logger.info(f"行为判定：{description}")
        return event
    
    def _calculate_distance(
        self, 
        det1: Detection, 
        det2: Detection
    ) -> float:
        """
        计算两个检测框之间的距离（中心点距离）
        
        Args:
            det1: 检测结果 1
            det2: 检测结果 2
        
        Returns:
            float: 欧几里得距离
        """
        x1, y1 = det1.center
        x2, y2 = det2.center
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    def _calculate_overlap_ratio(
        self, 
        det1: Detection, 
        det2: Detection
    ) -> float:
        """
        计算两个检测框的重叠率 (IoU - Intersection over Union)
        
        Args:
            det1: 检测结果 1
            det2: 检测结果 2
        
        Returns:
            float: IoU 值 (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = det1.bbox
        x1_2, y1_2, x2_2, y2_2 = det2.bbox
        
        # 计算交集
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # 计算并集
        area1 = det1.area
        area2 = det2.area
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    def _reset_interactions(self):
        """重置所有交互状态"""
        self.interactions.clear()
        self.current_alert_level = AlertLevel.NONE
    
    def get_current_alert(self) -> Tuple[AlertLevel, Optional[BehaviorEvent]]:
        """
        获取当前警报状态
        
        Returns:
            Tuple[AlertLevel, Optional[BehaviorEvent]]: (警报级别，最新事件)
        """
        latest_event = self.event_history[-1] if self.event_history else None
        return self.current_alert_level, latest_event
    
    def get_behavior_statistics(
        self, 
        time_window_seconds: float = 3600
    ) -> Dict[str, int]:
        """
        获取行为统计信息
        
        Args:
            time_window_seconds: 统计时间窗口（秒）
        
        Returns:
            Dict[str, int]: 各行为类型的计数
        """
        from datetime import timedelta
        
        cutoff_time = datetime.now() - timedelta(seconds=time_window_seconds)
        
        stats = {bt.value: 0 for bt in BehaviorType}
        
        for event in self.event_history:
            if event.timestamp >= cutoff_time:
                stats[event.behavior_type.value] += 1
        
        return stats
    
    # =========================================================================
    # 扩展接口：更复杂的行为识别
    # =========================================================================
    # 后续可添加：
    # 1. 基于光流法的动作识别（检测快速抓挠动作）
    # 2. 基于姿态估计的精细行为判定
    # 3. 机器学习分类器（RandomForest 等）进行行为分类
    # =========================================================================
    
    def analyze_with_optical_flow(
        self,
        prev_frame: Optional[cv2.Mat],
        curr_frame: cv2.Mat,
        cats: List[Detection],
        furniture: List[Detection]
    ) -> List[BehaviorEvent]:
        """
        结合光流法分析行为（预留接口）
        
        通过分析连续帧之间的运动模式，可以更准确地识别
        快速抓挠、啃咬等动作。
        
        Args:
            prev_frame: 上一帧图像
            curr_frame: 当前帧图像
            cats: 猫咪检测结果
            furniture: 家具检测结果
        
        Returns:
            List[BehaviorEvent]: 行为事件列表
        
        TODO: 后续实现光流法行为识别
        """
        logger.debug("[预留接口] 光流法行为分析")
        # 后续实现：
        # 1. 计算光流
        # flow = cv2.calcOpticalFlowFarneback(prev_frame, curr_frame, ...)
        # 2. 分析猫咪区域的光流模式
        # 3. 判定是否为抓挠动作（高频往复运动）
        return []