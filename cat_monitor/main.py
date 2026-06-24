"""
AI 喵管家主程序
Cat Monitor Main Program

整合摄像头读取、AI 推理、行为判定和素材存储模块，
实现完整的猫咪破坏行为监控系统。

核心功能：
- 实时视频流处理
- 猫咪与家具目标检测
- 破坏行为识别与判定
- 证据素材保存
- 警报通知（预留接口）
"""

import cv2
import logging
import argparse
import signal
import sys
from datetime import datetime
from typing import Optional, List
import time

from .camera_module import CameraModule, CameraOccupiedError
from .ai_inference_module import AIInferenceModule, Detection
from .behavior_analysis_module import BehaviorAnalysisModule, BehaviorEvent, AlertLevel
from .storage_module import StorageModule

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CatMonitor:
    """
    猫咪监控主类
    
    整合所有模块，实现完整的监控流程。
    
    工作流程：
    1. 从摄像头读取视频帧
    2. 使用 AI 模型检测猫咪和家具
    3. 分析猫咪与家具的交互行为
    4. 检测到异常行为时保存证据
    5. 触发警报通知
    
    Attributes:
        camera (CameraModule): 摄像头模块
        ai_inference (AIInferenceModule): AI 推理模块
        behavior_analysis (BehaviorAnalysisModule): 行为分析模块
        storage (StorageModule): 存储模块
    """
    
    def __init__(
        self,
        camera_source: int = 0,
        model_path: str = "models/cat_furniture_detector.pt",
        storage_root: str = "./storage",
        confidence_threshold: float = 0.5,
        enable_display: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化猫咪监控系统
        
        Args:
            camera_source: 摄像头源（设备索引或 RTSP 地址）
            model_path: YOLO 模型路径
            storage_root: 存储根目录
            confidence_threshold: 检测置信度阈值
            enable_display: 是否显示实时画面（服务器环境可禁用）
            max_retries: 摄像头打开失败时的最大重试次数
            retry_delay: 重试间隔（秒）
        """
        logger.info("初始化 CatMonitor...")
        
        # 初始化模块（带重试参数）
        self.camera = CameraModule(
            camera_source,
            max_retries=max_retries,
            retry_delay=retry_delay
        )
        self.ai_inference = AIInferenceModule(
            model_path=model_path,
            confidence_threshold=confidence_threshold
        )
        self.behavior_analysis = BehaviorAnalysisModule()
        self.storage = StorageModule(storage_root=storage_root)
        
        self.enable_display = enable_display
        self._running = False
        
        # 统计信息
        self.frame_count = 0
        self.alert_count = 0
        self.start_time: Optional[float] = None
        
        logger.info("CatMonitor 初始化完成")
    
    def start(self):
        """启动监控"""
        logger.info("启动猫咪监控...")
        self._running = True
        self.start_time = time.time()
        self.frame_count = 0
        self.alert_count = 0
        
        # 打开摄像头（带异常处理）
        try:
            # 先检查摄像头是否可用
            if not CameraModule.is_camera_available(self.camera.camera_id):
                # 尝试列出可用摄像头
                available = CameraModule.list_available_cameras()
                logger.error(
                    f"摄像头 {self.camera.camera_id} 不可用。"
                    f"系统可用摄像头：{available if available else '未检测到任何摄像头'}"
                )
                raise CameraOccupiedError(
                    f"摄像头 {self.camera.camera_id} 可能被其他程序占用或不存在"
                )
            
            if not self.camera.open():
                logger.error("无法打开摄像头")
                raise CameraOccupiedError("摄像头打开失败")
            
            logger.info("摄像头已成功打开")
        except CameraOccupiedError as e:
            logger.error(f"❌ 摄像头错误：{e}")
            logger.info("提示：请确保没有其他程序（如 Zoom、Teams）正在使用摄像头")
            return
        
        # 主循环
        self._main_loop()
    
    def stop(self):
        """停止监控"""
        logger.info("停止猫咪监控...")
        self._running = False
        
        # 关闭摄像头
        self.camera.release()
        
        # 打印统计信息
        self._print_statistics()
    
    def _main_loop(self):
        """主处理循环"""
        prev_frame = None
        
        while self._running:
            # 读取帧
            ret, frame = self.camera.read()
            if not ret:
                logger.warning("无法读取帧，重试...")
                time.sleep(0.1)
                continue
            
            if frame is None:
                logger.warning("无法读取帧，重试...")
                time.sleep(0.1)
                continue
            
            self.frame_count += 1
            
            # AI 推理：检测猫咪和家具
            all_detections = self.ai_inference.detect(frame)
            # 分离猫咪和家具检测结果
            cats = [d for d in all_detections if d.label == "cat"]
            furniture = [d for d in all_detections if d.label in ["couch", "chair", "bed", "dining table"]]
            
            # 行为分析
            events = self.behavior_analysis.analyze(cats, furniture)
            
            # 处理检测到的事件
            for event in events:
                self._handle_behavior_event(frame, event)
            
            # 显示画面（如果启用）
            if self.enable_display:
                display_frame = self._draw_detections(
                    frame.copy(), cats, furniture, events
                )
                cv2.imshow('AI 喵管家 - Cat Monitor', display_frame)
                
                # 处理键盘输入
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.stop()
                    break
                elif key == ord('s'):
                    # 手动保存截图
                    self.storage.save_screenshot(
                        frame,
                        BehaviorEvent(
                            behavior_type=None,
                            alert_level=AlertLevel.NONE,
                            timestamp=datetime.now(),
                            cat_detection=None,
                            furniture_detection=None,
                            duration_seconds=0,
                            description="手动截图"
                        )
                    )
            
            # 清理存储空间（每 60 帧执行一次）
            if self.frame_count % 60 == 0:
                self.storage.cleanup_old_files()
                self.storage.cleanup_storage_space()
        
        # 关闭窗口
        if self.enable_display:
            cv2.destroyAllWindows()
    
    def _handle_behavior_event(self, frame: cv2.Mat, event: BehaviorEvent):
        """
        处理检测到的行为事件
        
        Args:
            frame: 当前帧
            event: 行为事件
        """
        logger.info(f"检测到行为：{event.description}")
        
        # 根据警报级别处理
        if event.alert_level == AlertLevel.HIGH:
            # 高级别警报：保存截图 + 录制视频
            self.storage.save_screenshot(frame, event)
            self.alert_count += 1
            
            logger.warning(f"⚠️ 高级别警报：{event.description}")
            
        elif event.alert_level == AlertLevel.MEDIUM:
            # 中级别警报：保存截图
            self.storage.save_screenshot(frame, event)
            self.alert_count += 1
            
            logger.info(f"⚡ 中级别警报：{event.description}")
            
        elif event.alert_level == AlertLevel.LOW:
            # 低级别警报：仅记录
            logger.debug(f"ℹ️ 低级别警报：{event.description}")
    
    def _draw_detections(
        self,
        frame: cv2.Mat,
        cats: List[Detection],
        furniture: List[Detection],
        events: List[BehaviorEvent]
    ) -> cv2.Mat:
        """
        在帧上绘制检测结果
        
        Args:
            frame: 原始帧
            cats: 猫咪检测结果
            furniture: 家具检测结果
            events: 行为事件
        
        Returns:
            cv2.Mat: 绘制后的帧
        """
        # 绘制猫咪检测框（绿色）
        for cat in cats:
            x1, y1, x2, y2 = map(int, cat.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Cat: {cat.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制家具检测框（蓝色）
        for furn in furniture:
            x1, y1, x2, y2 = map(int, furn.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{furn.label}: {furn.confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示当前警报状态
        alert_level, latest_event = self.behavior_analysis.get_current_alert()
        alert_color = {
            AlertLevel.NONE: (0, 255, 0),
            AlertLevel.LOW: (0, 255, 255),
            AlertLevel.MEDIUM: (0, 128, 255),
            AlertLevel.HIGH: (0, 0, 255)
        }.get(alert_level, (0, 255, 0))
        
        status_text = f"Alert: {alert_level.value.upper()}"
        cv2.putText(frame, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, alert_color, 2)
        
        # 显示统计信息
        stats_text = f"FPS: {self._calculate_fps():.1f} | Alerts: {self.alert_count}"
        cv2.putText(frame, stats_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame
    
    def _calculate_fps(self) -> float:
        """计算帧率"""
        if self.start_time is None:
            return 0.0
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            return 0.0
        return self.frame_count / elapsed
    
    def _print_statistics(self):
        """打印统计信息"""
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        logger.info("=" * 50)
        logger.info("监控统计信息")
        logger.info("=" * 50)
        logger.info(f"运行时长：{elapsed:.1f} 秒")
        logger.info(f"处理帧数：{self.frame_count}")
        logger.info(f"平均 FPS: {fps:.2f}")
        logger.info(f"警报次数：{self.alert_count}")
        
        # 存储统计
        storage_stats = self.storage.get_statistics()
        logger.info(f"保存截图：{storage_stats['images_count']} 张")
        logger.info(f"保存视频：{storage_stats['videos_count']} 个")
        logger.info(f"存储空间：{storage_stats['total_size_gb']:.3f} GB")
        logger.info("=" * 50)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="AI 喵管家 - 猫咪破坏行为监控系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m cat_monitor.main                          # 使用默认摄像头
  python -m cat_monitor.main -c 1                     # 使用摄像头 1
  python -m cat_monitor.main -c rtsp://...            # 使用 RTSP 流
  python -m cat_monitor.main --no-display             # 无显示模式（服务器）
  python -m cat_monitor.main --confidence 0.7         # 设置置信度阈值
        """
    )
    
    parser.add_argument(
        '-c', '--camera',
        type=str,
        default='0',
        help='摄像头源：设备索引 (0,1,2...) 或 RTSP 地址 (默认：0)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='models/cat_furniture_detector.pt',
        help='YOLO 模型路径 (默认：models/cat_furniture_detector.pt)'
    )
    
    parser.add_argument(
        '-s', '--storage',
        type=str,
        default='./storage',
        help='存储根目录 (默认：./storage)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='检测置信度阈值 (默认：0.5)'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='禁用显示窗口（服务器模式）'
    )
    
    parser.add_argument(
        '--retries',
        type=int,
        default=3,
        help='摄像头打开失败时的最大重试次数 (默认：3)'
    )
    
    parser.add_argument(
        '--retry-delay',
        type=float,
        default=1.0,
        help='重试间隔秒数 (默认：1.0)'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 解析摄像头源
    camera_source = args.camera
    if camera_source.isdigit():
        camera_source = int(camera_source)
    
    # 创建监控器（带重试参数）
    monitor = CatMonitor(
        camera_source=camera_source,
        model_path=args.model,
        storage_root=args.storage,
        confidence_threshold=args.confidence,
        enable_display=not args.no_display,
        max_retries=args.retries,
        retry_delay=args.retry_delay
    )
    
    # 处理退出信号
    def signal_handler(sig, frame):
        logger.info("收到退出信号")
        monitor.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动监控
    try:
        monitor.start()
    except Exception as e:
        logger.error(f"监控异常：{e}")
        monitor.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()