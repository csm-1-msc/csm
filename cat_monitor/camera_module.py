"""
摄像头读取模块
Camera Module

负责摄像头的初始化、视频帧读取和释放。
支持电脑 USB 摄像头，预留树莓派摄像头接口。

硬件抽象层设计，便于后续迁移到树莓派环境。
"""

import cv2
import logging
import platform
import time
from typing import Optional, Tuple, List
from datetime import datetime

# 配置日志
logger = logging.getLogger(__name__)


class CameraOccupiedError(Exception):
    """摄像头被占用异常"""
    pass


class CameraNotFoundError(Exception):
    """摄像头未找到异常"""
    pass


class CameraModule:
    """
    摄像头模块类
    
    封装摄像头操作，提供统一的帧读取接口。
    支持 Windows 电脑摄像头，预留树莓派 CSI 摄像头接口。
    
    Attributes:
        camera_id (int): 摄像头设备 ID，Windows 通常为 0
        width (int): 帧宽度
        height (int): 帧高度
        fps (int): 帧率
        cap: OpenCV VideoCapture 对象
    """
    
    # 类变量，跟踪已打开的摄像头
    _opened_cameras: set = set()
    
    def __init__(
        self, 
        camera_id: int = 0, 
        width: int = 640, 
        height: int = 480, 
        fps: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        """
        初始化摄像头模块
        
        Args:
            camera_id: 摄像头设备 ID
                - Windows: 通常为 0 (内置摄像头) 或 1+ (外接 USB 摄像头)
                - Linux/Raspberry Pi: 0 或 /dev/video0 路径
            width: 帧宽度 (像素)
            height: 帧高度 (像素)
            fps: 目标帧率
            max_retries: 打开失败时的最大重试次数
            retry_delay: 重试间隔（秒）
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_opened = False
        
        logger.info(f"CameraModule 初始化：camera_id={camera_id}, resolution={width}x{height}, "
                   f"max_retries={max_retries}, retry_delay={retry_delay}s")
    
    @classmethod
    def list_available_cameras(cls) -> List[int]:
        """
        列出系统中所有可用的摄像头设备
        
        Returns:
            List[int]: 可用的摄像头 ID 列表
        """
        available = []
        # 检查前 10 个可能的摄像头 ID
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
    
    @classmethod
    def is_camera_available(cls, camera_id: int) -> bool:
        """
        检查指定摄像头是否可用（未被占用）
        
        Args:
            camera_id: 摄像头设备 ID
            
        Returns:
            bool: 摄像头是否可用
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.release()
                return True
            return False
        except Exception:
            return False
    
    def open(self) -> bool:
        """
        打开摄像头（带重试机制）
        
        Returns:
            bool: 是否成功打开
        """
        # 检查摄像头是否已被本模块其他实例打开
        if self.camera_id in CameraModule._opened_cameras:
            logger.warning(f"摄像头 ID {self.camera_id} 已被本模块其他实例打开")
            # 尝试先检查是否真的被占用
            if not self.is_camera_available(self.camera_id):
                logger.error(f"摄像头 ID {self.camera_id} 确实被其他程序占用")
                return False
        
        # 带重试的打开逻辑
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"尝试打开摄像头 ID {self.camera_id} (第 {attempt}/{self.max_retries} 次)")
                
                # Windows 环境下使用 DirectShow 后端可提高兼容性
                backend = cv2.CAP_DSHOW if self._is_windows() else cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.camera_id, backend)
                
                if not self.cap.isOpened():
                    logger.warning(f"无法打开摄像头设备 ID: {self.camera_id}")
                    if attempt < self.max_retries:
                        logger.info(f"等待 {self.retry_delay} 秒后重试...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"达到最大重试次数，摄像头打开失败")
                        return False
                
                # 设置分辨率
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)
                
                # 获取实际设置的参数（可能与请求值不同）
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                self.is_opened = True
                CameraModule._opened_cameras.add(self.camera_id)
                logger.info(f"摄像头已打开：实际分辨率={actual_width}x{actual_height}, FPS={actual_fps}")
                return True
                
            except CameraOccupiedError as e:
                logger.error(f"摄像头被占用：{e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue
            except Exception as e:
                logger.error(f"打开摄像头时发生错误：{e}")
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay)
                continue
        
        return False
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat]]:
        """
        读取一帧图像
        
        Returns:
            Tuple[bool, Optional[cv2.Mat]]: (成功标志，图像帧)
                - 成功时返回 (True, frame)
                - 失败时返回 (False, None)
        """
        if not self.is_opened or self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        if ret:
            return True, frame
        else:
            logger.warning("读取帧失败，可能摄像头已断开")
            # 尝试重新打开
            self.is_opened = False
            return False, None
    
    def release(self):
        """
        释放摄像头资源
        
        必须在程序退出前调用，否则可能导致摄像头被占用。
        """
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False
            CameraModule._opened_cameras.discard(self.camera_id)
            logger.info("摄像头已释放")
    
    def get_frame_size(self) -> Tuple[int, int]:
        """
        获取帧尺寸
        
        Returns:
            Tuple[int, int]: (宽度，高度)
        """
        if self.cap is None:
            return self.width, self.height
        return (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
    
    @staticmethod
    def _is_windows() -> bool:
        """判断当前是否为 Windows 系统"""
        return platform.system() == "Windows"
    
    # =========================================================================
    # 扩展接口：树莓派摄像头支持
    # =========================================================================
    # 后续迁移到树莓派时，可在此添加 CSI 摄像头或 USB 摄像头的特殊配置
    # 例如：
    #   - 树莓派 CSI 摄像头：使用 libcamera 或 picamera2
    #   - 调整曝光、增益等参数
    # =========================================================================
    
    def set_exposure(self, value: int):
        """
        设置曝光值（预留接口）
        
        Args:
            value: 曝光值
        """
        if self.cap is not None:
            # OpenCV 曝光控制因驱动而异，此处为示例
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # 禁用自动曝光
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            logger.debug(f"曝光值设置为：{value}")
    
    def __enter__(self):
        """上下文管理器入口"""
        if not self.open():
            raise CameraOccupiedError(f"无法打开摄像头 ID {self.camera_id}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()