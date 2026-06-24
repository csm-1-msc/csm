"""
素材存储模块
Storage Module

负责保存猫咪破坏行为的证据素材，包括截图和视频片段。
支持本地存储，预留云存储接口。

核心功能：
- 保存异常行为截图
- 录制短视频片段
- 管理存储文件（压缩、清理）
- 生成素材索引
"""

import os
import cv2
import json
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import shutil

from .behavior_analysis_module import BehaviorEvent, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class MediaFile:
    """媒体文件信息数据类"""
    file_path: str
    file_type: str  # "image" or "video"
    timestamp: datetime
    event_type: str
    file_size: int = 0
    duration: float = 0.0  # 视频时长（秒）
    thumbnail_path: Optional[str] = None
    
    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            "file_path": self.file_path,
            "file_type": self.file_type,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "file_size": self.file_size,
            "duration": self.duration,
            "thumbnail_path": self.thumbnail_path
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MediaFile":
        """从字典创建实例"""
        return cls(
            file_path=data["file_path"],
            file_type=data["file_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            event_type=data["event_type"],
            file_size=data.get("file_size", 0),
            duration=data.get("duration", 0.0),
            thumbnail_path=data.get("thumbnail_path")
        )


class StorageModule:
    """
    素材存储模块类
    
    管理猫咪破坏行为证据素材的存储。
    
    目录结构：
    storage_root/
    ├── images/           # 截图文件夹
    │   ├── 2024-01-15/
    │   │   ├── cat_scratching_sofa_143052.jpg
    │   │   └── ...
    │   └── ...
    ├── videos/           # 视频片段文件夹
    │   ├── 2024-01-15/
    │   │   ├── cat_biting_chair_143052.mp4
    │   │   └── ...
    │   └── ...
    ├── thumbnails/       # 缩略图文件夹
    │   └── ...
    └── index.json        # 素材索引文件
    
    Attributes:
        storage_root (str): 存储根目录路径
        max_storage_days (int): 素材保留天数
        max_storage_size_gb (float): 最大存储空间（GB）
        image_quality (int): JPEG 图片质量 (1-100)
        video_fps (int): 视频帧率
        video_codec (str): 视频编码格式
    """
    
    def __init__(
        self,
        storage_root: str = "./storage",
        max_storage_days: int = 30,
        max_storage_size_gb: float = 5.0,
        image_quality: int = 90,
        video_fps: int = 20,
        video_codec: str = "mp4v"
    ):
        """
        初始化存储模块
        
        Args:
            storage_root: 存储根目录路径
            max_storage_days: 素材保留天数，超过此天数的文件将被清理
            max_storage_size_gb: 最大存储空间（GB），超过时将清理旧文件
            image_quality: JPEG 图片质量 (1-100)
            video_fps: 视频录制帧率
            video_codec: 视频编码格式 ("mp4v", "X264", "H264" 等)
        """
        self.storage_root = Path(storage_root)
        self.max_storage_days = max_storage_days
        self.max_storage_size_gb = max_storage_size_gb
        self.image_quality = image_quality
        self.video_fps = video_fps
        self.video_codec = video_codec
        
        # 子目录
        self.images_dir = self.storage_root / "images"
        self.videos_dir = self.storage_root / "videos"
        self.thumbnails_dir = self.storage_root / "thumbnails"
        self.index_path = self.storage_root / "index.json"
        
        # 素材索引
        self.media_index: List[MediaFile] = []
        
        # 视频录制器
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._recording_frames: List[cv2.Mat] = []
        self._is_recording = False
        
        logger.info(f"StorageModule 初始化：storage_root={storage_root}")
        
        # 初始化目录和索引
        self._initialize()
    
    def _initialize(self):
        """初始化存储目录和加载索引"""
        # 创建目录
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.thumbnails_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载索引
        self._load_index()
        
        logger.info("存储目录初始化完成")
    
    def _load_index(self):
        """加载素材索引"""
        if self.index_path.exists():
            try:
                with open(self.index_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.media_index = [
                        MediaFile.from_dict(item) for item in data.get("media_files", [])
                    ]
                logger.info(f"已加载 {len(self.media_index)} 个素材索引")
            except Exception as e:
                logger.error(f"加载索引文件失败：{e}")
                self.media_index = []
        else:
            self.media_index = []
            logger.info("索引文件不存在，创建新索引")
    
    def _save_index(self):
        """保存素材索引"""
        try:
            with open(self.index_path, 'w', encoding='utf-8') as f:
                json.dump(
                    {"media_files": [m.to_dict() for m in self.media_index]},
                    f,
                    ensure_ascii=False,
                    indent=2
                )
            logger.debug("索引已保存")
        except Exception as e:
            logger.error(f"保存索引文件失败：{e}")
    
    def save_screenshot(
        self,
        frame: cv2.Mat,
        event: BehaviorEvent,
        sub_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        保存截图
        
        Args:
            frame: 图像帧 (BGR 格式)
            event: 触发保存的行为事件
            sub_dir: 子目录名（默认为日期）
        
        Returns:
            Optional[str]: 保存的文件路径，失败返回 None
        """
        if sub_dir is None:
            sub_dir = datetime.now().strftime("%Y-%m-%d")
        
        # 创建日期子目录
        date_dir = self.images_dir / sub_dir
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%H%M%S")
        event_type = event.behavior_type.value
        furniture = event.furniture_detection.label if event.furniture_detection else "unknown"
        filename = f"cat_{event_type}_{furniture}_{timestamp}.jpg"
        file_path = date_dir / filename
        
        # 保存截图
        try:
            cv2.imwrite(str(file_path), frame, [cv2.IMWRITE_JPEG_QUALITY, self.image_quality])
            
            # 生成缩略图
            thumbnail_path = self._generate_thumbnail(frame, file_path.stem + "_thumb.jpg")
            
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 添加到索引
            media_file = MediaFile(
                file_path=str(file_path),
                file_type="image",
                timestamp=datetime.now(),
                event_type=event_type,
                file_size=file_size,
                thumbnail_path=thumbnail_path
            )
            self.media_index.append(media_file)
            self._save_index()
            
            logger.info(f"截图已保存：{file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存截图失败：{e}")
            return None
    
    def start_recording(self, frame_size: Tuple[int, int]):
        """
        开始录制视频
        
        Args:
            frame_size: 帧尺寸 (宽度，高度)
        """
        if self._is_recording:
            logger.warning("已经在录制中")
            return
        
        self._recording_frames = []
        self._is_recording = True
        logger.info("开始录制视频")
    
    def add_frame_to_recording(self, frame: cv2.Mat):
        """
        添加帧到录制队列
        
        Args:
            frame: 图像帧
        """
        if not self._is_recording:
            return
        self._recording_frames.append(frame)
    
    def stop_and_save_recording(
        self,
        event: BehaviorEvent,
        sub_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        停止录制并保存视频
        
        Args:
            event: 触发保存的行为事件
            sub_dir: 子目录名
        
        Returns:
            Optional[str]: 保存的文件路径，失败返回 None
        """
        if not self._is_recording or not self._recording_frames:
            self._is_recording = False
            self._recording_frames = []
            return None
        
        self._is_recording = False
        
        if sub_dir is None:
            sub_dir = datetime.now().strftime("%Y-%m-%d")
        
        # 创建日期子目录
        date_dir = self.videos_dir / sub_dir
        date_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名
        timestamp = datetime.now().strftime("%H%M%S")
        event_type = event.behavior_type.value
        furniture = event.furniture_detection.label if event.furniture_detection else "unknown"
        filename = f"cat_{event_type}_{furniture}_{timestamp}.mp4"
        file_path = date_dir / filename
        
        try:
            # 获取帧尺寸
            height, width = self._recording_frames[0].shape[:2]
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            out = cv2.VideoWriter(
                str(file_path),
                fourcc,
                self.video_fps,
                (width, height)
            )
            
            # 写入帧
            for frame in self._recording_frames:
                out.write(frame)
            
            out.release()
            
            # 计算视频时长
            duration = len(self._recording_frames) / self.video_fps
            
            # 生成缩略图
            thumbnail_path = self._generate_thumbnail(
                self._recording_frames[len(self._recording_frames) // 2],
                file_path.stem + "_thumb.jpg"
            )
            
            # 获取文件大小
            file_size = file_path.stat().st_size
            
            # 添加到索引
            media_file = MediaFile(
                file_path=str(file_path),
                file_type="video",
                timestamp=datetime.now(),
                event_type=event_type,
                file_size=file_size,
                duration=duration,
                thumbnail_path=thumbnail_path
            )
            self.media_index.append(media_file)
            self._save_index()
            
            logger.info(f"视频已保存：{file_path}, 时长={duration:.1f}秒")
            
            self._recording_frames = []
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存视频失败：{e}")
            self._recording_frames = []
            return None
    
    def _generate_thumbnail(
        self,
        frame: cv2.Mat,
        filename: str,
        size: Tuple[int, int] = (160, 120)
    ) -> str:
        """
        生成缩略图
        
        Args:
            frame: 图像帧
            filename: 输出文件名
            size: 缩略图尺寸
        
        Returns:
            str: 缩略图路径
        """
        date_str = datetime.now().strftime("%Y-%m-%d")
        thumb_dir = self.thumbnails_dir / date_str
        thumb_dir.mkdir(parents=True, exist_ok=True)
        
        # 调整大小
        thumbnail = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        
        # 保存
        thumb_path = thumb_dir / filename
        cv2.imwrite(str(thumb_path), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        return str(thumb_path)
    
    def cleanup_old_files(self):
        """清理过期文件"""
        cutoff_date = datetime.now() - timedelta(days=self.max_storage_days)
        
        files_to_remove = []
        for media in self.media_index:
            if media.timestamp < cutoff_date:
                files_to_remove.append(media)
        
        # 删除文件
        for media in files_to_remove:
            try:
                # 删除主文件
                if os.path.exists(media.file_path):
                    os.remove(media.file_path)
                
                # 删除缩略图
                if media.thumbnail_path and os.path.exists(media.thumbnail_path):
                    os.remove(media.thumbnail_path)
                
                self.media_index.remove(media)
                logger.debug(f"已删除过期文件：{media.file_path}")
            except Exception as e:
                logger.error(f"删除文件失败：{e}")
        
        if files_to_remove:
            self._save_index()
            logger.info(f"清理了 {len(files_to_remove)} 个过期文件")
    
    def cleanup_storage_space(self):
        """清理存储空间，当超过最大容量时删除旧文件"""
        # 计算当前存储空间
        total_size = self._calculate_storage_size()
        
        if total_size < self.max_storage_size_gb * 1024 * 1024 * 1024:
            return  # 未超限
        
        logger.info(f"存储空间超限 ({total_size / 1024 / 1024 / 1024:.2f}GB)，开始清理")
        
        # 按时间排序，删除最旧的文件
        sorted_media = sorted(self.media_index, key=lambda x: x.timestamp)
        
        for media in sorted_media:
            if total_size < self.max_storage_size_gb * 1024 * 1024 * 1024:
                break
            
            try:
                if os.path.exists(media.file_path):
                    file_size = os.path.getsize(media.file_path)
                    os.remove(media.file_path)
                    total_size -= file_size
                    
                    if media.thumbnail_path and os.path.exists(media.thumbnail_path):
                        os.remove(media.thumbnail_path)
                    
                    self.media_index.remove(media)
                    logger.debug(f"已删除文件释放空间：{media.file_path}")
            except Exception as e:
                logger.error(f"删除文件失败：{e}")
        
        self._save_index()
        logger.info(f"存储空间清理完成，当前大小：{total_size / 1024 / 1024 / 1024:.2f}GB")
    
    def _calculate_storage_size(self) -> int:
        """计算当前存储总大小（字节）"""
        total_size = 0
        
        for directory in [self.images_dir, self.videos_dir, self.thumbnails_dir]:
            if directory.exists():
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        total_size += file_path.stat().st_size
        
        return total_size
    
    def get_media_files(
        self,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[MediaFile]:
        """
        查询媒体文件
        
        Args:
            event_type: 事件类型过滤
            start_date: 开始日期
            end_date: 结束日期
            limit: 最大返回数量
        
        Returns:
            List[MediaFile]: 媒体文件列表
        """
        results = self.media_index.copy()
        
        if event_type:
            results = [m for m in results if m.event_type == event_type]
        
        if start_date:
            results = [m for m in results if m.timestamp >= start_date]
        
        if end_date:
            results = [m for m in results if m.timestamp <= end_date]
        
        # 按时间倒序排列
        results.sort(key=lambda x: x.timestamp, reverse=True)
        
        return results[:limit]
    
    def get_statistics(self) -> dict:
        """
        获取存储统计信息
        
        Returns:
            dict: 统计信息字典
        """
        total_size = self._calculate_storage_size()
        
        # 按事件类型统计
        event_stats = {}
        for media in self.media_index:
            event_type = media.event_type
            if event_type not in event_stats:
                event_stats[event_type] = {"count": 0, "size": 0}
            event_stats[event_type]["count"] += 1
            event_stats[event_type]["size"] += media.file_size
        
        return {
            "total_files": len(self.media_index),
            "total_size_bytes": total_size,
            "total_size_gb": total_size / 1024 / 1024 / 1024,
            "images_count": len([m for m in self.media_index if m.file_type == "image"]),
            "videos_count": len([m for m in self.media_index if m.file_type == "video"]),
            "event_stats": event_stats,
            "storage_root": str(self.storage_root)
        }
    
    # =========================================================================
    # 扩展接口：云存储
    # =========================================================================
    # 后续可添加云存储上传功能，如：
    # - AWS S3
    # - Google Cloud Storage
    # - 阿里云 OSS
    # - 七牛云
    # =========================================================================
    
    def upload_to_cloud(self, file_path: str, cloud_provider: str = "s3"):
        """
        上传文件到云存储（预留接口）
        
        Args:
            file_path: 本地文件路径
            cloud_provider: 云服务商 ("s3", "gcs", "oss", "qiniu")
        
        TODO: 后续实现云存储上传
        """
        logger.info(f"[预留接口] 上传文件到{cloud_provider}: {file_path}")
        pass
    
    def sync_to_cloud(self):
        """
        同步所有素材到云存储（预留接口）
        
        TODO: 后续实现云同步
        """
        logger.info("[预留接口] 同步素材到云存储")
        pass