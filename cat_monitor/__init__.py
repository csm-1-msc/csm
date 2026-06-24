"""
猫咪破坏家具 AI 监控系统
Cat Furniture Destruction AI Monitoring System

本模块用于监控猫咪破坏家具行为，支持电脑 USB 摄像头实时监测，
后续可无缝迁移至树莓派 + Hailo AI Kit 硬件环境。

Author: CSM Team
Version: 1.0.0
"""

from .camera_module import CameraModule
from .ai_inference_module import AIInferenceModule
from .behavior_analysis_module import BehaviorAnalysisModule
from .storage_module import StorageModule

__version__ = "1.0.0"
__all__ = [
    "CameraModule",
    "AIInferenceModule", 
    "BehaviorAnalysisModule",
    "StorageModule"
]