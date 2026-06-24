# 🐱 AI 喵管家（AI Meow Butler）

> **Slogan**: 有 AI 喵陪，猫咪不拆家，吃饭也准时

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Version](https://img.shields.io/badge/version-v1.0.0-green.svg)

## 📖 项目简介

AI 喵管家是一个专为上班族设计的智能猫咪陪伴系统，通过 AI 技术解决猫咪独处时的三大痛点：
- 🏠 **拆家问题**：猫咪精力过剩，乱抓家具、搞破坏
- 🍽️ **喂食问题**：无法精准定时投喂，影响猫咪肠胃健康
- ⚠️ **异常预警**：无法及时发现猫咪行为异常

## 🎯 核心功能

### 1️⃣ AI 智能逗猫互动
基于 **RandomForest 行为分类模型** + **YOLOv8 动作识别**，自动判定猫咪无聊/精力旺盛状态，触发激光逗猫功能，帮助猫咪消耗过剩精力。

**核心文件**: `MeowMateAI/action_recognition.py`

### 2️⃣ AI 智能精准投喂
采用 **ARIMA 时间序列预测模型**，学习猫咪进食规律，结合主人作息动态定时定量投喂。

**核心文件**: `MeowMateAI/feed_guide.py`

### 3️⃣ AI 行为异常预警
**IsolationForest 异常检测模型** + 规则阈值双重校验，识别抓挠过多、进食不足等异常情况，及时推送预警提醒。

**核心文件**: `MeowMateAI/laser_interaction.py`

### 4️⃣ 📹 Cat Monitor 实时监控（NEW!）
基于 **YOLOv8 目标检测** + **行为分析引擎**，实时监测猫咪与家具的交互行为，自动识别破坏行为并保存证据。

**核心模块**: `cat_monitor/`
- `camera_module.py` - 摄像头视频流读取（带重试机制和占用检测）
- `ai_inference_module.py` - YOLOv8 AI 推理
- `behavior_analysis_module.py` - 行为分析与警报判定
- `storage_module.py` - 证据素材存储管理
- `main.py` - 主程序入口

**测试脚本**: `test_behavior_analysis.py` - 行为分析模块单元测试（5 个测试用例）

**日志输出**: `logs/behavior_log.csv` - CSV 格式行为日志（时间戳、行为类型、警报级别、持续时间、描述）

## 🛠️ AI 技术栈

| 功能 | AI 模型 | 说明 |
|------|--------|------|
| 动作识别 | YOLOv8 | 精准识别猫抓家具/蹭物体行为 |
| 状态分析 | RandomForest | 判定猫咪情绪/行为状态 |
| 投喂预测 | ARIMA | 生成个性化动态投喂计划 |
| 异常检测 | IsolationForest | 识别猫咪行为异常 |

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Windows / macOS / Linux

### Windows 用户一键安装（推荐）

> 💡 Windows 用户可使用 PowerShell 脚本一键安装，最简单！

1. **克隆仓库**
```powershell
git clone https://github.com/csm-1-msc/csm.git
cd csm
```

2. **运行一键安装脚本**
```powershell
# 允许执行脚本（首次需要）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# 运行安装脚本（按提示确认）
.\install_dependencies.ps1

# 或跳过确认直接安装
.\install_dependencies.ps1 -SkipConfirm
```

  3. **启动程序**
  ```powershell
  # Run in foreground (with display window)
  .\start_cat_monitor.ps1

  # Run in background (no display, for servers)
  .\start_cat_monitor.ps1 -NoDisplay

  # Show help information
  .\start_cat_monitor.ps1 -Help

  # Use camera device 1
  .\start_cat_monitor.ps1 -Camera 1

  # Set confidence threshold to 0.7
  .\start_cat_monitor.ps1 -Confidence 0.7
  ```

### 手动安装步骤（所有平台）

1. **克隆仓库**
```bash
git clone https://github.com/csm-1-msc/csm.git
cd csm
```

2. **创建虚拟环境（推荐）**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **安装依赖**

> 💡 中国用户建议使用清华镜像源加速下载：

```bash
# 步骤 1: 升级 pip
python -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 2: 安装基础库
pip install opencv-python numpy colorama -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 3: 安装数据分析库
pip install scikit-learn pandas statsmodels -i https://pypi.tuna.tsinghua.edu.cn/simple

# 步骤 4: 安装 PyTorch CPU 版
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 步骤 5: 安装 ultralytics (YOLOv8)
pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
```

4. **运行示例**
```bash
# 猫咪行为识别测试
python MeowMateAI/action_recognition.py

# 智能投喂预测
python MeowMateAI/feed_guide.py

# 行为异常检测
python MeowMateAI/laser_interaction.py
```

## 📁 项目结构

```
csm/
├── README.md                 # 项目说明文档
├── LICENSE                   # MIT 开源协议
├── SECURITY.md               # 安全与敏感信息检测报告
├── requirements.txt          # 项目依赖列表
├── .env.example              # 环境变量示例模板
├── .gitignore               # Git 忽略文件配置
├── MeowMateAI/              # 核心 AI 功能模块
│   ├── action_recognition.py # AI 智能逗猫互动（行为识别）
│   ├── feed_guide.py        # AI 智能精准投喂（定时投喂）
│   └── laser_interaction.py # AI 行为异常预警（激光互动）
└── cat_monitor/             # 📹 Cat Monitor 实时监控模块
    ├── __init__.py          # 包初始化
    ├── main.py              # 主程序入口
    ├── camera_module.py     # 摄像头视频流读取
    ├── ai_inference_module.py  # YOLOv8 AI 推理
    ├── behavior_analysis_module.py  # 行为分析与警报判定
    └── storage_module.py    # 证据素材存储管理
```

## 📝 使用示例

### 猫咪行为识别
```python
from MeowMateAI.action_recognition import train_cat_behavior_model, ai_interact_with_cat

# 训练模型
model = train_cat_behavior_model()

# 识别猫咪状态并互动
features = [0.2, 9, 6, 11]  # [趴卧时长占比，踱步次数，抓挠次数，叫声次数]
result = ai_interact_with_cat(features, model)
```

### 智能投喂预测
```python
from MeowMateAI.feed_guide import ai_predict_feed_plan
import pandas as pd

# 准备进食数据
feed_data = pd.DataFrame({
    "feed_amount": [30, 0, 0, 0, 0, 0, 40, 0, 0, 25],
})

# 预测最佳投喂方案
plan = ai_predict_feed_plan(feed_data)
print(plan)
```

### 异常行为检测
```python
from MeowMateAI.laser_interaction import ai_detect_abnormal

# 检测抓挠异常
result = ai_detect_abnormal([5, 90])  # [抓挠次数，进食量]
print(result)
```

### 📹 Cat Monitor 实时监控

```bash
# 使用默认摄像头启动监控
python -m cat_monitor.main

# 使用指定摄像头（设备索引 1）
python -m cat_monitor.main -c 1

# 使用 RTSP 网络摄像头
python -m cat_monitor.main -c rtsp://192.168.1.100:554/stream

# 无显示模式（服务器/后台运行）
python -m cat_monitor.main --no-display

# 设置置信度阈值为 0.7
python -m cat_monitor.main --confidence 0.7

# 指定模型和存储路径
python -m cat_monitor.main -m models/my_model.pt -s /path/to/storage

# 设置摄像头重试次数和间隔（解决占用问题）
python -m cat_monitor.main --retries 5 --retry-delay 2.0
```

**快捷键**:
- `Q` - 退出程序
- `S` - 手动保存当前帧截图

**命令行参数**:
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c, --camera` | 摄像头源（设备索引或 RTSP 地址） | `0` |
| `-m, --model` | YOLO 模型路径 | `models/cat_furniture_detector.pt` |
| `-s, --storage` | 存储根目录 | `./storage` |
| `--confidence` | 检测置信度阈值 | `0.5` |
| `--no-display` | 禁用显示窗口（服务器模式） | `false` |
| `--retries` | 摄像头打开失败时的最大重试次数 | `3` |
| `--retry-delay` | 重试间隔秒数 | `1.0` |

**常见问题**:
- **摄像头被占用**: 关闭其他可能使用摄像头的程序（如 Zoom、Teams、微信等），或使用 `--retries` 参数增加重试次数
- **无法打开摄像头**: 检查摄像头设备索引是否正确，使用 `list_available_cameras()` 方法查看可用摄像头
- **无显示窗口**: 服务器环境请使用 `--no-display` 参数

**输出说明**:
- 截图保存在 `storage/screenshots/` 目录
- 视频保存在 `storage/videos/` 目录
- 按日期自动分类存储

## 🔒 安全说明

本项目已进行敏感信息扫描，未发现 API 密钥、密码、内网 IP 等敏感信息泄露。

详细检测报告请查看 [`SECURITY.md`](SECURITY.md)

如需配置敏感信息（如 API 密钥），请：
1. 复制 `.env.example` 为 `.env`
2. 填写必要的配置项
3. `.env` 文件已被加入 `.gitignore`，不会被提交到版本控制

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 开源协议

本项目采用 **MIT License** - 详见 [`LICENSE`](LICENSE) 文件

## 📮 联系方式

- 项目仓库：https://github.com/csm-1-msc/csm
- 提交 Issue：https://github.com/csm-1-msc/csm/issues

---

<div align="center">

**Made with ❤️ for Cat Lovers**

⭐ 如果这个项目对你有帮助，请给一个 Star！

</div>