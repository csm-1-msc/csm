#   AI 行为异常预警（激光互动）
import numpy as np
from sklearn.ensemble import IsolationForest

# 模拟正常行为数据（每日抓挠次数, 每日进食量g）
# 补充更多样本，让AI模型学习更充分
normal_data = np.array([
    [2, 90], [1, 85], [3, 95], [2, 88],
    [1, 92], [2, 87], [3, 89], [2, 91]
])

# 训练异常检测模型，优化参数提升小样本敏感度
model = IsolationForest(
    contamination=0.3,  # 提高异常样本比例阈值
    n_estimators=100,   # 增加决策树数量，提升稳定性
    random_state=42     # 固定随机种子，保证结果可复现
)
model.fit(normal_data)

# 定义基础规则阈值（兜底判断）
SCRATCH_THRESHOLD = 3  # 抓挠次数超过3次为异常
EAT_THRESHOLD = 80     # 进食量低于80g为异常

# AI 检测异常行为（规则+AI双重校验）
def ai_detect_abnormal(behavior_data):
    scratch_times, eat_amount = behavior_data
    
    # 第一步：基础规则判断（兜底）
    if scratch_times > SCRATCH_THRESHOLD:
        return "预警：猫咪抓挠次数过多，易拆家！建议启动AI逗猫"
    if eat_amount < EAT_THRESHOLD:
        return "预警：猫咪进食量不足，检查投喂是否正常！"
    
    # 第二步：AI模型检测（识别潜在异常）
    ai_result = model.predict([behavior_data])[0]
    if ai_result == -1:
        return "预警：猫咪行为异常，建议关注状态！"
    
    # 无异常
    return "猫咪行为正常，无需干预"

# 测试1：抓挠异常（5次，进食90g）
print("测试1 - 抓挠异常：", ai_detect_abnormal([5, 90]))
# 测试2：进食异常（2次，进食70g）
print("测试2 - 进食异常：", ai_detect_abnormal([2, 70]))
# 测试3：正常行为（2次，进食88g）
print("测试3 - 正常行为：", ai_detect_abnormal([2, 88]))
# 测试4：AI识别潜在异常（4次抓挠，85g进食）
print("测试4 - 潜在异常：", ai_detect_abnormal([4, 85]))

