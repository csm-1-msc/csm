#   AI 智能精准投喂（定时投喂）
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# 模拟猫咪进食数据（时间, 投喂量g, 剩余量g, 进食时长s）
feed_data = pd.DataFrame({
    "time": pd.date_range("2026-01-01 08:00", periods=10, freq="h"),  # 修正：H→h
    "feed_amount": [30, 0, 0, 0, 0, 0, 40, 0, 0, 25],
    "remain_amount": [5, 0, 0, 0, 0, 0, 10, 0, 0, 0],
    "eat_duration": [20, 0, 0, 0, 0, 0, 25, 0, 0, 18]
})
# 将time列设为索引，满足ARIMA的时间序列要求
feed_data = feed_data.set_index("time")

# AI 预测最佳投喂时间和份量
def ai_predict_feed_plan(data):
    # 提取有效进食时段的投喂量序列
    feed_series = data["feed_amount"][data["feed_amount"] > 0]
    # 重置索引，确保序列连续
    feed_series = feed_series.reset_index(drop=True)
    
    # ARIMA 模型预测下一次投喂量
    model = ARIMA(feed_series, order=(1, 0, 0))
    model_fit = model.fit()
    next_feed_amount = int(model_fit.forecast(steps=1).iloc[0])  # 修正：用iloc取第一个值
    
    # 结合作息（主人上班8点出门），推荐投喂时间
    next_feed_time = "07:30" if pd.Timestamp.now().hour < 8 else "18:30"
    return f"AI推荐投喂：{next_feed_time}，份量{next_feed_amount}g"

# 测试：生成投喂计划
print(ai_predict_feed_plan(feed_data))

