import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
import akshare as ak

# ==========================================
# 1. 配置区：从 GitHub Secrets 读取信息
# ==========================================
# 这样写即使你不小心把代码公开，别人也拿不到你的 Token
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def send_to_telegram(photo_path, caption):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("❌ 错误：未找到 TELEGRAM_TOKEN 或 CHAT_ID 环境变量")
        return {"ok": False}
    
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        try:
            res = requests.post(url, files=files, data=data)
            return res.json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

# ==========================================
# 2. 第一步：抓取数据
# ==========================================
def get_real_data():
    print(">>> 正在从 AkShare 抓取上证指数最新日线...")
    df = ak.stock_zh_index_daily(symbol="sh000001")
    df = df.rename(columns={'date': 'timestamps'})
    cols = ['timestamps', 'open', 'high', 'low', 'close', 'volume']
    df = df[cols].copy()
    df['timestamps'] = df['timestamps'].astype(str)
    
    os.makedirs("./data", exist_ok=True)
    save_path = "./data/sse_daily_latest.csv"
    df.to_csv(save_path, index=False)
    print(f"✅ 数据抓取成功！最新日期: {df['timestamps'].iloc[-1]}")
    return save_path

# ==========================================
# 3. 第二步：Kronos 推理
# ==========================================
def run_prediction(data_path):
    # GitHub Actions 环境没有 GPU，强制使用 CPU
    device = "cpu" 
    print(f">>> 正在准备模型推理 (使用设备: {device})...")

    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    lookback, pred_len = 300, 30
    x_df = df.tail(lookback).reset_index(drop=True)
    x_ts = pd.Series(x_df['timestamps'])
    
    y_ts = pd.date_range(start=x_ts.iloc[-1] + pd.Timedelta(days=1), periods=pred_len, freq="D")
    y_ts = pd.Series(y_ts)

    pred_df = predictor.predict(
        df=x_df[['open', 'high', 'low', 'close', 'volume']], 
        x_timestamp=x_ts, 
        y_timestamp=y_ts, 
        pred_len=pred_len
    )

    # --- 优化后的绘图逻辑 ---
    plt.figure(figsize=(12, 6))
    
    # 获取最近 100 天历史数据用于对比显示
    hist_show_len = 100
    hist_x = x_df['timestamps'].iloc[-hist_show_len:]
    hist_y = x_df['close'].iloc[-hist_show_len:]
    
    plt.plot(hist_x, hist_y, label='History (Last 100D)', color='#1f77b4', linewidth=2)
    
    # 预测曲线的 x 轴是刚才生成的 y_ts
    plt.plot(y_ts, pred_df['close'], label='AI Predict (Next 30D)', 
             color='#d62728', linestyle='--', linewidth=2)
    
    plt.title(f"SSE Index Daily Forecast (Base on {df['timestamps'].iloc[-1]})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    img_path = "sse_daily_forecast.png"
    plt.savefig(img_path)
    plt.close()

    # 计算涨跌百分比
    start_p = x_df['close'].iloc[-1]
    end_p = pred_df['close'].iloc[-1]
    change = ((end_p / start_p) - 1) * 100

    msg = (f"📊 *Kronos 上证指数日线预测*\n"
           f"📅 截止日期：{df['timestamps'].iloc[-1].split()[0]}\n"
           f"🔮 预测长度：未来 30 个周期\n"
           f"📈 预估末端涨跌幅：{change:+.2f}%")
    
    res = send_to_telegram(img_path, msg)
    if res.get("ok"):
        print("✅ 预测结果已成功推送到 Telegram！")
    else:
        print(f"❌ 推送失败，Telegram 返回值: {res}")

# ==========================================
# 4. 运行入口
# ==========================================
if __name__ == "__main__":
    try:
        path = get_real_data()
        run_prediction(path)
    except Exception as e:
        print(f"💥 运行崩溃: {e}")
