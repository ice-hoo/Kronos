import os
import smtplib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import requests
import akshare as ak
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

# ==========================================
# 1. 配置区：从 GitHub Secrets 读取信息
# ==========================================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
EMAIL_USER = os.getenv("EMAIL_USER")     # QQ邮箱账号
EMAIL_PASS = os.getenv("EMAIL_PASS")     # QQ邮箱授权码
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER") # 接收邮箱

# ==========================================
# 2. 推送函数定义
# ==========================================

def send_to_telegram(photo_path, caption):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ 未配置 Telegram 环境，跳过")
        return {"ok": False}
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(photo_path, 'rb') as photo:
        files = {'photo': photo}
        data = {'chat_id': CHAT_ID, 'caption': caption, 'parse_mode': 'Markdown'}
        try:
            return requests.post(url, files=files, data=data).json()
        except Exception as e:
            return {"ok": False, "error": str(e)}

def send_to_email(photo_path, text):
    if not EMAIL_USER or not EMAIL_PASS:
        print("⚠️ 未配置邮件环境，跳过")
        return
    msg = MIMEMultipart()
    msg['Subject'] = f"A-share 预测日报 - {pd.Timestamp.now().strftime('%Y-%m-%d')}"
    msg['From'] = EMAIL_USER
    msg['To'] = EMAIL_RECEIVER
    msg.attach(MIMEText(text, 'plain'))
    
    with open(photo_path, 'rb') as f:
        image = MIMEImage(f.read(), name=os.path.basename(photo_path))
        msg.attach(image)
        
    try:
        with smtplib.SMTP_SSL("smtp.qq.com", 465) as server:
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        print("✅ QQ 邮箱推送成功")
    except Exception as e:
        print(f"❌ 邮件推送失败: {e}")

# ==========================================
# 3. 核心逻辑：数据与推理
# ==========================================

def get_real_data():
    print(">>> 正在从 AkShare 抓取数据...")
    df = ak.stock_zh_index_daily(symbol="sh000001")
    df = df.rename(columns={'date': 'timestamps'})
    df = df[['timestamps', 'open', 'high', 'low', 'close', 'volume']].copy()
    df['timestamps'] = df['timestamps'].astype(str)
    os.makedirs("./data", exist_ok=True)
    save_path = "./data/sse_daily_latest.csv"
    df.to_csv(save_path, index=False)
    return save_path

def run_prediction(data_path):
    device = "cpu" 
    print(f">>> 准备模型推理 (Device: {device})...")

    from model.kronos import Kronos, KronosTokenizer, KronosPredictor
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model = Kronos.from_pretrained("NeoQuasar/Kronos-small").to(device)
    predictor = KronosPredictor(model, tokenizer, device=device, max_context=512)

    df = pd.read_csv(data_path)
    df['timestamps'] = pd.to_datetime(df['timestamps'])

    x_df = df.tail(300).reset_index(drop=True)
    x_ts = pd.Series(x_df['timestamps'])
    y_ts = pd.Series(pd.date_range(start=x_ts.iloc[-1] + pd.Timedelta(days=1), periods=30, freq="D"))

    pred_df = predictor.predict(
        df=x_df[['open', 'high', 'low', 'close', 'volume']], 
        x_timestamp=x_ts, y_timestamp=y_ts, pred_len=30
    )

    # 绘图
    plt.figure(figsize=(12, 6))
    hist_show = 100
    plt.plot(x_df['timestamps'].iloc[-hist_show:], x_df['close'].iloc[-hist_show:], label='History', color='#1f77b4')
    plt.plot(y_ts, pred_df['close'], label='AI Predict', color='#d62728', linestyle='--')
    plt.title(f"SSE Index Forecast (Base: {df['timestamps'].iloc[-1].date()})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    img_path = "sse_daily_forecast.png"
    plt.savefig(img_path)
    plt.close()

    # 数据准备
    change = ((pred_df['close'].iloc[-1] / x_df['close'].iloc[-1]) - 1) * 100
    last_date = pd.to_datetime(df['timestamps'].iloc[-1]).date()
    summary_text = (f"📅 截止日期：{last_date}\n"
                    f"🔮 预测周期：未来 30 天\n"
                    f"📈 预估末端涨跌幅：{change:+.2f}%")

    # 执行推送
    print(">>> 启动推送流程...")
    # 1. Telegram
    tg_res = send_to_telegram(img_path, f"📊 *Kronos 上证预测*\n{summary_text}")
    if tg_res.get("ok"): print("✅ Telegram 推送成功")
    
    # 2. QQ 邮箱
    send_to_email(img_path, summary_text)

if __name__ == "__main__":
    try:
        path = get_real_data()
        run_prediction(path)
    except Exception as e:
        print(f"💥 程序崩溃: {str(e)}")
