#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
download_data.py

從 Binance API 下載 OHLCV 數據並保存為 CSV
功能：
  - 支援分批下載（Binance API 限制每次最多 1000 根 K 線）
  - 自動處理時間範圍分割
  - 合併數據並去重
  - 輸出標準格式 CSV

使用方式：
  python download_data.py --symbol BTCUSDT --interval 1d --start 2020-01-01
"""

import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import os
import sys

# =========================================================
# Binance API 配置
# =========================================================
API_KEY = "F7KYhuoYDaEEBuBU9guF6OUn0C3ibxtKDCVwadXh3GyGpsDcZcpATpgDI5Sk6jBT"
API_SECRET = "qqHYDDpj5VZGcpKEaeYHO7czLzfEFI5LdSLPiuC4VOlxKmJDBeUmsvOPenJS8U6q"
BASE_URL = "https://api.binance.com"

# API 限制：每次最多 1000 根 K 線
MAX_KLINES_PER_REQUEST = 1000

# =========================================================
# 時間轉換工具
# =========================================================
def datetime_to_timestamp(dt_str: str) -> int:
    """將日期字符串轉換為毫秒時間戳"""
    dt = datetime.strptime(dt_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

def timestamp_to_datetime(timestamp: int) -> str:
    """將毫秒時間戳轉換為日期字符串"""
    dt = datetime.fromtimestamp(timestamp / 1000)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

# =========================================================
# 計算時間間隔
# =========================================================
def get_interval_ms(interval: str) -> int:
    """根據間隔字符串計算毫秒數"""
    interval_map = {
        "1m": 60 * 1000,
        "3m": 3 * 60 * 1000,
        "5m": 5 * 60 * 1000,
        "15m": 15 * 60 * 1000,
        "30m": 30 * 60 * 1000,
        "1h": 60 * 60 * 1000,
        "2h": 2 * 60 * 60 * 1000,
        "4h": 4 * 60 * 60 * 1000,
        "6h": 6 * 60 * 60 * 1000,
        "8h": 8 * 60 * 60 * 1000,
        "12h": 12 * 60 * 60 * 1000,
        "1d": 24 * 60 * 60 * 1000,
        "3d": 3 * 24 * 60 * 60 * 1000,
        "1w": 7 * 24 * 60 * 60 * 1000,
        "1M": 30 * 24 * 60 * 60 * 1000,
    }
    return interval_map.get(interval, 24 * 60 * 60 * 1000)  # 預設 1 天

# =========================================================
# 單次 API 請求
# =========================================================
def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int = None, limit: int = 1000):
    """
    從 Binance API 獲取 K 線數據
    """
    url = f"{BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time,
        "limit": limit
    }
    
    if end_time:
        params["endTime"] = end_time
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[錯誤] API 請求失敗: {e}")
        return None

# =========================================================
# 分批下載主函數
# =========================================================
def download_historical_data(symbol: str, interval: str, start_date: str, end_date: str = None):
    """
    分批下載歷史數據
    """
    print(f"[開始] 下載 {symbol} {interval} 數據，起始日期: {start_date}")
    
    # 轉換時間
    start_timestamp = datetime_to_timestamp(start_date)
    if end_date:
        end_timestamp = datetime_to_timestamp(end_date)
    else:
        end_timestamp = int(datetime.now().timestamp() * 1000)
    
    # 計算間隔毫秒數
    interval_ms = get_interval_ms(interval)
    
    # 計算每批次的時間範圍（1000 根 K 線）
    batch_duration = interval_ms * MAX_KLINES_PER_REQUEST
    
    all_data = []
    current_start = start_timestamp
    batch_count = 0
    
    while current_start < end_timestamp:
        batch_count += 1
        current_end = min(current_start + batch_duration, end_timestamp)
        
        print(f"[批次 {batch_count}] 下載 {timestamp_to_datetime(current_start)} 到 {timestamp_to_datetime(current_end)}")
        
        # 發送 API 請求
        klines = fetch_klines(symbol, interval, current_start, current_end, MAX_KLINES_PER_REQUEST)
        
        if klines is None:
            print(f"[警告] 批次 {batch_count} 下載失敗，跳過")
            current_start = current_end
            continue
        
        if len(klines) == 0:
            print(f"[完成] 沒有更多數據")
            break
        
        # 添加到總數據
        all_data.extend(klines)
        print(f"[批次 {batch_count}] 獲得 {len(klines)} 根 K 線")
        
        # 更新下一批次的開始時間
        last_timestamp = klines[-1][0]  # 最後一根 K 線的時間戳
        current_start = last_timestamp + interval_ms
        
        # API 限制：避免請求過於頻繁
        time.sleep(0.1)
    
    print(f"[完成] 總共下載 {len(all_data)} 根 K 線")
    return all_data

# =========================================================
# 數據處理與保存
# =========================================================
def process_and_save_data(klines_data: list, symbol: str, output_path: str = "data/ohlcv.parquet"):
    """
    處理 K 線數據並保存為 Parquet 或 CSV
    """
    if not klines_data:
        print("[錯誤] 沒有數據可處理")
        return
    
    # 轉換為 DataFrame
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    df = pd.DataFrame(klines_data, columns=columns)
    
    # 選擇需要的欄位並轉換數據類型
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    
    # 重命名為 datetime（統一格式）
    df = df.rename(columns={"timestamp": "datetime"})
    
    # 轉換時間戳為日期時間
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms")
    
    # 轉換價格和成交量為數值
    price_cols = ["open", "high", "low", "close"]
    df[price_cols] = df[price_cols].astype(float)
    df["volume"] = df["volume"].astype(float)
    
    # 去重並排序
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    
    # 創建輸出目錄
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 根據文件擴展名保存
    if output_path.endswith('.parquet'):
        df.to_parquet(output_path, index=False)
        print(f"[完成] 數據已保存至 {output_path} (Parquet格式)")
    else:
        df.to_csv(output_path, index=False)
        print(f"[完成] 數據已保存至 {output_path} (CSV格式)")
    
    print(f"[統計] 共 {len(df)} 筆記錄，時間範圍: {df['datetime'].min()} 到 {df['datetime'].max()}")
    
    return df

# =========================================================
# 互動式輸入函數
# =========================================================
def get_user_input():
    """
    互動式獲取用戶輸入參數
    """
    print("=== Binance 數據下載工具 ===")
    print()
    
    # 交易對輸入
    print("支援的交易對範例：BTCUSDT, ETHUSDT, BNBUSDT, ADAUSDT, SOLUSDT")
    symbol = input("請輸入交易對符號 [預設: BTCUSDT]: ").strip().upper()
    if not symbol:
        symbol = "BTCUSDT"
    
    # K線間隔輸入
    print("\n支援的K線間隔：")
    print("分鐘線: 1m, 3m, 5m, 15m, 30m")
    print("小時線: 1h, 2h, 4h, 6h, 8h, 12h")
    print("日線: 1d, 3d")
    print("週線: 1w")
    print("月線: 1M")
    interval = input("請輸入K線間隔 [預設: 1d]: ").strip().lower()
    if not interval:
        interval = "1d"
    
    # 開始日期輸入
    print("\n日期格式：YYYY-MM-DD (例如: 2020-01-01)")
    start_date = input("請輸入開始日期 [預設: 2020-01-01]: ").strip()
    if not start_date:
        start_date = "2020-01-01"
    
    # 結束日期輸入
    end_date = input("請輸入結束日期 [預設: 當前日期，直接按Enter]: ").strip()
    if not end_date:
        end_date = None
    
    # 輸出路徑輸入
    output_path = input("請輸入輸出路徑 [預設: data/ohlcv.parquet]: ").strip()
    if not output_path:
        output_path = "data/ohlcv.parquet"
    
    # 確認輸入
    print("\n=== 確認下載參數 ===")
    print(f"交易對: {symbol}")
    print(f"K線間隔: {interval}")
    print(f"開始日期: {start_date}")
    print(f"結束日期: {end_date if end_date else '當前日期'}")
    print(f"輸出路徑: {output_path}")
    
    confirm = input("\n確認開始下載？ [y/N]: ").strip().lower()
    if confirm not in ['y', 'yes', '是']:
        print("已取消下載")
        return None
    
    return {
        'symbol': symbol,
        'interval': interval,
        'start': start_date,
        'end': end_date,
        'output': output_path
    }

# =========================================================
# 主程式入口
# =========================================================
def main():
    parser = argparse.ArgumentParser(description="從 Binance 下載歷史 K 線數據")
    parser.add_argument("--interactive", "-i", action="store_true", help="使用互動式輸入模式")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="交易對符號，預設 BTCUSDT")
    parser.add_argument("--interval", type=str, default="1d", help="K 線間隔，預設 1d")
    parser.add_argument("--start", type=str, default="2020-01-01", help="開始日期 YYYY-MM-DD，預設 2020-01-01")
    parser.add_argument("--end", type=str, default=None, help="結束日期 YYYY-MM-DD，預設為當前日期")
    parser.add_argument("--output", type=str, default="data/ohlcv.parquet", help="輸出路徑，預設 data/ohlcv.parquet (支持 .parquet 或 .csv)")
    
    args = parser.parse_args()
    
    # 如果使用互動模式或沒有提供任何參數，則進入互動式輸入
    if args.interactive or len(os.sys.argv) == 1:
        params = get_user_input()
        if params is None:
            return
        symbol = params['symbol']
        interval = params['interval']
        start_date = params['start']
        end_date = params['end']
        output_path = params['output']
    else:
        # 使用命令行參數
        symbol = args.symbol
        interval = args.interval
        start_date = args.start
        end_date = args.end
        output_path = args.output
    
    # 下載數據
    klines_data = download_historical_data(symbol, interval, start_date, end_date)
    
    # 處理並保存
    if klines_data:
        process_and_save_data(klines_data, symbol, output_path)
    else:
        print("[錯誤] 下載失敗，沒有數據")

if __name__ == "__main__":
    main()