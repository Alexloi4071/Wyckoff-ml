#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_data_incremental.py

實時增量更新 Parquet 數據文件
- 自動檢測最後更新時間
- 只下載新數據
- 直接更新同一個文件
- 支持自動備份
"""

import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# Binance API 配置
BASE_URL = "https://api.binance.com"
MAX_KLINES_PER_REQUEST = 1000

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
    }
    return interval_map.get(interval, 60 * 1000)

def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int = None, limit: int = 1000):
    """從 Binance API 獲取 K 線數據"""
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

def download_new_data(symbol: str, interval: str, start_timestamp: int, end_timestamp: int):
    """下載新數據"""
    interval_ms = get_interval_ms(interval)
    batch_duration = interval_ms * MAX_KLINES_PER_REQUEST
    
    all_data = []
    current_start = start_timestamp
    batch_count = 0
    
    while current_start < end_timestamp:
        batch_count += 1
        current_end = min(current_start + batch_duration, end_timestamp)
        
        klines = fetch_klines(symbol, interval, current_start, current_end, MAX_KLINES_PER_REQUEST)
        
        if klines is None or len(klines) == 0:
            break
        
        all_data.extend(klines)
        
        last_timestamp = klines[-1][0]
        current_start = last_timestamp + interval_ms
        
        time.sleep(0.1)  # API 限制
    
    return all_data

def update_parquet_file(file_path: str, symbol: str = "BTCUSDT", interval: str = "1m", 
                        backup: bool = True, verbose: bool = True):
    """
    實時更新 Parquet 文件
    
    Parameters:
    -----------
    file_path : str
        Parquet 文件路徑
    symbol : str
        交易對符號
    interval : str
        K線間隔
    backup : bool
        是否備份舊文件
    verbose : bool
        是否顯示詳細信息
    """
    file_path = Path(file_path)
    
    if verbose:
        print("="*70)
        print("實時數據更新工具")
        print("="*70)
    
    # 檢查文件是否存在
    if not file_path.exists():
        print(f"[錯誤] 文件不存在: {file_path}")
        print(f"[提示] 請先使用 download_data.py 下載初始數據")
        return False
    
    # 讀取現有數據
    if verbose:
        print(f"\n[1/5] 讀取現有數據: {file_path}")
    
    df_old = pd.read_parquet(file_path)
    df_old['datetime'] = pd.to_datetime(df_old['datetime'])
    
    last_time = df_old['datetime'].max()
    data_count_old = len(df_old)
    
    if verbose:
        print(f"  現有數據: {data_count_old:,} 筆")
        print(f"  最後時間: {last_time}")
    
    # 計算需要更新的時間範圍
    # 從最後時間的前一天開始（避免遺漏）
    start_time = last_time - timedelta(days=1)
    end_time = datetime.now()
    
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    if verbose:
        print(f"\n[2/5] 下載新數據")
        print(f"  時間範圍: {start_time} 到 {end_time}")
    
    # 下載新數據
    klines_data = download_new_data(symbol, interval, start_timestamp, end_timestamp)
    
    if not klines_data:
        print(f"[完成] 沒有新數據需要更新")
        return True
    
    if verbose:
        print(f"  下載完成: {len(klines_data)} 筆")
    
    # 處理新數據
    if verbose:
        print(f"\n[3/5] 處理新數據")
    
    columns = [
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ]
    
    df_new = pd.DataFrame(klines_data, columns=columns)
    df_new = df_new[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df_new = df_new.rename(columns={"timestamp": "datetime"})
    df_new["datetime"] = pd.to_datetime(df_new["datetime"], unit="ms")
    
    price_cols = ["open", "high", "low", "close"]
    df_new[price_cols] = df_new[price_cols].astype(float)
    df_new["volume"] = df_new["volume"].astype(float)
    
    # 合併數據
    if verbose:
        print(f"\n[4/5] 合併數據")
    
    df_combined = pd.concat([df_old, df_new], ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=["datetime"]).sort_values("datetime")
    df_combined = df_combined.reset_index(drop=True)
    
    new_records = len(df_combined) - data_count_old
    
    if verbose:
        print(f"  合併前: {data_count_old:,} 筆")
        print(f"  合併後: {len(df_combined):,} 筆")
        print(f"  新增: {new_records:,} 筆")
    
    # 保存數據
    if verbose:
        print(f"\n[5/5] 保存數據")
    
    # 備份
    if backup and new_records > 0:
        backup_path = file_path.parent / f"{file_path.stem}_backup{file_path.suffix}"
        # 如果備份文件已存在，先刪除
        if backup_path.exists():
            backup_path.unlink()
        file_path.rename(backup_path)
        if verbose:
            print(f"  ✓ 已備份到: {backup_path}")
    
    # 保存更新後的數據
    df_combined.to_parquet(file_path, index=False)
    
    if verbose:
        print(f"  ✓ 已保存到: {file_path}")
        print(f"\n" + "="*70)
        print("更新統計")
        print("="*70)
        print(f"原始數據: {data_count_old:,} 筆")
        print(f"新增數據: {new_records:,} 筆")
        print(f"最新時間: {df_combined['datetime'].max()}")
        print(f"最新價格: ${df_combined['close'].iloc[-1]:,.2f}")
        print("="*70)
        
        if new_records > 0:
            print(f"\n✅ 數據更新完成！")
        else:
            print(f"\n✅ 數據已是最新！")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="實時增量更新 Parquet 數據文件")
    parser.add_argument("--file", type=str, default="data/btcusdt_1m.parquet", 
                       help="Parquet 文件路徑，預設 data/btcusdt_1m.parquet")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", 
                       help="交易對符號，預設 BTCUSDT")
    parser.add_argument("--interval", type=str, default="1m", 
                       help="K線間隔，預設 1m")
    parser.add_argument("--no-backup", action="store_true", 
                       help="不備份舊文件")
    parser.add_argument("--quiet", "-q", action="store_true", 
                       help="靜默模式，只顯示錯誤")
    
    args = parser.parse_args()
    
    success = update_parquet_file(
        file_path=args.file,
        symbol=args.symbol,
        interval=args.interval,
        backup=not args.no_backup,
        verbose=not args.quiet
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
