#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py
主程式：整合 features.py + labels.py + utils.py
功能：
  - 載入 OHLCV 數據
  - 根據配置生成特徵與標籤
  - 輸出 CSV
"""

import pandas as pd
import yaml
from pathlib import Path

# 匯入自定義模組
from utils import validate_bounds, load_ohlcv
from features import make_features
from labels import make_labels

# =========================================================
# 1. 預設全域配置（不用 YAML 也能直接跑）
# =========================================================
default_cfg = {
    "general": {
        "seed": 42,                       # 隨機種子 (1~1e9)，預設 42
        "input_csv": "data/btcusdt_1m.csv", # 輸入檔案路徑（1分鐘數據）
        "time_col": "timestamp",          # 時間欄位名稱
        "open_col": "open",               # 開盤價欄位
        "high_col": "high",               # 最高價欄位
        "low_col": "low",                 # 最低價欄位
        "close_col": "close",             # 收盤價欄位
        "volume_col": "volume",           # 成交量欄位
        "start_date": None,               # 起始日期 (YYYY-MM-DD 或 None)
        "end_date": None,                 # 結束日期
        "dropna": True                    # 是否移除 NaN 資料
    },
    "preprocess": {
        "tz_localize": "UTC",             # 時區名稱 (例如 "UTC")
        "resample": ["1H", "4H", "1D"],   # 多時間框架列表，從1分鐘數據重採樣
        "log_price": False                # 是否對價格取對數
    },
    "features": {                         # 特徵開關與參數
        "returns": {"enabled": True, "windows": [20, 60], "method": "log"},
        "zscore": {"enabled": True, "window": 60},
        "rsi": {"enabled": True, "window": 14},
        "atr": {"enabled": True, "window": 14, "normalize": "close"},
        "bbands": {"enabled": True, "window": 20, "k": 2.0},
        "volume_percentile": {"enabled": True, "window": 60},
        "volatility_percentile": {"enabled": True, "window": 60, "base": "atr"},
        "range_contraction": {"enabled": False, "window": 20},
        "wick_ratios": {"enabled": False, "clip": 5.0},
        "obv_delta": {"enabled": False, "window": 20}
    },
    "labels": {                            # 標籤開關與參數
        "future_return_binary": {
            "enabled": True, "method": "log", "side": "upper",
            "pctile": 70, "ref_window": 252, "horizons": [20]
        },
        "breakout_confirmation": {
            "enabled": True, "lookback": 60, "confirm_days": 3, "min_breakout_pct": 0.5
        },
        "max_drawdown_within_horizon": {
            "enabled": True, "horizon": 20, "threshold_pct": -5.0
        },
        "stop_target_hit_first": {
            "enabled": True, "horizon": 20, "atr_window": 14,
            "stop_atr": 1.5, "target_atr": 2.5
        },
        "regime_label": {
            "enabled": True, "atr_window": 14, "vol_upper_pctile": 80, "trend_ma_window": 50
        }
    },
    "output": {
        "base_dir": "out",                    # 基礎輸出目錄
        "base_name": "btcusdt",               # 基礎文件名
        "features_csv": "out/features.csv",   # 後向兼容（單一時間框架時使用）
        "labels_csv": "out/labels.csv"        # 後向兼容（單一時間框架時使用）
    }
}

# =========================================================
# 2. 主流程
# =========================================================
def run_pipeline(cfg: dict, resample_freq: str = None):
    """
    執行特徵+標籤生成流程
    cfg: 設定字典
    resample_freq: 可選的重採樣頻率，覆蓋 cfg 中的設定
    """
    # 如果提供了 resample_freq，則覆蓋 cfg
    if resample_freq:
        cfg["preprocess"]["resample"] = resample_freq

    # 驗證參數範圍
    validate_bounds(cfg)

    # 載入資料
    df = load_ohlcv(cfg)

    # 篩選日期
    sd = cfg["general"].get("start_date")
    ed = cfg["general"].get("end_date")
    if sd:
        df = df.loc[df.index >= pd.to_datetime(sd)]
    if ed:
        df = df.loc[df.index <= pd.to_datetime(ed)]

    # 生成特徵
    feats = make_features(df, cfg)

    # 生成標籤
    labels = make_labels(df, cfg)

    # 去除 NaN 並對齊
    if cfg["general"].get("dropna", True):
        valid_idx = feats.dropna().index.intersection(labels.dropna().index)
        feats = feats.loc[valid_idx]
        labels = labels.loc[valid_idx]

    # 調整輸出路徑以包含時間框架目錄結構
    if resample_freq:
        # 創建時間框架專用目錄
        timeframe_dir = Path(cfg["output"]["base_dir"]) / resample_freq
        timeframe_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成帶時間框架的文件名
        base_name = cfg["output"]["base_name"]
        feats_csv = timeframe_dir / f"{base_name}_features_{resample_freq}.csv"
        labels_csv = timeframe_dir / f"{base_name}_labels_{resample_freq}.csv"
    else:
        # 使用原始路徑
        feats_csv = cfg["output"]["features_csv"]
        labels_csv = cfg["output"]["labels_csv"]
        Path(feats_csv).parent.mkdir(parents=True, exist_ok=True)

    # 輸出結果（支持CSV和Parquet）
    # 根據配置文件中的擴展名決定輸出格式
    if str(feats_csv).endswith('.parquet'):
        feats.to_parquet(feats_csv, index=True)
        labels.to_parquet(labels_csv, index=True)
    else:
        feats.to_csv(feats_csv, index_label="ts")
        labels.to_csv(labels_csv, index_label="ts")
    
    print(f"[DONE] Features saved to: {feats_csv}")
    print(f"[DONE] Labels saved to: {labels_csv}")

# =========================================================
# 3. 執行入口
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quant ML Feature+Label Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        help="YAML 配置檔路徑，留空則使用內建 default_cfg"
    )
    args = parser.parse_args()

    # 如果指定了 YAML，就讀 YAML；否則用 default_cfg
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = default_cfg  # 使用內建預設值

    # 檢查是否有多個 resample 頻率
    resample_list = cfg["preprocess"].get("resample", None)
    if isinstance(resample_list, list):
        for freq in resample_list:
            print(f"[PROCESSING] Resample frequency: {freq}")
            run_pipeline(cfg.copy(), freq)  # 使用 cfg 複本避免修改
    else:
        run_pipeline(cfg)
