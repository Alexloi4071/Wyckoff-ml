#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils.py

專案工具函式：
 - 參數檢查（validate_bounds）
 - 資料載入與基本清洗（load_ohlcv）
 - 技術指標計算（ATR、RSI、rolling 百分位）
全中文註解 + 每個可調參數的範圍提示
"""

import pandas as pd
import numpy as np

# =========================================================
# 1. 百分位滾動計算
# =========================================================
def pct_rank_rolling(series: pd.Series, window: int = 60) -> pd.Series:
    """
    計算當期數值在過去 window 期內的百分位
    window: 視窗長度，範圍 5~5000，預設 60
    """
    return series.rolling(window, min_periods=window).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

# =========================================================
# 2. 平均真實波幅（ATR）
# =========================================================
def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    計算 ATR
    window: 範圍 2~200，預設 14
    """
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window, min_periods=window).mean()

# =========================================================
# 3. RSI
# =========================================================
def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """
    計算 RSI
    window: 範圍 2~200，預設 14
    """
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window, min_periods=window).mean()
    avg_loss = loss.rolling(window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# =========================================================
# 4. 資料載入與基本處理
# =========================================================
def load_ohlcv(cfg: dict) -> pd.DataFrame:
    """
    從 CSV/Parquet 載入 OHLCV，並依配置進行時間轉換與重採樣
    """
    g = cfg["general"]
    
    # 根據文件擴展名選擇讀取方式
    input_file = g["input_csv"]
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
    else:
        df = pd.read_csv(input_file)

    # 重新命名欄位
    df = df.rename(columns={
        g["time_col"]: "ts",
        g["open_col"]: "open",
        g["high_col"]: "high",
        g["low_col"]: "low",
        g["close_col"]: "close",
        g["volume_col"]: "volume"
    })

    # 設置時間索引
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts").set_index("ts")

    # 時區設定
    tz = cfg["preprocess"].get("tz_localize")
    if tz:
        df.index = df.index.tz_localize(tz)

    # 重採樣
    freq = cfg["preprocess"].get("resample")
    if freq:
        # 轉換時間框架格式：15m -> 15min, 1H -> 1h, 4H -> 4h, 1D -> 1D
        freq_map = {
            '15m': '15min',
            '1H': '1h',
            '4H': '4h',
            '1D': '1D'
        }
        pandas_freq = freq_map.get(freq, freq)
        
        o = df["open"].resample(pandas_freq).first()
        h = df["high"].resample(pandas_freq).max()
        l = df["low"].resample(pandas_freq).min()
        c = df["close"].resample(pandas_freq).last()
        v = df["volume"].resample(pandas_freq).sum()
        df = pd.concat([o, h, l, c, v], axis=1)
        df.columns = ["open", "high", "low", "close", "volume"]

    # 是否取對數價
    if cfg["preprocess"].get("log_price", False):
        df["close"] = np.log(df["close"])

    return df

# =========================================================
# 5. 參數範圍檢查
# =========================================================
def validate_bounds(cfg: dict):
    """
    驗證配置檔中的主要可調參數是否在允許範圍內
    發現不合法會直接 raise ValueError
    """
    # --- General ---
    seed = int(cfg["general"].get("seed", 42))
    if not (1 <= seed <= 1e9):
        raise ValueError(f"general.seed 超出範圍 (1~1e9): {seed}")

    # --- Features ---
    f = cfg["features"]
    
    # returns
    if f["returns"]["enabled"]:
        for w in f["returns"]["windows"]:
            if not (2 <= w <= 1000):
                raise ValueError(f"features.returns.windows 超出範圍 (2~1000): {w}")
        if f["returns"]["method"] not in ["log", "simple"]:
            raise ValueError(f"features.returns.method 無效: {f['returns']['method']}")

    # zscore
    if f["zscore"]["enabled"]:
        w = f["zscore"]["window"]
        if not (5 <= w <= 500):
            raise ValueError(f"features.zscore.window 超出範圍 (5~500): {w}")

    # rsi
    if f["rsi"]["enabled"]:
        w = f["rsi"]["window"]
        if not (2 <= w <= 200):
            raise ValueError(f"features.rsi.window 超出範圍 (2~200): {w}")

    # atr
    if f["atr"]["enabled"]:
        w = f["atr"]["window"]
        if not (2 <= w <= 200):
            raise ValueError(f"features.atr.window 超出範圍 (2~200): {w}")
        if f["atr"]["normalize"] not in ["close", "none"]:
            raise ValueError(f"features.atr.normalize 無效: {f['atr']['normalize']}")

    # bbands
    if f["bbands"]["enabled"]:
        w = f["bbands"]["window"]
        if not (5 <= w <= 500):
            raise ValueError(f"features.bbands.window 超出範圍 (5~500): {w}")
        k = f["bbands"]["k"]
        if not (0.5 <= k <= 4.0):
            raise ValueError(f"features.bbands.k 超出範圍 (0.5~4.0): {k}")

    # volume_percentile
    if f["volume_percentile"]["enabled"]:
        w = f["volume_percentile"]["window"]
        if not (5 <= w <= 2000):
            raise ValueError(f"features.volume_percentile.window 超出範圍 (5~2000): {w}")

    # volatility_percentile
    if f["volatility_percentile"]["enabled"]:
        w = f["volatility_percentile"]["window"]
        if not (5 <= w <= 2000):
            raise ValueError(f"features.volatility_percentile.window 超出範圍 (5~2000): {w}")
        if f["volatility_percentile"]["base"] not in ["atr", "absret"]:
            raise ValueError(f"features.volatility_percentile.base 無效: {f['volatility_percentile']['base']}")

    # range_contraction
    if f["range_contraction"]["enabled"]:
        w = f["range_contraction"]["window"]
        if not (5 <= w <= 500):
            raise ValueError(f"features.range_contraction.window 超出範圍 (5~500): {w}")

    # wick_ratios
    if f["wick_ratios"]["enabled"]:
        clip = f["wick_ratios"]["clip"]
        if not (0.5 <= clip <= 20.0):
            raise ValueError(f"features.wick_ratios.clip 超出範圍 (0.5~20.0): {clip}")

    # obv_delta
    if f["obv_delta"]["enabled"]:
        w = f["obv_delta"]["window"]
        if not (2 <= w <= 1000):
            raise ValueError(f"features.obv_delta.window 超出範圍 (2~1000): {w}")

    # --- Labels ---
    l = cfg["labels"]
    
    # future_return_binary
    if l["future_return_binary"]["enabled"]:
        if not (1 <= l["future_return_binary"]["pctile"] <= 99):
            raise ValueError("labels.future_return_binary.pctile 超出範圍 (1~99)")
        if not (50 <= l["future_return_binary"]["ref_window"] <= 5000):
            raise ValueError("labels.future_return_binary.ref_window 超出範圍 (50~5000)")

    # breakout_confirmation
    if l["breakout_confirmation"]["enabled"]:
        if not (5 <= l["breakout_confirmation"]["lookback"] <= 2000):
            raise ValueError("labels.breakout_confirmation.lookback 超出範圍 (5~2000)")
        if not (1 <= l["breakout_confirmation"]["confirm_days"] <= 20):
            raise ValueError("labels.breakout_confirmation.confirm_days 超出範圍 (1~20)")

    # max_drawdown_within_horizon
    if l["max_drawdown_within_horizon"]["enabled"]:
        if not (1 <= l["max_drawdown_within_horizon"]["horizon"] <= 2000):
            raise ValueError("labels.max_drawdown_within_horizon.horizon 超出範圍 (1~2000)")

    # stop_target_hit_first
    if l["stop_target_hit_first"]["enabled"]:
        if not (1 <= l["stop_target_hit_first"]["horizon"] <= 2000):
            raise ValueError("labels.stop_target_hit_first.horizon 超出範圍 (1~2000)")

    # regime_label
    if l["regime_label"]["enabled"]:
        if not (2 <= l["regime_label"]["atr_window"] <= 200):
            raise ValueError("labels.regime_label.atr_window 超出範圍 (2~200)")

    print("[CHECK COMPLETE] All parameters are within valid ranges.")
