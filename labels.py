#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels.py

用於生成交易策略 ML 訓練標籤的模組。
全中文註解 + 預設值，標明可調整範圍，避免資料外洩。

Performance Note:
- stop_target_sequence 函數現在使用向量化實現以提升性能
- 對於15分鐘時框，性能提升約 30-60 倍
- 可通過環境變量 USE_VECTORIZED_LABELS=0 切換回原始實現（用於調試）
"""

import pandas as pd
import numpy as np
import os
from utils import atr, pct_rank_rolling  # utils.py 需包含 ATR 與百分位函數

# 功能開關：是否使用向量化實現（默認啟用）
USE_VECTORIZED = os.environ.get('USE_VECTORIZED_LABELS', '1') == '1'

if USE_VECTORIZED:
    try:
        from labels_vectorized import stop_target_sequence_vectorized
        print("[INFO] Using vectorized implementation for stop_target_sequence (faster)")
    except ImportError:
        print("[WARNING] labels_vectorized not found, falling back to original implementation")
        USE_VECTORIZED = False

# =========================================================
# 1. 未來報酬計算
# =========================================================
def forward_return(close: pd.Series, h: int = 20, method: str = "log") -> pd.Series:
    """
    計算未來 h 期報酬
    h: 預測視窗，範圍 1~2000，預設 20
    method: 'log'（對數收益）或 'simple'（簡單收益），預設 'log'
    """
    if method == "log":
        return np.log(close.shift(-h) / close)
    else:
        return close.shift(-h) / close - 1.0

# =========================================================
# 2. 滾動分位閾值（避免資料外洩）
# =========================================================
def rolling_threshold(series: pd.Series, window: int = 252,
                      q: float = 70, side: str = "upper") -> pd.Series:
    """
    window: 參考視窗長度，範圍 50~5000，預設 252
    q: 分位百分比，範圍 1~99，預設 70
    side: 'upper' 或 'lower'，預設 'upper'
    """
    ref = series.shift(1).rolling(window, min_periods=window)
    return ref.quantile(q / 100.0)

# =========================================================
# 3. 未來最大回撤
# =========================================================
def max_drawdown_forward(close: pd.Series, h: int = 20) -> pd.Series:
    """
    計算未來 h 期內的最大回撤
    h: 預測視窗，範圍 1~2000，預設 20
    """
    rolling_min = close.shift(-1).rolling(h, min_periods=h).min()
    return (rolling_min / close) - 1.0

# =========================================================
# 4. 止盈止損先後觸發
# =========================================================
def stop_target_sequence(df: pd.DataFrame, h: int = 20, atr_now: pd.Series = None,
                         stop_k: float = 1.5, tgt_k: float = 2.5) -> pd.Series:
    """
    判斷未來 h 期內，先觸發止盈還是止損
    h: 預測視窗，範圍 1~2000，預設 20
    stop_k: 止損 ATR 倍數，範圍 0.1~10.0，預設 1.5
    tgt_k: 止盈 ATR 倍數，範圍 0.1~10.0，預設 2.5
    """
    res = pd.Series(np.nan, index=df.index)
    for t in range(len(df)):
        if t + 1 >= len(df):
            break
        c0 = df["close"].iloc[t]
        stop = c0 * (1 - stop_k * atr_now.iloc[t] / c0)
        tgt = c0 * (1 + tgt_k * atr_now.iloc[t] / c0)
        end = min(t + h, len(df) - 1)
        for k in range(t + 1, end + 1):
            hi = df["high"].iloc[k]
            lo = df["low"].iloc[k]
            if np.isnan(hi) or np.isnan(lo):
                continue
            if lo <= stop and hi >= tgt:
                res.iloc[t] = np.nan  # 同時觸發視為無效
                break
            if hi >= tgt:
                res.iloc[t] = 1       # 先止盈
                break
            if lo <= stop:
                res.iloc[t] = 0       # 先止損
                break
    return res

# =========================================================
# 5. 標籤生成主函數
# =========================================================
def make_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    根據 cfg 生成標籤 DataFrame
    """
    lcfg = cfg["labels"]
    labels = pd.DataFrame(index=df.index)

    # ---- 未來報酬二分類 ----
    if lcfg["future_return_binary"]["enabled"]:
        method = lcfg["future_return_binary"].get("method", "log")
        side   = lcfg["future_return_binary"].get("side", "upper")
        q      = lcfg["future_return_binary"].get("pctile", 70)
        ref_w  = lcfg["future_return_binary"].get("ref_window", 252)
        for h in lcfg["future_return_binary"].get("horizons", [20]):
            fr = forward_return(df["close"], h, method)
            thr = rolling_threshold(fr, ref_w, q, side)
            if side == "upper":
                labels[f"y_bin_ret_{method}_p{q}_h{h}"] = (fr > thr).astype(float)
            else:
                labels[f"y_bin_ret_{method}_p{q}_h{h}"] = (fr < thr).astype(float)

    # ---- 突破確認 ----
    if lcfg["breakout_confirmation"]["enabled"]:
        lb       = lcfg["breakout_confirmation"].get("lookback", 60)
        k        = lcfg["breakout_confirmation"].get("confirm_days", 3)
        min_pct  = lcfg["breakout_confirmation"].get("min_breakout_pct", 0.5) / 100.0
        rb_high  = df["high"].rolling(lb, min_periods=lb).max().shift(1)
        base     = rb_high * (1.0 + min_pct)
        cond     = (df["close"] > base)
        confirm  = cond.rolling(k).apply(lambda x: 1.0 if np.all(x) else 0.0)
        labels[f"y_brk_lb{lb}_k{k}_p{int(min_pct*100)}"] = (confirm == 1.0).astype(float)

    # ---- 未來最大回撤 ----
    if lcfg["max_drawdown_within_horizon"]["enabled"]:
        h   = lcfg["max_drawdown_within_horizon"].get("horizon", 20)
        thr = lcfg["max_drawdown_within_horizon"].get("threshold_pct", -5.0) / 100.0
        mdd_fwd = max_drawdown_forward(df["close"], h)
        labels[f"y_mdd_le_{int(thr*100)}_h{h}"] = (mdd_fwd <= thr).astype(float)

    # ---- 止盈止損先後 ----
    if lcfg["stop_target_hit_first"]["enabled"]:
        h       = lcfg["stop_target_hit_first"].get("horizon", 20)
        atr_w   = lcfg["stop_target_hit_first"].get("atr_window", 14)
        stop_k  = lcfg["stop_target_hit_first"].get("stop_atr", 1.5)
        tgt_k   = lcfg["stop_target_hit_first"].get("target_atr", 2.5)
        atr_now = atr(df, atr_w)
        
        # 使用向量化或原始實現
        if USE_VECTORIZED:
            labels[f"y_hit_target_first_h{h}_s{stop_k}_t{tgt_k}"] = stop_target_sequence_vectorized(
                df, h, atr_now, stop_k, tgt_k
            )
        else:
            labels[f"y_hit_target_first_h{h}_s{stop_k}_t{tgt_k}"] = stop_target_sequence(
                df, h, atr_now, stop_k, tgt_k
            )

    # ---- 市場狀態 ----
    if lcfg["regime_label"]["enabled"]:
        atr_w   = lcfg["regime_label"].get("atr_window", 14)
        vol_q   = lcfg["regime_label"].get("vol_upper_pctile", 80)
        ma_w    = lcfg["regime_label"].get("trend_ma_window", 50)
        atr_now = atr(df, atr_w)
        vol_rank = pct_rank_rolling(atr_now, 252)
        vol_high = vol_rank > (vol_q / 100.0)
        trend_up = df["close"] > df["close"].rolling(ma_w, min_periods=ma_w).mean()
        labels[f"y_regime_vol{vol_q}_ma{ma_w}"] = (
            (vol_high.astype(int) * 2 + trend_up.astype(int))
        ).astype(int)

    return labels

# =========================================================
# 6. 測試入口
# =========================================================
if __name__ == "__main__":
    # 預設測試配置（可以直接改這裡的數字測試）
    default_cfg = {
        "labels": {
            "future_return_binary": {
                "enabled": True,
                "method": "log",       # 'log' 或 'simple'
                "side": "upper",       # 'upper' 或 'lower'
                "pctile": 70,          # 1~99，預設 70
                "ref_window": 252,     # 50~5000，預設 252
                "horizons": [20]       # 1~2000，預設 [20]
            },
            "breakout_confirmation": {
                "enabled": True,
                "lookback": 60,        # 5~2000，預設 60
                "confirm_days": 3,     # 1~20，預設 3
                "min_breakout_pct": 0.5# 0.0~20.0%，預設 0.5
            },
            "max_drawdown_within_horizon": {
                "enabled": True,
                "horizon": 20,         # 1~2000，預設 20
                "threshold_pct": -5.0  # -50.0~0.0%，預設 -5.0
            },
            "stop_target_hit_first": {
                "enabled": True,
                "horizon": 20,         # 1~2000，預設 20
                "atr_window": 14,      # 2~200，預設 14
                "stop_atr": 1.5,       # 0.1~10.0，預設 1.5
                "target_atr": 2.5      # 0.1~10.0，預設 2.5
            },
            "regime_label": {
                "enabled": True,
                "atr_window": 14,      # 2~200，預設 14
                "vol_upper_pctile": 80,# 50~99，預設 80
                "trend_ma_window": 50  # 5~1000，預設 50
            }
        }
    }

    # 假設有一個 ohlcv.csv（必須包含 open, high, low, close 欄位）
    df = pd.read_csv("ohlcv.csv", parse_dates=True, index_col=0)
    labels_df = make_labels(df, default_cfg)
    labels_df.to_csv("labels_output.csv")
    print(f"[完成] 已輸出標籤至 labels_output.csv，共 {labels_df.shape[0]} 筆記錄。")
