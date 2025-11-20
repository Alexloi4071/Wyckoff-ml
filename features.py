import numpy as np
import pandas as pd
from utils import pct_rank_rolling, atr, rsi  # 引入工具函數

def make_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    根據 YAML 配置生成特徵
    df: OHLCV 數據
    cfg: YAML 配置字典
    """
    fcfg = cfg["features"]
    feats = pd.DataFrame(index=df.index)

    # ====== 1. 收益率特徵 ======
    if fcfg["returns"]["enabled"]:
        method = fcfg["returns"].get("method", "log")  # 預設 "log"（可選 "log" 或 "simple"）
        for w in fcfg["returns"].get("windows", [20, 60]):  # 預設 [20, 60]，範圍 2~1000
            if method == "log":
                feats[f"f_ret_log_{w}"] = np.log(df["close"] / df["close"].shift(w))
            else:
                feats[f"f_ret_smp_{w}"] = df["close"].pct_change(w)

    # ====== 2. Z-score 特徵 ======
    if fcfg["zscore"]["enabled"]:
        w = fcfg["zscore"].get("window", 60)  # 預設 60，範圍 5~500
        ma = df["close"].rolling(w).mean()
        sd = df["close"].rolling(w).std()
        feats[f"f_z_{w}"] = (df["close"] - ma) / sd

    # ====== 3. RSI 特徵 ======
    if fcfg["rsi"]["enabled"]:
        w = fcfg["rsi"].get("window", 14)  # 預設 14，範圍 2~200
        feats[f"f_rsi_{w}"] = rsi(df["close"], w)

    # ====== 4. ATR 特徵 ======
    atr_series = None
    if fcfg["atr"]["enabled"]:
        w = fcfg["atr"].get("window", 14)  # 預設 14，範圍 2~200
        atr_series = atr(df, w)
        normalize = fcfg["atr"].get("normalize", "close")  # 預設 "close"（可選 "none" 或 "close"）
        if normalize == "close":
            feats[f"f_atr_{w}_p"] = atr_series / df["close"]
        else:
            feats[f"f_atr_{w}"] = atr_series

    # ====== 5. 布林帶特徵 ======
    if fcfg["bbands"]["enabled"]:
        w = fcfg["bbands"].get("window", 20)  # 預設 20，範圍 5~500
        k = fcfg["bbands"].get("k", 2.0)      # 預設 2.0，範圍 0.5~4.0
        ma = df["close"].rolling(w).mean()
        sd = df["close"].rolling(w).std()
        upper = ma + k * sd
        lower = ma - k * sd
        feats[f"f_bb_percB_{w}_{k}"] = (df["close"] - lower) / (upper - lower)
        feats[f"f_bb_width_{w}_{k}"] = (upper - lower) / ma

    # ====== 6. 成交量分位數 ======
    if fcfg["volume_percentile"]["enabled"]:
        w = fcfg["volume_percentile"].get("window", 60)  # 預設 60，範圍 5~2000
        feats[f"f_vol_pct_{w}"] = pct_rank_rolling(df["volume"], w)

    # ====== 7. 波動率分位數 ======
    if fcfg["volatility_percentile"]["enabled"]:
        w = fcfg["volatility_percentile"].get("window", 60)  # 預設 60，範圍 5~2000
        base = fcfg["volatility_percentile"].get("base", "atr")  # 預設 "atr"（可選 "atr" 或 "absret"）
        if base == "atr":
            vol_series = atr_series if atr_series is not None else atr(df, fcfg["atr"].get("window", 14))
        else:
            vol_series = df["close"].pct_change().abs()
        feats[f"f_vola_pct_{base}_{w}"] = pct_rank_rolling(vol_series, w)

    # ====== 8. 區間收縮特徵 ======
    if fcfg["range_contraction"]["enabled"]:
        w = fcfg["range_contraction"].get("window", 20)  # 預設 20，範圍 5~500
        rolling_rng = (df["high"].rolling(w).max() - df["low"].rolling(w).min())
        atr_w = atr_series if atr_series is not None else atr(df, fcfg["atr"].get("window", 14))
        feats[f"f_rng_ctr_{w}"] = rolling_rng / (atr_w * np.sqrt(w))

    # ====== 9. 上下影線比例 ======
    if fcfg["wick_ratios"]["enabled"]:
        clip_val = fcfg["wick_ratios"].get("clip", 5.0)  # 預設 5.0，範圍 0.5~20.0
        rng = (df["high"] - df["low"]).replace(0, np.nan)
        upper_wick = (df["high"] - df[["open", "close"]].max(axis=1)) / rng
        lower_wick = (df[["open", "close"]].min(axis=1) - df["low"]) / rng
        feats["f_wick_up"] = upper_wick.clip(-clip_val, clip_val)
        feats["f_wick_dn"] = lower_wick.clip(-clip_val, clip_val)

    # ====== 10. OBV 變化率 ======
    if fcfg["obv_delta"]["enabled"]:
        w = fcfg["obv_delta"].get("window", 20)  # 預設 20，範圍 2~1000
        direction = np.sign(df["close"].diff().fillna(0))
        obv = (direction * df["volume"]).cumsum()
        feats[f"f_obv_d_{w}"] = obv.diff(w) / (abs(obv).rolling(w).mean())

    return feats
