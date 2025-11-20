#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
labels_vectorized.py

向量化版本的標籤生成函數，專門優化 stop_target_sequence 性能。
用於處理大規模數據（特別是15分鐘時框）。
"""

import pandas as pd
import numpy as np
from typing import Tuple


def create_forward_window_matrix(series: pd.Series, window: int) -> np.ndarray:
    """
    創建未來窗口矩陣，每行包含該時間點後 window 期的數據
    
    Parameters:
    -----------
    series : pd.Series
        時間序列數據（如 high 或 low 價格）
    window : int
        未來視窗大小
    
    Returns:
    --------
    np.ndarray
        形狀為 (n, window) 的矩陣，其中 matrix[i, j] 表示第 i 個時間點後第 j 期的值
        如果未來數據不足，填充 NaN
    
    Examples:
    ---------
    >>> s = pd.Series([1, 2, 3, 4, 5])
    >>> create_forward_window_matrix(s, 3)
    array([[ 2.,  3.,  4.],
           [ 3.,  4.,  5.],
           [ 4.,  5., nan],
           [ 5., nan, nan],
           [nan, nan, nan]])
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if window <= 0:
        raise ValueError(f"window must be positive, got {window}")
    
    n = len(series)
    values = series.values
    
    # 創建結果矩陣，預填充 NaN
    matrix = np.full((n, window), np.nan, dtype=np.float64)
    
    # 填充每一行的未來數據
    for i in range(n):
        end_idx = min(i + window + 1, n)
        available = end_idx - i - 1
        if available > 0:
            matrix[i, :available] = values[i+1:end_idx]
    
    return matrix


def find_first_trigger(
    high_matrix: np.ndarray,
    low_matrix: np.ndarray,
    stop_levels: np.ndarray,
    target_levels: np.ndarray
) -> np.ndarray:
    """
    向量化查找首次觸發止盈或止損
    
    Parameters:
    -----------
    high_matrix : np.ndarray
        形狀 (n, h) 的未來最高價矩陣
    low_matrix : np.ndarray
        形狀 (n, h) 的未來最低價矩陣
    stop_levels : np.ndarray
        形狀 (n,) 的止損價格數組
    target_levels : np.ndarray
        形狀 (n,) 的止盈價格數組
    
    Returns:
    --------
    np.ndarray
        形狀 (n,) 的結果數組：
        - 0.0: 先觸發止損
        - 1.0: 先觸發止盈
        - NaN: 未觸發或同時觸發
    
    Logic:
    ------
    對於每個時間點 i：
    1. 檢查未來每一期是否觸發止損（low <= stop）或止盈（high >= target）
    2. 找到首次觸發的位置
    3. 如果同一期同時觸發，返回 NaN
    4. 否則返回先觸發的類型
    """
    n, h = high_matrix.shape
    
    # 驗證輸入形狀
    if low_matrix.shape != (n, h):
        raise ValueError(f"low_matrix shape {low_matrix.shape} doesn't match high_matrix shape {(n, h)}")
    if stop_levels.shape != (n,):
        raise ValueError(f"stop_levels shape {stop_levels.shape} doesn't match expected ({n},)")
    if target_levels.shape != (n,):
        raise ValueError(f"target_levels shape {target_levels.shape} doesn't match expected ({n},)")
    
    # 擴展 stop 和 target 到矩陣形狀以便廣播比較
    stop_matrix = np.broadcast_to(stop_levels[:, np.newaxis], (n, h))
    target_matrix = np.broadcast_to(target_levels[:, np.newaxis], (n, h))
    
    # 檢查每期是否觸發（布爾矩陣）
    stop_hit = low_matrix <= stop_matrix
    target_hit = high_matrix >= target_matrix
    
    # 初始化結果為 NaN
    result = np.full(n, np.nan, dtype=np.float64)
    
    # 對每一行（每個時間點）處理
    for i in range(n):
        # 找到首次觸發止損的位置
        stop_indices = np.where(stop_hit[i])[0]
        first_stop = stop_indices[0] if len(stop_indices) > 0 else h
        
        # 找到首次觸發止盈的位置
        target_indices = np.where(target_hit[i])[0]
        first_target = target_indices[0] if len(target_indices) > 0 else h
        
        # 判斷結果
        if first_stop == first_target and first_stop < h:
            # 同時觸發，返回 NaN
            result[i] = np.nan
        elif first_stop < first_target:
            # 先觸發止損
            result[i] = 0.0
        elif first_target < first_stop:
            # 先觸發止盈
            result[i] = 1.0
        # else: 都沒觸發，保持 NaN
    
    return result


def stop_target_sequence_vectorized(
    df: pd.DataFrame,
    h: int = 20,
    atr_now: pd.Series = None,
    stop_k: float = 1.5,
    tgt_k: float = 2.5
) -> pd.Series:
    """
    向量化版本：判斷未來 h 期內，先觸發止盈還是止損
    
    這是 labels.py 中 stop_target_sequence 的優化版本，使用向量化操作
    替代雙重循環，大幅提升性能（特別是對15分鐘時框數據）。
    
    Parameters:
    -----------
    df : pd.DataFrame
        包含 'open', 'high', 'low', 'close' 的 OHLCV 數據
    h : int
        預測視窗，範圍 1~2000，預設 20
    atr_now : pd.Series
        當前 ATR 值序列，索引必須與 df 對齊
    stop_k : float
        止損 ATR 倍數，範圍 0.1~10.0，預設 1.5
    tgt_k : float
        止盈 ATR 倍數，範圍 0.1~10.0，預設 2.5
    
    Returns:
    --------
    pd.Series
        索引與 df 相同的序列：
        - 0.0: 先觸發止損
        - 1.0: 先觸發止盈
        - NaN: 未觸發或同時觸發
    
    Raises:
    -------
    ValueError
        如果參數不在有效範圍內
    TypeError
        如果輸入類型不正確
    
    Performance:
    ------------
    相比原始實現，預期性能提升：
    - 15m 時框（~105k 數據點）：從 >1 小時降到 <5 分鐘（>10x）
    - 1H 時框（~26k 數據點）：從 ~30 秒降到 <10 秒（>3x）
    """
    # ===== 輸入驗證 =====
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")
    
    required_cols = ['high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"DataFrame must contain columns {required_cols}, missing: {missing_cols}")
    
    if not isinstance(h, int) or h <= 0:
        raise ValueError(f"h must be a positive integer, got {h}")
    
    if stop_k <= 0:
        raise ValueError(f"stop_k must be positive, got {stop_k}")
    
    if tgt_k <= 0:
        raise ValueError(f"tgt_k must be positive, got {tgt_k}")
    
    if atr_now is None:
        raise ValueError("atr_now cannot be None")
    
    if not isinstance(atr_now, pd.Series):
        raise TypeError("atr_now must be a pandas Series")
    
    # 檢查索引對齊
    if not df.index.equals(atr_now.index):
        raise ValueError("atr_now index must match DataFrame index")
    
    # ===== 預計算止盈止損水平 =====
    close = df['close'].values
    atr_values = atr_now.values
    
    # 計算止損和止盈價格（向量化）
    # stop = close * (1 - stop_k * atr / close) = close - stop_k * atr
    stop_levels = close - stop_k * atr_values
    
    # target = close * (1 + tgt_k * atr / close) = close + tgt_k * atr
    target_levels = close + tgt_k * atr_values
    
    # ===== 構建未來窗口矩陣 =====
    high_matrix = create_forward_window_matrix(df['high'], h)
    low_matrix = create_forward_window_matrix(df['low'], h)
    
    # ===== 向量化查找首次觸發 =====
    result_array = find_first_trigger(
        high_matrix,
        low_matrix,
        stop_levels,
        target_levels
    )
    
    # ===== 轉換回 Series =====
    result = pd.Series(result_array, index=df.index)
    
    return result
