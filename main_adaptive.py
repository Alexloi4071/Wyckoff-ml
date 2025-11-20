#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main_adaptive.py (重構版)
集成時框自適應參數管理
"""

import pandas as pd
import yaml
from pathlib import Path
import copy

from utils import validate_bounds, load_ohlcv
from features import make_features
from labels import make_labels
from timeframe_adapter import TimeframeAdapter

def run_pipeline(base_cfg: dict, timeframe: str):
    """
    執行特徵+標籤生成流程(時框自適應版)
    
    Args:
        base_cfg: 基礎配置字典
        timeframe: 時框字符串 (例如 '15m', '1h')
    """
    # 初始化時框適配器
    adapter = TimeframeAdapter()
    
    # 深拷貝配置避免污染
    cfg = copy.deepcopy(base_cfg)
    
    # 用時框特定參數更新配置
    cfg = adapter.update_config(cfg, timeframe)
    
    # 設置重採樣頻率
    cfg["preprocess"]["resample"] = timeframe
    
    print(f"\n{'='*60}")
    print(f"處理時框: {timeframe}")
    print(f"{'='*60}")
    print(f"標籤horizon: {cfg['timeframe_params']['label']['horizon']}")
    print(f"止損/止盈: {cfg['timeframe_params']['risk']['stop_loss_pct']}% / "
          f"{cfg['timeframe_params']['risk']['take_profit_pct']}%")
    print(f"RSI窗口: {cfg['timeframe_params']['features']['rsi_window']}")
    
    # 驗證參數
    validate_bounds(cfg)
    
    # 載入數據
    df = load_ohlcv(cfg)
    
    # 日期篩選
    sd = cfg["general"].get("start_date")
    ed = cfg["general"].get("end_date")
    if sd:
        sd_dt = pd.to_datetime(sd)
        if df.index.tz is not None:
            sd_dt = sd_dt.tz_localize('UTC')
        df = df.loc[df.index >= sd_dt]
    if ed:
        ed_dt = pd.to_datetime(ed)
        if df.index.tz is not None:
            ed_dt = ed_dt.tz_localize('UTC')
        df = df.loc[df.index <= ed_dt]
    
    print(f"數據範圍: {df.index[0]} 到 {df.index[-1]}")
    print(f"數據點數: {len(df)}")
    
    # 生成特徵
    feats = make_features(df, cfg)
    print(f"特徵數量: {feats.shape[1]}")
    
    # 生成標籤
    labels = make_labels(df, cfg)
    print(f"標籤數量: {labels.shape[1]}")
    
    # 對齊並去除NaN
    if cfg["general"].get("dropna", True):
        valid_idx = feats.dropna().index.intersection(labels.dropna().index)
        feats = feats.loc[valid_idx]
        labels = labels.loc[valid_idx]
        print(f"有效數據點: {len(valid_idx)} (去除NaN後)")
    
    # 生成輸出路徑
    base_dir = Path(cfg["output"]["base_dir"])
    tf_dir = base_dir / timeframe
    tf_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = cfg["output"]["base_name"]
    feats_csv = tf_dir / f"{base_name}_features_{timeframe}.csv"
    labels_csv = tf_dir / f"{base_name}_labels_{timeframe}.csv"
    params_yaml = tf_dir / f"{base_name}_params_{timeframe}.yaml"
    
    # 保存結果
    feats.to_csv(feats_csv, index_label="timestamp")
    labels.to_csv(labels_csv, index_label="timestamp")
    
    # 保存該時框的完整參數(用於回測)
    with open(params_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(cfg['timeframe_params'], f, allow_unicode=True)
    
    print(f"✓ 特徵已保存: {feats_csv}")
    print(f"✓ 標籤已保存: {labels_csv}")
    print(f"✓ 參數已保存: {params_yaml}")
    
    return feats, labels, cfg

# =========================================================
# 執行入口
# =========================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="多時框量化ML流程")
    parser.add_argument(
        "--config",
        type=str,
        default="config_base.yaml",
        help="基礎配置文件路徑"
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        nargs='+',
        default=['15m', '1h', '4h', '1D'],
        help="要處理的時框列表"
    )
    args = parser.parse_args()
    
    # 載入基礎配置
    with open(args.config, 'r', encoding='utf-8') as f:
        base_cfg = yaml.safe_load(f)
    
    # 對每個時框獨立處理
    results = {}
    for tf in args.timeframes:
        try:
            feats, labels, cfg = run_pipeline(base_cfg, tf)
            results[tf] = {
                'features': feats,
                'labels': labels,
                'config': cfg
            }
        except Exception as e:
            print(f"✗ 處理時框 {tf} 時出錯: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"處理完成! 成功處理 {len(results)}/{len(args.timeframes)} 個時框")
    print(f"{'='*60}")
