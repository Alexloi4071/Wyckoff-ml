#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析新生成的數據統計
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml

print('='*80)
print('新生成數據統計分析報告')
print('='*80)

timeframes = ['15m', '1h', '4h', '1D']

for tf in timeframes:
    print(f'\n{"="*80}')
    print(f'時框: {tf.upper()}')
    print(f'{"="*80}')
    
    # 讀取數據
    features_path = Path(f'out/{tf}/btcusdt_features_{tf}.csv')
    labels_path = Path(f'out/{tf}/btcusdt_labels_{tf}.csv')
    params_path = Path(f'out/{tf}/btcusdt_params_{tf}.yaml')
    
    if not features_path.exists() or not labels_path.exists():
        print(f'  ⚠️  數據文件不存在')
        continue
    
    # 讀取特徵和標籤
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    
    # 讀取參數
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    
    print(f'\n【基本信息】')
    print(f'  數據點數: {len(features):,}')
    print(f'  特徵數量: {features.shape[1] - 1}')  # 減去timestamp列
    print(f'  標籤數量: {labels.shape[1] - 1}')
    print(f'  時間範圍: {features.iloc[0, 0]} 到 {features.iloc[-1, 0]}')
    
    print(f'\n【時框特定參數】')
    print(f'  Horizon: {params["label"]["horizon"]}')
    print(f'  Percentile: {params["label"]["percentile"]}%')
    print(f'  Ref Window: {params["label"]["ref_window"]}')
    print(f'  Stop Loss: {params["risk"]["stop_loss_pct"]}%')
    print(f'  Take Profit: {params["risk"]["take_profit_pct"]}%')
    print(f'  盈虧比: 1:{params["risk"]["take_profit_pct"]/params["risk"]["stop_loss_pct"]:.2f}')
    print(f'  Time Stop: {params["risk"]["time_stop_bars"]} bars')
    print(f'  Max Position: {params["risk"]["max_position_pct"]*100:.0f}%')
    
    print(f'\n【特徵統計】')
    # 選擇幾個關鍵特徵進行統計
    key_features = ['f_rsi_14', 'f_atr_14_p', 'f_vol_pct_60', 'f_ret_log_20']
    for feat in key_features:
        # 找到匹配的列
        matching_cols = [col for col in features.columns if feat in col]
        if matching_cols:
            col = matching_cols[0]
            data = features[col].dropna()
            if len(data) > 0:
                print(f'  {col}:')
                print(f'    均值: {data.mean():.4f}')
                print(f'    標準差: {data.std():.4f}')
                print(f'    最小值: {data.min():.4f}')
                print(f'    最大值: {data.max():.4f}')
    
    print(f'\n【標籤統計】')
    for col in labels.columns:
        if col.startswith('y_'):
            data = labels[col].dropna()
            if len(data) > 0:
                # 計算正樣本比例
                if data.dtype in [np.float64, np.int64]:
                    positive_ratio = (data > 0).sum() / len(data) * 100
                    print(f'  {col}:')
                    print(f'    正樣本比例: {positive_ratio:.2f}%')
                    print(f'    均值: {data.mean():.4f}')
                    print(f'    標準差: {data.std():.4f}')
                    
                    # 如果是二分類標籤，顯示分佈
                    if set(data.unique()).issubset({0.0, 1.0}):
                        print(f'    分佈: 0={((data==0).sum()):,}, 1={((data==1).sum()):,}')

print(f'\n{"="*80}')
print('統計分析完成！')
print('='*80)

# 生成對比表格
print(f'\n{"="*80}')
print('時框參數對比表')
print('='*80)

print(f'\n{"時框":<8} {"Horizon":<10} {"Percentile":<12} {"Stop Loss":<12} {"Take Profit":<12} {"盈虧比":<10}')
print('-'*80)

for tf in timeframes:
    params_path = Path(f'out/{tf}/btcusdt_params_{tf}.yaml')
    if params_path.exists():
        with open(params_path, 'r') as f:
            params = yaml.safe_load(f)
        
        horizon = params["label"]["horizon"]
        pct = params["label"]["percentile"]
        stop = params["risk"]["stop_loss_pct"]
        take = params["risk"]["take_profit_pct"]
        ratio = take / stop
        
        print(f'{tf:<8} {horizon:<10} {pct}%{"":<9} {stop}%{"":<9} {take}%{"":<9} 1:{ratio:.2f}')

print('\n' + '='*80)
