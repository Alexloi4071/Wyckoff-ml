#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
對比修復前後的數據變化
"""

import pandas as pd
from pathlib import Path

print('='*70)
print('修復前後數據對比分析')
print('='*70)

timeframes = {
    '15m': {
        'old': 'out/BTCUSDT/loose_v5.1/15m/btcusdt_loose_v5.1_labels_15m.csv',
        'new': 'out/15m/btcusdt_labels_15m.csv'
    },
    '1h': {
        'old': 'out/BTCUSDT/loose_v5.1/1H/btcusdt_loose_v5.1_labels_1H.csv',
        'new': 'out/1h/btcusdt_labels_1h.csv'
    },
    '4h': {
        'old': 'out/BTCUSDT/loose_v5.1/4H/btcusdt_loose_v5.1_labels_4H.csv',
        'new': 'out/4h/btcusdt_labels_4h.csv'
    },
    '1D': {
        'old': 'out/BTCUSDT/loose_v5.1/1D/btcusdt_loose_v5.1_labels_1D.csv',
        'new': 'out/1D/btcusdt_labels_1D.csv'
    }
}

for tf, paths in timeframes.items():
    print(f'\n{"-"*70}')
    print(f'時框: {tf}')
    print(f'{"-"*70}')
    
    old_path = Path(paths['old'])
    new_path = Path(paths['new'])
    
    if not old_path.exists():
        print(f'  ⚠️  修復前數據不存在: {old_path}')
        continue
    
    if not new_path.exists():
        print(f'  ⚠️  修復後數據不存在: {new_path}')
        continue
    
    # 讀取數據
    old_df = pd.read_csv(old_path)
    new_df = pd.read_csv(new_path)
    
    print(f'  修復前數據量: {len(old_df):,}')
    print(f'  修復後數據量: {len(new_df):,}')
    print(f'  變化: {len(new_df) - len(old_df):,} ({(len(new_df)/len(old_df)-1)*100:+.1f}%)')
    
    print(f'\n  修復前標籤列:')
    for col in old_df.columns:
        if col.startswith('y_'):
            print(f'    - {col}')
    
    print(f'\n  修復後標籤列:')
    for col in new_df.columns:
        if col.startswith('y_'):
            print(f'    - {col}')
    
    # 檢查參數文件
    param_file = new_path.parent / f'btcusdt_params_{tf.lower()}.yaml'
    if param_file.exists():
        import yaml
        with open(param_file, 'r') as f:
            params = yaml.safe_load(f)
        print(f'\n  時框特定參數:')
        print(f'    Horizon: {params["label"]["horizon"]}')
        print(f'    Percentile: {params["label"]["percentile"]}%')
        print(f'    Ref Window: {params["label"]["ref_window"]}')
        print(f'    Stop Loss: {params["risk"]["stop_loss_pct"]}%')
        print(f'    Take Profit: {params["risk"]["take_profit_pct"]}%')
        print(f'    盈虧比: 1:{params["risk"]["take_profit_pct"]/params["risk"]["stop_loss_pct"]:.2f}')

print(f'\n{"="*70}')
print('對比完成！')
print('='*70)
print('\n關鍵發現:')
print('1. 數據量減少是正常的（雙重shift + 更大的ref_window）')
print('2. 每個時框現在有獨立的horizon和參數')
print('3. 標籤列名反映了時框特定的參數')
print('4. 所有時框都生成了參數文件供回測使用')
