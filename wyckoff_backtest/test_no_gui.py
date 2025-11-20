#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
測試腳本：驗證無GUI環境下的回測
"""
import os
import sys

# 設置環境變量（模擬無GUI環境）
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''

print("=" * 60)
print("測試：無GUI環境下的回測")
print("=" * 60)
print(f"MPLBACKEND: {os.environ.get('MPLBACKEND')}")
print(f"DISPLAY: {os.environ.get('DISPLAY')}")
print()

# 導入並運行回測
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, "wyckoff_backtest/run_backtest.py", 
         "--config", "wyckoff_backtest/backtest_new_1d.yaml"],
        capture_output=False,
        text=True,
        timeout=300
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("✅ 測試成功：回測在無GUI環境下正常完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ 測試失敗：回測返回錯誤")
        print("=" * 60)
        sys.exit(1)
        
except subprocess.TimeoutExpired:
    print("\n" + "=" * 60)
    print("❌ 測試失敗：回測超時（可能卡在GUI顯示）")
    print("=" * 60)
    sys.exit(1)
    
except Exception as e:
    print("\n" + "=" * 60)
    print(f"❌ 測試失敗：{e}")
    print("=" * 60)
    sys.exit(1)
