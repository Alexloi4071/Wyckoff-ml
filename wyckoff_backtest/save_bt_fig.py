
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_bt_fig.py
自動保存 Backtrader 回測 K 線與交易細節圖
"""
import os
import pathlib

# 必須在導入 matplotlib 之前設置環境變量
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''  # 禁用 DISPLAY（Linux/Mac）

import matplotlib
matplotlib.use('Agg', force=True)  # 強制使用非交互後端
import matplotlib.pyplot as plt
plt.ioff()  # 關閉交互模式

def save_backtrader_figures(cerebro, output_dir="BackTest_Result", file_prefix="backtest_kline"):
    """
    自動保存 Backtrader 的所有 K線/回測細節圖（無GUI彈窗）
    - cerebro: Backtrader Cerebro 實例
    - output_dir: 圖檔儲存目錄
    - file_prefix: 檔名前綴
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 取得所有 Figure，完全禁用顯示
        # iplot=False: 不使用 Jupyter
        # use=None: 不使用特定後端
        # show=False: 不顯示圖表
        figs = cerebro.plot(
            style="candlestick", 
            iplot=False, 
            use=None,
            show=False,
            **{'plotdist': 0.5}  # 額外參數確保不顯示
        )
        
        if not isinstance(figs, (list, tuple)):
            figs = [figs]

        idx = 0
        for group in figs:
            if isinstance(group, (list, tuple)):
                for fig in group:
                    path = output_dir / f"{file_prefix}_figure_{idx}.png"
                    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                    print(f"[K線儲存] 已保存: {path}")
                    plt.close(fig)
                    idx += 1
            else:
                path = output_dir / f"{file_prefix}_figure_{idx}.png"
                group.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
                print(f"[K線儲存] 已保存: {path}")
                plt.close(group)
                idx += 1
        
        # 確保關閉所有圖表
        plt.close('all')
        
    except Exception as e:
        print(f"[警告] 圖表保存過程中出現錯誤: {e}")
        plt.close('all')  # 即使出錯也要關閉所有圖表

if __name__ == '__main__':
    print("請在 run_backtest.py 完成回測後引用 save_backtrader_figures 函式儲存圖表。")
