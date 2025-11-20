
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
save_bt_fig.py
自動保存 Backtrader 回測 K 線與交易細節圖
"""
import pathlib
import matplotlib
matplotlib.use('Agg')  # 設置非交互後端
import matplotlib.pyplot as plt
plt.ioff()  # 關閉交互模式

def save_backtrader_figures(cerebro, output_dir="BackTest_Result", file_prefix="backtest_kline"):
    """
    自動保存 Backtrader 的所有 K線/回測細節圖
    - cerebro: Backtrader Cerebro 實例
    - output_dir: 圖檔儲存目錄
    - file_prefix: 檔名前綴
    """
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 取得所有 Figure，show=False 無彈窗
    figs = cerebro.plot(style="candlestick", iplot=False, use=False, show=False)
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
    plt.close('all')

if __name__ == '__main__':
    print("請在 run_backtest.py 完成回測後引用 save_backtrader_figures 函式儲存圖表。")
