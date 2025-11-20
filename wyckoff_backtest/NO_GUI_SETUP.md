# 無GUI環境運行指南

## 問題說明

在無GUI環境（VM、服務器、Docker）下運行回測時，Backtrader 的 `cerebro.plot()` 方法會嘗試顯示圖表窗口，導致程序卡死或報錯。

## 解決方案

### 方案1：禁用K線圖生成（推薦）

**默認行為**：K線圖生成已被禁用，避免GUI彈窗問題。

回測會正常完成並生成：
- ✅ 文字報告
- ✅ 分析圖表（PNG格式）
- ✅ 交易記錄CSV
- ✅ 權益曲線CSV
- ❌ Backtrader K線圖（已禁用）

### 方案2：啟用K線圖（僅在有GUI環境下）

如果你在有GUI的環境下運行，可以在配置文件中啟用：

```yaml
output:
  results_dir: "BackTest_Result"
  save_kline_chart: true  # 啟用K線圖保存（會彈出窗口）
  # ... 其他配置
```

## 環境變量設置

代碼已自動設置以下環境變量：

```python
os.environ['MPLBACKEND'] = 'Agg'  # 使用非交互後端
os.environ['DISPLAY'] = ''         # 禁用顯示
```

## 測試無GUI環境

運行測試腳本：

```bash
python wyckoff_backtest/test_no_gui.py
```

## Docker/VM 環境建議

### Dockerfile 示例

```dockerfile
FROM python:3.9-slim

# 安裝必要的依賴（不包含GUI）
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 設置環境變量
ENV MPLBACKEND=Agg
ENV DISPLAY=

# 複製代碼
COPY . /app
WORKDIR /app

# 安裝Python依賴
RUN pip install --no-cache-dir -r requirements.txt

# 運行回測
CMD ["python", "wyckoff_backtest/run_backtest.py", "--config", "wyckoff_backtest/backtest_new_1d.yaml"]
```

### SSH遠程運行

```bash
# 設置環境變量
export MPLBACKEND=Agg
export DISPLAY=

# 運行回測
python wyckoff_backtest/run_backtest.py --config wyckoff_backtest/backtest_new_1d.yaml
```

## 配置文件示例

```yaml
general:
  data_path: "data/btcusdt_1m.parquet"
  features_path: "out/1d/btcusdt_features_1d.csv"
  labels_path: "out/1d/btcusdt_labels_1d.csv"
  start_date: "2023-01-15"
  # ... 其他配置

output:
  results_dir: "BackTest_Result"
  generate_report: true
  save_trades: true
  save_equity_curve: true
  plot_results: true          # 生成分析圖表（PNG）
  save_kline_chart: false     # 禁用Backtrader K線圖（避免GUI彈窗）
  gui_mode: false             # 非GUI模式
  # ... 其他配置
```

## 常見問題

### Q: 為什麼還是會彈出窗口？

A: 如果配置文件中 `save_kline_chart: true`，則會嘗試生成K線圖。請確保設置為 `false` 或完全移除該選項（默認為false）。

### Q: 如何在無GUI環境查看結果？

A: 所有結果都保存為文件：
- `backtest_report_*.txt` - 文字報告
- `analysis_charts_*.png` - 分析圖表
- `trades_*.csv` - 交易記錄
- `equity_*.csv` - 權益曲線

### Q: 分析圖表和K線圖有什麼區別？

A: 
- **分析圖表** (`analysis_charts_*.png`): 由 `analyzers.py` 生成，包含收益曲線、回撤等，不會彈窗
- **K線圖** (`*_figure_*.png`): 由 Backtrader 的 `cerebro.plot()` 生成，會嘗試顯示GUI窗口

### Q: 如何完全禁用所有圖表？

A: 在配置文件中設置：

```yaml
output:
  plot_results: false      # 禁用分析圖表
  save_kline_chart: false  # 禁用K線圖
```

## 驗證設置

運行以下命令確認環境：

```bash
python -c "import matplotlib; print('Backend:', matplotlib.get_backend())"
```

應該輸出：`Backend: agg`

## 總結

✅ **默認配置已優化為無GUI環境**  
✅ **所有必要的分析結果都會保存為文件**  
✅ **不會因為圖表顯示而卡死**  

如需在有GUI環境下查看K線圖，手動設置 `save_kline_chart: true` 即可。
