#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
backtest_engine.py

Backtrader回測引擎核心模組
- 支援多特徵、多標籤的ML策略回測
- 靈活的配置驅動架構
- 預留ML模型集成接口
- 完整的分析報告和可視化
"""

import backtrader as bt
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from ml_strategy import MLSignalStrategy
from data_handler import MLDataFeed
from analyzers import MLAnalyzers
from fix_backtrader_plot import BacktestPlotter
from timeframe_utils import TimeframeUtils

class BacktestEngine:
    """
    Backtrader回測引擎
    
    Features:
    - 配置檔驅動的靈活架構
    - 支援多特徵、多標籤策略
    - 完整的風險管理
    - ML模型集成準備
    - 豐富的分析報告
    """
    
    def __init__(self, config_path: str):
        """
        初始化回測引擎
        
        Args:
            config_path: YAML配置檔路徑
        """
        self.config = self._load_config(config_path)
        self.cerebro = None
        self.results = None
        self.data_feed = None
        
        # 存儲時間框架用於時間單位轉換
        self.timeframe = self.config['general'].get('timeframe', None)
        
        # 繪圖器 - 按照 backtrader_plot_fix_doc.md 要求設置
        self.plotter = BacktestPlotter(
            gui_mode=self.config['output'].get('gui_mode', False)
        )
        
    def _load_config(self, config_path: str) -> dict:
        """載入配置檔"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 驗證必要配置項
        required_keys = ['general', 'broker', 'strategy', 'output']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"配置檔缺少必要項目: {key}")
        
        return config
        
    def setup_cerebro(self):
        """設置Cerebro回測引擎"""
        self.cerebro = bt.Cerebro()
        
        # 配置券商
        broker_cfg = self.config['broker']
        self.cerebro.broker.setcash(broker_cfg['cash'])
        self.cerebro.broker.setcommission(commission=broker_cfg['commission'])
        
        # 設置滑點（如果需要）
        if broker_cfg.get('slippage', 0) > 0:
            self.cerebro.broker.set_slippage_perc(
                perc=broker_cfg['slippage'],
                slip_open=True,
                slip_limit=True,
                slip_match=True
            )
        
        print(f"[券商設置] 初始資金: ${broker_cfg['cash']:,.2f}")
        print(f"[券商設置] 手續費率: {broker_cfg['commission']*100:.3f}%")
        
    def load_data(self):
        """載入數據並創建數據源"""
        general_cfg = self.config['general']
        
        # 使用自定義數據處理器
        self.data_feed = MLDataFeed(
            data_path=general_cfg['data_path'],
            features_path=general_cfg['features_path'], 
            labels_path=general_cfg['labels_path'],
            datetime_col=general_cfg.get('datetime_col', 'ts'),  # 向後兼容
            ohlcv_datetime_col=general_cfg.get('ohlcv_datetime_col'),
            features_datetime_col=general_cfg.get('features_datetime_col'),
            labels_datetime_col=general_cfg.get('labels_datetime_col'),
            start_date=general_cfg.get('start_date'),
            end_date=general_cfg.get('end_date')
        )
        
        # 獲取Backtrader數據源
        data_bt = self.data_feed.get_backtrader_data()
        self.cerebro.adddata(data_bt, name=general_cfg['symbol'])
        
        print(f"[數據載入] 時間範圍: {general_cfg.get('start_date', 'All')} 到 {general_cfg.get('end_date', 'All')}")
        print(f"[數據載入] 特徵數量: {len(self.data_feed.features.columns)}")
        print(f"[數據載入] 標籤數量: {len(self.data_feed.labels.columns)}")
        
    def add_strategy(self):
        """添加策略"""
        strategy_cfg = self.config['strategy']
        
        # 使用基礎策略（僅做多）
        strategy_class = MLSignalStrategy
        print(f"[策略類型] 基礎策略（僅做多）")
        
        # 將配置和數據傳遞給策略
        self.cerebro.addstrategy(
            strategy_class,
            config=self.config,
            ml_data=self.data_feed
        )
        
        print(f"[策略配置] 策略: {strategy_cfg['strategy_name']}")
        print(f"[策略配置] 主信號: {strategy_cfg['signals']['primary_signal']['label_col']}")
        
    def add_analyzers(self):
        """添加分析器"""
        analyzers = MLAnalyzers(self.config)
        
        # 添加標準分析器
        analyzer_list = analyzers.get_analyzers()
        for name, analyzer_class, params in analyzer_list:
            self.cerebro.addanalyzer(analyzer_class, **params, _name=name)
            
        print(f"[分析器] 已添加 {len(analyzer_list)} 個分析器")
        
    def run_backtest(self):
        """執行回測"""
        print(f"\n{'='*60}")
        print("[開始回測]")
        print('='*60)
        
        start_time = datetime.now()
        
        # 執行回測
        self.results = self.cerebro.run()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n[回測完成] 耗時: {duration.total_seconds():.2f}秒")
        
    def analyze_results(self):
        """分析結果並生成報告"""
        if not self.results:
            print("[錯誤] 沒有回測結果可分析")
            return
            
        # 獲取策略實例
        strategy = self.results[0]
        
        # 獲取最終資金
        final_value = self.cerebro.broker.getvalue()
        initial_cash = self.config['broker']['cash']
        total_return = (final_value - initial_cash) / initial_cash * 100
        
        print(f"\n{'='*60}")
        print("[回測結果摘要]")
        print('='*60)
        print(f"初始資金: ${initial_cash:,.2f}")
        print(f"最終資金: ${final_value:,.2f}")
        print(f"總收益率: {total_return:.2f}%")
        
        # 分析器結果
        self._print_analyzer_results(strategy)
        
        # 儲存詳細結果
        if self.config['output']['generate_report']:
            self._save_results(strategy)
            
    def _print_analyzer_results(self, strategy):
        """打印分析器結果"""
        print(f"\n[詳細分析]")
        print("-" * 40)
        
        # SharpeRatio
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe.get_analysis()
            if 'sharperatio' in sharpe and sharpe['sharperatio'] is not None:
                print(f"夏普比率: {sharpe['sharperatio']:.4f}")
        
        # DrawDown
        if hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown.get_analysis()
            if dd.max.drawdown is not None:
                print(f"最大回撤: {dd.max.drawdown:.2f}%")
            if dd.max.len is not None:
                # 使用 TimeframeUtils 轉換時間單位
                duration_str = TimeframeUtils.format_duration(dd.max.len, self.timeframe)
                print(f"最大回撤期間: {duration_str}")
        
        # TradeAnalyzer
        if hasattr(strategy.analyzers, 'trades'):
            trades = strategy.analyzers.trades.get_analysis()
            total_trades = trades.total.closed if trades.total.closed else 0
            won_trades = trades.won.total if hasattr(trades, 'won') and trades.won.total else 0
            
            if total_trades > 0:
                win_rate = (won_trades / total_trades) * 100
                print(f"總交易次數: {total_trades}")
                print(f"勝率: {win_rate:.2f}%")
                
                if hasattr(trades, 'won') and trades.won.total and trades.won.pnl.average is not None:
                    avg_win = trades.won.pnl.average
                    print(f"平均盈利: ${avg_win:.2f}")
                    
                if hasattr(trades, 'lost') and trades.lost.total and trades.lost.pnl.average is not None:
                    avg_loss = trades.lost.pnl.average  
                    print(f"平均虧損: ${avg_loss:.2f}")
                    
                    if avg_loss != 0 and avg_win is not None:
                        # 盈虧比 (Win/Loss Ratio) = 平均盈利 / 平均虧損絕對值
                        win_loss_ratio = avg_win / abs(avg_loss)
                        print(f"盈虧比: {win_loss_ratio:.2f}")
                        
                        # 盈利因子 (Profit Factor) = 總盈利 / 總虧損絕對值
                        profit_factor = abs(avg_win * won_trades / (avg_loss * (total_trades - won_trades)))
                        print(f"盈利因子: {profit_factor:.2f}")
        
    def _save_results(self, strategy):
        """儲存詳細結果"""
        output_cfg = self.config['output']
        base_results_dir = Path(output_cfg['results_dir'])
        
        # 生成日期和版本號目錄
        current_date = datetime.now().strftime("%Y%m%d")
        
        if output_cfg.get('create_timestamp_dir', False):
            # 使用版本管理系統，並存儲到config中供其他地方使用
            results_dir = self._create_versioned_directory(base_results_dir, current_date)
            # 將版本目錄存儲到config中，供報告生成器使用
            self.config['_internal_results_dir'] = str(results_dir)
        else:
            results_dir = base_results_dir
            
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件前綴 - 使用配置中的file_prefix
        file_prefix = output_cfg.get('file_prefix', 'backtest')
        timestamp = datetime.now().strftime("%H%M%S")
        if output_cfg.get('timestamp', True):
            prefix = f"{file_prefix}_{timestamp}"
        else:
            prefix = file_prefix
        
        # 儲存交易記錄
        if output_cfg.get('save_trades', True):
            trades_file = results_dir / f"{prefix}_trades.csv"
            self._save_trades(strategy, trades_file)
            
        # 儲存權益曲線
        if output_cfg.get('save_equity_curve', True):
            equity_file = results_dir / f"{prefix}_equity.csv"
            self._save_equity_curve(equity_file)
            
        print(f"\n[結果儲存] 檔案前綴: {prefix}")
        print(f"[結果儲存] 儲存目錄: {results_dir}")
        
        # ✅ 新增：生成完整的詳細報告（與V2-V4一致）
        if output_cfg.get('generate_report', True):
            print(f"\n[報告生成] 開始生成詳細報告...")
            try:
                from analyzers import ReportGenerator
                report_gen = ReportGenerator(self.config)
                report_gen.generate_report(self.results, results_dir)
                print(f"[報告生成] 詳細報告生成完成")
            except Exception as e:
                print(f"[報告生成] 警告: 報告生成失敗 - {e}")
                import traceback
                traceback.print_exc()
        
    def _create_versioned_directory(self, base_dir: Path, date_str: str) -> Path:
        """
        創建版本化的目錄結構
        
        Args:
            base_dir: 基礎目錄
            date_str: 日期字符串 (YYYYMMDD)
            
        Returns:
            Path: 版本化的目錄路徑
        """
        # 查找當天已存在的版本
        existing_versions = []
        pattern = f"{date_str}_V*"
        
        if base_dir.exists():
            for dir_path in base_dir.glob(pattern):
                if dir_path.is_dir():
                    try:
                        # 提取版本號
                        version_part = dir_path.name.split('_V')[1]
                        version_num = int(version_part)
                        existing_versions.append(version_num)
                    except (IndexError, ValueError):
                        continue
        
        # 確定新版本號
        if existing_versions:
            next_version = max(existing_versions) + 1
        else:
            next_version = 1
            
        # 創建新的版本目錄
        version_dir = base_dir / f"{date_str}_V{next_version}"
        
        print(f"[版本管理] 創建版本目錄: {version_dir.name}")
        return version_dir
        
    def _save_trades(self, strategy, filepath):
        """儲存交易記錄"""
        # 這裡需要從策略中提取交易記錄
        # 具體實現依賴於策略類的設計
        trades_data = getattr(strategy, 'trades_log', [])
        
        if trades_data:
            df = pd.DataFrame(trades_data)
            df.to_csv(filepath, index=False)
            print(f"交易記錄已儲存: {filepath}")
        else:
            print("無交易記錄可儲存")
            
    def _save_equity_curve(self, filepath):
        """儲存權益曲線"""
        # 從Cerebro獲取權益曲線數據
        # 這需要在策略中記錄每日權益
        equity_data = getattr(self.results[0], 'equity_log', [])
        
        if equity_data:
            df = pd.DataFrame(equity_data)
            df.to_csv(filepath, index=False)
            print(f"權益曲線已儲存: {filepath}")
        else:
            print("無權益曲線數據可儲存")
    
    def plot_results(self):
        """使用 BacktestPlotter 繪製回測結果圖表 - 按照 backtrader_plot_fix_doc.md 實現"""
        if not self.config['output'].get('plot_results', True):
            return
            
        try:
            # 按照文檔要求獲取配置
            if '_internal_results_dir' in self.config:
                output_dir = Path(self.config['_internal_results_dir'])
            else:
                output_dir = Path(self.config['output']['results_dir'])
            
            prefix = self.config['output'].get('file_prefix', 'backtest')
            
            # 按照文檔要求調用 BacktestPlotter
            self.plotter.safe_plot_and_save(self.cerebro, output_dir, prefix)
            
        except Exception as e:
            print(f"[繪圖] 錯誤: {e}")
            
    def run_full_backtest(self):
        """執行完整的回測流程"""
        try:
            # 1. 設置引擎
            self.setup_cerebro()
            
            # 2. 載入數據
            self.load_data()
            
            # 3. 添加策略
            self.add_strategy()
            
            # 4. 添加分析器
            self.add_analyzers()
            
            # 5. 執行回測
            self.run_backtest()
            
            # 6. 分析結果
            self.analyze_results()
            
            # 7. 繪製圖表 (安全模式)
            try:
                if self.config['output'].get('plot_results', True):
                    print("[繪圖] 開始生成圖表...")
                    self.plot_results()
                    print("[繪圖] 圖表生成完成")
                else:
                    print("[繪圖] 已跳過圖表生成")
            except Exception as plot_error:
                print(f"[警告] 繪圖過程出錯，但回測繼續: {plot_error}")
            
            print(f"\n{'='*60}")
            print("[回測流程完成]")
            print('='*60)
            
            return self.results
            
        except Exception as e:
            print(f"[錯誤] 回測執行失敗: {e}")
            raise e


def main():
    """主函數 - 支援命令行參數"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtrader回測引擎')
    parser.add_argument('--config', type=str, default='backtest_config.yaml',
                        help='YAML配置檔路徑 (預設: backtest_config.yaml)')
    
    args = parser.parse_args()
    config_path = args.config
    
    if not Path(config_path).exists():
        print(f"[錯誤] 找不到配置檔: {config_path}")
        return
        
    # 創建回測引擎並執行
    engine = BacktestEngine(config_path)
    results = engine.run_full_backtest()
    
    return results


if __name__ == "__main__":
    main()