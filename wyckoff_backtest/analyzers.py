#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyzers.py

Backtrader回測分析器模組
- 提供豐富的性能分析指標
- 支援自定義分析器
- 生成詳細的回測報告
- 可視化分析結果
"""

import backtrader as bt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import platform
from timeframe_utils import TimeframeUtils

# 設置matplotlib後端和字體
matplotlib.use('Agg')  # 使用非交互式後端
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']  # 設置字體
plt.rcParams['axes.unicode_minus'] = False  # 正確顯示負號

# 如果是Windows系統，嘗試設置中文字體
if platform.system() == 'Windows':
    try:
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
    except:
        pass

# 安全導入seaborn，避免版本兼容問題
try:
    import seaborn as sns
    HAS_SEABORN = True
except (ImportError, AttributeError) as e:
    print(f"[警告] Seaborn導入失敗，將使用matplotlib替代: {e}")
    HAS_SEABORN = False
    sns = None


class MLAnalyzers:
    """
    ML策略分析器管理器
    
    提供多種分析器用於評估策略性能
    """
    
    def __init__(self, config):
        """
        初始化分析器管理器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.output_cfg = config.get('output', {})
        
    def get_analyzers(self):
        """
        獲取所有需要的分析器
        
        Returns:
            list: (name, analyzer_class, params) 的列表
        """
        analyzers = []
        
        # 基礎分析器
        analyzers.extend([
            ('sharpe', bt.analyzers.SharpeRatio, {'timeframe': bt.TimeFrame.Days}),
            ('drawdown', bt.analyzers.DrawDown, {}),
            ('trades', bt.analyzers.TradeAnalyzer, {}),
            ('returns', bt.analyzers.Returns, {}),
            ('positions', bt.analyzers.PositionsValue, {}),
            ('transactions', bt.analyzers.Transactions, {}),
        ])
        
        # 根據配置添加額外分析器
        metrics = self.output_cfg.get('metrics', [])
        
        if 'calmar_ratio' in metrics:
            analyzers.append(('calmar', CalmarRatio, {}))
            
        if 'win_rate' in metrics or 'profit_factor' in metrics:
            analyzers.append(('winloss', WinLossAnalyzer, {}))
            
        # 自定義ML分析器
        analyzers.append(('ml_performance', MLPerformanceAnalyzer, {}))
        
        return analyzers


class CalmarRatio(bt.Analyzer):
    """
    Calmar比率分析器
    Calmar Ratio = Annual Return / Maximum Drawdown
    """
    
    def __init__(self):
        super().__init__()
        self.ret = bt.analyzers.Returns()
        self.dd = bt.analyzers.DrawDown()
        
    def start(self):
        super().start()
        
    def next(self):
        super().next()
        
    def stop(self):
        super().stop()
        
    def get_analysis(self):
        returns = self.ret.get_analysis()
        drawdown = self.dd.get_analysis()
        
        annual_return = returns.get('rtot', 0) * 252  # 假設252個交易日
        max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
        
        if max_drawdown == 0:
            calmar = 0
        else:
            calmar = annual_return / (max_drawdown / 100)
            
        return {
            'calmar_ratio': calmar,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown
        }


class WinLossAnalyzer(bt.Analyzer):
    """
    勝負分析器
    計算勝率、盈虧比等指標
    """
    
    def __init__(self):
        super().__init__()
        self.trades = []
        
    def notify_trade(self, trade):
        if trade.isclosed:
            self.trades.append({
                'pnl': trade.pnl,
                'pnlcomm': trade.pnlcomm,
                'size': trade.size,
                'price': trade.price,
                'value': trade.value,
                'commission': trade.commission,
                'dtopen': trade.dtopen,
                'dtclose': trade.dtclose,
                'baropen': trade.baropen,
                'barclose': trade.barclose,
                'is_win': trade.pnlcomm > 0
            })
            
    def get_analysis(self):
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0
            }
            
        df = pd.DataFrame(self.trades)
        
        total_trades = len(df)
        winning_trades = df[df['is_win'] == True]
        losing_trades = df[df['is_win'] == False]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = winning_trades['pnlcomm'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnlcomm'].mean() if len(losing_trades) > 0 else 0
        
        # 盈虧比 (Profit Factor)
        total_wins = winning_trades['pnlcomm'].sum() if len(winning_trades) > 0 else 0
        total_losses = abs(losing_trades['pnlcomm'].sum()) if len(losing_trades) > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': df['pnlcomm'].max(),
            'largest_loss': df['pnlcomm'].min(),
            'total_pnl': df['pnlcomm'].sum(),
            'avg_trade': df['pnlcomm'].mean()
        }


class MLPerformanceAnalyzer(bt.Analyzer):
    """
    ML策略性能分析器
    專門分析ML信號的表現
    """
    
    def __init__(self):
        super().__init__()
        self.signal_data = []
        self.portfolio_values = []
        
    def next(self):
        # 記錄當前組合價值
        self.portfolio_values.append({
            'datetime': self.strategy.data.datetime.datetime(0),
            'value': self.strategy.broker.get_value(),
            'cash': self.strategy.broker.get_cash()
        })
        
        # 如果策略有信號記錄，則收集
        if hasattr(self.strategy, 'signals_log'):
            current_signals = getattr(self.strategy, 'signals_log', [])
            if len(current_signals) > len(self.signal_data):
                self.signal_data = current_signals.copy()
                
    def get_analysis(self):
        """返回ML特定的分析結果"""
        analysis = {
            'portfolio_history': self.portfolio_values,
            'signal_analysis': self._analyze_signals(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        return analysis
        
    def _analyze_signals(self):
        """分析信號表現"""
        if not self.signal_data:
            return {}
            
        signals_df = pd.DataFrame(self.signal_data)
        
        if len(signals_df) == 0:
            return {}
            
        analysis = {
            'total_signals': len(signals_df),
            'signal_strength_stats': {
                'mean': signals_df['signal_strength'].mean(),
                'std': signals_df['signal_strength'].std(),
                'min': signals_df['signal_strength'].min(),
                'max': signals_df['signal_strength'].max()
            }
        }
        
        # 信號分佈分析
        strong_signals = signals_df[abs(signals_df['signal_strength']) > 0.5]
        analysis['strong_signals_pct'] = len(strong_signals) / len(signals_df)
        
        return analysis
        
    def _calculate_performance_metrics(self):
        """計算性能指標，包含圖表繪製所需的詳細數據"""
        if len(self.portfolio_values) < 2:
            return {}
            
        values_df = pd.DataFrame(self.portfolio_values)
        values_df['datetime'] = pd.to_datetime(values_df['datetime'])
        values_df = values_df.set_index('datetime')
        
        # 計算日收益率
        values_df['returns'] = values_df['value'].pct_change()
        daily_returns = values_df['returns'].dropna()
        
        if len(daily_returns) == 0:
            return {}
            
        # 計算回撤序列 (用於回撤圖表)
        peak = values_df['value'].expanding(min_periods=1).max()
        drawdown_series = (values_df['value'] - peak) / peak
        
        # 計算月度收益 (用於月度收益分佈圖)
        try:
            # 重採樣至月度並計算月度收益率
            monthly_returns = values_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly_returns_list = monthly_returns.dropna().tolist()
        except Exception:
            monthly_returns_list = []
        
        # 基本統計指標
        metrics = {
            'total_return': (values_df['value'].iloc[-1] / values_df['value'].iloc[0] - 1),
            'volatility': daily_returns.std() * np.sqrt(252),  # 年化波動率
            'sharpe_estimate': daily_returns.mean() / daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0,
            'max_drawdown_pct': drawdown_series.min(),
            
            # 圖表繪製數據
            'drawdown_series': drawdown_series,    # 完整回撤時間序列
            'monthly_returns': monthly_returns_list,  # 月度收益列表
            'daily_returns': daily_returns,        # 日收益率序列
            'portfolio_values': values_df['value']  # 組合價值序列
        }
        
        return metrics
        
    def _calculate_max_drawdown(self, values):
        """計算最大回撤"""
        peak = values.expanding(min_periods=1).max()
        drawdown = (values - peak) / peak
        return drawdown.min()


class ReportGenerator:
    """
    回測報告生成器
    """
    
    def __init__(self, config):
        self.config = config
        self.output_cfg = config.get('output', {})
        # 存儲時間框架用於時間單位轉換
        self.timeframe = config['general'].get('timeframe', None)
        
    def generate_report(self, strategy_results, output_dir):
        """
        生成完整的回測報告
        
        Args:
            strategy_results: 策略回測結果
            output_dir: 輸出目錄
        """
        strategy = strategy_results[0]
        
        # 生成文字報告
        self._generate_text_report(strategy, output_dir)
        
        # 生成圖表報告
        if self.output_cfg.get('plot_results', True):
            self._generate_chart_report(strategy, output_dir)
            
        # 導出數據
        self._export_analysis_data(strategy, output_dir)
        
    def _generate_text_report(self, strategy, output_dir):
        """生成文字報告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = output_dir / f"backtest_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("Backtrader ML策略回測報告\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write("[基本信息]\n")
            f.write(f"策略名稱: {self.config['strategy']['strategy_name']}\n")
            f.write(f"回測期間: {self.config['general'].get('start_date', 'All')} 到 {self.config['general'].get('end_date', 'All')}\n")
            f.write(f"初始資金: ${self.config['broker']['cash']:,.2f}\n")
            f.write(f"手續費率: {self.config['broker']['commission']*100:.3f}%\n\n")
            
            # 績效指標
            self._write_performance_metrics(f, strategy)
            
            # 交易分析
            self._write_trade_analysis(f, strategy)
            
            # 信號分析
            self._write_signal_analysis(f, strategy)
            
        print(f"文字報告已生成: {report_file}")
        
    def _write_performance_metrics(self, f, strategy):
        """寫入績效指標"""
        f.write("============================================================\n")
        f.write("[回測結果摘要]\n")
        f.write("============================================================\n")
        
        # 基本收益
        final_value = strategy.broker.get_value()
        initial_value = self.config['broker']['cash']
        total_return = (final_value / initial_value - 1) * 100
        
        f.write(f"初始資金: ${initial_value:,.2f}\n")
        f.write(f"最終資金: ${final_value:,.2f}\n")
        f.write(f"總收益率: {total_return:.2f}%\n")
        f.write("\n")
        
        f.write("[詳細分析]\n")
        f.write("----------------------------------------\n")
        
        # 夏普比率
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe.get_analysis()
            f.write(f"夏普比率: {sharpe.get('sharperatio', 0):.4f}\n")
            
        # 最大回撤
        if hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown.get_analysis()
            f.write(f"最大回撤: {dd.max.drawdown:.2f}%\n")
            # 使用 TimeframeUtils 轉換時間單位
            duration_str = TimeframeUtils.format_duration(dd.max.len, self.timeframe)
            f.write(f"最大回撤期間: {duration_str}\n")
        
        # 交易統計（從策略獲取或計算）
        self._write_detailed_trade_stats(f, strategy)
        
        f.write("\n")
    
    def _write_detailed_trade_stats(self, f, strategy):
        """寫入詳細交易統計 - 使用與terminal相同的數據源"""
        # 使用與terminal輸出相同的數據來源
        if hasattr(strategy.analyzers, 'trades'):
            trades = strategy.analyzers.trades.get_analysis()
            total_trades = trades.total.closed if hasattr(trades.total, 'closed') else 0
            
            if total_trades > 0:
                won_trades = trades.won.total if hasattr(trades.won, 'total') else 0
                win_rate = (won_trades / total_trades) * 100
                
                f.write(f"總交易次數: {total_trades}\n")
                f.write(f"勝率: {win_rate:.2f}%\n")
                
                # 獲取盈虧數據
                if hasattr(trades.won, 'pnl') and hasattr(trades.lost, 'pnl'):
                    avg_win = trades.won.pnl.average if hasattr(trades.won.pnl, 'average') else 0
                    avg_loss = trades.lost.pnl.average if hasattr(trades.lost.pnl, 'average') else 0
                    
                    f.write(f"平均盈利: ${avg_win:.2f}\n")
                    f.write(f"平均虧損: ${avg_loss:.2f}\n")
                    
                    if avg_loss != 0:
                        # 盈虧比 (Win/Loss Ratio)
                        win_loss_ratio = avg_win / abs(avg_loss)
                        f.write(f"盈虧比: {win_loss_ratio:.2f}\n")
                        
                        # 盈利因子 (Profit Factor)
                        profit_factor = abs(avg_win * won_trades / (avg_loss * (total_trades - won_trades)))
                        f.write(f"盈利因子: {profit_factor:.2f}\n")
        
    def _write_trade_analysis(self, f, strategy):
        """寫入交易分析"""
        f.write("[交易分析]\n")
        
        if hasattr(strategy.analyzers, 'winloss'):
            wl = strategy.analyzers.winloss.get_analysis()
            
            f.write(f"總交易次數: {wl['total_trades']}\n")
            f.write(f"獲利交易: {wl['winning_trades']}\n")
            f.write(f"虧損交易: {wl['losing_trades']}\n")
            f.write(f"勝率: {wl['win_rate']*100:.2f}%\n")
            f.write(f"盈虧比: {wl['profit_factor']:.4f}\n")
            f.write(f"平均獲利: ${wl['avg_win']:.2f}\n")
            f.write(f"平均虧損: ${wl['avg_loss']:.2f}\n")
            f.write(f"最大單筆獲利: ${wl['largest_win']:.2f}\n")
            f.write(f"最大單筆虧損: ${wl['largest_loss']:.2f}\n")
            
        f.write("\n")
        
    def _write_signal_analysis(self, f, strategy):
        """寫入信號分析"""
        f.write("[信號分析]\n")
        
        if hasattr(strategy.analyzers, 'ml_performance'):
            ml_perf = strategy.analyzers.ml_performance.get_analysis()
            
            signal_analysis = ml_perf.get('signal_analysis', {})
            if signal_analysis:
                f.write(f"總信號數量: {signal_analysis['total_signals']}\n")
                f.write(f"強信號比例: {signal_analysis['strong_signals_pct']*100:.2f}%\n")
                
                stats = signal_analysis['signal_strength_stats']
                f.write(f"信號強度統計:\n")
                f.write(f"  平均值: {stats['mean']:.4f}\n")
                f.write(f"  標準差: {stats['std']:.4f}\n")
                f.write(f"  最小值: {stats['min']:.4f}\n")
                f.write(f"  最大值: {stats['max']:.4f}\n")
                
        f.write("\n")
        
    def _generate_chart_report(self, strategy, output_dir):
        """生成圖表報告"""
        try:
            # 確保使用非交互後端
            import matplotlib
            matplotlib.use('Agg')
            plt.ioff()
            
            # 設置圖表樣式
            try:
                plt.style.use('seaborn-v0_8')
            except OSError:
                # 如果seaborn樣式不可用，使用默認樣式
                plt.style.use('default')
                print("[警告] 使用默認matplotlib樣式，因為seaborn樣式不可用")
            
            # 創建多子圖
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Backtest Analysis Charts', fontsize=16)  # 使用英文避免編碼問題
            
            # 1. 權益曲線
            self._plot_equity_curve(axes[0, 0], strategy)
            
            # 2. 回撤圖
            self._plot_drawdown(axes[0, 1], strategy)
            
            # 3. 月度收益
            self._plot_monthly_returns(axes[1, 0], strategy)
            
            # 4. 信號分佈
            self._plot_signal_distribution(axes[1, 1], strategy)
            
            # 保存圖表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            chart_file = output_dir / f"analysis_charts_{timestamp}.png"
            plt.tight_layout()
            plt.savefig(chart_file, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            
            # 立即關閉圖表並清理所有資源
            plt.close(fig)
            plt.close('all')
            
            print(f"分析圖表已生成: {chart_file}")
            
        except Exception as e:
            print(f"[警告] 圖表生成失敗: {e}")
            
    def _plot_equity_curve(self, ax, strategy):
        """繪製權益曲線"""
        try:
            if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'ml_performance'):
                analysis = strategy.analyzers.ml_performance.get_analysis()
                
                # 嘗試從新的數據結構獲取組合價值
                performance_metrics = analysis.get('performance_metrics', {})
                portfolio_values = performance_metrics.get('portfolio_values')
                
                if portfolio_values is not None and len(portfolio_values) > 0:
                    # 從pandas Series繪製
                    ax.plot(portfolio_values.index, portfolio_values.values, 
                           label='Portfolio Value', color='blue', linewidth=2)
                    ax.set_title('Equity Curve')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Portfolio Value ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # 格式化Y軸數值
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                    return True
                
                # 回退到原來的數據結構
                portfolio_history = analysis.get('portfolio_history', [])
                if portfolio_history:
                    df = pd.DataFrame(portfolio_history)
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    
                    ax.plot(df['datetime'], df['value'], label='Portfolio Value', 
                           color='blue', linewidth=2)
                    ax.set_title('Equity Curve')
                    ax.set_xlabel('Time')
                    ax.set_ylabel('Portfolio Value ($)')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                    return True
            
            # 如果沒有數據，顯示占位文字
            ax.set_title('Equity Curve')
            ax.text(0.5, 0.5, 'No portfolio data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
            
        except Exception as e:
            ax.set_title('Equity Curve')
            ax.text(0.5, 0.5, f'Error loading equity data:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
                
    def _plot_drawdown(self, ax, strategy):
        """繪製回撤圖"""
        try:
            # 從MLPerformanceAnalyzer獲取回撤數據
            if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'ml_performance'):
                analysis = strategy.analyzers.ml_performance.get_analysis()
                performance_metrics = analysis.get('performance_metrics', {})
                drawdown_series = performance_metrics.get('drawdown_series')
                
                if drawdown_series is not None and len(drawdown_series) > 0:
                    # 繪製回撤曲線
                    ax.plot(drawdown_series.index, drawdown_series.values, 
                           color='red', linewidth=1.5, label='Drawdown')
                    
                    # 填充負回撤區域
                    ax.fill_between(drawdown_series.index, drawdown_series.values, 0, 
                                   where=(drawdown_series.values < 0), 
                                   color='red', alpha=0.3)
                    
                    ax.set_title('Drawdown Analysis')
                    ax.set_ylabel('Drawdown')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # 設置Y軸格式為百分比
                    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
                    
                    return True
            
            # 如果沒有數據，顯示占位文字
            ax.set_title('Drawdown Analysis')
            ax.text(0.5, 0.5, 'No drawdown data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
            
        except Exception as e:
            ax.set_title('Drawdown Analysis')
            ax.text(0.5, 0.5, f'Error loading drawdown data:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
                
    def _plot_monthly_returns(self, ax, strategy):
        """繪製月度收益分佈"""
        try:
            # 從MLPerformanceAnalyzer獲取月度收益數據
            if hasattr(strategy, 'analyzers') and hasattr(strategy.analyzers, 'ml_performance'):
                analysis = strategy.analyzers.ml_performance.get_analysis()
                performance_metrics = analysis.get('performance_metrics', {})
                monthly_returns = performance_metrics.get('monthly_returns', [])
                
                if monthly_returns and len(monthly_returns) > 0:
                    # 繪製月度收益分佈直方圖
                    n, bins, patches = ax.hist(monthly_returns, bins=min(20, len(monthly_returns)), 
                                             edgecolor='black', alpha=0.7, color='skyblue')
                    
                    # 設置標題和標籤
                    ax.set_title('Monthly Returns Distribution')
                    ax.set_xlabel('Monthly Return')
                    ax.set_ylabel('Frequency')
                    ax.grid(True, alpha=0.3)
                    
                    # 設置X軸格式為百分比
                    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
                    
                    # 添加統計信息
                    mean_return = np.mean(monthly_returns)
                    std_return = np.std(monthly_returns)
                    ax.axvline(mean_return, color='red', linestyle='--', 
                              label=f'Mean: {mean_return:.2%}')
                    ax.legend()
                    
                    return True
            
            # 如果沒有數據，顯示占位文字
            ax.set_title('Monthly Returns Distribution')
            ax.text(0.5, 0.5, 'No monthly returns data available', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
            
        except Exception as e:
            ax.set_title('Monthly Returns Distribution')
            ax.text(0.5, 0.5, f'Error loading monthly returns:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes)
            return False
                
    def _plot_signal_distribution(self, ax, strategy):
        """繪製信號分佈"""
        try:
            if hasattr(strategy, 'signals_log') and strategy.signals_log:
                signals_df = pd.DataFrame(strategy.signals_log)
                
                # 繪製信號強度分佈直方圖
                n, bins, patches = ax.hist(signals_df['signal_strength'], bins=20, 
                                         alpha=0.7, edgecolor='black', color='steelblue')
                
                # 設置標題和標籤（使用英文避免編碼問題）
                ax.set_title('Signal Strength Distribution', fontsize=12, fontweight='bold')
                ax.set_xlabel('Signal Strength', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # 添加統計信息
                mean_strength = signals_df['signal_strength'].mean()
                ax.axvline(mean_strength, color='red', linestyle='--', 
                          label=f'Mean: {mean_strength:.3f}')
                ax.legend()
                
                return True
            else:
                # 如果沒有信號數據，顯示占位文字
                ax.set_title('Signal Strength Distribution', fontsize=12, fontweight='bold')
                ax.text(0.5, 0.5, 'No signal data available', 
                        ha='center', va='center', transform=ax.transAxes, fontsize=10)
                return False
                
        except Exception as e:
            ax.set_title('Signal Strength Distribution', fontsize=12, fontweight='bold')
            ax.text(0.5, 0.5, f'Error loading signal data:\n{str(e)}', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=8)
            return False
            
    def _export_analysis_data(self, strategy, output_dir):
        """導出分析數據"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 導出交易記錄
        if hasattr(strategy, 'trades_log') and strategy.trades_log:
            trades_df = pd.DataFrame(strategy.trades_log)
            trades_file = output_dir / f"trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False)
            print(f"交易記錄已導出: {trades_file}")
            
        # 導出信號記錄
        if hasattr(strategy, 'signals_log') and strategy.signals_log:
            signals_df = pd.DataFrame(strategy.signals_log)
            signals_file = output_dir / f"signals_{timestamp}.csv"
            signals_df.to_csv(signals_file, index=False)
            print(f"信號記錄已導出: {signals_file}")
            
        # 導出權益記錄
        if hasattr(strategy, 'equity_log') and strategy.equity_log:
            equity_df = pd.DataFrame(strategy.equity_log)
            equity_file = output_dir / f"equity_{timestamp}.csv"
            equity_df.to_csv(equity_file, index=False)
            print(f"權益記錄已導出: {equity_file}")


if __name__ == "__main__":
    # 測試分析器
    print("分析器模組載入成功")