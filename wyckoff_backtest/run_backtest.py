#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_backtest.py

Backtrader ML策略回測主執行腳本
- 整合所有模組
- 支援命令列參數
- 完整的回測流程
- 批次回測功能
- 參數優化準備
"""

# ------ 強制設置 matplotlib 非交互後端 ------
import os
import matplotlib
matplotlib.use('Agg')          # 必須在導入pyplot前設置
import matplotlib.pyplot as plt
plt.ioff()                     # 關閉交互模式
os.environ['MPLBACKEND'] = 'Agg'
# -------------------------------------------

import argparse
import sys
from pathlib import Path
import yaml
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 導入自定義模組
from backtest_engine import BacktestEngine
from analyzers import ReportGenerator

def _create_versioned_directory(base_dir: Path, date_str: str) -> Path:
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


def validate_config(config_path):
    """
    驗證配置文件
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        bool: 驗證結果
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 檢查必要的配置項
        required_sections = ['general', 'broker', 'strategy', 'output']
        for section in required_sections:
            if section not in config:
                print(f"[錯誤] 配置文件缺少 '{section}' 部分")
                return False
                
        # 檢查必要的文件路徑
        general = config['general']
        required_files = [
            ('data_path', '價格數據'),
            ('features_path', '特徵數據'),
            ('labels_path', '標籤數據')
        ]
        
        for file_key, file_desc in required_files:
            if file_key not in general:
                print(f"[錯誤] 配置文件缺少 '{file_key}'")
                return False
                
            file_path = Path(general[file_key])
            if not file_path.exists():
                print(f"[警告] {file_desc}文件不存在: {file_path}")
                # 不直接返回False，讓用戶決定是否繼續
                
        print(f"[驗證] 配置文件驗證通過: {config_path}")
        return True
        
    except Exception as e:
        print(f"[錯誤] 配置文件驗證失敗: {e}")
        return False


def run_single_backtest(config_path, verbose=True):
    """
    執行單次回測
    
    Args:
        config_path: 配置文件路徑
        verbose: 是否顯示詳細輸出
        
    Returns:
        tuple: (dict: 回測結果摘要, BacktestEngine: 引擎對象)
    """
    if not validate_config(config_path):
        return (None, None)
        
    try:
        # 創建回測引擎
        engine = BacktestEngine(config_path)
        
        # 執行回測
        results = engine.run_full_backtest()
        
        if results:
            strategy = results[0]
            
            # 生成報告
            if engine.config['output'].get('generate_report', True):
                report_gen = ReportGenerator(engine.config)
                
                # 使用已創建的版本目錄或創建新的
                output_cfg = engine.config['output']
                
                if '_internal_results_dir' in engine.config:
                    # 使用引擎已創建的版本目錄
                    output_dir = Path(engine.config['_internal_results_dir'])
                elif output_cfg.get('create_timestamp_dir', False):
                    # 如果引擎沒有創建，則自己創建
                    base_output_dir = Path(output_cfg['results_dir'])
                    current_date = datetime.now().strftime("%Y%m%d")
                    output_dir = _create_versioned_directory(base_output_dir, current_date)
                else:
                    output_dir = Path(output_cfg['results_dir'])
                    
                output_dir.mkdir(parents=True, exist_ok=True)
                report_gen.generate_report(results, output_dir)
                
            # 返回結果摘要
            final_value = strategy.broker.get_value()
            initial_value = engine.config['broker']['cash']
            total_return = (final_value / initial_value - 1) * 100
            
            summary = {
                'success': True,
                'initial_value': initial_value,
                'final_value': final_value,
                'total_return': total_return,
                'config_path': config_path,
                'timestamp': datetime.now()
            }
            
            # 添加分析器結果
            if hasattr(strategy.analyzers, 'sharpe'):
                sharpe = strategy.analyzers.sharpe.get_analysis()
                summary['sharpe_ratio'] = sharpe.get('sharperatio', 0)
                
            if hasattr(strategy.analyzers, 'drawdown'):
                dd = strategy.analyzers.drawdown.get_analysis()
                summary['max_drawdown'] = dd.max.drawdown
                
            if hasattr(strategy.analyzers, 'winloss'):
                wl = strategy.analyzers.winloss.get_analysis()
                summary['win_rate'] = wl.get('win_rate', 0)
                summary['profit_factor'] = wl.get('profit_factor', 0)
                summary['total_trades'] = wl.get('total_trades', 0)
                
            return (summary, engine)
            
        else:
            return ({'success': False, 'error': '回測執行失敗'}, None)
            
    except Exception as e:
        print(f"[錯誤] 回測執行異常: {e}")
        return ({'success': False, 'error': str(e)}, None)


def run_batch_backtest(config_paths, output_summary=True):
    """
    執行批次回測
    
    Args:
        config_paths: 配置文件路徑列表
        output_summary: 是否輸出匯總報告
        
    Returns:
        list: 所有回測結果
    """
    results = []
    
    print(f"\n{'='*60}")
    print(f"[批次回測] 開始執行 {len(config_paths)} 個配置")
    print('='*60)
    
    for i, config_path in enumerate(config_paths, 1):
        print(f"\n[{i}/{len(config_paths)}] 執行配置: {config_path}")
        print("-" * 40)
        
        backtest_result = run_single_backtest(config_path)
        if backtest_result and backtest_result[0]:
            result, _ = backtest_result  # 批次回測不需要 engine 對象
            results.append(result)
            
            if result['success']:
                print(f"✓ 完成 - 總收益率: {result['total_return']:+.2f}%")
            else:
                print(f"✗ 失敗 - {result.get('error', '未知錯誤')}")
        else:
            print(f"✗ 失敗 - 配置文件問題")
            
    # 輸出匯總報告
    if output_summary and results:
        print_batch_summary(results)
        
    return results


def print_batch_summary(results):
    """打印批次回測匯總"""
    print(f"\n{'='*60}")
    print("[批次回測匯總]")
    print('='*60)
    
    successful = [r for r in results if r.get('success', False)]
    failed = [r for r in results if not r.get('success', False)]
    
    print(f"總配置數: {len(results)}")
    print(f"成功: {len(successful)}")
    print(f"失敗: {len(failed)}")
    
    if successful:
        print(f"\n[成功回測統計]")
        returns = [r['total_return'] for r in successful]
        print(f"收益率範圍: {min(returns):+.2f}% 到 {max(returns):+.2f}%")
        print(f"平均收益率: {sum(returns)/len(returns):+.2f}%")
        
        # 排序顯示最佳表現
        sorted_results = sorted(successful, key=lambda x: x['total_return'], reverse=True)
        
        print(f"\n[最佳表現前3名]")
        for i, result in enumerate(sorted_results[:3], 1):
            config_name = Path(result['config_path']).stem
            print(f"{i}. {config_name}: {result['total_return']:+.2f}%")
            if 'sharpe_ratio' in result:
                print(f"   夏普比率: {result['sharpe_ratio']:.4f}")
            if 'max_drawdown' in result:
                print(f"   最大回撤: {result['max_drawdown']:.2f}%")


def create_sample_configs():
    """創建示例配置文件"""
    configs_dir = Path("sample_configs")
    configs_dir.mkdir(exist_ok=True)
    
    # 基礎配置
    base_config = {
        'general': {
            'data_path': 'data/btcusdt_1m_3years.csv',
            'features_path': 'out/BTCUSDT/loose/1H/btcusdt_loose_features_1H.csv',
            'labels_path': 'out/BTCUSDT/loose/1H/btcusdt_loose_labels_1H.csv',
            'start_date': '2022-01-01',
            'end_date': '2024-12-31',
            'datetime_col': 'ts',
            'symbol': 'BTCUSDT',
            'timeframe': '1H'
        },
        'broker': {
            'cash': 100000.0,
            'commission': 0.001,
            'slippage': 0.0001
        },
        'output': {
            'results_dir': 'backtest_results',
            'strategy_prefix': 'MLStrategy',
            'generate_report': True,
            'save_trades': True,
            'save_equity_curve': True,
            'plot_results': True,
            'metrics': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor']
        }
    }
    
    # 保守策略
    conservative_config = base_config.copy()
    conservative_config['strategy'] = {
        'strategy_name': 'Conservative_ML_Strategy',
        'signals': {
            'primary_signal': {
                'label_col': 'y_bin_ret_log_p80_h20',
                'threshold': 0.7,
                'weight': 1.0
            },
            'secondary_signals': []
        },
        'feature_filters': {
            'volume_filter': {
                'feature_col': 'f_vol_pct_60',
                'min_threshold': 0.5,
                'enabled': True
            }
        },
        'position': {
            'size_pct': 0.8,
            'max_positions': 1,
            'size_method': 'percent'
        },
        'risk': {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'max_drawdown': 0.10,
            'trailing_stop': 0.015,
            'atr_stop': {
                'enabled': False
            }
        },
        'ml_model': {
            'enabled': False
        }
    }
    
    # 積極策略
    aggressive_config = base_config.copy()
    aggressive_config['strategy'] = {
        'strategy_name': 'Aggressive_ML_Strategy',
        'signals': {
            'primary_signal': {
                'label_col': 'y_bin_ret_log_p70_h10',
                'threshold': 0.5,
                'weight': 1.0
            },
            'secondary_signals': [
                {
                    'label_col': 'y_brk_lb60_k2_p0',
                    'threshold': 0.5,
                    'weight': 0.3
                }
            ]
        },
        'feature_filters': {
            'volume_filter': {
                'feature_col': 'f_vol_pct_60',
                'min_threshold': 0.2,
                'enabled': True
            }
        },
        'position': {
            'size_pct': 0.95,
            'max_positions': 1,
            'size_method': 'percent'
        },
        'risk': {
            'stop_loss': 0.04,
            'take_profit': 0.08,
            'max_drawdown': 0.20,
            'trailing_stop': 0.025,
            'atr_stop': {
                'enabled': True,
                'atr_feature': 'f_atr_14_p',
                'atr_multiplier': 2.0
            }
        },
        'ml_model': {
            'enabled': False
        }
    }
    
    # 保存配置文件
    with open(configs_dir / "conservative.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(conservative_config, f, default_flow_style=False, allow_unicode=True)
        
    with open(configs_dir / "aggressive.yaml", 'w', encoding='utf-8') as f:
        yaml.dump(aggressive_config, f, default_flow_style=False, allow_unicode=True)
        
    print(f"[配置生成] 示例配置文件已創建在: {configs_dir}")
    print("- conservative.yaml: 保守策略")
    print("- aggressive.yaml: 積極策略")


def main():
    """主函數"""
    engine = None  # 初始化 engine 變量
    parser = argparse.ArgumentParser(
        description="Backtrader ML策略回測系統",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  # 單次回測
  python run_backtest.py --config backtest_config.yaml
  
  # 批次回測
  python run_backtest.py --batch sample_configs/
  
  # 創建示例配置
  python run_backtest.py --create-samples
  
  # 驗證配置文件
  python run_backtest.py --validate backtest_config.yaml
        """
    )
    
    parser.add_argument('--config', '-c', type=str,
                       help='單次回測配置文件路徑')
                       
    parser.add_argument('--batch', '-b', type=str,
                       help='批次回測配置文件目錄或文件列表')
                       
    parser.add_argument('--create-samples', action='store_true',
                       help='創建示例配置文件')
                       
    parser.add_argument('--validate', '-v', type=str,
                       help='驗證配置文件')
                       
    parser.add_argument('--verbose', action='store_true',
                       help='顯示詳細輸出')
    
    args = parser.parse_args()
    
    # 顯示標題
    print("=" * 60)
    print("Backtrader ML策略回測系統")
    print("=" * 60)
    
    if args.create_samples:
        create_sample_configs()
        return
        
    if args.validate:
        if validate_config(args.validate):
            print("✓ 配置文件驗證通過")
        else:
            print("✗ 配置文件驗證失敗")
        return
        
    if args.config:
        # 單次回測 - 一次性獲取結果和引擎對象
        print(f"[模式] 單次回測")
        backtest_result = run_single_backtest(args.config, args.verbose)
        
        if backtest_result and backtest_result[0] and backtest_result[0].get('success', False):
            result, engine = backtest_result
            print(f"\n[回測完成] 總收益率: {result['total_return']:+.2f}%")
            # engine 對象已經包含執行過回測的 cerebro，可以直接使用
        else:
            print(f"\n[回測失敗]")
            sys.exit(1)
            
    elif args.batch:
        # 批次回測
        print(f"[模式] 批次回測")
        
        batch_path = Path(args.batch)
        
        if batch_path.is_dir():
            # 目錄模式 - 找所有yaml文件
            config_files = list(batch_path.glob("*.yaml")) + list(batch_path.glob("*.yml"))
            config_paths = [str(f) for f in config_files]
        elif batch_path.is_file():
            # 單文件模式
            config_paths = [str(batch_path)]
        else:
            print(f"[錯誤] 找不到批次路徑: {batch_path}")
            sys.exit(1)
            
        if not config_paths:
            print(f"[錯誤] 沒有找到配置文件")
            sys.exit(1)
            
        results = run_batch_backtest(config_paths)
        
        successful = [r for r in results if r.get('success', False)]
        if successful:
            print(f"\n[批次完成] {len(successful)}/{len(results)} 成功")
        else:
            print(f"\n[批次失敗] 沒有成功的回測")
            sys.exit(1)
            
    else:
        # 沒有指定模式，顯示幫助
        parser.print_help()
        
        # 檢查是否存在默認配置
        default_config = Path("backtest_config.yaml")
        if default_config.exists():
            print(f"\n[提示] 發現默認配置文件: {default_config}")
            print("運行: python run_backtest.py --config backtest_config.yaml")
        else:
            print(f"\n[提示] 創建示例配置:")
            print("運行: python run_backtest.py --create-samples")
    
    return engine  # 返回 engine 對象供後續使用


if __name__ == "__main__":
    engine = main()   # 一定要這樣從 main() 取得 engine
    
    # 只在成功創建 engine 且 cerebro 有效時保存圖表
    if engine is not None and hasattr(engine, 'cerebro') and engine.cerebro is not None:
        from save_bt_fig import save_backtrader_figures
        try:
            # 使用與回測報告相同的版本化目錄
            if '_internal_results_dir' in engine.config:
                output_dir = engine.config['_internal_results_dir']
            else:
                output_dir = engine.config['output']['results_dir']
                
            save_backtrader_figures(
                engine.cerebro,
                output_dir=output_dir,
                file_prefix=engine.config['output'].get('file_prefix','backtest')
            )
        except Exception as e:
            print(f"[警告] K線圖表保存失敗: {e}")
    else:
        print("[信息] 跳過 K線圖表保存 (非單次回測或 cerebro 未初始化)")