#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
optimization.py

參數優化模組 - 支援多種優化方法
- Optuna貝葉斯優化
- 網格搜索
- 隨機搜索  
- 時間序列交叉驗證
- 多目標優化
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union, Callable
import json

warnings.filterwarnings('ignore')

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[警告] Optuna未安裝，貝葉斯優化功能不可用")

from backtest_engine import BacktestEngine


class OptimizationEngine:
    """
    參數優化引擎
    
    支持多種優化方法和策略評估指標
    """
    
    def __init__(self, base_config_path: str, optimization_config: Dict = None):
        """
        初始化優化引擎
        
        Args:
            base_config_path: 基礎配置文件路徑
            optimization_config: 優化配置字典
        """
        self.base_config_path = base_config_path
        self.base_config = self._load_base_config()
        
        # 優化配置
        if optimization_config:
            self.opt_config = optimization_config
        else:
            self.opt_config = self.base_config.get('optimization', {})
            
        # 結果存儲
        self.results = []
        self.best_params = None
        self.best_score = None
        
        # 設置輸出目錄
        self.output_dir = Path("optimization_results")
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"[優化引擎] 初始化完成")
        print(f"[優化方法] {self.opt_config.get('method', 'optuna')}")
        print(f"[優化目標] {self.opt_config.get('objective', 'sharpe_ratio')}")
        
    def _load_base_config(self) -> Dict:
        """載入基礎配置"""
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
            
    def _create_modified_config(self, params: Dict) -> Dict:
        """
        根據參數創建修改後的配置
        
        Args:
            params: 參數字典
            
        Returns:
            Dict: 修改後的配置
        """
        config = self.base_config.copy()
        
        # 應用參數到配置
        for param_name, param_value in params.items():
            self._set_nested_config(config, param_name, param_value)
            
        return config
        
    def _set_nested_config(self, config: Dict, key: str, value):
        """設置嵌套配置值"""
        keys = key.split('.')
        current = config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def _evaluate_strategy(self, params: Dict) -> Dict:
        """
        評估策略參數
        
        Args:
            params: 參數字典
            
        Returns:
            Dict: 評估結果
        """
        try:
            # 創建臨時配置
            temp_config = self._create_modified_config(params)
            
            # 保存臨時配置文件
            temp_config_path = self.output_dir / f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.yaml"
            with open(temp_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(temp_config, f, default_flow_style=False)
                
            try:
                # 執行回測
                engine = BacktestEngine(str(temp_config_path))
                results = engine.run_full_backtest()
                
                if results:
                    strategy = results[0]
                    
                    # 計算評估指標
                    metrics = self._calculate_metrics(strategy, temp_config)
                    
                    # 清理臨時文件
                    temp_config_path.unlink()
                    
                    return {
                        'success': True,
                        'params': params,
                        'metrics': metrics,
                        'primary_score': metrics.get(self.opt_config.get('objective', 'sharpe_ratio'), 0)
                    }
                else:
                    # 清理臨時文件
                    temp_config_path.unlink()
                    return {
                        'success': False,
                        'params': params,
                        'error': '回測失敗'
                    }
                    
            except Exception as e:
                # 清理臨時文件
                if temp_config_path.exists():
                    temp_config_path.unlink()
                raise e
                
        except Exception as e:
            return {
                'success': False,
                'params': params,
                'error': str(e)
            }
            
    def _calculate_metrics(self, strategy, config: Dict) -> Dict:
        """計算策略評估指標"""
        metrics = {}
        
        # 基本收益指標
        final_value = strategy.broker.get_value()
        initial_value = config['broker']['cash']
        total_return = (final_value / initial_value - 1)
        
        metrics['total_return'] = total_return
        metrics['annual_return'] = total_return * 252 / len(strategy.data)  # 粗略估算年化收益
        
        # 夏普比率
        if hasattr(strategy.analyzers, 'sharpe'):
            sharpe = strategy.analyzers.sharpe.get_analysis()
            metrics['sharpe_ratio'] = sharpe.get('sharperatio', 0)
        else:
            metrics['sharpe_ratio'] = 0
            
        # 最大回撤
        if hasattr(strategy.analyzers, 'drawdown'):
            dd = strategy.analyzers.drawdown.get_analysis()
            metrics['max_drawdown'] = dd.max.drawdown / 100  # 轉為小數
        else:
            metrics['max_drawdown'] = 0
            
        # Calmar比率
        if metrics['max_drawdown'] != 0:
            metrics['calmar_ratio'] = metrics['annual_return'] / abs(metrics['max_drawdown'])
        else:
            metrics['calmar_ratio'] = 0
            
        # 交易統計
        if hasattr(strategy.analyzers, 'winloss'):
            wl = strategy.analyzers.winloss.get_analysis()
            metrics['win_rate'] = wl.get('win_rate', 0)
            metrics['profit_factor'] = wl.get('profit_factor', 0)
            metrics['total_trades'] = wl.get('total_trades', 0)
            metrics['avg_trade'] = wl.get('avg_trade', 0)
        else:
            metrics.update({
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'avg_trade': 0
            })
            
        return metrics
        
    def run_optuna_optimization(self, n_trials: int = 100) -> Dict:
        """
        運行Optuna貝葉斯優化
        
        Args:
            n_trials: 試驗次數
            
        Returns:
            Dict: 優化結果
        """
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna未安裝，請先安裝: pip install optuna")
            
        def objective(trial):
            """Optuna目標函數"""
            params = {}
            param_space = self.opt_config.get('param_space', {})
            
            for param_name, param_config in param_space.items():
                if isinstance(param_config, list) and len(param_config) == 3:
                    # [min, max, step] 格式
                    min_val, max_val, step = param_config
                    
                    if isinstance(min_val, float) or isinstance(max_val, float):
                        params[param_name] = trial.suggest_float(
                            param_name, min_val, max_val, step=step
                        )
                    else:
                        params[param_name] = trial.suggest_int(
                            param_name, min_val, max_val, step=step
                        )
                elif isinstance(param_config, list):
                    # 離散選擇
                    params[param_name] = trial.suggest_categorical(
                        param_name, param_config
                    )
                    
            # 評估參數組合
            result = self._evaluate_strategy(params)
            
            if result['success']:
                score = result['primary_score']
                
                # 記錄結果
                self.results.append(result)
                
                # 多目標優化支持
                if self.opt_config.get('multi_objective', False):
                    # 返回多個目標值
                    objectives = self.opt_config.get('objectives', ['sharpe_ratio', 'max_drawdown'])
                    values = []
                    for obj in objectives:
                        if obj == 'max_drawdown':
                            # 最大回撤要最小化，所以取負值
                            values.append(-result['metrics'].get(obj, 0))
                        else:
                            values.append(result['metrics'].get(obj, 0))
                    return values
                else:
                    return score
            else:
                # 失敗的情況返回最差分數
                return -999999
                
        # 創建研究對象
        study_name = f"ml_strategy_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.opt_config.get('multi_objective', False):
            study = optuna.create_study(
                directions=['maximize'] * len(self.opt_config.get('objectives', ['sharpe_ratio'])),
                study_name=study_name
            )
        else:
            study = optuna.create_study(
                direction='maximize',
                study_name=study_name
            )
            
        # 運行優化
        print(f"\n[Optuna優化] 開始優化，試驗次數: {n_trials}")
        study.optimize(objective, n_trials=n_trials)
        
        # 處理結果
        if self.opt_config.get('multi_objective', False):
            best_trials = study.best_trials
            self.best_params = [trial.params for trial in best_trials]
            self.best_score = [trial.values for trial in best_trials]
        else:
            self.best_params = study.best_params
            self.best_score = study.best_value
            
        # 保存優化歷史
        self._save_optimization_results(study, 'optuna')
        
        return {
            'method': 'optuna',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': n_trials,
            'study': study
        }
        
    def run_grid_search(self) -> Dict:
        """
        運行網格搜索優化
        
        Returns:
            Dict: 優化結果
        """
        param_space = self.opt_config.get('param_space', {})
        
        # 生成參數網格
        param_grid = self._generate_parameter_grid(param_space)
        
        print(f"\n[網格搜索] 開始優化，參數組合數: {len(param_grid)}")
        
        best_score = -999999
        best_params = None
        
        for i, params in enumerate(param_grid, 1):
            print(f"[{i}/{len(param_grid)}] 測試參數組合: {params}")
            
            result = self._evaluate_strategy(params)
            
            if result['success']:
                score = result['primary_score']
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                print(f"  分數: {score:.4f}")
            else:
                print(f"  失敗: {result.get('error', '未知錯誤')}")
                
        self.best_params = best_params
        self.best_score = best_score
        
        # 保存結果
        self._save_optimization_results(None, 'grid_search')
        
        return {
            'method': 'grid_search',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'total_combinations': len(param_grid)
        }
        
    def _generate_parameter_grid(self, param_space: Dict) -> List[Dict]:
        """生成參數網格"""
        import itertools
        
        param_names = []
        param_values = []
        
        for param_name, param_config in param_space.items():
            param_names.append(param_name)
            
            if isinstance(param_config, list) and len(param_config) == 3:
                # [min, max, step] 格式
                min_val, max_val, step = param_config
                values = np.arange(min_val, max_val + step, step).tolist()
                param_values.append(values)
            elif isinstance(param_config, list):
                # 離散選擇
                param_values.append(param_config)
                
        # 生成所有組合
        combinations = list(itertools.product(*param_values))
        
        param_grid = []
        for combination in combinations:
            params = dict(zip(param_names, combination))
            param_grid.append(params)
            
        return param_grid
        
    def run_random_search(self, n_trials: int = 50) -> Dict:
        """
        運行隨機搜索優化
        
        Args:
            n_trials: 隨機試驗次數
            
        Returns:
            Dict: 優化結果
        """
        print(f"\n[隨機搜索] 開始優化，試驗次數: {n_trials}")
        
        param_space = self.opt_config.get('param_space', {})
        
        best_score = -999999
        best_params = None
        
        for i in range(n_trials):
            # 隨機生成參數
            params = self._generate_random_params(param_space)
            
            print(f"[{i+1}/{n_trials}] 測試參數組合: {params}")
            
            result = self._evaluate_strategy(params)
            
            if result['success']:
                score = result['primary_score']
                self.results.append(result)
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
                print(f"  分數: {score:.4f}")
            else:
                print(f"  失敗: {result.get('error', '未知錯誤')}")
                
        self.best_params = best_params
        self.best_score = best_score
        
        # 保存結果
        self._save_optimization_results(None, 'random_search')
        
        return {
            'method': 'random_search',
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_trials': n_trials
        }
        
    def _generate_random_params(self, param_space: Dict) -> Dict:
        """生成隨機參數"""
        params = {}
        
        for param_name, param_config in param_space.items():
            if isinstance(param_config, list) and len(param_config) == 3:
                # [min, max, step] 格式
                min_val, max_val, step = param_config
                
                # 計算可能的值
                if isinstance(min_val, float) or isinstance(max_val, float):
                    n_steps = int((max_val - min_val) / step)
                    random_step = np.random.randint(0, n_steps + 1)
                    params[param_name] = min_val + random_step * step
                else:
                    params[param_name] = np.random.randint(min_val, max_val + 1)
                    
            elif isinstance(param_config, list):
                # 離散選擇
                params[param_name] = np.random.choice(param_config)
                
        return params
        
    def _save_optimization_results(self, study=None, method: str = 'unknown'):
        """保存優化結果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存詳細結果
        results_df = pd.DataFrame([
            {
                **result['params'],
                **result['metrics'],
                'success': result['success'],
                'primary_score': result.get('primary_score', 0)
            }
            for result in self.results if result['success']
        ])
        
        if len(results_df) > 0:
            results_file = self.output_dir / f"optimization_results_{method}_{timestamp}.csv"
            results_df.to_csv(results_file, index=False)
            print(f"[結果保存] 詳細結果: {results_file}")
            
        # 保存最佳參數
        best_params_file = self.output_dir / f"best_params_{method}_{timestamp}.yaml"
        best_result = {
            'method': method,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_config': self.opt_config,
            'timestamp': timestamp
        }
        
        with open(best_params_file, 'w', encoding='utf-8') as f:
            yaml.dump(best_result, f, default_flow_style=False)
            
        print(f"[結果保存] 最佳參數: {best_params_file}")
        
        # 如果是Optuna，保存研究對象
        if study is not None and method == 'optuna':
            study_file = self.output_dir / f"optuna_study_{timestamp}.pkl"
            import pickle
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            print(f"[結果保存] Optuna研究: {study_file}")
            
    def create_optimized_config(self, output_path: str = None) -> str:
        """
        創建使用最佳參數的配置文件
        
        Args:
            output_path: 輸出路徑
            
        Returns:
            str: 配置文件路徑
        """
        if self.best_params is None:
            raise ValueError("尚未進行優化或優化失敗")
            
        optimized_config = self._create_modified_config(self.best_params)
        
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f"optimized_config_{timestamp}.yaml"
        else:
            output_path = Path(output_path)
            
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_config, f, default_flow_style=False)
            
        print(f"[配置生成] 最佳配置: {output_path}")
        return str(output_path)
        
    def run_optimization(self) -> Dict:
        """
        根據配置運行相應的優化方法
        
        Returns:
            Dict: 優化結果
        """
        if not self.opt_config.get('enabled', False):
            print("[警告] 優化未啟用")
            return {}
            
        method = self.opt_config.get('method', 'optuna')
        n_trials = self.opt_config.get('n_trials', 100)
        
        if method == 'optuna':
            return self.run_optuna_optimization(n_trials)
        elif method == 'grid':
            return self.run_grid_search()
        elif method == 'random':
            return self.run_random_search(n_trials)
        else:
            raise ValueError(f"不支持的優化方法: {method}")


def main():
    """主函數 - 優化示例"""
    # 示例使用
    base_config = "backtest_config.yaml"
    
    if not Path(base_config).exists():
        print(f"[錯誤] 找不到基礎配置文件: {base_config}")
        return
        
    # 創建優化引擎
    optimizer = OptimizationEngine(base_config)
    
    # 運行優化
    result = optimizer.run_optimization()
    
    if result:
        print(f"\n[優化完成]")
        print(f"最佳參數: {result['best_params']}")
        print(f"最佳分數: {result['best_score']}")
        
        # 創建最佳配置
        optimized_config_path = optimizer.create_optimized_config()
        print(f"最佳配置已保存: {optimized_config_path}")


if __name__ == "__main__":
    main()