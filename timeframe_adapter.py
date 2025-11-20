#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
timeframe_adapter.py
時框自適應參數管理器
"""

import yaml
from typing import Dict, Any
from pathlib import Path

class TimeframeAdapter:
    """時框參數自適應管理"""
    
    # 時框標準化映射
    TIMEFRAME_MAP = {
        '15m': '15min',
        '15min': '15min',
        '1h': '1h',
        '1H': '1h',
        '4h': '4h',
        '4H': '4h',
        '1d': '1D',
        '1D': '1D',
        'daily': '1D'
    }
    
    def __init__(self, config_path: str = None):
        """
        初始化時框適配器
        Args:
            config_path: 時框參數配置文件路徑
        """
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                self.params = yaml.safe_load(f)
        else:
            self.params = self._get_default_params()
    
    def _get_default_params(self) -> Dict:
        """獲取默認參數矩陣"""
        return {
            '15min': {
                'label': {
                    'horizon': 8,
                    'ref_window': 672,
                    'percentile': 65,
                    'method': 'log'
                },
                'features': {
                    'rsi_window': 24,
                    'atr_window': 24,
                    'ma_windows': [48, 96],
                    'bb_window': 40,
                    'bb_k': 2.0,
                    'volume_window': 96
                },
                'risk': {
                    'stop_loss_pct': 2.0,
                    'take_profit_pct': 6.0,
                    'trailing_stop_pct': 1.5,
                    'time_stop_bars': 120,
                    'max_position_pct': 0.95
                },
                'filter': {
                    'min_volume_pct': 30,
                    'rsi_lower': 25,
                    'rsi_upper': 85,
                    'min_signal_strength': 0.40
                }
            },
            '1h': {
                'label': {
                    'horizon': 12,
                    'ref_window': 504,
                    'percentile': 70,
                    'method': 'log'
                },
                'features': {
                    'rsi_window': 14,
                    'atr_window': 14,
                    'ma_windows': [20, 50],
                    'bb_window': 20,
                    'bb_k': 2.0,
                    'volume_window': 60
                },
                'risk': {
                    'stop_loss_pct': 1.5,
                    'take_profit_pct': 5.0,
                    'trailing_stop_pct': 1.2,
                    'time_stop_bars': 30,
                    'max_position_pct': 0.90
                },
                'filter': {
                    'min_volume_pct': 20,
                    'rsi_lower': 30,
                    'rsi_upper': 80,
                    'min_signal_strength': 0.35
                }
            },
            '4h': {
                'label': {
                    'horizon': 10,
                    'ref_window': 378,
                    'percentile': 70,
                    'method': 'log'
                },
                'features': {
                    'rsi_window': 14,
                    'atr_window': 14,
                    'ma_windows': [12, 26],
                    'bb_window': 20,
                    'bb_k': 2.0,
                    'volume_window': 42
                },
                'risk': {
                    'stop_loss_pct': 1.2,
                    'take_profit_pct': 4.0,
                    'trailing_stop_pct': 1.0,
                    'time_stop_bars': 8,
                    'max_position_pct': 0.85
                },
                'filter': {
                    'min_volume_pct': 15,
                    'rsi_lower': 30,
                    'rsi_upper': 75,
                    'min_signal_strength': 0.30
                }
            },
            '1D': {
                'label': {
                    'horizon': 5,
                    'ref_window': 252,
                    'percentile': 75,
                    'method': 'log'
                },
                'features': {
                    'rsi_window': 14,
                    'atr_window': 14,
                    'ma_windows': [20, 50],
                    'bb_window': 20,
                    'bb_k': 2.0,
                    'volume_window': 60
                },
                'risk': {
                    'stop_loss_pct': 0.8,
                    'take_profit_pct': 3.0,
                    'trailing_stop_pct': 0.6,
                    'time_stop_bars': 3,
                    'max_position_pct': 0.75
                },
                'filter': {
                    'min_volume_pct': 10,
                    'rsi_lower': 30,
                    'rsi_upper': 70,
                    'min_signal_strength': 0.25
                }
            }
        }
    
    def get_params(self, timeframe: str) -> Dict[str, Any]:
        """
        獲取指定時框的完整參數
        Args:
            timeframe: 時框字符串 (例如 '15m', '1H', '4h', '1D')
        Returns:
            該時框的參數字典
        """
        # 標準化時框名稱
        std_tf = self.TIMEFRAME_MAP.get(timeframe, timeframe)
        
        if std_tf not in self.params:
            raise ValueError(f"不支持的時框: {timeframe} (標準化為 {std_tf})")
        
        return self.params[std_tf]
    
    def update_config(self, base_config: Dict, timeframe: str) -> Dict:
        """
        用時框特定參數更新基礎配置
        Args:
            base_config: 基礎配置字典
            timeframe: 時框字符串
        Returns:
            更新後的配置字典
        """
        import copy
        config = copy.deepcopy(base_config)
        tf_params = self.get_params(timeframe)
        
        # 更新標籤參數
        if 'labels' in config:
            for label_type in config['labels']:
                if isinstance(config['labels'][label_type], dict):
                    if label_type == 'future_return_binary' and 'label' in tf_params:
                        config['labels'][label_type]['horizons'] = [tf_params['label']['horizon']]
                        config['labels'][label_type]['ref_window'] = tf_params['label']['ref_window']
                        config['labels'][label_type]['pctile'] = tf_params['label']['percentile']
                    
                    if label_type == 'stop_target_hit_first' and 'label' in tf_params:
                        config['labels'][label_type]['horizon'] = tf_params['label']['horizon']
        
        # 更新特徵參數
        if 'features' in config and 'features' in tf_params:
            if 'rsi' in config['features']:
                config['features']['rsi']['window'] = tf_params['features']['rsi_window']
            if 'atr' in config['features']:
                config['features']['atr']['window'] = tf_params['features']['atr_window']
            if 'bbands' in config['features']:
                config['features']['bbands']['window'] = tf_params['features']['bb_window']
            if 'volume_percentile' in config['features']:
                config['features']['volume_percentile']['window'] = tf_params['features']['volume_window']
        
        # 添加時框特定參數(用於回測)
        config['timeframe_params'] = tf_params
        
        return config
    
    @staticmethod
    def get_optimal_timeframes() -> list:
        """返回推薦的時框組合"""
        return ['15min', '1h', '4h', '1D']
    
    def validate_params(self, timeframe: str) -> bool:
        """
        驗證時框參數的合理性
        Args:
            timeframe: 時框字符串
        Returns:
            True如果參數有效
        """
        try:
            params = self.get_params(timeframe)
            
            # 檢查必要字段
            assert 'label' in params
            assert 'features' in params
            assert 'risk' in params
            assert 'filter' in params
            
            # 檢查風險管理邏輯
            assert params['risk']['stop_loss_pct'] > 0
            assert params['risk']['take_profit_pct'] > params['risk']['stop_loss_pct']
            assert 0 < params['risk']['max_position_pct'] <= 1.0
            
            # 檢查過濾條件
            assert 0 < params['filter']['min_volume_pct'] <= 100
            assert params['filter']['rsi_lower'] < params['filter']['rsi_upper']
            
            return True
        except (AssertionError, KeyError) as e:
            print(f"參數驗證失敗: {e}")
            return False


# ============================================
# 使用示例
# ============================================
if __name__ == "__main__":
    adapter = TimeframeAdapter()
    
    # 測試所有時框
    for tf in ['15m', '1h', '4h', '1D']:
        print(f"\n{'='*50}")
        print(f"時框: {tf}")
        print(f"{'='*50}")
        
        params = adapter.get_params(tf)
        print(f"標籤horizon: {params['label']['horizon']}")
        print(f"止損: {params['risk']['stop_loss_pct']}%")
        print(f"止盈: {params['risk']['take_profit_pct']}%")
        print(f"盈虧比: 1:{params['risk']['take_profit_pct']/params['risk']['stop_loss_pct']:.2f}")
        
        # 驗證參數
        is_valid = adapter.validate_params(tf)
        print(f"參數驗證: {'✓ 通過' if is_valid else '✗ 失敗'}")
