#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ml_strategy.py

基於機器學習信號的Backtrader交易策略
- 支援多特徵、多標籤的信號組合
- 靈活的風險管理
- 預留ML模型預測接口
- 完整的交易記錄和分析
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MLSignalStrategy(bt.Strategy):
    """
    機器學習信號策略
    
    Features:
    - 多信號組合決策
    - 動態特徵過濾
    - 靈活的風險管理
    - ATR基礎的動態止損
    - ML模型集成接口
    """
    
    def __init__(self, config, ml_data):
        """
        初始化策略
        
        Args:
            config: 策略配置字典
            ml_data: ML數據處理器實例
        """
        self.config = config
        self.ml_data = ml_data
        self.strategy_cfg = config['strategy']
        self.risk_cfg = self.strategy_cfg['risk']
        
        # 交易狀態
        self.in_position = False
        self.entry_price = None
        self.entry_bar = None
        self.entry_signal_strength = None
        self.stop_price = None
        self.target_price = None
        
        # 移動止盈追蹤
        self.highest_price = None
        self.trailing_tp_active = False
        
        # 記錄系統
        self.trades_log = []
        self.equity_log = []
        self.signals_log = []
        
        # ML模型 (如果啟用)
        self.ml_model = None
        if self.strategy_cfg.get('ml_model', {}).get('enabled', False):
            self._load_ml_model()
            
        print(f"[策略初始化] {self.strategy_cfg['strategy_name']}")
        print(f"[策略初始化] 風險管理: 止損{self.risk_cfg['stop_loss']*100:.1f}%, 止盈{self.risk_cfg['take_profit']*100:.1f}%")
        
    def _load_ml_model(self):
        """載入ML模型 (預留接口)"""
        ml_cfg = self.strategy_cfg['ml_model']
        model_path = ml_cfg.get('model_path')
        
        if model_path:
            try:
                # 根據模型類型載入
                model_type = ml_cfg.get('model_type', 'xgboost')
                
                if model_type == 'xgboost':
                    import xgboost as xgb
                    self.ml_model = xgb.Booster()
                    self.ml_model.load_model(model_path)
                    
                elif model_type == 'lightgbm':
                    import lightgbm as lgb
                    self.ml_model = lgb.Booster(model_file=model_path)
                    
                elif model_type == 'sklearn':
                    import joblib
                    self.ml_model = joblib.load(model_path)
                    
                print(f"[ML模型] 已載入 {model_type} 模型: {model_path}")
                
            except Exception as e:
                print(f"[警告] ML模型載入失敗: {e}")
                self.ml_model = None
        
    def next(self):
        """策略主邏輯 - 每個時間點調用"""
        current_idx = len(self.data) - 1
        current_time = self.data.datetime.datetime(0)
        current_price = self.data.close[0]
        
        # 記錄當前權益
        current_value = self.broker.get_value()
        self.equity_log.append({
            'datetime': current_time,
            'portfolio_value': current_value,
            'cash': self.broker.get_cash(),
            'price': current_price
        })
        
        # 獲取當前特徵和標籤
        features, labels = self._get_current_features_labels(current_idx)
        if features is None or labels is None:
            return
            
        # 計算信號強度
        signal_strength = self._calculate_signal_strength(features, labels)
        
        # 記錄信號
        self.signals_log.append({
            'datetime': current_time,
            'signal_strength': signal_strength,
            'price': current_price,
            'features': dict(features),
            'labels': dict(labels)
        })
        
        # 風險管理檢查
        if self.in_position:
            self._check_risk_management()
            
        # 交易決策
        if not self.in_position:
            self._check_entry_signal(signal_strength, current_price, current_time)
        else:
            self._check_exit_signal(signal_strength, current_price, current_time)
            
    def _get_current_features_labels(self, idx):
        """獲取當前時間點的特徵和標籤"""
        try:
            # 從ml_data獲取對應的特徵和標籤
            if idx < len(self.ml_data.features):
                features = self.ml_data.features.iloc[idx]
                labels = self.ml_data.labels.iloc[idx]
                return features, labels
            return None, None
            
        except Exception as e:
            # print(f"[警告] 獲取特徵標籤失敗: {e}")
            return None, None
            
    def _calculate_signal_strength(self, features, labels):
        """
        計算信號強度
        
        Returns:
            float: 信號強度 (-1 到 1，正數為買入信號，負數為賣出信號)
        """
        signal_strength = 0.0
        
        # 1. 主信號
        primary_signal = self.strategy_cfg['signals']['primary_signal']
        primary_col = primary_signal['label_col']
        primary_threshold = primary_signal['threshold']
        primary_weight = primary_signal['weight']
        
        if primary_col in labels and not pd.isna(labels[primary_col]):
            if labels[primary_col] > primary_threshold:
                signal_strength += primary_weight
        
        # 2. 輔助信號
        secondary_signals = self.strategy_cfg['signals'].get('secondary_signals', [])
        for signal in secondary_signals:
            signal_col = signal['label_col']
            threshold = signal['threshold']
            weight = signal['weight']
            
            if signal_col in labels and not pd.isna(labels[signal_col]):
                if labels[signal_col] > threshold:
                    signal_strength += weight
                else:
                    signal_strength -= abs(weight) * 0.2  # 負向影響較小
        
        # 3. 特徵過濾
        if not self._check_feature_filters(features):
            signal_strength *= 0.3  # 特徵過濾不通過，信號強度大幅削弱
            
        # 4. ML模型預測 (如果啟用)
        if self.ml_model is not None:
            ml_signal = self._get_ml_prediction(features)
            if ml_signal is not None:
                ml_weight = 0.4  # ML模型權重
                signal_strength = signal_strength * (1 - ml_weight) + ml_signal * ml_weight
        
        # 限制在 [-1, 1] 範圍內
        return np.clip(signal_strength, -1.0, 1.0)
        
    def _check_feature_filters(self, features):
        """檢查特徵過濾條件"""
        feature_filters = self.strategy_cfg.get('feature_filters', {})
        
        for filter_name, filter_cfg in feature_filters.items():
            if not filter_cfg.get('enabled', True):
                continue
                
            feature_col = filter_cfg['feature_col']
            if feature_col not in features or pd.isna(features[feature_col]):
                continue
                
            value = features[feature_col]
            
            # 檢查最小閾值
            min_threshold = filter_cfg.get('min_threshold')
            if min_threshold is not None and value < min_threshold:
                return False
                
            # 檢查最大閾值
            max_threshold = filter_cfg.get('max_threshold')
            if max_threshold is not None and value > max_threshold:
                return False
                
        return True
        
    def _get_ml_prediction(self, features):
        """獲取ML模型預測 (預留接口)"""
        try:
            if self.ml_model is None:
                return None
                
            ml_cfg = self.strategy_cfg['ml_model']
            feature_columns = ml_cfg.get('feature_columns', [])
            threshold = ml_cfg.get('prediction_threshold', 0.6)
            
            # 準備特徵數據
            if feature_columns:
                feature_data = features[feature_columns].values.reshape(1, -1)
            else:
                feature_data = features.values.reshape(1, -1)
            
            # 模型預測
            model_type = ml_cfg.get('model_type', 'xgboost')
            
            if model_type == 'xgboost':
                import xgboost as xgb
                dtest = xgb.DMatrix(feature_data)
                prediction = self.ml_model.predict(dtest)[0]
                
            elif model_type == 'lightgbm':
                prediction = self.ml_model.predict(feature_data)[0]
                
            elif model_type == 'sklearn':
                prediction = self.ml_model.predict_proba(feature_data)[0][1]
                
            # 轉換為信號強度
            if prediction > threshold:
                return (prediction - threshold) / (1 - threshold)
            elif prediction < (1 - threshold):
                return (prediction - (1 - threshold)) / (1 - threshold)
            else:
                return 0.0
                
        except Exception as e:
            # print(f"[警告] ML預測失敗: {e}")
            return None
            
    def _check_entry_signal(self, signal_strength, current_price, current_time):
        """檢查入場信號"""
        # 入場門檻
        entry_threshold = 0.5
        
        if signal_strength > entry_threshold:
            # 計算部位大小 (傳入signal_strength用於signal_based sizing)
            position_size = self._calculate_position_size(current_price, signal_strength)
            
            if position_size > 0:
                # 執行買入
                order = self.buy(size=position_size)
                
                if order:
                    self.in_position = True
                    self.entry_price = current_price
                    self.entry_bar = len(self.data)
                    self.entry_signal_strength = signal_strength  # 記錄入場信號強度
                    
                    # 設置止損止盈
                    self._set_stop_target(current_price)
                    
                    # 記錄交易
                    trade_log = {
                        'datetime': current_time,
                        'action': 'BUY',
                        'price': current_price,
                        'size': position_size,
                        'signal_strength': signal_strength,
                        'portfolio_value': self.broker.get_value()
                    }
                    self.trades_log.append(trade_log)
                    
                    print(f"[買入] {current_time.strftime('%Y-%m-%d %H:%M')} | "
                          f"價格: ${current_price:.2f} | "
                          f"數量: {position_size:.4f} | "
                          f"信號強度: {signal_strength:.3f}")
                          
    def _check_exit_signal(self, signal_strength, current_price, current_time):
        """檢查出場信號"""
        # 出場門檻
        exit_threshold = -0.3
        
        if signal_strength < exit_threshold:
            self._execute_sell(current_price, current_time, 'SIGNAL', signal_strength)
            
    def _check_risk_management(self):
        """風險管理檢查"""
        if not self.in_position:
            return
            
        current_price = self.data.close[0]
        current_time = self.data.datetime.datetime(0)
        
        # 更新最高價
        if self.highest_price is None or current_price > self.highest_price:
            self.highest_price = current_price
        
        # 時間止損檢查 (優先級最高)
        time_stop_cfg = self.risk_cfg.get('time_stop', {})
        if time_stop_cfg.get('enabled', False):
            time_stop_triggered, time_stop_reason = self._check_time_stop(current_price)
            if time_stop_triggered:
                self._execute_sell(current_price, current_time, time_stop_reason)
                return
        
        # 增強移動止盈檢查 (優先級第二)
        trailing_tp_cfg = self.risk_cfg.get('trailing_take_profit', {})
        if trailing_tp_cfg.get('enabled', False):
            tp_triggered, tp_reason = self._check_trailing_take_profit(current_price)
            if tp_triggered:
                self._execute_sell(current_price, current_time, tp_reason)
                return
        
        # 止損檢查
        if self.stop_price and current_price <= self.stop_price:
            self._execute_sell(current_price, current_time, 'STOP_LOSS')
            return
            
        # 固定止盈檢查
        if self.target_price and current_price >= self.target_price:
            self._execute_sell(current_price, current_time, 'TAKE_PROFIT')
            return
            
        # 移動止損 (如果啟用)
        trailing_stop = self.risk_cfg.get('trailing_stop', 0)
        if trailing_stop > 0:
            new_stop = current_price * (1 - trailing_stop)
            if self.stop_price is None or new_stop > self.stop_price:
                self.stop_price = new_stop
                
        # ATR動態止損 (如果啟用)
        atr_cfg = self.risk_cfg.get('atr_stop', {})
        if atr_cfg.get('enabled', False):
            self._update_atr_stop(current_price)
            
    def _calculate_position_size(self, current_price, signal_strength=None):
        """計算部位大小"""
        position_cfg = self.strategy_cfg['position']
        available_cash = self.broker.get_cash()
        
        size_method = position_cfg.get('size_method', 'percent')
        
        if size_method == 'percent':
            size_pct = position_cfg.get('size_pct', 0.95)
            position_value = available_cash * size_pct
            position_size = position_value / current_price
            
        elif size_method == 'fixed':
            position_size = position_cfg.get('fixed_size', 1000) / current_price
            
        elif size_method == 'signal_based':
            # V5新增：根據信號強度動態調整倉位
            base_size = position_cfg.get('base_size', 0.5)
            max_size = position_cfg.get('max_size', 0.95)
            
            if signal_strength is not None:
                # 信號強度從0.5到1.0，映射到base_size到max_size
                # signal_strength = 0.5 -> base_size
                # signal_strength = 1.0 -> max_size
                normalized_signal = (signal_strength - 0.5) / 0.5  # 0到1
                normalized_signal = max(0, min(1, normalized_signal))  # 限制範圍
                size_pct = base_size + (max_size - base_size) * normalized_signal
            else:
                size_pct = base_size
            
            position_value = available_cash * size_pct
            position_size = position_value / current_price
            
        elif size_method == 'kelly':
            # Kelly公式計算 (簡化版)
            # 需要歷史勝率和賠率數據
            win_rate = 0.6  # 可從歷史數據計算
            avg_win = 0.05  # 平均盈利率
            avg_loss = 0.03  # 平均虧損率
            
            if avg_loss > 0:
                kelly_pct = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
                kelly_pct = max(0, min(kelly_pct, 0.25))  # 限制最大25%
                position_value = available_cash * kelly_pct
                position_size = position_value / current_price
            else:
                position_size = 0
                
        else:
            position_size = 0
            
        return max(0, position_size)
        
    def _set_stop_target(self, entry_price):
        """設置止損止盈價格"""
        stop_loss_pct = self.risk_cfg['stop_loss']
        take_profit_pct = self.risk_cfg['take_profit']
        
        self.stop_price = entry_price * (1 - stop_loss_pct)
        self.target_price = entry_price * (1 + take_profit_pct)
        
    def _check_time_stop(self, current_price):
        """
        檢查時間止損
        
        Returns:
            (bool, str): (是否觸發, 觸發原因)
        """
        time_stop_cfg = self.risk_cfg.get('time_stop', {})
        
        # 如果沒有入場bar，返回False
        if self.entry_bar is None or self.entry_price is None:
            return False, None
        
        # 計算持倉時間（bar數）
        bars_held = len(self.data) - self.entry_bar
        
        # 最大持倉時間
        max_bars = time_stop_cfg.get('max_bars', 30)
        if bars_held >= max_bars:
            profit_pct = (current_price - self.entry_price) / self.entry_price * 100
            return True, f'TIME_STOP_MAX (持倉{bars_held}bars, 盈虧: {profit_pct:+.2f}%)'
        
        # 最小盈利時間止損
        min_profit_bars = time_stop_cfg.get('min_profit_bars', 20)
        min_profit_threshold = time_stop_cfg.get('min_profit_threshold', 0.005)
        
        if bars_held >= min_profit_bars:
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct < min_profit_threshold:
                return True, f'TIME_STOP_MIN_PROFIT (持倉{bars_held}bars, 盈利不足: {profit_pct*100:+.2f}%)'
        
        return False, None
    
    def _check_trailing_take_profit(self, current_price):
        """
        檢查增強移動止盈
        
        Returns:
            (bool, str): (是否觸發, 觸發原因)
        """
        trailing_tp_cfg = self.risk_cfg.get('trailing_take_profit', {})
        
        # 如果沒有入場價格，返回False
        if self.entry_price is None:
            return False, None
        
        # 計算當前盈利百分比
        profit_pct = (current_price - self.entry_price) / self.entry_price
        
        # 啟動條件：盈利達到activation_pct
        activation_pct = trailing_tp_cfg.get('activation_pct', 0.02)
        if profit_pct >= activation_pct:
            self.trailing_tp_active = True
        
        # 如果移動止盈已啟動且有最高價記錄
        if self.trailing_tp_active and self.highest_price is not None:
            trail_pct = trailing_tp_cfg.get('trail_pct', 0.015)
            
            # 計算從最高點的回撤
            drawdown_from_high = (self.highest_price - current_price) / self.highest_price
            
            # 如果回撤超過trail_pct，觸發止盈
            if drawdown_from_high >= trail_pct:
                profit_locked = (current_price - self.entry_price) / self.entry_price * 100
                return True, f'TRAILING_TP (鎖定利潤: {profit_locked:.2f}%)'
        
        return False, None
    
    def _update_atr_stop(self, current_price):
        """更新ATR動態止損"""
        try:
            current_idx = len(self.data) - 1
            features, _ = self._get_current_features_labels(current_idx)
            
            if features is not None:
                atr_cfg = self.risk_cfg['atr_stop']
                atr_feature = atr_cfg['atr_feature']
                atr_multiplier = atr_cfg['atr_multiplier']
                
                if atr_feature in features and not pd.isna(features[atr_feature]):
                    atr_value = features[atr_feature] * current_price
                    new_stop = current_price - (atr_value * atr_multiplier)
                    
                    if self.stop_price is None or new_stop > self.stop_price:
                        self.stop_price = new_stop
                        
        except Exception as e:
            # print(f"[警告] ATR止損更新失敗: {e}")
            pass
            
    def _execute_sell(self, current_price, current_time, reason, signal_strength=None):
        """執行賣出操作"""
        if not self.in_position:
            return
            
        # 獲取當前持倉
        position = self.getposition()
        if position.size <= 0:
            return
            
        # 執行賣出
        order = self.sell(size=position.size)
        
        if order:
            # 計算盈虧
            pnl = (current_price - self.entry_price) * position.size
            pnl_pct = (current_price / self.entry_price - 1) * 100
            
            # 記錄交易
            trade_log = {
                'datetime': current_time,
                'action': 'SELL',
                'price': current_price,
                'size': position.size,
                'entry_price': self.entry_price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'reason': reason,
                'signal_strength': signal_strength,
                'bars_held': len(self.data) - self.entry_bar,
                'portfolio_value': self.broker.get_value()
            }
            self.trades_log.append(trade_log)
            
            print(f"[賣出] {current_time.strftime('%Y-%m-%d %H:%M')} | "
                  f"價格: ${current_price:.2f} | "
                  f"盈虧: {pnl_pct:+.2f}% | "
                  f"原因: {reason}")
            
            # 重置狀態
            self.in_position = False
            self.entry_price = None
            self.entry_bar = None
            self.stop_price = None
            self.target_price = None
            self.highest_price = None
            self.trailing_tp_active = False
            
    def notify_order(self, order):
        """訂單狀態通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return
            
        if order.status == order.Completed:
            if order.isbuy():
                pass  # 買入完成已在_check_entry_signal處理
            elif order.issell():
                pass  # 賣出完成已在_execute_sell處理
                
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print(f"[警告] 訂單失敗: {order.status}")
            
    def get_analysis_data(self):
        """獲取策略分析數據"""
        return {
            'trades_log': self.trades_log,
            'equity_log': self.equity_log,
            'signals_log': self.signals_log
        }