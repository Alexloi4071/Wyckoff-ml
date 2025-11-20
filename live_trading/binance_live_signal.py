#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
binance_live_signal.py

实时从 Binance 获取数据并生成 V5.1 策略交易信号

功能:
- 实时获取 Binance 1H K线数据
- 计算技术指标特征
- 计算标签（预测信号）
- 根据 V5.1 策略生成交易信号
- 输出交易建议
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import yaml
import warnings
warnings.filterwarnings('ignore')

class BinanceLiveSignal:
    """
    Binance 实时交易信号生成器
    """
    
    def __init__(self, config_path='live_config.yaml'):
        """
        初始化
        
        Args:
            config_path: 配置文件路径
        """
        self.config = self._load_config(config_path)
        self.exchange = self._init_exchange()
        self.symbol = self.config['symbol']
        self.timeframe = self.config['timeframe']
        
        # 策略配置
        self.strategy_cfg = self.config['strategy']
        
        print(f"[初始化] Binance 实时信号系统")
        print(f"[交易对] {self.symbol}")
        print(f"[时间框架] {self.timeframe}")
        
    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            print(f"[警告] 配置文件不存在，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self):
        """获取默认配置（V5.1策略）"""
        return {
            'symbol': 'BTC/USDT',
            'timeframe': '1h',
            'lookback_bars': 200,  # 获取历史数据量
            
            'strategy': {
                'signals': {
                    'primary_signal': {
                        'label_col': 'y_bin_ret_log_p60_h10',
                        'threshold': 0.35,
                        'weight': 0.6
                    },
                    'secondary_signals': [
                        {
                            'label_col': 'y_brk_lb60_k2_p0',
                            'threshold': 0.35,
                            'weight': 0.25
                        },
                        {
                            'label_col': 'y_regime_vol70_ma50',
                            'threshold': 2,
                            'weight': 0.15
                        }
                    ]
                },
                'feature_filters': {
                    'volume_filter': {
                        'enabled': True,
                        'feature_col': 'volume_percentile',
                        'min_threshold': 0.2
                    },
                    'volatility_filter': {
                        'enabled': True,
                        'feature_col': 'volatility_percentile',
                        'min_threshold': 0.15,
                        'max_threshold': 0.95
                    },
                    'trend_filter': {
                        'enabled': True,
                        'feature_col': 'rsi',
                        'min_threshold': 30,
                        'max_threshold': 80
                    }
                },
                'risk': {
                    'stop_loss': 0.012,
                    'take_profit': 0.045
                }
            }
        }
    
    def _init_exchange(self):
        """初始化 Binance 交易所连接"""
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        
        # 测试连接
        try:
            exchange.load_markets()
            print(f"[连接] Binance 连接成功")
        except Exception as e:
            print(f"[错误] Binance 连接失败: {e}")
            
        return exchange
    
    def fetch_ohlcv(self, limit=None):
        """
        获取 OHLCV 数据
        
        Args:
            limit: 获取的K线数量
            
        Returns:
            pd.DataFrame: OHLCV 数据
        """
        if limit is None:
            limit = self.config.get('lookback_bars', 200)
        
        try:
            # 获取数据
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol,
                self.timeframe,
                limit=limit
            )
            
            # 转换为 DataFrame
            df = pd.DataFrame(
                ohlcv,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
            )
            
            # 转换时间戳
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('datetime')
            
            print(f"[数据] 获取 {len(df)} 根K线，最新时间: {df.index[-1]}")
            
            return df
            
        except Exception as e:
            print(f"[错误] 获取数据失败: {e}")
            return None
    
    def calculate_features(self, df):
        """
        计算技术指标特征
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: 包含特征的 DataFrame
        """
        features = df.copy()
        
        # 1. RSI
        features['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        # 2. 成交量百分位
        features['volume_percentile'] = df['volume'].rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 3. 波动率百分位
        returns = df['close'].pct_change()
        volatility = returns.rolling(20).std()
        features['volatility_percentile'] = volatility.rolling(60).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1]
        )
        
        # 4. ATR (Average True Range)
        features['atr'] = self._calculate_atr(df, period=14)
        features['atr_pct'] = features['atr'] / df['close']
        
        # 5. 移动平均线
        features['ma_20'] = df['close'].rolling(20).mean()
        features['ma_50'] = df['close'].rolling(50).mean()
        features['ma_200'] = df['close'].rolling(200).mean()
        
        # 6. 布林带
        bb_period = 20
        bb_std = 2
        ma = df['close'].rolling(bb_period).mean()
        std = df['close'].rolling(bb_period).std()
        features['bb_upper'] = ma + (std * bb_std)
        features['bb_lower'] = ma - (std * bb_std)
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # 7. MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_hist'] = features['macd'] - features['macd_signal']
        
        return features
    
    def calculate_labels(self, df):
        """
        计算标签（预测信号）- 修复版：不使用未来数据
        
        Args:
            df: 包含特征的 DataFrame
            
        Returns:
            pd.DataFrame: 包含标签的 DataFrame
        """
        labels = pd.DataFrame(index=df.index)
        
        # 1. 主信号: y_bin_ret_log_p60_h10
        # ✅ 修复：使用历史10根K线的动量，而非未来收益
        historical_returns = np.log(df['close'] / df['close'].shift(10))
        # 使用历史数据计算阈值
        threshold_60 = historical_returns.shift(1).rolling(100).quantile(0.6)
        labels['y_bin_ret_log_p60_h10'] = (historical_returns > threshold_60).astype(float)
        
        # 2. 辅助信号1: y_brk_lb60_k2_p0
        # ✅ 修复：使用历史最高点（shift(1)确保不使用当前bar）
        lookback_high = df['high'].shift(1).rolling(60).max()
        atr = self._calculate_atr(df, period=14)
        breakout_threshold = lookback_high + 2 * atr
        labels['y_brk_lb60_k2_p0'] = (df['close'] > breakout_threshold).astype(float)
        
        # 3. 辅助信号2: y_regime_vol70_ma50
        # ✅ 修复：波动率状态（使用历史数据）
        volatility = df['close'].pct_change().rolling(20).std()
        vol_ma50 = volatility.rolling(50).mean()
        vol_threshold_high = vol_ma50 * 1.5
        vol_threshold_low = vol_ma50 * 0.5
        
        labels['y_regime_vol70_ma50'] = 1  # 默认中波动
        labels.loc[volatility > vol_threshold_high, 'y_regime_vol70_ma50'] = 2  # 高波动
        labels.loc[volatility < vol_threshold_low, 'y_regime_vol70_ma50'] = 0  # 低波动
        
        return labels
    
    def _calculate_rsi(self, prices, period=14):
        """计算 RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, df, period=14):
        """计算 ATR"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        return atr
    
    def calculate_signal_strength(self, features, labels):
        """
        计算信号强度（V5.1策略）
        
        Args:
            features: 特征数据
            labels: 标签数据
            
        Returns:
            float: 信号强度 [-1.0, 1.0]
        """
        signal_strength = 0.0
        
        # 1. 主信号
        primary_cfg = self.strategy_cfg['signals']['primary_signal']
        primary_col = primary_cfg['label_col']
        primary_threshold = primary_cfg['threshold']
        primary_weight = primary_cfg['weight']
        
        if primary_col in labels and not pd.isna(labels[primary_col]):
            if labels[primary_col] > primary_threshold:
                signal_strength += primary_weight
        
        # 2. 辅助信号
        secondary_signals = self.strategy_cfg['signals'].get('secondary_signals', [])
        for signal in secondary_signals:
            signal_col = signal['label_col']
            threshold = signal['threshold']
            weight = signal['weight']
            
            if signal_col in labels and not pd.isna(labels[signal_col]):
                if labels[signal_col] > threshold:
                    signal_strength += weight
                else:
                    signal_strength -= abs(weight) * 0.2
        
        # 3. 特征过滤
        if not self._check_feature_filters(features):
            signal_strength *= 0.3
        
        # 限制在 [-1, 1] 范围
        return np.clip(signal_strength, -1.0, 1.0)
    
    def _check_feature_filters(self, features):
        """检查特征过滤条件"""
        feature_filters = self.strategy_cfg.get('feature_filters', {})
        
        for filter_name, filter_cfg in feature_filters.items():
            if not filter_cfg.get('enabled', True):
                continue
            
            feature_col = filter_cfg['feature_col']
            if feature_col not in features or pd.isna(features[feature_col]):
                continue
            
            value = features[feature_col]
            
            # 检查最小阈值
            min_threshold = filter_cfg.get('min_threshold')
            if min_threshold is not None and value < min_threshold:
                return False
            
            # 检查最大阈值
            max_threshold = filter_cfg.get('max_threshold')
            if max_threshold is not None and value > max_threshold:
                return False
        
        return True
    
    def generate_trading_signal(self):
        """
        生成交易信号
        
        Returns:
            dict: 交易信号信息
        """
        # 1. 获取数据
        df = self.fetch_ohlcv()
        if df is None or len(df) == 0:
            return None
        
        # 2. 计算特征
        features_df = self.calculate_features(df)
        
        # 3. 计算标签
        labels_df = self.calculate_labels(features_df)
        
        # 4. 获取最新数据
        latest_features = features_df.iloc[-1]
        latest_labels = labels_df.iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # 5. 计算信号强度
        signal_strength = self.calculate_signal_strength(latest_features, latest_labels)
        
        # 6. 生成交易建议
        signal_info = self._generate_signal_info(
            signal_strength,
            current_price,
            latest_features,
            latest_labels
        )
        
        return signal_info
    
    def _generate_signal_info(self, signal_strength, current_price, features, labels):
        """生成信号信息"""
        # 入场阈值
        entry_threshold = 0.5
        
        # 判断信号类型
        if signal_strength > entry_threshold:
            signal_type = 'BUY'
            signal_color = '🟢'
        elif signal_strength < -0.3:
            signal_type = 'SELL'
            signal_color = '🔴'
        else:
            signal_type = 'HOLD'
            signal_color = '🟡'
        
        # 计算止损止盈价格
        risk_cfg = self.strategy_cfg['risk']
        stop_loss_pct = risk_cfg['stop_loss']
        take_profit_pct = risk_cfg['take_profit']
        
        stop_loss_price = current_price * (1 - stop_loss_pct)
        take_profit_price = current_price * (1 + take_profit_pct)
        
        # 构建信号信息
        signal_info = {
            'timestamp': datetime.now(),
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'current_price': current_price,
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'features': {
                'rsi': features.get('rsi'),
                'volume_percentile': features.get('volume_percentile'),
                'volatility_percentile': features.get('volatility_percentile'),
                'bb_position': features.get('bb_position'),
                'macd_hist': features.get('macd_hist')
            },
            'labels': {
                'y_bin_ret_log_p60_h10': labels.get('y_bin_ret_log_p60_h10'),
                'y_brk_lb60_k2_p0': labels.get('y_brk_lb60_k2_p0'),
                'y_regime_vol70_ma50': labels.get('y_regime_vol70_ma50')
            }
        }
        
        return signal_info
    
    def print_signal(self, signal_info):
        """打印交易信号"""
        if signal_info is None:
            print("[错误] 无法生成信号")
            return
        
        print("\n" + "="*60)
        print(f"📊 {signal_info['symbol']} 实时交易信号 ({signal_info['timeframe']})")
        print("="*60)
        print(f"⏰ 时间: {signal_info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"💰 当前价格: ${signal_info['current_price']:,.2f}")
        print()
        
        # 信号类型
        signal_type = signal_info['signal_type']
        signal_strength = signal_info['signal_strength']
        
        if signal_type == 'BUY':
            print(f"🟢 信号: {signal_type} (强度: {signal_strength:.3f})")
            print(f"   建议: 做多入场")
            print(f"   止损: ${signal_info['stop_loss']:,.2f} (-{self.strategy_cfg['risk']['stop_loss']*100:.1f}%)")
            print(f"   止盈: ${signal_info['take_profit']:,.2f} (+{self.strategy_cfg['risk']['take_profit']*100:.1f}%)")
        elif signal_type == 'SELL':
            print(f"🔴 信号: {signal_type} (强度: {signal_strength:.3f})")
            print(f"   建议: 平仓/观望")
        else:
            print(f"🟡 信号: {signal_type} (强度: {signal_strength:.3f})")
            print(f"   建议: 持有/观望")
        
        print()
        print("📈 技术指标:")
        features = signal_info['features']
        print(f"   RSI: {features['rsi']:.2f}")
        print(f"   成交量百分位: {features['volume_percentile']:.2%}")
        print(f"   波动率百分位: {features['volatility_percentile']:.2%}")
        print(f"   布林带位置: {features['bb_position']:.2%}")
        print(f"   MACD柱: {features['macd_hist']:.4f}")
        
        print()
        print("🎯 预测标签:")
        labels = signal_info['labels']
        print(f"   未来收益预测: {labels['y_bin_ret_log_p60_h10']:.2f}")
        print(f"   突破信号: {labels['y_brk_lb60_k2_p0']:.2f}")
        print(f"   波动率状态: {labels['y_regime_vol70_ma50']:.0f} (0=低, 1=中, 2=高)")
        
        print("="*60)
    
    def run_continuous(self, interval_seconds=3600):
        """
        持续运行，定期生成信号
        
        Args:
            interval_seconds: 更新间隔（秒），默认3600秒（1小时）
        """
        print(f"\n[启动] 实时信号系统")
        print(f"[更新间隔] {interval_seconds}秒 ({interval_seconds/60:.0f}分钟)")
        print(f"[按 Ctrl+C 停止]\n")
        
        try:
            while True:
                # 生成信号
                signal_info = self.generate_trading_signal()
                
                # 打印信号
                self.print_signal(signal_info)
                
                # 等待下一次更新
                print(f"\n⏳ 等待 {interval_seconds/60:.0f} 分钟后更新...")
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n\n[停止] 用户中断")
        except Exception as e:
            print(f"\n[错误] {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance 实时交易信号生成器')
    parser.add_argument('--config', type=str, default='live_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--once', action='store_true',
                       help='只运行一次（不持续）')
    parser.add_argument('--interval', type=int, default=3600,
                       help='更新间隔（秒），默认3600秒（1小时）')
    
    args = parser.parse_args()
    
    # 创建信号生成器
    signal_generator = BinanceLiveSignal(args.config)
    
    if args.once:
        # 只运行一次
        signal_info = signal_generator.generate_trading_signal()
        signal_generator.print_signal(signal_info)
    else:
        # 持续运行
        signal_generator.run_continuous(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
