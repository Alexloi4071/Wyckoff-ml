#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
test_signal.py

快速测试 Binance 实时信号系统
"""

from binance_live_signal import BinanceLiveSignal

def test_connection():
    """测试 Binance 连接"""
    print("="*60)
    print("测试 1: Binance 连接")
    print("="*60)
    
    try:
        generator = BinanceLiveSignal()
        print("✅ 连接成功")
        return generator
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return None

def test_fetch_data(generator):
    """测试数据获取"""
    print("\n" + "="*60)
    print("测试 2: 获取 OHLCV 数据")
    print("="*60)
    
    try:
        df = generator.fetch_ohlcv(limit=100)
        if df is not None and len(df) > 0:
            print(f"✅ 成功获取 {len(df)} 根K线")
            print(f"   时间范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"   最新价格: ${df['close'].iloc[-1]:,.2f}")
            return df
        else:
            print("❌ 数据为空")
            return None
    except Exception as e:
        print(f"❌ 获取失败: {e}")
        return None

def test_calculate_features(generator, df):
    """测试特征计算"""
    print("\n" + "="*60)
    print("测试 3: 计算技术指标")
    print("="*60)
    
    try:
        features = generator.calculate_features(df)
        print(f"✅ 成功计算 {len(features.columns)} 个特征")
        
        # 显示最新指标
        latest = features.iloc[-1]
        print(f"\n   最新指标:")
        print(f"   - RSI: {latest['rsi']:.2f}")
        print(f"   - 成交量百分位: {latest['volume_percentile']:.2%}")
        print(f"   - 波动率百分位: {latest['volatility_percentile']:.2%}")
        print(f"   - 布林带位置: {latest['bb_position']:.2%}")
        
        return features
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_calculate_labels(generator, df):
    """测试标签计算"""
    print("\n" + "="*60)
    print("测试 4: 计算预测标签")
    print("="*60)
    
    try:
        labels = generator.calculate_labels(df)
        print(f"✅ 成功计算 {len(labels.columns)} 个标签")
        
        # 显示最新标签
        latest = labels.iloc[-1]
        print(f"\n   最新标签:")
        print(f"   - 未来收益预测: {latest['y_bin_ret_log_p60_h10']:.2f}")
        print(f"   - 突破信号: {latest['y_brk_lb60_k2_p0']:.2f}")
        print(f"   - 波动率状态: {latest['y_regime_vol70_ma50']:.0f}")
        
        return labels
    except Exception as e:
        print(f"❌ 计算失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_generate_signal(generator):
    """测试信号生成"""
    print("\n" + "="*60)
    print("测试 5: 生成交易信号")
    print("="*60)
    
    try:
        signal_info = generator.generate_trading_signal()
        if signal_info:
            print("✅ 信号生成成功")
            generator.print_signal(signal_info)
            return signal_info
        else:
            print("❌ 信号生成失败")
            return None
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """主测试函数"""
    print("\n" + "🚀 "*20)
    print("Binance 实时信号系统 - 测试程序")
    print("🚀 "*20 + "\n")
    
    # 测试1: 连接
    generator = test_connection()
    if generator is None:
        print("\n❌ 测试失败: 无法连接到 Binance")
        return
    
    # 测试2: 获取数据
    df = test_fetch_data(generator)
    if df is None:
        print("\n❌ 测试失败: 无法获取数据")
        return
    
    # 测试3: 计算特征
    features = test_calculate_features(generator, df)
    if features is None:
        print("\n❌ 测试失败: 无法计算特征")
        return
    
    # 测试4: 计算标签
    labels = test_calculate_labels(generator, df)
    if labels is None:
        print("\n❌ 测试失败: 无法计算标签")
        return
    
    # 测试5: 生成信号
    signal_info = test_generate_signal(generator)
    if signal_info is None:
        print("\n❌ 测试失败: 无法生成信号")
        return
    
    # 全部测试通过
    print("\n" + "="*60)
    print("✅ 所有测试通过！系统运行正常")
    print("="*60)
    print("\n💡 提示:")
    print("   - 运行 'python binance_live_signal.py --once' 查看实时信号")
    print("   - 运行 'python binance_live_signal.py' 持续监控")
    print()

if __name__ == "__main__":
    main()
