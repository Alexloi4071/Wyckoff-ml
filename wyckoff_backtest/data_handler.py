#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
data_handler.py

ML數據處理器 - 處理特徵和標籤數據
- 載入和合併OHLCV、特徵、標籤數據
- 轉換為Backtrader可用格式
- 數據對齊和清洗
- 支援多種數據格式
"""

import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MLDataFeed:
    """
    ML數據饋送器
    
    負責載入和處理ML所需的多種數據：
    - OHLCV價格數據
    - 特徵數據 (features)
    - 標籤數據 (labels)
    """
    
    def __init__(self, data_path, features_path, labels_path, 
                 datetime_col='ts', ohlcv_datetime_col=None, 
                 features_datetime_col=None, labels_datetime_col=None,
                 start_date=None, end_date=None):
        """
        初始化數據饋送器
        
        Args:
            data_path: OHLCV數據路徑
            features_path: 特徵數據路徑
            labels_path: 標籤數據路徑
            datetime_col: 時間欄位名稱 (向後兼容)
            ohlcv_datetime_col: OHLCV數據時間列名
            features_datetime_col: 特徵數據時間列名
            labels_datetime_col: 標籤數據時間列名
            start_date: 開始日期
            end_date: 結束日期
        """
        self.data_path = data_path
        self.features_path = features_path
        self.labels_path = labels_path
        
        # 支持新的分別指定時間列名，或使用舊的統一列名
        self.ohlcv_datetime_col = ohlcv_datetime_col or datetime_col
        self.features_datetime_col = features_datetime_col or datetime_col
        self.labels_datetime_col = labels_datetime_col or datetime_col
        
        self.start_date = start_date
        self.end_date = end_date
        
        # 數據容器
        self.ohlcv_data = None
        self.features = None
        self.labels = None
        self.aligned_data = None
        
        # 載入所有數據
        self._load_all_data()
        
    def _load_all_data(self):
        """載入所有數據並對齊"""
        print(f"[數據載入] 開始載入數據...")
        
        # 1. 載入OHLCV數據
        self._load_ohlcv_data()
        
        # 2. 載入特徵數據
        self._load_features()
        
        # 3. 載入標籤數據
        self._load_labels()
        
        # 4. 對齊所有數據
        self._align_data()
        
        print(f"[數據載入] 完成，最終數據量: {len(self.aligned_data)} 筆")
        
    def _load_ohlcv_data(self):
        """載入OHLCV價格數據（支持CSV和Parquet）"""
        try:
            # 根據文件擴展名選擇讀取方式
            if self.data_path.endswith('.parquet'):
                df = pd.read_parquet(self.data_path)
            else:
                df = pd.read_csv(self.data_path)
            
            # 處理時間欄位
            if self.ohlcv_datetime_col in df.columns:
                df[self.ohlcv_datetime_col] = pd.to_datetime(df[self.ohlcv_datetime_col])
                # 統一時區處理：如果沒有時區信息，添加UTC時區
                if df[self.ohlcv_datetime_col].dt.tz is None:
                    df[self.ohlcv_datetime_col] = df[self.ohlcv_datetime_col].dt.tz_localize('UTC')
                df = df.set_index(self.ohlcv_datetime_col)
            else:
                # 如果沒有指定時間欄位，假設index就是時間
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
            # 確保有必要的OHLCV欄位
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"OHLCV數據缺少欄位: {missing_cols}")
            
            # 排序並過濾日期範圍
            df = df.sort_index()
            
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize('UTC')
                df = df[df.index >= start_dt]
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')
                df = df[df.index <= end_dt]
                
            self.ohlcv_data = df[required_cols].copy()
            
            print(f"[OHLCV數據] 載入成功，共 {len(self.ohlcv_data)} 筆數據")
            
        except Exception as e:
            raise Exception(f"載入OHLCV數據失敗: {e}")
            
    def _load_features(self):
        """載入特徵數據（支持CSV和Parquet）"""
        try:
            # 根據文件擴展名選擇讀取方式
            if self.features_path.endswith('.parquet'):
                df = pd.read_parquet(self.features_path)
            else:
                df = pd.read_csv(self.features_path)
            
            # 處理時間索引
            if self.features_datetime_col in df.columns:
                df[self.features_datetime_col] = pd.to_datetime(df[self.features_datetime_col])
                # 統一時區處理
                if df[self.features_datetime_col].dt.tz is None:
                    df[self.features_datetime_col] = df[self.features_datetime_col].dt.tz_localize('UTC')
                df = df.set_index(self.features_datetime_col)
            else:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
            # 排序
            df = df.sort_index()
            
            # 過濾日期範圍
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize('UTC')
                df = df[df.index >= start_dt]
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')
                df = df[df.index <= end_dt]
                
            self.features = df.copy()
            
            print(f"[特徵數據] 載入成功，共 {len(self.features)} 筆數據，{len(self.features.columns)} 個特徵")
            
        except Exception as e:
            raise Exception(f"載入特徵數據失敗: {e}")
            
    def _load_labels(self):
        """載入標籤數據（支持CSV和Parquet）"""
        try:
            # 根據文件擴展名選擇讀取方式
            if self.labels_path.endswith('.parquet'):
                df = pd.read_parquet(self.labels_path)
            else:
                df = pd.read_csv(self.labels_path)
            
            # 處理時間索引
            if self.labels_datetime_col in df.columns:
                df[self.labels_datetime_col] = pd.to_datetime(df[self.labels_datetime_col])
                # 統一時區處理
                if df[self.labels_datetime_col].dt.tz is None:
                    df[self.labels_datetime_col] = df[self.labels_datetime_col].dt.tz_localize('UTC')
                df = df.set_index(self.labels_datetime_col)
            else:
                df.index = pd.to_datetime(df.index)
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
            # 排序
            df = df.sort_index()
            
            # 過濾日期範圍
            if self.start_date:
                start_dt = pd.to_datetime(self.start_date)
                if start_dt.tz is None:
                    start_dt = start_dt.tz_localize('UTC')
                df = df[df.index >= start_dt]
            if self.end_date:
                end_dt = pd.to_datetime(self.end_date)
                if end_dt.tz is None:
                    end_dt = end_dt.tz_localize('UTC')
                df = df[df.index <= end_dt]
                
            self.labels = df.copy()
            
            print(f"[標籤數據] 載入成功，共 {len(self.labels)} 筆數據，{len(self.labels.columns)} 個標籤")
            
        except Exception as e:
            raise Exception(f"載入標籤數據失敗: {e}")
            
    def _align_data(self):
        """對齊所有數據"""
        try:
            # 找到共同的時間索引
            common_index = self.ohlcv_data.index.intersection(
                self.features.index
            ).intersection(
                self.labels.index
            )
            
            if len(common_index) == 0:
                raise ValueError("沒有找到共同的時間索引")
                
            # 對齊數據
            self.ohlcv_data = self.ohlcv_data.loc[common_index]
            self.features = self.features.loc[common_index]
            self.labels = self.labels.loc[common_index]
            
            # 合併為一個DataFrame
            self.aligned_data = pd.concat([
                self.ohlcv_data,
                self.features,
                self.labels
            ], axis=1)
            
            # 處理缺失值
            self.aligned_data = self.aligned_data.ffill().dropna()
            
            print(f"[數據對齊] 對齊後共 {len(self.aligned_data)} 筆有效數據")
            
        except Exception as e:
            raise Exception(f"數據對齊失敗: {e}")
            
    def get_backtrader_data(self):
        """
        獲取Backtrader可用的數據源
        
        Returns:
            bt.feeds.PandasData: Backtrader數據源
        """
        if self.aligned_data is None:
            raise ValueError("數據尚未載入或對齊")
            
        # 準備Backtrader數據
        bt_data = self.aligned_data[['open', 'high', 'low', 'close', 'volume']].copy()
        
        # 確保數據類型正確
        for col in bt_data.columns:
            bt_data[col] = pd.to_numeric(bt_data[col], errors='coerce')
            
        # 移除任何剩餘的NaN
        bt_data = bt_data.dropna()
        
        # 創建Backtrader數據源
        data_feed = bt.feeds.PandasData(
            dataname=bt_data,
            datetime=None,  # 使用index作為datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )
        
        return data_feed
        
    def get_feature_at_index(self, index):
        """
        獲取指定索引位置的特徵
        
        Args:
            index: 數據索引
            
        Returns:
            pd.Series: 特徵數據
        """
        if index < len(self.features):
            return self.features.iloc[index]
        return None
        
    def get_label_at_index(self, index):
        """
        獲取指定索引位置的標籤
        
        Args:
            index: 數據索引
            
        Returns:
            pd.Series: 標籤數據
        """
        if index < len(self.labels):
            return self.labels.iloc[index]
        return None
        
    def get_data_info(self):
        """
        獲取數據信息摘要
        
        Returns:
            dict: 數據信息
        """
        if self.aligned_data is None:
            return {}
            
        info = {
            'total_records': len(self.aligned_data),
            'date_range': {
                'start': self.aligned_data.index.min().strftime('%Y-%m-%d'),
                'end': self.aligned_data.index.max().strftime('%Y-%m-%d')
            },
            'ohlcv_columns': list(self.ohlcv_data.columns),
            'feature_columns': list(self.features.columns),
            'label_columns': list(self.labels.columns),
            'feature_count': len(self.features.columns),
            'label_count': len(self.labels.columns)
        }
        
        return info
        
    def validate_data_quality(self):
        """
        驗證數據質量
        
        Returns:
            dict: 數據質量報告
        """
        if self.aligned_data is None:
            return {'status': 'error', 'message': '數據未載入'}
            
        report = {
            'status': 'success',
            'total_records': len(self.aligned_data),
            'missing_data': {},
            'data_types': {},
            'value_ranges': {}
        }
        
        # 檢查缺失數據
        for col in self.aligned_data.columns:
            missing_count = self.aligned_data[col].isna().sum()
            if missing_count > 0:
                report['missing_data'][col] = missing_count
                
        # 檢查數據類型
        for col in self.aligned_data.columns:
            report['data_types'][col] = str(self.aligned_data[col].dtype)
            
        # 檢查數值範圍 (僅數值列)
        numeric_cols = self.aligned_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            series = self.aligned_data[col]
            report['value_ranges'][col] = {
                'min': float(series.min()),
                'max': float(series.max()),
                'mean': float(series.mean()),
                'std': float(series.std())
            }
            
        # 檢查是否有異常值
        anomalies = []
        for col in ['open', 'high', 'low', 'close']:
            if col in self.aligned_data.columns:
                # 檢查價格是否為正數
                if (self.aligned_data[col] <= 0).any():
                    anomalies.append(f'{col}列包含非正數值')
                    
                # 檢查high >= low
                if col in ['high', 'low'] and 'high' in self.aligned_data.columns and 'low' in self.aligned_data.columns:
                    if (self.aligned_data['high'] < self.aligned_data['low']).any():
                        anomalies.append('存在high < low的異常數據')
                        
        if anomalies:
            report['anomalies'] = anomalies
            report['status'] = 'warning'
            
        return report
        
    def export_aligned_data(self, filepath):
        """
        導出對齊後的完整數據
        
        Args:
            filepath: 導出文件路徑
        """
        if self.aligned_data is None:
            raise ValueError("沒有數據可導出")
            
        self.aligned_data.to_csv(filepath)
        print(f"[數據導出] 已導出到: {filepath}")


def test_data_handler():
    """測試數據處理器"""
    # 測試用例
    config = {
        'data_path': 'data/btcusdt_1m_3years.csv',
        'features_path': 'out/BTCUSDT/loose/1H/btcusdt_loose_features_1H.csv',
        'labels_path': 'out/BTCUSDT/loose/1H/btcusdt_loose_labels_1H.csv',
        'start_date': '2022-01-01',
        'end_date': '2023-12-31'
    }
    
    try:
        # 創建數據處理器
        ml_data = MLDataFeed(**config)
        
        # 驗證數據質量
        quality_report = ml_data.validate_data_quality()
        print("\n數據質量報告:")
        for key, value in quality_report.items():
            print(f"{key}: {value}")
            
        # 獲取數據信息
        info = ml_data.get_data_info()
        print(f"\n數據信息摘要:")
        for key, value in info.items():
            print(f"{key}: {value}")
            
        # 獲取Backtrader數據源
        bt_data = ml_data.get_backtrader_data()
        print(f"\nBacktrader數據源創建成功")
        
        return ml_data
        
    except Exception as e:
        print(f"測試失敗: {e}")
        return None


if __name__ == "__main__":
    # 運行測試
    ml_data = test_data_handler()