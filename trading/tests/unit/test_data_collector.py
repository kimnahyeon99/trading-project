"""
DataCollector 클래스에 대한 단위 테스트
"""
import os
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from pathlib import Path

from src.data_collection.data_collector import DataCollector
from src.config.config import TARGET_SYMBOLS

class TestDataCollector(unittest.TestCase):
    """DataCollector 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.api_key = "test_api_key"
        self.test_symbols = ["AAPL", "MSFT"]
        
        # 테스트용 임시 디렉토리 설정
        self.test_data_dir = Path("./test_data")
        if not os.path.exists(self.test_data_dir):
            os.makedirs(self.test_data_dir)
    
    def tearDown(self):
        """테스트 정리"""
        # 테스트 후 임시 파일 정리
        import shutil
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)
    
    @patch('src.data_collection.data_collector.TimeSeries')
    def test_init(self, mock_time_series):
        """초기화 테스트"""
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        
        self.assertEqual(collector.api_key, self.api_key)
        self.assertEqual(collector.symbols, self.test_symbols)
        mock_time_series.assert_called_once_with(key=self.api_key, output_format='pandas')
    
    @patch('src.data_collection.data_collector.TimeSeries')
    def test_collect_daily_data(self, mock_time_series):
        """일별 데이터 수집 테스트"""
        # 목 데이터 설정
        mock_data = pd.DataFrame({
            '1. open': [100.0, 101.0, 102.0],
            '2. high': [105.0, 106.0, 107.0],
            '3. low': [98.0, 99.0, 100.0],
            '4. close': [103.0, 104.0, 105.0],
            '5. volume': [1000, 1100, 1200]
        }, index=pd.date_range('2023-01-01', periods=3))
        
        mock_instance = mock_time_series.return_value
        mock_instance.get_daily.return_value = (mock_data, {})
        
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        result = collector.collect_daily_data("AAPL")
        
        # 결과 검증
        self.assertFalse(result.empty)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['open', 'high', 'low', 'close', 'volume'])
        mock_instance.get_daily.assert_called_once_with(symbol="AAPL", outputsize='full')
    
    @patch('src.data_collection.data_collector.TimeSeries')
    @patch('src.data_collection.data_collector.time.sleep')
    def test_collect_all_symbols(self, mock_sleep, mock_time_series):
        """모든 심볼 데이터 수집 테스트"""
        # 목 데이터 설정
        mock_data1 = pd.DataFrame({
            '1. open': [100.0, 101.0],
            '2. high': [105.0, 106.0],
            '3. low': [98.0, 99.0],
            '4. close': [103.0, 104.0],
            '5. volume': [1000, 1100]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_data2 = pd.DataFrame({
            '1. open': [200.0, 201.0],
            '2. high': [205.0, 206.0],
            '3. low': [198.0, 199.0],
            '4. close': [203.0, 204.0],
            '5. volume': [2000, 2100]
        }, index=pd.date_range('2023-01-01', periods=2))
        
        mock_instance = mock_time_series.return_value
        mock_instance.get_daily.side_effect = [(mock_data1, {}), (mock_data2, {})]
        
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        result = collector.collect_all_symbols()
        
        # 결과 검증
        self.assertEqual(len(result), 2)
        self.assertIn("AAPL", result)
        self.assertIn("MSFT", result)
        self.assertEqual(len(result["AAPL"]), 2)
        self.assertEqual(len(result["MSFT"]), 2)
        self.assertEqual(mock_sleep.call_count, 2)
    
    @patch('src.data_collection.data_collector.TimeSeries')
    @patch('src.data_collection.data_collector.save_to_csv')
    def test_save_data(self, mock_save_to_csv, mock_time_series):
        """데이터 저장 테스트"""
        # 테스트 데이터 생성
        test_data = {
            "AAPL": pd.DataFrame({
                'open': [100.0, 101.0],
                'high': [105.0, 106.0],
                'low': [98.0, 99.0],
                'close': [103.0, 104.0],
                'volume': [1000, 1100]
            }, index=pd.date_range('2023-01-01', periods=2)),
            "MSFT": pd.DataFrame({
                'open': [200.0, 201.0],
                'high': [205.0, 206.0],
                'low': [198.0, 199.0],
                'close': [203.0, 204.0],
                'volume': [2000, 2100]
            }, index=pd.date_range('2023-01-01', periods=2))
        }
        
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        collector.save_data(test_data, subdir="test_subdir")
        
        # save_to_csv 함수 호출 검증
        self.assertEqual(mock_save_to_csv.call_count, 2)
    
    @patch('src.data_collection.data_collector.TimeSeries')
    @patch('src.data_collection.data_collector.os.listdir')
    @patch('src.data_collection.data_collector.os.path.isdir')
    @patch('src.data_collection.data_collector.load_from_csv')
    def test_load_data(self, mock_load_from_csv, mock_isdir, mock_listdir, mock_time_series):
        """데이터 로드 테스트"""
        # 목 설정
        mock_listdir.return_value = ["20230101", "20230102"]
        mock_isdir.return_value = True
        
        # 테스트 데이터 생성
        test_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=2),
            'open': [100.0, 101.0],
            'high': [105.0, 106.0],
            'low': [98.0, 99.0],
            'close': [103.0, 104.0],
            'volume': [1000, 1100]
        })
        mock_load_from_csv.return_value = test_df
        
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        result = collector.load_data("AAPL")
        
        # 결과 검증
        self.assertFalse(result.empty)
        mock_load_from_csv.assert_called_once()
    
    @patch('src.data_collection.data_collector.TimeSeries')
    @patch('src.data_collection.data_collector.DataCollector.collect_all_symbols')
    @patch('src.data_collection.data_collector.DataCollector.save_data')
    def test_collect_and_save(self, mock_save_data, mock_collect_all_symbols, mock_time_series):
        """데이터 수집 및 저장 통합 테스트"""
        # 테스트 데이터 생성
        test_data = {
            "AAPL": pd.DataFrame({
                'open': [100.0, 101.0],
                'high': [105.0, 106.0],
                'low': [98.0, 99.0],
                'close': [103.0, 104.0],
                'volume': [1000, 1100]
            }, index=pd.date_range('2023-01-01', periods=2))
        }
        mock_collect_all_symbols.return_value = test_data
        
        collector = DataCollector(api_key=self.api_key, symbols=self.test_symbols)
        result = collector.collect_and_save()
        
        # 함수 호출 검증
        mock_collect_all_symbols.assert_called_once()
        mock_save_data.assert_called_once_with(test_data)
        self.assertEqual(result, test_data)

if __name__ == "__main__":
    unittest.main() 