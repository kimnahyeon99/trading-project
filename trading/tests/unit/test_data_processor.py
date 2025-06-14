"""
DataProcessor 클래스에 대한 단위 테스트
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import os

from src.preprocessing.data_processor import DataProcessor

class TestDataProcessor(unittest.TestCase):
    """DataProcessor 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 임시 디렉토리
        self.test_dir = tempfile.mkdtemp()
        
        # 테스트 데이터 생성
        self.window_size = 5
        self.processor = DataProcessor(window_size=self.window_size)
        
        # 샘플 주식 데이터 생성
        dates = pd.date_range('2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 210, 100),
            'low': np.random.uniform(90, 190, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 일부 결측치 추가
        self.test_data.loc[self.test_data.index[10], 'close'] = np.nan
        self.test_data.loc[self.test_data.index[20], 'volume'] = np.nan
        
        # 일부 이상치 추가
        self.test_data.loc[self.test_data.index[30], 'close'] = -1
        
        self.test_symbol = 'TEST'
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """초기화 테스트"""
        processor = DataProcessor(window_size=10)
        self.assertEqual(processor.window_size, 10)
        self.assertEqual(len(processor.scalers), 0)
    
    def test_preprocess_data(self):
        """데이터 전처리 테스트"""
        # 전처리 실행
        processed_data = self.processor.preprocess_data(self.test_data)
        
        # 결과 검증
        self.assertEqual(len(processed_data), len(self.test_data))
        self.assertFalse(processed_data.isna().any().any())  # 결측치가 없어야 함
        self.assertTrue((processed_data['close'] >= 0).all())  # 음수 값이 없어야 함
    
    def test_extract_features(self):
        """특성 추출 테스트"""
        # 전처리 데이터 준비
        processed_data = self.processor.preprocess_data(self.test_data)
        
        # 특성 추출 실행
        featured_data = self.processor.extract_features(processed_data)
        
        # 결과 검증
        self.assertGreater(len(featured_data.columns), len(processed_data.columns))
        self.assertTrue('ma5' in featured_data.columns)
        self.assertTrue('rsi' in featured_data.columns)
        self.assertTrue('daily_return' in featured_data.columns)
        self.assertFalse(featured_data.isna().any().any())  # 결측치가 없어야 함
    
    def test_normalize_features(self):
        """특성 정규화 테스트"""
        # 전처리 및 특성 추출 데이터 준비
        processed_data = self.processor.preprocess_data(self.test_data)
        featured_data = self.processor.extract_features(processed_data)
        
        # 정규화 실행
        normalized_data = self.processor.normalize_features(featured_data, self.test_symbol)
        
        # 결과 검증
        self.assertEqual(normalized_data.shape, featured_data.shape)
        self.assertTrue(self.test_symbol in self.processor.scalers)
        
        # 값이 0~1 범위 내에 있는지 확인
        self.assertTrue((normalized_data.min() >= 0).all())
        self.assertTrue((normalized_data.max() <= 1).all())
    
    def test_create_window_samples(self):
        """윈도우 샘플 생성 테스트"""
        # 전처리, 특성 추출, 정규화 데이터 준비
        processed_data = self.processor.preprocess_data(self.test_data)
        featured_data = self.processor.extract_features(processed_data)
        normalized_data = self.processor.normalize_features(featured_data, self.test_symbol)
        
        # 윈도우 샘플 생성 실행
        X, y = self.processor.create_window_samples(normalized_data)
        
        # 결과 검증
        expected_samples = len(normalized_data) - self.window_size
        self.assertEqual(len(X), expected_samples)
        self.assertEqual(len(y), expected_samples)
        self.assertEqual(X.shape[1], self.window_size)  # 윈도우 크기
        self.assertEqual(X.shape[2], len(normalized_data.columns))  # 특성 수
    
    def test_split_data(self):
        """데이터 분할 테스트"""
        # 윈도우 샘플 준비
        processed_data = self.processor.preprocess_data(self.test_data)
        featured_data = self.processor.extract_features(processed_data)
        normalized_data = self.processor.normalize_features(featured_data, self.test_symbol)
        X, y = self.processor.create_window_samples(normalized_data)
        
        # 데이터 분할 실행
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.processor.split_data(X, y)
        
        # 결과 검증
        self.assertEqual(len(X_train) + len(X_valid) + len(X_test), len(X))
        self.assertEqual(len(y_train) + len(y_valid) + len(y_test), len(y))
        
        # 비율 확인 (오차 허용)
        self.assertAlmostEqual(len(X_train) / len(X), 0.7, delta=0.01)
        self.assertAlmostEqual(len(X_valid) / len(X), 0.15, delta=0.01)
        self.assertAlmostEqual(len(X_test) / len(X), 0.15, delta=0.01)
    
    def test_process_symbol_data(self):
        """단일 심볼 데이터 처리 테스트"""
        # 전체 처리 과정 실행
        result = self.processor.process_symbol_data(self.test_data, self.test_symbol)
        
        # 결과 검증
        self.assertIsInstance(result, dict)
        self.assertTrue('processed_data' in result)
        self.assertTrue('featured_data' in result)
        self.assertTrue('normalized_data' in result)
        self.assertTrue('X_train' in result)
        self.assertTrue('X_valid' in result)
        self.assertTrue('X_test' in result)
        self.assertTrue('y_train' in result)
        self.assertTrue('y_valid' in result)
        self.assertTrue('y_test' in result)
    
    def test_process_all_symbols(self):
        """모든 심볼 데이터 처리 테스트"""
        # 여러 심볼 데이터 준비
        data_dict = {
            'TEST1': self.test_data.copy(),
            'TEST2': self.test_data.copy()
        }
        
        # 전체 처리 과정 실행
        results = self.processor.process_all_symbols(data_dict)
        
        # 결과 검증
        self.assertEqual(len(results), 2)
        self.assertTrue('TEST1' in results)
        self.assertTrue('TEST2' in results)
        self.assertTrue('X_train' in results['TEST1'])
        self.assertTrue('X_train' in results['TEST2'])
    
    @patch('src.preprocessing.data_processor.save_to_csv')
    def test_save_processed_data(self, mock_save_to_csv):
        """처리된 데이터 저장 테스트"""
        # 처리 결과 준비
        result = self.processor.process_symbol_data(self.test_data, self.test_symbol)
        results = {self.test_symbol: result}
        
        # 저장 실행
        self.processor.save_processed_data(results, base_dir=self.test_dir)
        
        # 저장 함수 호출 확인
        self.assertEqual(mock_save_to_csv.call_count, 3)  # processed_data, featured_data, normalized_data

if __name__ == "__main__":
    unittest.main() 