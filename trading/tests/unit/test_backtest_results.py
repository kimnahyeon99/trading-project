#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import unittest
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
from unittest.mock import patch, MagicMock

# 상위 디렉토리를 path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.config.config import config
from src.dashboard.data_manager import DataManager
from src.backtesting.backtester import Backtester
from src.utils.logger import Logger


class BacktestResultsTest(unittest.TestCase):
    """백테스트 결과 처리 관련 테스트 클래스"""
    
    @classmethod
    def setUpClass(cls):
        """테스트 클래스 설정"""
        # 테스트 디렉토리 생성
        cls.test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_data')
        cls.backtest_dir = os.path.join(cls.test_dir, 'backtesting')
        os.makedirs(cls.backtest_dir, exist_ok=True)
        
        # 테스트용 로거 생성
        cls.logger = Logger(os.path.join(cls.test_dir, 'test.log'))
        
        # 더미 데이터 생성
        cls.create_dummy_backtest_results()
    
    @classmethod
    def tearDownClass(cls):
        """테스트 클래스 정리"""
        # 테스트 디렉토리 삭제
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_dummy_backtest_results(cls):
        """테스트용 더미 백테스트 결과 생성"""
        # 모델 ID
        model_ids = ['sac_v1', 'sac_v2']
        
        for model_id in model_ids:
            # 날짜와 범위 설정
            start_date = '2023-01-01'
            end_date = '2023-06-30'
            
            # 백테스트 결과 데이터
            backtest_data = {
                'model_id': model_id,
                'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'start_date': start_date,
                'end_date': end_date,
                'initial_balance': 10000.0,
                'final_balance': 12500.0,
                'metrics': {
                    'cumulative_return': 0.25,
                    'annual_return': 0.5,
                    'annual_volatility': 0.2,
                    'sharpe_ratio': 2.5,
                    'max_drawdown': -0.1,
                    'win_rate': 0.6,
                    'total_trades': 50
                },
                'portfolio_values': [10000.0 + 50.0 * i for i in range(50)],
                'timestamps': [(datetime.strptime(start_date, '%Y-%m-%d') + 
                               pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(50)],
                'trades': [
                    {
                        'timestamp': (datetime.strptime(start_date, '%Y-%m-%d') + 
                                    pd.Timedelta(days=i*7)).strftime('%Y-%m-%d'),
                        'action': 1.0 if i % 2 == 0 else -1.0,
                        'price': 100 + i,
                        'shares': 10,
                        'cost': (100 + i) * 10,
                        'portfolio_value': 10000.0 + 500.0 * i
                    }
                    for i in range(5)
                ]
            }
            
            # 저장
            file_path = os.path.join(cls.backtest_dir, f'backtest_{model_id}_{datetime.now().strftime("%Y%m%d")}.json')
            with open(file_path, 'w') as f:
                json.dump(backtest_data, f, indent=4)
    
    def test_backtest_results_loading(self):
        """백테스트 결과 로딩 테스트"""
        # DataManager 인스턴스 생성
        data_manager = DataManager(data_dir=self.test_dir, logger=self.logger)
        
        # 백테스트 결과 로드
        results = data_manager.get_backtest_results(refresh=True)
        
        # 결과 검증
        self.assertIsNotNone(results)
        self.assertTrue(len(results) > 0)
        self.assertIn('sac_v1', results)
        self.assertIn('sac_v2', results)
        
        # 특정 모델 결과 로드
        model_results = data_manager.get_backtest_results(model_id='sac_v1', refresh=True)
        self.assertIsNotNone(model_results)
        self.assertIn('sac_v1', model_results)
        
        # 메트릭스 검증
        self.assertEqual(model_results['sac_v1']['initial_balance'], 10000.0)
        self.assertEqual(model_results['sac_v1']['final_balance'], 12500.0)
        self.assertEqual(model_results['sac_v1']['metrics']['cumulative_return'], 0.25)
        self.assertEqual(model_results['sac_v1']['metrics']['sharpe_ratio'], 2.5)
    
    def test_performance_metrics_calculation(self):
        """성능 지표 계산 테스트"""
        # DataManager 인스턴스 생성
        data_manager = DataManager(data_dir=self.test_dir, logger=self.logger)
        
        # 성능 지표 조회
        metrics = data_manager.get_performance_metrics(model_id='sac_v1')
        
        # 결과 검증
        self.assertIsNotNone(metrics)
        self.assertIn('cumulative_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        
        # 값 검증
        self.assertEqual(metrics['cumulative_return'], 0.25)
        self.assertEqual(metrics['sharpe_ratio'], 2.5)
        self.assertEqual(metrics['max_drawdown'], -0.1)
        self.assertEqual(metrics['win_rate'], 0.6)
    
    def test_direct_json_file_access(self):
        """직접 JSON 파일 액세스 테스트"""
        # 백테스트 디렉토리의 모든 JSON 파일 확인
        json_files = [f for f in os.listdir(self.backtest_dir) if f.endswith('.json')]
        self.assertTrue(len(json_files) >= 2, "백테스트 결과 파일이 충분히 존재해야 합니다")
        
        # 첫 번째 파일 읽기
        first_file = os.path.join(self.backtest_dir, json_files[0])
        with open(first_file, 'r') as f:
            data = json.load(f)
        
        # 데이터 검증
        self.assertIn('model_id', data)
        self.assertIn('metrics', data)
        self.assertIn('portfolio_values', data)
        self.assertIn('timestamps', data)
        
        # 메트릭스 데이터 확인
        self.assertIn('cumulative_return', data['metrics'])
        self.assertIn('sharpe_ratio', data['metrics'])
        
        # 타임스탬프와 포트폴리오 값 길이 일치 확인
        self.assertEqual(len(data['timestamps']), len(data['portfolio_values']))
    
    def test_backtester_save_load_consistency(self):
        """Backtester 저장 및 로드 일관성 테스트"""
        # 테스트용 결과 데이터
        results = {
            'portfolio_values': [10000.0 + 100.0 * i for i in range(30)],
            'returns': [0.01 * i for i in range(30)],
            'timestamps': [(datetime(2023, 1, 1) + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)],
            'metrics': {
                'cumulative_return': 0.3,
                'annual_return': 0.6,
                'annual_volatility': 0.15,
                'sharpe_ratio': 4.0,
                'max_drawdown': -0.05,
                'win_rate': 0.7,
                'total_trades': 20
            },
            'trades': [
                {
                    'timestamp': (datetime(2023, 1, 1) + pd.Timedelta(days=i*3)).strftime('%Y-%m-%d'),
                    'action': 1.0 if i % 2 == 0 else -1.0,
                    'price': 100 + i*5,
                    'shares': 5,
                    'portfolio_value': 10000.0 + 300.0 * i
                }
                for i in range(10)
            ]
        }
        
        # Mock backtester로 결과 저장
        backtester_mock = MagicMock()
        backtester_mock.results = results
        backtester_mock.agent = MagicMock()
        backtester_mock.agent.model_name = 'sac_test'
        
        # 결과 저장
        save_path = os.path.join(self.backtest_dir, 'backtest_sac_test_consistency.json')
        Backtester.save_results(backtester_mock, save_path)
        
        # 파일이 생성되었는지 확인
        self.assertTrue(os.path.exists(save_path), "백테스트 결과 파일이 생성되지 않았습니다")
        
        # 파일 직접 읽기
        with open(save_path, 'r') as f:
            saved_data = json.load(f)
        
        # 데이터 검증
        self.assertEqual(saved_data['model_id'], 'sac_test')
        self.assertEqual(saved_data['metrics']['cumulative_return'], results['metrics']['cumulative_return'])
        self.assertEqual(saved_data['metrics']['sharpe_ratio'], results['metrics']['sharpe_ratio'])
        self.assertEqual(len(saved_data['portfolio_values']), len(results['portfolio_values']))
        
        # DataManager로 로드
        data_manager = DataManager(data_dir=self.test_dir, logger=self.logger)
        loaded_results = data_manager.get_backtest_results(refresh=True)
        
        # 결과 검증
        self.assertIn('sac_test', loaded_results)
        loaded_data = loaded_results['sac_test']
        
        # 주요 지표 비교
        self.assertEqual(loaded_data['metrics']['cumulative_return'], results['metrics']['cumulative_return'])
        self.assertEqual(loaded_data['metrics']['sharpe_ratio'], results['metrics']['sharpe_ratio'])
        self.assertEqual(loaded_data['metrics']['max_drawdown'], results['metrics']['max_drawdown'])
        self.assertEqual(loaded_data['metrics']['win_rate'], results['metrics']['win_rate'])
        
        # 포트폴리오 값 비교
        self.assertEqual(len(loaded_data['portfolio_values']), len(results['portfolio_values']))
        self.assertEqual(loaded_data['portfolio_values'][0], results['portfolio_values'][0])
        self.assertEqual(loaded_data['portfolio_values'][-1], results['portfolio_values'][-1])


if __name__ == '__main__':
    unittest.main() 