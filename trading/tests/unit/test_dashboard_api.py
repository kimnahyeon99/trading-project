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

from flask import Flask
from flask.testing import FlaskClient
import plotly

from src.dashboard.dashboard_app import DashboardApp, create_app
from src.dashboard.data_manager import DataManager
from src.utils.logger import Logger


class TestDashboardAPI(unittest.TestCase):
    """대시보드 API 기능 테스트 클래스"""
    
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
        # 모델 ID 목록
        model_ids = ['sac_v1', 'sac_v2']
        
        # 날짜와 범위 설정
        start_date = '2023-01-01'
        end_date = '2023-06-30'
        
        for model_id in model_ids:
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
                    'total_trades': 50,
                    'profit_factor': 1.5
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
        
        # 트레이딩 통계 디렉토리 생성
        trading_stats_dir = os.path.join(cls.test_dir, 'live_trading')
        os.makedirs(trading_stats_dir, exist_ok=True)
        
        # 트레이딩 통계 데이터
        stats_data = {
            'portfolio_values': [10000.0 + 100.0 * i for i in range(30)],
            'timestamps': [(datetime(2023, 1, 1) + pd.Timedelta(days=i)).strftime('%Y-%m-%d') for i in range(30)],
            'returns': [0.01 * i for i in range(30)],
            'drawdowns': [0.0] + [0.02 * (i % 5) for i in range(1, 30)],
            'trades': [
                {
                    'timestamp': (datetime(2023, 1, 1) + pd.Timedelta(days=i*3)).strftime('%Y-%m-%d'),
                    'symbol': 'BTC',
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'quantity': 0.1,
                    'price': 40000 + 1000 * i,
                    'fee': 20.0,
                    'pnl': 100.0 * i
                }
                for i in range(10)
            ],
            'positions': {
                'BTC': {
                    'quantity': 0.5,
                    'avg_entry_price': 42000.0,
                    'current_price': 45000.0,
                    'unrealized_pnl': 1500.0
                },
                'ETH': {
                    'quantity': 5.0,
                    'avg_entry_price': 2800.0,
                    'current_price': 3000.0,
                    'unrealized_pnl': 1000.0
                }
            },
            'trading_stats': {
                'portfolio_value': 12900.0,
                'cash_balance': 5000.0,
                'equity_value': 7900.0,
                'daily_pnl': 300.0,
                'total_pnl': 2900.0
            }
        }
        
        # 트레이딩 통계 저장
        stats_path = os.path.join(trading_stats_dir, f'trading_stats_{datetime.now().strftime("%Y%m%d")}.json')
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=4)
    
    def setUp(self):
        """각 테스트 전 설정"""
        # DataManager 인스턴스 생성
        self.data_manager = DataManager(data_dir=self.test_dir, logger=self.logger)
        
        # 플라스크 앱 생성
        app = create_app(self.data_manager)
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    def test_backtest_results_api(self):
        """백테스트 결과 API 테스트"""
        # API 호출
        response = self.client.get('/api/backtest-results')
        
        # 상태 코드 확인
        self.assertEqual(response.status_code, 200)
        
        # 응답 데이터 확인
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
        self.assertIn('sac_v1', data)
        self.assertIn('sac_v2', data)
        
        # 결과 데이터 검증
        result = data['sac_v1']
        self.assertEqual(result['model_id'], 'sac_v1')
        self.assertEqual(result['initial_balance'], 10000.0)
        self.assertEqual(result['final_balance'], 12500.0)
        self.assertIn('metrics', result)
        self.assertEqual(result['metrics']['cumulative_return'], 0.25)
        
        # 특정 모델 결과 API 호출
        model_response = self.client.get('/api/backtest-results?model_id=sac_v1')
        self.assertEqual(model_response.status_code, 200)
        model_data = json.loads(model_response.data)
        self.assertIn('sac_v1', model_data)
    
    def test_performance_metrics_api(self):
        """성능 지표 API 테스트"""
        # API 호출
        response = self.client.get('/api/performance-metrics?model_id=sac_v1')
        
        # 상태 코드 확인
        self.assertEqual(response.status_code, 200)
        
        # 응답 데이터 확인
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)
        
        # 성능 지표 검증
        self.assertIn('cumulative_return', data)
        self.assertIn('annual_return', data)
        self.assertIn('sharpe_ratio', data)
        self.assertIn('max_drawdown', data)
        self.assertIn('win_rate', data)
        
        # 값 검증
        self.assertEqual(data['cumulative_return'], 0.25)
        self.assertEqual(data['sharpe_ratio'], 2.5)
        self.assertEqual(data['max_drawdown'], -0.1)
    
    def test_portfolio_chart_api(self):
        """포트폴리오 차트 API 테스트"""
        # API 호출
        response = self.client.get('/api/charts/portfolio')
        
        # 상태 코드 확인
        self.assertEqual(response.status_code, 200)
        
        # 응답 데이터 확인
        data = json.loads(response.data)
        
        # 데이터가 있거나 의미 있는 오류 메시지가 있어야 함
        if 'error' in data:
            # 오류 메시지가 의미 있어야 함
            self.assertIsInstance(data['error'], str)
            self.assertTrue(len(data['error']) > 0)
        else:
            # Plotly 차트 데이터 구조 확인
            self.assertIsInstance(data, dict)
            self.assertIn('data', data)
            self.assertIn('layout', data)
    
    def test_model_comparison_chart_api(self):
        """모델 비교 차트 API 테스트"""
        # API 호출
        response = self.client.get('/api/charts/model-comparison')
        
        # 상태 코드 확인
        self.assertEqual(response.status_code, 200)
        
        # 응답 데이터 확인
        data = json.loads(response.data)
        
        # 데이터가 있거나 의미 있는 오류 메시지가 있어야 함
        if 'error' in data:
            # 오류 메시지가 의미 있어야 함
            self.assertIsInstance(data['error'], str)
            self.assertTrue(len(data['error']) > 0)
        else:
            # Plotly 차트 데이터 구조 확인
            self.assertIsInstance(data, dict)
            self.assertIn('data', data)
            self.assertIn('layout', data)
    
    def test_radar_chart_api(self):
        """레이더 차트 API 테스트"""
        # API 호출
        response = self.client.get('/api/charts/radar')
        
        # 상태 코드 확인
        self.assertEqual(response.status_code, 200)
        
        # 응답 데이터 확인
        data = json.loads(response.data)
        
        # 데이터가 있거나 의미 있는 오류 메시지가 있어야 함
        if 'error' in data:
            # 오류 메시지가 의미 있어야 함
            self.assertIsInstance(data['error'], str)
            self.assertTrue(len(data['error']) > 0)
        else:
            # Plotly 차트 데이터 구조 확인
            self.assertIsInstance(data, dict)
            self.assertIn('data', data)
            self.assertIn('layout', data)
    
    def test_api_error_handling(self):
        """API 오류 처리 테스트"""
        # 잘못된 모델 ID로 요청
        response = self.client.get('/api/performance-metrics?model_id=nonexistent_model')
        
        # 상태 코드 확인 (에러여도 200 반환, 내용에 에러 표시)
        self.assertEqual(response.status_code, 200)
        
        # 데이터가 비어있거나 에러 메시지가 있는지 확인
        data = json.loads(response.data)
        self.assertTrue(len(data) == 0 or 'error' in data)
    
    def test_api_endpoints_exist(self):
        """모든 API 엔드포인트 존재 테스트"""
        # 주요 API 엔드포인트 목록
        endpoints = [
            '/api/trading-stats',
            '/api/models',
            '/api/backtest-results',
            '/api/performance-metrics',
            '/api/market-data',
            '/api/charts/portfolio',
            '/api/charts/returns',
            '/api/charts/drawdown',
            '/api/charts/trade-distribution',
            '/api/charts/price',
            '/api/charts/model-comparison',
            '/api/charts/radar'
        ]
        
        # 각 엔드포인트 테스트
        for endpoint in endpoints:
            response = self.client.get(endpoint)
            # 서버 오류가 없어야 함 (200 또는 404 응답은 허용)
            self.assertNotEqual(response.status_code, 500, f"{endpoint} 엔드포인트에서 서버 오류 발생")


if __name__ == '__main__':
    unittest.main() 