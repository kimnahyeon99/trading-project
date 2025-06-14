"""
TradingEnvironment 클래스에 대한 단위 테스트
"""
import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os

from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment

class TestTradingEnvironment(unittest.TestCase):
    """TradingEnvironment 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 데이터 생성
        self.window_size = 10
        self.feature_dim = 5
        
        # 샘플 주식 데이터 생성
        dates = pd.date_range('2023-01-01', periods=100)
        self.test_data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 210, 100),
            'low': np.random.uniform(90, 190, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # 환경 생성
        self.env = TradingEnvironment(
            data=self.test_data,
            window_size=self.window_size,
            initial_balance=10000.0,
            max_trading_units=10,
            transaction_fee_percent=0.001,
            symbol="TEST"
        )
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.env.window_size, self.window_size)
        self.assertEqual(self.env.initial_balance, 10000.0)
        self.assertEqual(self.env.max_trading_units, 10)
        self.assertEqual(self.env.transaction_fee_percent, 0.001)
        self.assertEqual(self.env.symbol, "TEST")
        self.assertEqual(self.env.feature_dim, 5)  # 5개 특성
        self.assertEqual(self.env.data_length, 100)  # 100일치 데이터
    
    def test_reset(self):
        """환경 초기화 테스트"""
        observation = self.env.reset()
        
        # 초기 상태 확인
        self.assertEqual(self.env.current_step, 0)
        self.assertEqual(self.env.balance, 10000.0)
        self.assertEqual(self.env.shares_held, 0)
        
        # 관측값 형태 확인
        self.assertIn('market_data', observation)
        self.assertIn('portfolio_state', observation)
        self.assertEqual(observation['market_data'].shape, (self.window_size, self.feature_dim))
        self.assertEqual(observation['portfolio_state'].shape, (2,))
        
        # 히스토리 초기화 확인
        self.assertEqual(len(self.env.states_history), 1)
        self.assertEqual(len(self.env.actions_history), 0)
        self.assertEqual(len(self.env.rewards_history), 0)
        self.assertEqual(len(self.env.portfolio_values_history), 0)
    
    def test_step_buy(self):
        """매수 행동 테스트"""
        self.env.reset()
        
        # 매수 행동 (1.0)
        action = 1.0
        observation, reward, done, info = self.env.step(action)
        
        # 상태 변화 확인
        self.assertEqual(self.env.current_step, 1)
        self.assertLess(self.env.balance, 10000.0)  # 잔고 감소
        self.assertGreater(self.env.shares_held, 0)  # 주식 보유량 증가
        
        # 관측값 및 정보 확인
        self.assertIn('market_data', observation)
        self.assertIn('portfolio_state', observation)
        self.assertIn('balance', info)
        self.assertIn('shares_held', info)
        self.assertIn('portfolio_value', info)
        
        # 히스토리 업데이트 확인
        self.assertEqual(len(self.env.states_history), 2)
        self.assertEqual(len(self.env.actions_history), 1)
        self.assertEqual(len(self.env.rewards_history), 1)
        self.assertEqual(len(self.env.portfolio_values_history), 1)
    
    def test_step_sell(self):
        """매도 행동 테스트"""
        self.env.reset()
        
        # 먼저 매수 행동
        self.env.step(1.0)
        
        # 매도 행동 (-1.0)
        action = -1.0
        observation, reward, done, info = self.env.step(action)
        
        # 상태 변화 확인
        self.assertEqual(self.env.current_step, 2)
        self.assertGreaterEqual(self.env.balance, 0)  # 잔고 확인
        
        # 관측값 및 정보 확인
        self.assertIn('market_data', observation)
        self.assertIn('portfolio_state', observation)
        self.assertIn('balance', info)
        self.assertIn('shares_held', info)
        self.assertIn('portfolio_value', info)
        
        # 히스토리 업데이트 확인
        self.assertEqual(len(self.env.states_history), 3)
        self.assertEqual(len(self.env.actions_history), 2)
        self.assertEqual(len(self.env.rewards_history), 2)
        self.assertEqual(len(self.env.portfolio_values_history), 2)
    
    def test_step_hold(self):
        """홀드 행동 테스트"""
        self.env.reset()
        
        # 홀드 행동 (0.0)
        action = 0.0
        initial_balance = self.env.balance
        initial_shares_held = self.env.shares_held
        
        observation, reward, done, info = self.env.step(action)
        
        # 상태 변화 확인 (잔고와 보유량 변화 없음)
        self.assertEqual(self.env.current_step, 1)
        self.assertEqual(self.env.balance, initial_balance)
        self.assertEqual(self.env.shares_held, initial_shares_held)
    
    def test_episode_completion(self):
        """에피소드 완료 테스트"""
        self.env.reset()
        
        done = False
        step_count = 0
        
        # 에피소드 끝까지 진행
        while not done:
            action = np.random.uniform(-1.0, 1.0)
            _, _, done, _ = self.env.step(action)
            step_count += 1
        
        # 데이터 길이만큼 스텝이 진행되었는지 확인
        self.assertEqual(step_count, self.env.data_length - 1)
        self.assertEqual(self.env.current_step, self.env.data_length - 1)
    
    def test_get_portfolio_value(self):
        """포트폴리오 가치 계산 테스트"""
        self.env.reset()
        
        # 초기 포트폴리오 가치
        initial_value = self.env._get_portfolio_value()
        self.assertEqual(initial_value, 10000.0)
        
        # 매수 후 포트폴리오 가치
        self.env.step(1.0)
        after_buy_value = self.env._get_portfolio_value()
        
        # 매수 후에도 포트폴리오 가치는 거의 동일해야 함 (수수료로 인한 약간의 감소)
        self.assertAlmostEqual(after_buy_value, initial_value, delta=initial_value * 0.01)
    
    def test_calculate_reward(self):
        """보상 계산 테스트"""
        # 포트폴리오 가치 증가 시 양의 보상
        reward = self.env._calculate_reward(1000.0, 1100.0)
        self.assertGreater(reward, 0)
        
        # 포트폴리오 가치 감소 시 음의 보상
        reward = self.env._calculate_reward(1000.0, 900.0)
        self.assertLess(reward, 0)
        
        # 포트폴리오 가치 유지 시 0에 가까운 보상
        reward = self.env._calculate_reward(1000.0, 1000.0)
        self.assertAlmostEqual(reward, 0)
    
    def test_get_episode_data(self):
        """에피소드 데이터 반환 테스트"""
        self.env.reset()
        self.env.step(1.0)
        self.env.step(-0.5)
        
        episode_data = self.env.get_episode_data()
        
        self.assertIn('actions', episode_data)
        self.assertIn('rewards', episode_data)
        self.assertIn('portfolio_values', episode_data)
        
        self.assertEqual(len(episode_data['actions']), 2)
        self.assertEqual(len(episode_data['rewards']), 2)
        self.assertEqual(len(episode_data['portfolio_values']), 2)


class TestMultiAssetTradingEnvironment(unittest.TestCase):
    """MultiAssetTradingEnvironment 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 데이터 생성
        self.window_size = 10
        
        # 샘플 주식 데이터 생성 (2개 종목)
        dates = pd.date_range('2023-01-01', periods=100)
        self.test_data1 = pd.DataFrame({
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 210, 100),
            'low': np.random.uniform(90, 190, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        self.test_data2 = pd.DataFrame({
            'open': np.random.uniform(50, 100, 100),
            'high': np.random.uniform(55, 105, 100),
            'low': np.random.uniform(45, 95, 100),
            'close': np.random.uniform(50, 100, 100),
            'volume': np.random.randint(5000, 15000, 100)
        }, index=dates)
        
        # 데이터 딕셔너리 생성
        self.data_dict = {
            "STOCK1": self.test_data1,
            "STOCK2": self.test_data2
        }
        
        # 다중 자산 환경 생성
        self.multi_env = MultiAssetTradingEnvironment(
            data_dict=self.data_dict,
            window_size=self.window_size,
            initial_balance=10000.0,
            max_trading_units=10,
            transaction_fee_percent=0.001
        )
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.multi_env.window_size, self.window_size)
        self.assertEqual(self.multi_env.initial_balance, 10000.0)
        self.assertEqual(self.multi_env.n_assets, 2)
        self.assertEqual(len(self.multi_env.envs), 2)
        self.assertIn("STOCK1", self.multi_env.envs)
        self.assertIn("STOCK2", self.multi_env.envs)
    
    def test_reset(self):
        """환경 초기화 테스트"""
        observations = self.multi_env.reset()
        
        # 관측값 확인
        self.assertEqual(len(observations), 2)
        self.assertIn("STOCK1", observations)
        self.assertIn("STOCK2", observations)
        
        # 각 환경의 초기 상태 확인
        for symbol, env in self.multi_env.envs.items():
            self.assertEqual(env.current_step, 0)
            self.assertEqual(env.balance, 10000.0 / 2)  # 자산별로 균등 배분
            self.assertEqual(env.shares_held, 0)
    
    def test_step(self):
        """스텝 진행 테스트"""
        self.multi_env.reset()
        
        # 각 자산에 대한 행동 정의
        actions = {
            "STOCK1": 0.5,  # 매수
            "STOCK2": -0.3  # 매도
        }
        
        observations, reward, done, infos = self.multi_env.step(actions)
        
        # 관측값 및 정보 확인
        self.assertEqual(len(observations), 2)
        self.assertIn("STOCK1", observations)
        self.assertIn("STOCK2", observations)
        
        # 각 환경의 상태 변화 확인
        self.assertEqual(self.multi_env.envs["STOCK1"].current_step, 1)
        self.assertEqual(self.multi_env.envs["STOCK2"].current_step, 1)
        
        # 추가 정보 확인
        self.assertIn("STOCK1", infos)
        self.assertIn("STOCK2", infos)
        self.assertIn("total", infos)
        self.assertIn("portfolio_value", infos["total"])
        self.assertIn("total_return", infos["total"])
    
    def test_get_final_portfolio_value(self):
        """최종 포트폴리오 가치 테스트"""
        self.multi_env.reset()
        
        # 몇 번의 스텝 진행
        for _ in range(5):
            actions = {
                "STOCK1": np.random.uniform(-1.0, 1.0),
                "STOCK2": np.random.uniform(-1.0, 1.0)
            }
            self.multi_env.step(actions)
        
        # 최종 포트폴리오 가치 계산
        final_value = self.multi_env.get_final_portfolio_value()
        
        # 각 자산의 포트폴리오 가치 합과 동일한지 확인
        expected_value = sum(env.get_final_portfolio_value() for env in self.multi_env.envs.values())
        self.assertEqual(final_value, expected_value)

if __name__ == "__main__":
    unittest.main() 