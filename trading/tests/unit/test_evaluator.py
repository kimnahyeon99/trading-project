"""
Evaluator 클래스에 대한 단위 테스트
"""
import unittest
import torch
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os
from pathlib import Path

from src.evaluation.evaluator import Evaluator
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment

class TestEvaluator(unittest.TestCase):
    """Evaluator 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 임시 디렉토리
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.test_dir, "results")
        
        # 목(mock) 환경 생성
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = {"market_data": torch.zeros((10, 5)), "portfolio_state": torch.zeros(2)}
        self.mock_env.step.return_value = (
            {"market_data": torch.zeros((10, 5)), "portfolio_state": torch.zeros(2)},
            1.0,
            True,  # 한 스텝 후 종료
            {"portfolio_value": 10500.0}
        )
        
        # 목(mock) 에이전트 생성
        self.mock_agent = MagicMock()
        self.mock_agent.select_action.return_value = np.array([0.5])
        
        # 평가기 생성
        self.evaluator = Evaluator(
            agent=self.mock_agent,
            env=self.mock_env,
            results_dir=self.results_dir
        )
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.evaluator.agent, self.mock_agent)
        self.assertEqual(self.evaluator.env, self.mock_env)
        self.assertEqual(self.evaluator.results_dir, Path(self.results_dir))
    
    def test_evaluate(self):
        """평가 테스트"""
        # 평가 실행
        results = self.evaluator.evaluate(num_episodes=2)
        
        # 환경과 에이전트 메서드 호출 확인
        self.assertEqual(self.mock_env.reset.call_count, 2)  # 2개 에피소드
        self.assertEqual(self.mock_agent.select_action.call_count, 2)  # 각 에피소드에서 한 번씩 (첫 스텝 후 done=True로 설정)
        
        # 결과 확인
        self.assertIn('avg_reward', results)
        self.assertIn('avg_steps', results)
        self.assertIn('final_portfolio_value', results)
        self.assertIn('total_return', results)
        self.assertIn('sharpe_ratio', results)
        self.assertIn('max_drawdown', results)
        self.assertIn('portfolio_values', results)
        self.assertIn('actions', results)
        self.assertIn('rewards', results)
        
        # 값 확인
        self.assertEqual(results['avg_reward'], 1.0)  # 각 에피소드에서 1.0 보상, 평균 1.0
        self.assertEqual(results['avg_steps'], 1.0)  # 각 에피소드에서 1 스텝, 평균 1.0
        self.assertEqual(len(results['portfolio_values']), 2)  # 2개 에피소드, 각 1 스텝
        self.assertEqual(len(results['actions']), 2)
        self.assertEqual(len(results['rewards']), 2)
    
    @patch('src.evaluation.evaluator.plot_equity_curve')
    @patch('src.evaluation.evaluator.plt.savefig')
    @patch('src.evaluation.evaluator.pd.DataFrame.to_csv')
    def test_save_results(self, mock_to_csv, mock_savefig, mock_plot_equity_curve):
        """결과 저장 테스트"""
        # 테스트 결과 데이터
        results = {
            'avg_reward': 1.0,
            'avg_steps': 1.0,
            'final_portfolio_value': 10500.0,
            'total_return': 5.0,
            'sharpe_ratio': 2.0,
            'max_drawdown': 1.0,
            'portfolio_values': [10000.0, 10500.0],
            'actions': [np.array([0.5]), np.array([0.5])],
            'rewards': [1.0, 1.0]
        }
        
        # 결과 저장
        result_dir = self.evaluator.save_results(results, prefix="test_")
        
        # 디렉토리 생성 확인
        self.assertTrue(os.path.exists(result_dir))
        
        # plot_equity_curve 호출 확인
        mock_plot_equity_curve.assert_called_once()
        
        # savefig 호출 확인 (행동 분포, 누적 보상)
        self.assertEqual(mock_savefig.call_count, 2)
        
        # to_csv 호출 확인 (지표, 포트폴리오 가치, 행동 및 보상)
        self.assertEqual(mock_to_csv.call_count, 3)
    
    @patch('src.evaluation.evaluator.plt.hist')
    @patch('src.evaluation.evaluator.plt.savefig')
    def test_plot_action_distribution(self, mock_savefig, mock_hist):
        """행동 분포 시각화 테스트"""
        # _plot_action_distribution 메서드 원래 구현 복원
        self.evaluator._plot_action_distribution = Evaluator._plot_action_distribution.__get__(self.evaluator, Evaluator)
        
        # 테스트 행동 데이터
        actions = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
        
        # 시각화 실행
        self.evaluator._plot_action_distribution(actions, save_path="test.png")
        
        # hist 호출 확인
        mock_hist.assert_called_once()
        
        # savefig 호출 확인
        mock_savefig.assert_called_once_with("test.png", dpi=300, bbox_inches='tight')
    
    @patch('src.evaluation.evaluator.plt.plot')
    @patch('src.evaluation.evaluator.plt.savefig')
    def test_plot_cumulative_rewards(self, mock_savefig, mock_plot):
        """누적 보상 곡선 시각화 테스트"""
        # _plot_cumulative_rewards 메서드 원래 구현 복원
        self.evaluator._plot_cumulative_rewards = Evaluator._plot_cumulative_rewards.__get__(self.evaluator, Evaluator)
        
        # 테스트 보상 데이터
        rewards = [1.0, 2.0, 3.0]
        
        # 시각화 실행
        self.evaluator._plot_cumulative_rewards(rewards, save_path="test.png")
        
        # plot 호출 확인
        mock_plot.assert_called_once()
        
        # savefig 호출 확인
        mock_savefig.assert_called_once_with("test.png", dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    unittest.main() 