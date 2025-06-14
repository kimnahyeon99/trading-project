"""
Trainer 클래스에 대한 단위 테스트
"""
import unittest
import torch
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import shutil
import os
from pathlib import Path

from src.training.trainer import Trainer
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment

class TestTrainer(unittest.TestCase):
    """Trainer 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트용 임시 디렉토리
        self.test_dir = tempfile.mkdtemp()
        self.models_dir = os.path.join(self.test_dir, "models")
        self.results_dir = os.path.join(self.test_dir, "results")
        
        # 목(mock) 환경 생성
        self.mock_env = MagicMock()
        self.mock_env.reset.return_value = {"market_data": torch.zeros((10, 5)), "portfolio_state": torch.zeros(2)}
        self.mock_env.step.return_value = (
            {"market_data": torch.zeros((10, 5)), "portfolio_state": torch.zeros(2)},
            1.0,
            False,
            {}
        )
        
        # 목(mock) 에이전트 생성
        self.mock_agent = MagicMock()
        self.mock_agent.select_action.return_value = np.array([0.5])
        self.mock_agent.update_parameters.return_value = {
            "actor_loss": 0.1,
            "critic_loss": 0.2,
            "alpha_loss": 0.05,
            "entropy": 0.5
        }
        self.mock_agent.replay_buffer = MagicMock()
        self.mock_agent.replay_buffer.__len__.return_value = 1000
        
        # 트레이너 생성
        self.trainer = Trainer(
            agent=self.mock_agent,
            env=self.mock_env,
            num_episodes=5,
            batch_size=32,
            evaluate_interval=2,
            save_interval=5,
            max_steps=10,
            models_dir=self.models_dir,
            results_dir=self.results_dir
        )
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.trainer.agent, self.mock_agent)
        self.assertEqual(self.trainer.env, self.mock_env)
        self.assertEqual(self.trainer.num_episodes, 5)
        self.assertEqual(self.trainer.batch_size, 32)
        self.assertEqual(self.trainer.evaluate_interval, 2)
        self.assertEqual(self.trainer.save_interval, 5)
        self.assertEqual(self.trainer.max_steps, 10)
        self.assertEqual(self.trainer.models_dir, Path(self.models_dir))
        self.assertEqual(self.trainer.results_dir, Path(self.results_dir))
        
        # 학습 통계 초기화 확인
        self.assertEqual(len(self.trainer.episode_rewards), 0)
        self.assertEqual(len(self.trainer.episode_lengths), 0)
        self.assertEqual(len(self.trainer.eval_rewards), 0)
        self.assertEqual(len(self.trainer.train_losses), 0)
    
    @patch('src.training.trainer.time.time')
    def test_train(self, mock_time):
        """학습 테스트"""
        # time.time() 목 설정
        mock_time.side_effect = [0, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 10]
        
        # evaluate 메서드 목 설정
        self.trainer.evaluate = MagicMock(return_value=5.0)
        
        # _plot_training_curves 메서드 목 설정
        self.trainer._plot_training_curves = MagicMock()
        
        # 학습 실행
        stats = self.trainer.train()
        
        # 환경과 에이전트 메서드 호출 확인
        self.assertEqual(self.mock_env.reset.call_count, 5)  # 5개 에피소드
        self.assertEqual(self.mock_agent.select_action.call_count, 50)  # 5개 에피소드 x 10 스텝
        self.assertEqual(self.mock_agent.update_parameters.call_count, 50)  # 5개 에피소드 x 10 스텝
        
        # 통계 기록 확인
        self.assertEqual(len(self.trainer.episode_rewards), 5)
        self.assertEqual(len(self.trainer.episode_lengths), 5)
        self.assertEqual(len(self.trainer.train_losses), 5)
        
        # evaluate 메서드 호출 확인
        self.assertEqual(self.trainer.evaluate.call_count, 2)  # 에피소드 2, 4에서 호출
        self.assertEqual(len(self.trainer.eval_rewards), 2)
        
        # 모델 저장 확인
        self.assertEqual(self.mock_agent.save_model.call_count, 2)  # 에피소드 5와 최종 저장
        
        # 반환된 통계 확인
        self.assertIn("episode_rewards", stats)
        self.assertIn("episode_lengths", stats)
        self.assertIn("eval_rewards", stats)
        self.assertIn("actor_losses", stats)
        self.assertIn("critic_losses", stats)
        self.assertIn("alpha_losses", stats)
        self.assertIn("entropy_values", stats)
    
    def test_evaluate(self):
        """평가 테스트"""
        # evaluate 메서드 원래 구현 복원
        self.trainer.evaluate = Trainer.evaluate.__get__(self.trainer, Trainer)
        
        # 평가 실행
        reward = self.trainer.evaluate(num_episodes=2)
        
        # 환경과 에이전트 메서드 호출 확인
        self.assertEqual(self.mock_env.reset.call_count, 2)  # 2개 에피소드
        self.assertEqual(self.mock_agent.select_action.call_count, 2)  # 각 에피소드에서 한 번씩 (첫 스텝 후 done=True로 설정)
        
        # 반환값 확인
        self.assertEqual(reward, 1.0)  # 각 에피소드에서 1.0 보상, 평균 1.0
    
    @patch('src.training.trainer.plot_learning_curve')
    @patch('src.training.trainer.plt.savefig')
    @patch('src.training.trainer.torch.save')
    def test_plot_training_curves(self, mock_torch_save, mock_savefig, mock_plot_learning_curve):
        """학습 곡선 시각화 테스트"""
        # _plot_training_curves 메서드 원래 구현 복원
        self.trainer._plot_training_curves = Trainer._plot_training_curves.__get__(self.trainer, Trainer)
        
        # 학습 통계 설정
        self.trainer.episode_rewards = [1.0, 2.0, 3.0]
        self.trainer.episode_lengths = [10, 20, 30]
        self.trainer.eval_rewards = [1.5, 2.5]
        self.trainer.train_losses = [
            {"actor_loss": 0.1, "critic_loss": 0.2, "alpha_loss": 0.05, "entropy": 0.5},
            {"actor_loss": 0.09, "critic_loss": 0.18, "alpha_loss": 0.04, "entropy": 0.45},
            {"actor_loss": 0.08, "critic_loss": 0.16, "alpha_loss": 0.03, "entropy": 0.4}
        ]
        
        # 시각화 실행
        self.trainer._plot_training_curves("test_timestamp")
        
        # plot_learning_curve 호출 확인
        mock_plot_learning_curve.assert_called_once()
        
        # savefig 호출 확인 (에피소드 길이, actor 손실, critic 손실, alpha 손실, 엔트로피, 평가 보상)
        self.assertEqual(mock_savefig.call_count, 6)
        
        # torch.save 호출 확인
        mock_torch_save.assert_called_once()


if __name__ == "__main__":
    unittest.main() 