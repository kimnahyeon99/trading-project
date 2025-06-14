"""
SACAgent 클래스에 대한 단위 테스트
"""
import unittest
import torch
import numpy as np
import tempfile
import shutil
import os
from pathlib import Path

from src.models.sac_agent import SACAgent, ReplayBuffer
from src.config.config import DEVICE

class TestReplayBuffer(unittest.TestCase):
    """ReplayBuffer 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        self.buffer_capacity = 100
        self.buffer = ReplayBuffer(capacity=self.buffer_capacity)
        
        # 테스트용 임시 디렉토리
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """초기화 테스트"""
        self.assertEqual(self.buffer.capacity, self.buffer_capacity)
        self.assertEqual(len(self.buffer.buffer), 0)
        self.assertEqual(self.buffer.position, 0)
    
    def test_push(self):
        """샘플 추가 테스트"""
        # 샘플 추가
        state = torch.randn(10)
        action = torch.randn(1)
        reward = 1.0
        next_state = torch.randn(10)
        done = False
        
        self.buffer.push(state, action, reward, next_state, done)
        
        # 버퍼 크기 확인
        self.assertEqual(len(self.buffer), 1)
        self.assertEqual(self.buffer.position, 1)
        
        # 용량 초과 테스트
        for i in range(self.buffer_capacity * 2):
            self.buffer.push(state, action, reward, next_state, done)
        
        # 버퍼 크기는 용량을 초과하지 않아야 함
        self.assertEqual(len(self.buffer), self.buffer_capacity)
        self.assertEqual(self.buffer.position, self.buffer_capacity % self.buffer_capacity)
    
    def test_sample(self):
        """샘플링 테스트"""
        # 버퍼에 샘플 추가
        batch_size = 4
        for i in range(batch_size * 2):
            state = torch.randn(10)
            action = torch.randn(1)
            reward = float(i)
            next_state = torch.randn(10)
            done = i % 2 == 0
            
            self.buffer.push(state, action, reward, next_state, done)
        
        # 샘플링
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        
        # 결과 확인
        self.assertEqual(len(states), batch_size)
        self.assertEqual(len(actions), batch_size)
        self.assertEqual(len(rewards), batch_size)
        self.assertEqual(len(next_states), batch_size)
        self.assertEqual(len(dones), batch_size)
    
    def test_save_load(self):
        """저장 및 로드 테스트"""
        # 버퍼에 샘플 추가
        for i in range(10):
            state = torch.randn(10)
            action = torch.randn(1)
            reward = float(i)
            next_state = torch.randn(10)
            done = i % 2 == 0
            
            self.buffer.push(state, action, reward, next_state, done)
        
        # 버퍼 저장
        save_path = os.path.join(self.test_dir, "buffer.pkl")
        self.buffer.save(save_path)
        
        # 새 버퍼 생성 및 로드
        new_buffer = ReplayBuffer(capacity=self.buffer_capacity)
        new_buffer.load(save_path)
        
        # 버퍼 크기 확인
        self.assertEqual(len(new_buffer), len(self.buffer))
        self.assertEqual(new_buffer.position, len(self.buffer) % self.buffer_capacity)


class TestSACAgent(unittest.TestCase):
    """SACAgent 클래스 테스트"""
    
    def setUp(self):
        """테스트 설정"""
        # 테스트 파라미터
        self.state_dim = 10
        self.action_dim = 1
        self.hidden_dim = 64
        self.batch_size = 4
        
        # 일반 에이전트 생성
        self.agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # CNN 에이전트 생성
        self.window_size = 20
        self.feature_dim = 5
        self.cnn_agent = SACAgent(
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            input_shape=(self.window_size, self.feature_dim),
            use_cnn=True
        )
        
        # 테스트용 임시 디렉토리
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """테스트 정리"""
        # 임시 디렉토리 삭제
        shutil.rmtree(self.test_dir)
    
    def test_init(self):
        """초기화 테스트"""
        # 일반 에이전트 확인
        self.assertEqual(self.agent.action_dim, self.action_dim)
        self.assertFalse(self.agent.use_cnn)
        
        # CNN 에이전트 확인
        self.assertEqual(self.cnn_agent.action_dim, self.action_dim)
        self.assertTrue(self.cnn_agent.use_cnn)
    
    def test_select_action(self):
        """행동 선택 테스트"""
        # 일반 에이전트 행동 선택
        state = np.random.randn(self.state_dim)
        action = self.agent.select_action(state)
        
        # 행동 형태 확인
        self.assertEqual(action.shape, (self.action_dim,))
        self.assertTrue(np.all(action >= -1.0) and np.all(action <= 1.0))
        
        # CNN 에이전트 행동 선택
        cnn_state = {
            'market_data': np.random.randn(self.window_size, self.feature_dim),
            'portfolio_state': np.random.randn(2)
        }
        cnn_action = self.cnn_agent.select_action(cnn_state)
        
        # 행동 형태 확인
        self.assertEqual(cnn_action.shape, (self.action_dim,))
        self.assertTrue(np.all(cnn_action >= -1.0) and np.all(cnn_action <= 1.0))
    
    def test_update_parameters_empty_buffer(self):
        """빈 버퍼로 파라미터 업데이트 테스트"""
        # 빈 버퍼로 업데이트
        stats = self.agent.update_parameters(batch_size=self.batch_size)
        
        # 결과 확인 (업데이트가 수행되지 않아야 함)
        self.assertEqual(stats['actor_loss'], 0.0)
        self.assertEqual(stats['critic_loss'], 0.0)
        self.assertEqual(stats['alpha_loss'], 0.0)
        self.assertEqual(stats['entropy'], 0.0)
    
    def test_update_parameters(self):
        """파라미터 업데이트 테스트"""
        # 일반 에이전트 버퍼에 샘플 추가
        for _ in range(self.batch_size * 2):
            state = np.random.randn(self.state_dim)
            action = np.random.randn(self.action_dim)
            reward = np.random.randn(1)
            next_state = np.random.randn(self.state_dim)
            done = np.random.randint(0, 2, (1,))
            
            self.agent.replay_buffer.push(
                torch.FloatTensor(state).to(self.agent.device),
                torch.FloatTensor(action).to(self.agent.device),
                reward,
                torch.FloatTensor(next_state).to(self.agent.device),
                done
            )
        
        # 업데이트 수행
        stats = self.agent.update_parameters(batch_size=self.batch_size)
        
        # 결과 확인 (업데이트가 수행되어야 함)
        self.assertNotEqual(stats['actor_loss'], 0.0)
        self.assertNotEqual(stats['critic_loss'], 0.0)
        if self.agent.use_automatic_entropy_tuning:
            self.assertNotEqual(stats['alpha_loss'], 0.0)
        self.assertNotEqual(stats['entropy'], 0.0)
    
    def test_save_load_model(self):
        """모델 저장 및 로드 테스트"""
        # 모델 저장
        save_dir = os.path.join(self.test_dir, "models")
        model_path = self.agent.save_model(save_dir=save_dir)
        
        # 새 에이전트 생성
        new_agent = SACAgent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim
        )
        
        # 모델 로드
        new_agent.load_model(model_path)
        
        # 모델 파라미터 비교
        for param1, param2 in zip(self.agent.actor.parameters(), new_agent.actor.parameters()):
            self.assertTrue(torch.allclose(param1, param2))
        
        for param1, param2 in zip(self.agent.critic.parameters(), new_agent.critic.parameters()):
            self.assertTrue(torch.allclose(param1, param2))
    
    def test_get_latest_model_path(self):
        """최신 모델 경로 반환 테스트"""
        # 모델 저장
        save_dir = os.path.join(self.test_dir, "models")
        model_path1 = self.agent.save_model(save_dir=save_dir, prefix="test1_")
        model_path2 = self.agent.save_model(save_dir=save_dir, prefix="test2_")
        
        # 최신 모델 경로 확인
        latest_path1 = self.agent.get_latest_model_path(save_dir=save_dir, prefix="test1_")
        latest_path2 = self.agent.get_latest_model_path(save_dir=save_dir, prefix="test2_")
        
        self.assertEqual(latest_path1, model_path1)
        self.assertEqual(latest_path2, model_path2)
        
        # 존재하지 않는 접두사로 확인
        latest_path3 = self.agent.get_latest_model_path(save_dir=save_dir, prefix="nonexistent_")
        self.assertIsNone(latest_path3)


if __name__ == "__main__":
    unittest.main() 