"""
SAC 모델 학습을 위한 트레이너 모듈 (SAC 정책 준수)
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional
from pathlib import Path
import time

from src.config.ea_teb_config import (
    DEVICE,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    MODELS_DIR,
    RESULTS_DIR,
    LOGGER
)
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.utils.utils import create_directory, get_timestamp

class SACTrainer:
    """
    SAC 모델 학습을 위한 트레이너 클래스 (SAC 정책 준수)
    - 자동 엔트로피 튜닝 지원
    - 표준 SAC 학습 루프
    - 리스크 관리 통합
    """
    
    def __init__(
        self,
        agent: SACAgent,
        env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
        valid_env: Optional[Union[TradingEnvironment, MultiAssetTradingEnvironment]] = None,
        batch_size: int = BATCH_SIZE,
        num_episodes: int = NUM_EPISODES,
        evaluate_interval: int = EVALUATE_INTERVAL,
        save_interval: int = SAVE_MODEL_INTERVAL,
        max_steps: int = MAX_STEPS_PER_EPISODE,
        models_dir: Union[str, Path] = MODELS_DIR,
        results_dir: Union[str, Path] = RESULTS_DIR
    ):
        """트레이너 초기화"""
        self.agent = agent
        self.env = env
        self.valid_env = valid_env
        self.batch_size = batch_size
        self.num_episodes = num_episodes
        self.evaluate_interval = evaluate_interval
        self.save_interval = save_interval
        self.max_steps = max_steps
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        
        # 디렉토리 생성
        create_directory(self.models_dir)
        create_directory(self.results_dir)
        
        # 학습 통계
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_rewards: List[float] = []
        self.train_losses: List[Dict[str, float]] = []
        
        LOGGER.info("✅ SAC Trainer 초기화 완료")
        LOGGER.info(f"   📊 총 에피소드: {num_episodes}")
        LOGGER.info(f"   🎯 배치 크기: {batch_size}")
        LOGGER.info(f"   🔄 평가 간격: {evaluate_interval}")
    
    def train(self) -> Dict[str, List[float]]:
        """
        SAC 모델 학습 수행
        
        Returns:
            학습 통계 딕셔너리
        """
        start_time = time.time()
        timestamp = get_timestamp()
        
        LOGGER.info("🚀 SAC 학습 시작")
        
        for episode in range(1, self.num_episodes + 1):
            episode_start_time = time.time()
            
            # 환경 리셋
            state = self.env.reset()
            
            episode_reward = 0.0
            episode_loss = {"actor_loss": 0.0, "critic_loss": 0.0, "alpha_loss": 0.0, "entropy": 0.0}
            episode_steps = 0
            done = False
            
            # 에피소드 진행
            while not done and episode_steps < self.max_steps:
                # 행동 선택
                action = self.agent.select_action(state)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = self.env.step(action)
                
                # 경험 저장
                self.agent.add_experience(state, action, reward, next_state, done)
                
                # 모델 업데이트 (충분한 경험이 쌓인 후)
                if len(self.agent.replay_buffer) > self.batch_size:
                    loss = self.agent.update_parameters(self.batch_size)
                    
                    # 손실 누적
                    for k, v in loss.items():
                        episode_loss[k] += v
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
            
            # 에피소드 통계 기록
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_steps)
            
            # 손실 평균 계산
            if episode_steps > 0:
                for k in episode_loss:
                    episode_loss[k] /= episode_steps
            self.train_losses.append(episode_loss)
            
            # 진행 상황 로깅
            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                episode_time = time.time() - episode_start_time
                
                LOGGER.info(
                    f"Episode {episode:4d} | "
                    f"Reward: {episode_reward:8.2f} | "
                    f"Avg(10): {avg_reward:8.2f} | "
                    f"Steps: {episode_steps:3d} | "
                    f"Time: {episode_time:.1f}s"
                )
                
                # 손실 정보 (SAC 특화)
                if episode_loss["actor_loss"] > 0:
                    LOGGER.info(
                        f"         Losses -> "
                        f"Actor: {episode_loss['actor_loss']:.4f} | "
                        f"Critic: {episode_loss['critic_loss']:.4f} | "
                        f"Alpha: {episode_loss['alpha_loss']:.4f} | "
                        f"Entropy: {episode_loss['entropy']:.4f}"
                    )
            
            # 주기적 평가
            if episode % self.evaluate_interval == 0 and self.valid_env:
                eval_reward = self.evaluate()
                self.eval_rewards.append(eval_reward)
                LOGGER.info(f"📊 평가 결과 (Episode {episode}): {eval_reward:.2f}")
            
            # 주기적 모델 저장
            if episode % self.save_interval == 0:
                model_path = self.agent.save_model(
                    save_dir=str(self.models_dir),
                    prefix=f"episode_{episode}_",
                    model_type=getattr(self.agent, 'model_type', 'sac')
                )
                LOGGER.info(f"💾 모델 저장: {model_path}")
            
            # 학습 곡선 업데이트
            if episode % 50 == 0:
                self._plot_training_curves(timestamp)
        
        # 최종 모델 저장
        final_model_path = self.agent.save_model(
            save_dir=str(self.models_dir),
            prefix='final_',
            model_type=getattr(self.agent, 'model_type', 'sac')
        )
        
        total_time = time.time() - start_time
        LOGGER.info(f"🎉 SAC 학습 완료!")
        LOGGER.info(f"   ⏱️  총 학습 시간: {total_time/3600:.2f}시간")
        LOGGER.info(f"   💾 최종 모델: {final_model_path}")
        
        # 최종 학습 곡선 저장
        self._plot_training_curves(timestamp)
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'eval_rewards': self.eval_rewards,
            'train_losses': self.train_losses
        }
    
    def evaluate(self, num_episodes: int = 1) -> float:
        """
        모델 평가 수행
        
        Args:
            num_episodes: 평가할 에피소드 수
            
        Returns:
            평균 보상
        """
        if not self.valid_env:
            LOGGER.warning("검증 환경이 설정되지 않았습니다.")
            return 0.0
        
        total_reward = 0.0
        
        for _ in range(num_episodes):
            state = self.valid_env.reset()
            episode_reward = 0.0
            done = False
            steps = 0
            
            while not done and steps < self.max_steps:
                # 평가 모드로 행동 선택 (탐험 없음)
                action = self.agent.select_action(state, evaluate=True)
                next_state, reward, done, _ = self.valid_env.step(action)
                state = next_state
                episode_reward += reward
                steps += 1
            
            total_reward += episode_reward
        
        avg_reward = total_reward / num_episodes
        return avg_reward
    
    def _plot_training_curves(self, timestamp: str) -> None:
        """학습 곡선 시각화"""
        if not self.episode_rewards:
            return
        
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'SAC Training Progress - {timestamp}', fontsize=16)
            
            # 에피소드 보상
            axes[0, 0].plot(self.episode_rewards, alpha=0.6)
            if len(self.episode_rewards) > 10:
                # 이동평균 추가
                window = min(50, len(self.episode_rewards) // 10)
                moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(self.episode_rewards)), moving_avg, 'r-', linewidth=2)
            axes[0, 0].set_title('Episode Rewards')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
            
            # 에피소드 길이
            axes[0, 1].plot(self.episode_lengths, 'g-', alpha=0.7)
            axes[0, 1].set_title('Episode Lengths')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Steps')
            axes[0, 1].grid(True, alpha=0.3)
            
            # 평가 보상 (있는 경우)
            if self.eval_rewards:
                eval_episodes = [i * self.evaluate_interval for i in range(1, len(self.eval_rewards) + 1)]
                axes[1, 0].plot(eval_episodes, self.eval_rewards, 'b-o', linewidth=2, markersize=4)
                axes[1, 0].set_title('Evaluation Rewards')
                axes[1, 0].set_xlabel('Episode')
                axes[1, 0].set_ylabel('Eval Reward')
                axes[1, 0].grid(True, alpha=0.3)
            
            # 손실 (SAC 특화)
            if self.train_losses:
                episodes = list(range(1, len(self.train_losses) + 1))
                actor_losses = [loss.get('actor_loss', 0) for loss in self.train_losses]
                critic_losses = [loss.get('critic_loss', 0) for loss in self.train_losses]
                alpha_losses = [loss.get('alpha_loss', 0) for loss in self.train_losses]
                
                axes[1, 1].plot(episodes, actor_losses, label='Actor Loss', alpha=0.7)
                axes[1, 1].plot(episodes, critic_losses, label='Critic Loss', alpha=0.7)
                axes[1, 1].plot(episodes, alpha_losses, label='Alpha Loss', alpha=0.7)
                axes[1, 1].set_title('SAC Losses')
                axes[1, 1].set_xlabel('Episode')
                axes[1, 1].set_ylabel('Loss')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 저장
            plot_path = self.results_dir / f"sac_training_curves_{timestamp}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            LOGGER.debug(f"📊 학습 곡선 저장: {plot_path}")
            
        except Exception as e:
            LOGGER.warning(f"학습 곡선 생성 실패: {e}")
    
    def get_training_stats(self) -> Dict[str, float]:
        """학습 통계 반환"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'mean_episode_length': np.mean(self.episode_lengths),
            'total_steps': sum(self.episode_lengths)
        }
