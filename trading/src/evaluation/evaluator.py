"""
SAC 모델 평가를 위한 평가기 모듈
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Optional, Any
from pathlib import Path
import time

from src.config.ea_teb_config import (
    DEVICE,
    RESULTS_DIR,
    LOGGER
)
from src.models.sac_agent import SACAgent
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.utils.utils import (
    create_directory, 
    plot_equity_curve, 
    calculate_sharpe_ratio, 
    calculate_max_drawdown, 
    get_timestamp,
    set_korean_font
)

class Evaluator:
    """
    SAC 모델 평가를 위한 평가기 클래스
    """
    
    def __init__(
        self,
        agent: SACAgent,
        env: Union[TradingEnvironment, MultiAssetTradingEnvironment],
        results_dir: Union[str, Path] = RESULTS_DIR,
        data_type: str = "test"  # 'train', 'valid', 'test' 중 선택
    ):
        """
        Evaluator 클래스 초기화
        
        Args:
            agent: 평가할 SAC 에이전트
            env: 평가에 사용할 트레이딩 환경
            results_dir: 결과 저장 디렉토리
            data_type: 평가에 사용할 데이터 유형 ('train', 'valid', 'test')
        """
        self.agent = agent
        self.env = env
        self.results_dir = Path(results_dir)
        self.data_type = data_type
        
        # 디렉토리 생성
        create_directory(self.results_dir)
        
        LOGGER.info(f"Evaluator 초기화 완료 (데이터 유형: {data_type})")
    
    def evaluate(self, num_episodes: int = 1, render: bool = False) -> Dict[str, Any]:
        """
        모델 평가 수행
        
        Args:
            num_episodes: 평가할 에피소드 수
            render: 환경 렌더링 여부
            
        Returns:
            평가 결과 딕셔너리
        """
        total_reward = 0
        total_steps = 0
        all_portfolio_values = []
        all_actions = []
        all_rewards = []
        
        LOGGER.info(f"평가 시작: {num_episodes}개 에피소드 ({self.data_type} 데이터)")
        
        for episode in range(1, num_episodes + 1):
            state = self.env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            # 에피소드 진행
            while not done:
                # 행동 선택 (평가 모드)
                action = self.agent.select_action(state, evaluate=True)
                
                # 환경에서 한 스텝 진행
                next_state, reward, done, info = self.env.step(action)
                
                if render:
                    self.env.render()
                
                state = next_state
                episode_reward += reward
                episode_steps += 1
                
                # 행동 및 보상 기록
                all_actions.append(action)
                all_rewards.append(reward)
                
                # 포트폴리오 가치 기록
                if isinstance(self.env, TradingEnvironment):
                    all_portfolio_values.append(info['portfolio_value'])
                else:
                    all_portfolio_values.append(info['total']['portfolio_value'])
            
            total_reward += episode_reward
            total_steps += episode_steps
            
            LOGGER.info(f"에피소드 {episode}/{num_episodes} - 보상: {episode_reward:.2f}, 스텝: {episode_steps}")
        
        # 평균 보상 및 스텝 계산
        avg_reward = total_reward / num_episodes
        avg_steps = total_steps / num_episodes
        
        # 성능 지표 계산
        final_portfolio_value = all_portfolio_values[-1]
        initial_portfolio_value = all_portfolio_values[0]
        total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value * 100
        
        # 일별 수익률 계산
        daily_returns = []
        for i in range(1, len(all_portfolio_values)):
            daily_return = (all_portfolio_values[i] - all_portfolio_values[i-1]) / all_portfolio_values[i-1]
            daily_returns.append(daily_return)
        
        # 샤프 비율 계산
        sharpe = calculate_sharpe_ratio(daily_returns)
        
        # 최대 낙폭 계산
        mdd = calculate_max_drawdown(all_portfolio_values)
        
        LOGGER.info(f"평가 완료 ({self.data_type} 데이터): 평균 보상 {avg_reward:.2f}, 총 수익률 {total_return:.2f}%, 샤프 비율 {sharpe:.2f}, MDD {mdd:.2f}%")
        
        # 결과 딕셔너리 반환
        return {
            'data_type': self.data_type,
            'avg_reward': avg_reward,
            'avg_steps': avg_steps,
            'final_portfolio_value': final_portfolio_value,
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd,
            'portfolio_values': all_portfolio_values,
            'actions': all_actions,
            'rewards': all_rewards
    
        }
    
    def save_results(self, results: Dict[str, Any], prefix: str = '') -> Path:
        """
        평가 결과 저장
        
        Args:
            results: 평가 결과 딕셔너리
            prefix: 파일명 접두사
            
        Returns:
            결과 저장 디렉토리 경로
        """
        timestamp = get_timestamp()
        result_dir = self.results_dir / f"{prefix}evaluation_{self.data_type}_{timestamp}"
        create_directory(result_dir)
        
        # 포트폴리오 가치 곡선 저장
        plot_equity_curve(results['portfolio_values'], save_path=result_dir / "equity_curve.png")
        
        # 행동 분포 시각화
        self._plot_action_distribution(results['actions'], save_path=result_dir / "action_distribution.png")
        
        # 누적 보상 곡선
        self._plot_cumulative_rewards(results['rewards'], save_path=result_dir / "cumulative_rewards.png")
        
        # 성능 지표 저장
        metrics = {
            '데이터 유형': results['data_type'],
            '평균 보상': results['avg_reward'],
            '평균 스텝': results['avg_steps'],
            '최종 포트폴리오 가치': results['final_portfolio_value'],
            '총 수익률 (%)': results['total_return'],
            '샤프 비율': results['sharpe_ratio'],
            '최대 낙폭 (%)': results['max_drawdown']
        }
        
        metrics_df = pd.DataFrame(list(metrics.items()), columns=['지표', '값'])
        metrics_df.to_csv(result_dir / "metrics.csv", index=False, encoding='utf-8-sig')
        
        # 포트폴리오 가치 데이터 저장
        portfolio_df = pd.DataFrame({
            'step': list(range(len(results['portfolio_values']))),
            'portfolio_value': results['portfolio_values']
        })
        portfolio_df.to_csv(result_dir / "portfolio_values.csv", index=False)
        
        # 행동 및 보상 데이터 저장
        actions_rewards_df = pd.DataFrame({
            'step': list(range(len(results['actions']))),
            'action': [a[0] if isinstance(a, np.ndarray) else a for a in results['actions']],
            'reward': results['rewards']
        })
        actions_rewards_df.to_csv(result_dir / "actions_rewards.csv", index=False)
        
        LOGGER.info(f"평가 결과 저장 완료: {result_dir}")
        
        return result_dir
    
    def _plot_action_distribution(self, actions: List[np.ndarray], save_path: Union[str, Path] = None) -> None:
        """
        행동 분포 시각화
        
        Args:
            actions: 행동 리스트
            save_path: 저장 경로
        """
        set_korean_font()
        
        # 행동 값 추출 (다차원 배열의 경우 첫 번째 차원만 사용)
        action_values = [a[0] if isinstance(a, np.ndarray) and a.size > 1 else a for a in actions]
        
        plt.figure(figsize=(10, 6))
        plt.hist(action_values, bins=50, alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title('행동 분포')
        plt.xlabel('행동 값')
        plt.ylabel('빈도')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _plot_cumulative_rewards(self, rewards: List[float], save_path: Union[str, Path] = None) -> None:
        """
        누적 보상 곡선 시각화
        
        Args:
            rewards: 보상 리스트
            save_path: 저장 경로
        """
        set_korean_font()
        
        cumulative_rewards = np.cumsum(rewards)
        
        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_rewards)
        plt.title('누적 보상')
        plt.xlabel('스텝')
        plt.ylabel('누적 보상')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close()
        
if __name__ == "__main__":
    # 모듈 테스트 코드
    from src.data_collection.data_collector import DataCollector
    from src.preprocessing.data_processor import DataProcessor
    
    # 데이터 수집 및 전처리
    # collector = DataCollector(symbols=["AAPL"])
    
    from src.config.config import TARGET_SYMBOLS
    collector = DataCollector(symbols=TARGET_SYMBOLS)
    data = collector.load_all_data()
    
    if not data:
        print("저장된 데이터가 없어 데이터를 수집합니다.")
        data = collector.collect_and_save()
    
    if data:
        # 데이터 전처리
        processor = DataProcessor()
        results = processor.process_all_symbols(data)
       
        test_symbol = TARGET_SYMBOLS[0]
        if test_symbol in results:
            valid_data = results[test_symbol]["valid"]
            raw_data = results[test_symbol]["featured_data"]
        
            # valid 범위에 해당하는 raw_data 잘라냄 (window_size 맞추려면 정확히 분할 인덱스 가져오는 게 좋음)
            train_len = len(results[test_symbol]["train"])
            valid_len = len(valid_data)
            raw_valid_data = raw_data.iloc[train_len:train_len + valid_len]
        
            env = TradingEnvironment(
                data=valid_data,
                raw_data=raw_valid_data,
                symbol=test_symbol
            )
            # 에이전트 생성
            state_dim = env.observation_space['market_data'].shape[0] * env.observation_space['market_data'].shape[1] + env.observation_space['portfolio_state'].shape[0]
            action_dim = 1
            
            # 에이전트 초기화 (랜덤 정책으로 테스트)
            agent = SACAgent(
                state_dim=state_dim,
                action_dim=action_dim
            )
            
            # 평가기 생성 및 평가
            evaluator = Evaluator(agent=agent, env=env)
            results = evaluator.evaluate(num_episodes=1, render=True)
            
            # 결과 저장
            evaluator.save_results(results, prefix="test_")
            
            print(f"평가 완료: 총 수익률 {results['total_return']:.2f}%, 샤프 비율 {results['sharpe_ratio']:.2f}") 