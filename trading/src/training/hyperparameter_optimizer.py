#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAC 하이퍼파라미터 최적화 스크립트
"""

import sys
import os
import time
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# 프로젝트 루트 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

# 경고 억제
warnings.filterwarnings('ignore')

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("⚠️ Optuna가 설치되지 않았습니다. pip install optuna를 실행하세요.")

import torch
import numpy as np
import pandas as pd

from src.config.ea_teb_config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER,
    INITIAL_BALANCE,
    WINDOW_SIZE,
    NUM_EPISODES,
    BATCH_SIZE,
    MAX_STEPS_PER_EPISODE,
    MODEL_DEFAULTS,
    get_model_config
)

from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.environment.trading_env import create_environment_from_results
from src.models.sac_agent import SACAgent
from src.utils.utils import create_directory, get_timestamp

class HyperparameterOptimizer:
    """
    SAC 모델을 위한 하이퍼파라미터 최적화 클래스
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        model_type: str = 'mlp',
        n_trials: int = 100,
        n_episodes_per_trial: int = 200,
        optimization_metric: str = 'sharpe_ratio',
        results_dir: str = None,
        use_pruning: bool = True,
        random_seed: int = 42
    ):
        """
        HyperparameterOptimizer 초기화
        
        Args:
            symbols: 최적화에 사용할 심볼 리스트
            model_type: 모델 타입 ('mlp', 'cnn', 'lstm', 'mamba')
            n_trials: 최적화 시도 횟수
            n_episodes_per_trial: 각 시도당 학습 에피소드 수
            optimization_metric: 최적화 지표 ('sharpe_ratio', 'total_return', 'final_portfolio_value')
            results_dir: 결과 저장 디렉토리
            use_pruning: 조기 종료 사용 여부
            random_seed: 랜덤 시드
        """
        
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna가 필요합니다. pip install optuna를 실행하세요.")
        
        self.symbols = symbols or TARGET_SYMBOLS[:2]  # 기본적으로 처음 2개 심볼만 사용
        self.model_type = model_type.lower()
        self.n_trials = n_trials
        self.n_episodes_per_trial = n_episodes_per_trial
        self.optimization_metric = optimization_metric
        self.use_pruning = use_pruning
        self.random_seed = random_seed
        
        # 결과 저장 디렉토리 설정
        if results_dir is None:
            timestamp = get_timestamp()
            results_dir = f"hyperopt_results_{self.model_type}_{timestamp}"
        self.results_dir = Path(results_dir)
        create_directory(self.results_dir)
        
        # 데이터 준비
        self.data_results = None
        self.best_trial_info = {}
        
        LOGGER.info(f"🔍 하이퍼파라미터 최적화 초기화 완료")
        LOGGER.info(f"   모델 타입: {self.model_type.upper()}")
        LOGGER.info(f"   심볼: {self.symbols}")
        LOGGER.info(f"   시도 횟수: {self.n_trials}")
        LOGGER.info(f"   에피소드/시도: {self.n_episodes_per_trial}")
        LOGGER.info(f"   최적화 지표: {self.optimization_metric}")
        
    def prepare_data(self):
        """데이터 수집 및 전처리"""
        LOGGER.info("📊 데이터 준비 중...")
        
        # 데이터 수집
        collector = DataCollector(symbols=self.symbols)
        raw_data = collector.load_all_data()
        
        if not raw_data:
            raise ValueError("데이터를 로드할 수 없습니다.")
        
        # 데이터 전처리
        processor = DataProcessor()
        self.data_results = processor.process_all_symbols(raw_data)
        
        LOGGER.info(f"✅ 데이터 준비 완료: {len(self.data_results)}개 심볼")
        
    def define_hyperparameter_space(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        모델 타입별 하이퍼파라미터 검색 공간 정의
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            하이퍼파라미터 딕셔너리
        """
        # 공통 하이퍼파라미터
        params = {
            # 학습률 (로그 스케일)
            'actor_lr': trial.suggest_float('actor_lr', 1e-5, 1e-2, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-5, 1e-2, log=True),
            'alpha_lr': trial.suggest_float('alpha_lr', 1e-5, 1e-2, log=True),
            
            # SAC 파라미터
            'gamma': trial.suggest_float('gamma', 0.85, 0.999),
            'tau': trial.suggest_float('tau', 0.001, 0.05),
            'alpha_init': trial.suggest_float('alpha_init', 0.05, 0.5),
            
            # 네트워크 구조
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512, 1024]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.0, 0.5),
            
            # 학습 파라미터
            'batch_size': trial.suggest_categorical('batch_size', [128, 256, 512, 1024]),
            'gradient_clip_norm': trial.suggest_float('gradient_clip_norm', 0.5, 5.0),
            
            # 버퍼 파라미터
            'buffer_capacity': trial.suggest_categorical('buffer_capacity', [50000, 100000, 500000, 1000000]),
        }
        
        # 모델별 특수 파라미터
        if self.model_type == 'cnn':
            params.update({
                'cnn_dropout_rate': trial.suggest_float('cnn_dropout_rate', 0.1, 0.4),
                # CNN은 이미 구조가 정의되어 있으므로 드롭아웃만 조정
            })
            
        elif self.model_type == 'lstm':
            params.update({
                'lstm_hidden_dim': trial.suggest_categorical('lstm_hidden_dim', [64, 128, 256, 512]),
                'num_lstm_layers': trial.suggest_int('num_lstm_layers', 1, 4),
                'lstm_dropout': trial.suggest_float('lstm_dropout', 0.0, 0.5),
            })
            
        elif self.model_type == 'mlp':
            # MLP는 상태 차원이 고정되어 있으므로 추가 파라미터 없음
            pass
        
        return params
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        최적화 목적 함수
        
        Args:
            trial: Optuna trial 객체
            
        Returns:
            최적화할 스코어 (높을수록 좋음)
        """
        try:
            # 하이퍼파라미터 샘플링
            params = self.define_hyperparameter_space(trial)
            
            # 심볼별 성능 평가
            symbol_scores = []
            
            for symbol in self.symbols:
                if symbol not in self.data_results:
                    continue
                
                try:
                    # 환경 생성
                    env = create_environment_from_results(
                        results=self.data_results,
                        symbol=symbol,
                        data_type='train',
                        log_level='minimal'  # 빠른 실행을 위해 로깅 최소화
                    )
                    
                    # 모델 생성
                    agent = self._create_agent(env, params)
                    
                    # 학습 및 평가
                    score = self._train_and_evaluate(agent, env, trial, symbol)
                    
                    if score is not None:
                        symbol_scores.append(score)
                        LOGGER.info(f"   {symbol}: {score:.4f}")
                    
                    # 메모리 정리
                    del agent, env
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                except Exception as e:
                    LOGGER.warning(f"   {symbol} 평가 실패: {e}")
                    import traceback
                    LOGGER.debug(f"   상세 오류: {traceback.format_exc()}")
                    continue
            
            if not symbol_scores:
                raise optuna.TrialPruned("모든 심볼에서 평가 실패")
            
            # 평균 스코어 반환
            final_score = np.mean(symbol_scores)
            
            # 최고 성능 추적
            if trial.number == 0 or final_score > self.best_trial_info.get('score', -np.inf):
                self.best_trial_info = {
                    'trial_number': trial.number,
                    'score': final_score,
                    'params': params,
                    'symbol_scores': dict(zip(self.symbols[:len(symbol_scores)], symbol_scores))
                }
                
                # 최고 성능 파라미터 저장
                self._save_best_params(params, final_score, trial.number)
            
            LOGGER.info(f"Trial {trial.number}: 평균 스코어 = {final_score:.4f}")
            return final_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            LOGGER.error(f"Trial {trial.number} 실패: {e}")
            raise optuna.TrialPruned(f"Trial 실행 중 오류: {e}")
    
    def _create_agent(self, env, params: Dict[str, Any]) -> SACAgent:
        """하이퍼파라미터로 에이전트 생성"""
        
        # 입력 형태 결정
        if self.model_type in ['cnn', 'lstm']:
            input_shape = (WINDOW_SIZE, env.feature_dim)
            state_dim = None
        else:  # mlp
            input_shape = None
            state_dim = WINDOW_SIZE * env.feature_dim + 2
        
        # 에이전트 생성
        agent = SACAgent(
            state_dim=state_dim,
            action_dim=1,
            hidden_dim=params['hidden_dim'],
            actor_lr=params['actor_lr'],
            critic_lr=params['critic_lr'],
            alpha_lr=params['alpha_lr'],
            gamma=params['gamma'],
            tau=params['tau'],
            alpha_init=params['alpha_init'],
            use_cnn=(self.model_type == 'cnn'),
            use_lstm=(self.model_type == 'lstm'),
            input_shape=input_shape,
            model_type=self.model_type,
            dropout_rate=params['dropout_rate'],
            gradient_clip_norm=params['gradient_clip_norm'],
            buffer_capacity=params['buffer_capacity'],
            # LSTM 전용 파라미터
            lstm_hidden_dim=params.get('lstm_hidden_dim', 128),
            num_lstm_layers=params.get('num_lstm_layers', 2),
            lstm_dropout=params.get('lstm_dropout', 0.2),
            device=DEVICE
        )
        
        # 하이퍼파라미터로 받은 배치 크기 적용
        agent.batch_size = params['batch_size']
        
        return agent
    
    def _train_and_evaluate(
        self, 
        agent: SACAgent, 
        env, 
        trial: optuna.Trial,
        symbol: str
    ) -> Optional[float]:
        """
        에이전트 학습 및 평가
        
        Returns:
            평가 스코어 (None if failed)
        """
        portfolio_values = []
        returns = []
        episode_rewards = []
        
        try:
            for episode in range(self.n_episodes_per_trial):
                state = env.reset()
                episode_reward = 0
                episode_returns = []
                
                for step in range(MAX_STEPS_PER_EPISODE):
                    # 행동 선택
                    action = agent.select_action(state, evaluate=False)
                    
                    # 환경에서 스텝 실행
                    next_state, reward, done, info = env.step(action)
                    
                    # 경험 저장
                    agent.add_experience(state, action, reward, next_state, done)
                    
                    # 모델 업데이트
                    if len(agent.replay_buffer) > agent.batch_size:
                        agent.update_parameters(agent.batch_size)
                    
                    episode_reward += reward
                    episode_returns.append(info['portfolio_value'])
                    
                    state = next_state
                    
                    if done:
                        break
                
                episode_rewards.append(episode_reward)
                portfolio_values.extend(episode_returns)
                
                # 중간 성능 체크 (Pruning)
                if self.use_pruning and episode > 20 and episode % 10 == 0:
                    intermediate_score = self._calculate_score(episode_rewards, portfolio_values)
                    trial.report(intermediate_score, episode)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"중간 성능 부족으로 조기 종료 (episode {episode})")
            
            # 최종 스코어 계산
            final_score = self._calculate_score(episode_rewards, portfolio_values)
            return final_score
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            LOGGER.warning(f"학습 중 오류 ({symbol}): {e}")
            return None
    
    def _calculate_score(self, episode_rewards: List[float], portfolio_values: List[float]) -> float:
        """성능 스코어 계산"""
        if not episode_rewards or not portfolio_values:
            return -np.inf
        
        try:
            if self.optimization_metric == 'total_return':
                initial_value = INITIAL_BALANCE
                final_value = portfolio_values[-1] if portfolio_values else initial_value
                return (final_value - initial_value) / initial_value
                
            elif self.optimization_metric == 'sharpe_ratio':
                returns = np.diff(portfolio_values) / portfolio_values[:-1] if len(portfolio_values) > 1 else [0]
                if len(returns) < 2 or np.std(returns) == 0:
                    return -np.inf
                return np.mean(returns) / np.std(returns) * np.sqrt(252)  # 연간화
                
            elif self.optimization_metric == 'final_portfolio_value':
                return portfolio_values[-1] if portfolio_values else 0
                
            else:  # 기본값: 평균 에피소드 보상
                return np.mean(episode_rewards)
                
        except Exception as e:
            LOGGER.warning(f"스코어 계산 오류: {e}")
            return -np.inf
    
    def _save_best_params(self, params: Dict[str, Any], score: float, trial_number: int):
        """최고 성능 파라미터 저장"""
        best_params_file = self.results_dir / "best_hyperparameters.json"
        
        save_data = {
            'trial_number': trial_number,
            'score': float(score),
            'optimization_metric': self.optimization_metric,
            'model_type': self.model_type,
            'symbols': self.symbols,
            'hyperparameters': params,
            'timestamp': datetime.now().isoformat(),
            'n_episodes_per_trial': self.n_episodes_per_trial
        }
        
        with open(best_params_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        LOGGER.info(f"💾 최고 성능 파라미터 저장됨: {best_params_file}")
    
    def optimize(self) -> Dict[str, Any]:
        """
        하이퍼파라미터 최적화 실행
        
        Returns:
            최적화 결과 딕셔너리
        """
        if self.data_results is None:
            self.prepare_data()
        
        LOGGER.info(f"🚀 하이퍼파라미터 최적화 시작")
        LOGGER.info(f"   모델: {self.model_type.upper()}")
        LOGGER.info(f"   목표: {self.optimization_metric}")
        
        # Optuna 스터디 생성
        sampler = TPESampler(seed=self.random_seed)
        
        if self.use_pruning:
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=20)
        else:
            pruner = optuna.pruners.NopPruner()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"sac_{self.model_type}_optimization"
        )
        
        # 최적화 실행
        start_time = time.time()
        
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=None,
            show_progress_bar=True
        )
        
        optimization_time = time.time() - start_time
        
        # 결과 정리 (완료된 trial이 없는 경우 처리)
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if not completed_trials:
            LOGGER.error("❌ 완료된 trial이 없습니다!")
            LOGGER.error(f"   총 시도: {len(study.trials)}")
            LOGGER.error(f"   실패: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
            LOGGER.error(f"   가지치기: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
            
            # 기본 결과 반환
            results = {
                'best_score': float('-inf'),
                'best_params': {},
                'model_type': self.model_type,
                'optimization_metric': self.optimization_metric,
                'symbols': self.symbols,
                'n_trials': len(study.trials),
                'optimization_time_hours': optimization_time / 3600,
                'study': study,
                'error': 'No completed trials'
            }
            return results
        
        best_trial = study.best_trial
        best_params = best_trial.params
        best_score = best_trial.value
        
        LOGGER.info(f"✅ 최적화 완료!")
        LOGGER.info(f"   소요 시간: {optimization_time/3600:.2f} 시간")
        LOGGER.info(f"   최고 스코어: {best_score:.4f}")
        LOGGER.info(f"   완료된 시도: {len(study.trials)}")
        LOGGER.info(f"   가지치기된 시도: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
        
        # 최종 결과 저장
        results = {
            'best_score': best_score,
            'best_params': best_params,
            'model_type': self.model_type,
            'optimization_metric': self.optimization_metric,
            'symbols': self.symbols,
            'n_trials': len(study.trials),
            'optimization_time_hours': optimization_time / 3600,
            'study': study
        }
        
        self._save_optimization_results(results, study)
        
        return results
    
    def _save_optimization_results(self, results: Dict[str, Any], study: optuna.Study):
        """최적화 결과 저장"""
        
        # 전체 결과 저장
        results_file = self.results_dir / "optimization_results.json"
        save_data = {k: v for k, v in results.items() if k != 'study'}  # study 객체 제외
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        # Optuna 스터디 저장
        study_file = self.results_dir / "optuna_study.pkl"
        optuna.pickle_storage.dump_study(study, study_file)
        
        # 시각화 생성
        self._create_optimization_plots(study)
        
        LOGGER.info(f"📁 결과 저장 완료: {self.results_dir}")
    
    def _create_optimization_plots(self, study: optuna.Study):
        """최적화 결과 시각화"""
        try:
            import matplotlib.pyplot as plt
            
            # 1. 최적화 히스토리
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Trial 값 히스토리
            trial_values = [t.value for t in study.trials if t.value is not None]
            trial_numbers = [t.number for t in study.trials if t.value is not None]
            
            axes[0, 0].plot(trial_numbers, trial_values, 'b-', alpha=0.6)
            axes[0, 0].axhline(y=study.best_value, color='r', linestyle='--', label=f'Best: {study.best_value:.4f}')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Score')
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # 파라미터 중요도 (상위 5개)
            if len(study.trials) >= 10:
                try:
                    importance = optuna.importance.get_param_importances(study)
                    top_params = list(importance.items())[:5]
                    
                    if top_params:
                        params, values = zip(*top_params)
                        axes[0, 1].barh(params, values)
                        axes[0, 1].set_xlabel('Importance')
                        axes[0, 1].set_title('Parameter Importance (Top 5)')
                except:
                    axes[0, 1].text(0.5, 0.5, 'Parameter importance\ncalculation failed', 
                                   ha='center', va='center', transform=axes[0, 1].transAxes)
            else:
                axes[0, 1].text(0.5, 0.5, 'Not enough trials\nfor importance analysis', 
                               ha='center', va='center', transform=axes[0, 1].transAxes)
            
            # 학습률 분포 (actor_lr 예시)
            lr_values = [t.params.get('actor_lr') for t in study.trials if t.params.get('actor_lr')]
            if lr_values:
                axes[1, 0].hist(lr_values, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].axvline(x=study.best_params.get('actor_lr', 0), color='r', linestyle='--', 
                                  label=f"Best: {study.best_params.get('actor_lr', 0):.2e}")
                axes[1, 0].set_xlabel('Actor Learning Rate')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Actor LR Distribution')
                axes[1, 0].legend()
                axes[1, 0].set_xscale('log')
            
            # Hidden dimension 분포
            hidden_dims = [t.params.get('hidden_dim') for t in study.trials if t.params.get('hidden_dim')]
            if hidden_dims:
                unique_dims, counts = np.unique(hidden_dims, return_counts=True)
                bars = axes[1, 1].bar(unique_dims.astype(str), counts, alpha=0.7, edgecolor='black')
                
                # 최고 성능 하이든 디멘션 강조
                best_hidden = study.best_params.get('hidden_dim')
                if best_hidden in unique_dims:
                    best_idx = np.where(unique_dims == best_hidden)[0][0]
                    bars[best_idx].set_color('red')
                    bars[best_idx].set_alpha(1.0)
                
                axes[1, 1].set_xlabel('Hidden Dimension')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Hidden Dimension Distribution')
                axes[1, 1].set_xticks(range(len(unique_dims)))
                axes[1, 1].set_xticklabels(unique_dims.astype(str))
            
            plt.tight_layout()
            plt.savefig(self.results_dir / "optimization_plots.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            LOGGER.info("📊 최적화 시각화 생성 완료")
            
        except Exception as e:
            LOGGER.warning(f"시각화 생성 실패: {e}")
    
    def load_best_hyperparameters(self, file_path: str = None) -> Dict[str, Any]:
        """저장된 최고 성능 하이퍼파라미터 로드"""
        if file_path is None:
            file_path = self.results_dir / "best_hyperparameters.json"
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            LOGGER.info(f"✅ 최고 성능 하이퍼파라미터 로드됨: {file_path}")
            LOGGER.info(f"   스코어: {data['score']:.4f}")
            LOGGER.info(f"   Trial: {data['trial_number']}")
            
            return data
            
        except FileNotFoundError:
            LOGGER.error(f"❌ 파일을 찾을 수 없습니다: {file_path}")
            return {}
        except Exception as e:
            LOGGER.error(f"❌ 파일 로드 실패: {e}")
            return {}


def run_hyperparameter_optimization(
    symbols: List[str] = None,
    model_type: str = 'mlp',
    n_trials: int = 50,
    n_episodes_per_trial: int = 100,
    optimization_metric: str = 'sharpe_ratio'
) -> Dict[str, Any]:
    """
    하이퍼파라미터 최적화 실행 함수
    
    Args:
        symbols: 사용할 심볼 리스트
        model_type: 모델 타입
        n_trials: 시도 횟수
        n_episodes_per_trial: 에피소드 수
        optimization_metric: 최적화 지표
        
    Returns:
        최적화 결과
    """
    
    LOGGER.info("🔍 하이퍼파라미터 최적화 시작")
    
    # 최적화 객체 생성
    optimizer = HyperparameterOptimizer(
        symbols=symbols or TARGET_SYMBOLS[:2],
        model_type=model_type,
        n_trials=n_trials,
        n_episodes_per_trial=n_episodes_per_trial,
        optimization_metric=optimization_metric,
        use_pruning=True,
        random_seed=42
    )
    
    # 최적화 실행
    results = optimizer.optimize()
    
    # 결과 출력
    LOGGER.info("=" * 60)
    LOGGER.info("🏆 하이퍼파라미터 최적화 결과")
    LOGGER.info("=" * 60)
    LOGGER.info(f"모델 타입: {model_type.upper()}")
    LOGGER.info(f"최고 스코어: {results['best_score']:.4f}")
    LOGGER.info(f"최적 파라미터:")
    
    for param, value in results['best_params'].items():
        if isinstance(value, float):
            LOGGER.info(f"  {param}: {value:.6f}")
        else:
            LOGGER.info(f"  {param}: {value}")
    
    LOGGER.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SAC 하이퍼파라미터 최적화')
    parser.add_argument('--model_type', choices=['mlp', 'cnn', 'lstm', 'mamba'], default='mlp',
                       help='최적화할 모델 타입')
    parser.add_argument('--symbols', nargs='+', default=None,
                       help='사용할 심볼 리스트')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='최적화 시도 횟수')
    parser.add_argument('--n_episodes', type=int, default=100,
                       help='각 시도당 학습 에피소드 수')
    parser.add_argument('--metric', choices=['sharpe_ratio', 'total_return', 'final_portfolio_value'],
                       default='sharpe_ratio', help='최적화 지표')
    
    args = parser.parse_args()
    
    # 최적화 실행
    results = run_hyperparameter_optimization(
        symbols=args.symbols,
        model_type=args.model_type,
        n_trials=args.n_trials,
        n_episodes_per_trial=args.n_episodes,
        optimization_metric=args.metric
    )
    
    print(f"\n🎉 최적화 완료! 결과는 {results.get('results_dir', 'hyperopt_results')} 디렉토리에 저장되었습니다.") 