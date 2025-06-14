"""
SAC 모델 학습 실행 스크립트 (통합 간소화 버전)
- 검증 없이 훈련에만 집중
- 학습 시간 측정 및 로깅
- 간단하고 명확한 로그 출력
- 통합된 모델 설정 사용
"""
import sys
import os
import time
from datetime import datetime, timedelta

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import argparse
import torch
import numpy as np
from pathlib import Path

from src.config.ea_teb_config import (
    DEVICE,
    HIDDEN_DIM,
    BATCH_SIZE,
    NUM_EPISODES,
    EVALUATE_INTERVAL,
    SAVE_MODEL_INTERVAL,
    MAX_STEPS_PER_EPISODE,
    TARGET_SYMBOLS,
    LOGGER,
    INITIAL_BALANCE,
    WINDOW_SIZE,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    LEARNING_RATE_ALPHA,
    ALPHA_INIT,
    GRADIENT_CLIP,
)
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment, MultiAssetTradingEnvironment
from src.models.sac_agent import SACAgent
from src.utils.utils import create_directory, get_timestamp

episode_actions_history: list = []

class TrainingTimer:
    """학습 시간 측정을 위한 클래스"""
    
    def __init__(self):
        self.start_time = None
        self.episode_start_time = None
        self.episode_times = []
        
    def start_training(self):
        """전체 학습 시작"""
        self.start_time = time.time()
        LOGGER.info(f"🚀 학습 시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    def start_episode(self):
        """에피소드 시작"""
        self.episode_start_time = time.time()
        
    def end_episode(self):
        """에피소드 종료"""
        if self.episode_start_time:
            episode_time = time.time() - self.episode_start_time
            self.episode_times.append(episode_time)
            return episode_time
        return 0
        
    def get_training_time(self):
        """전체 학습 시간 반환"""
        if self.start_time:
            return time.time() - self.start_time
        return 0
        
    def get_avg_episode_time(self):
        """평균 에피소드 시간 반환"""
        if self.episode_times:
            return np.mean(self.episode_times)
        return 0
        
    def get_eta(self, current_episode, total_episodes):
        """남은 시간 추정"""
        if len(self.episode_times) > 0:
            avg_time = self.get_avg_episode_time()
            remaining_episodes = total_episodes - current_episode
            return remaining_episodes * avg_time
        return 0
        
    def format_time(self, seconds):
        """시간을 보기 좋게 포맷"""
        return str(timedelta(seconds=int(seconds)))

def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='SAC 모델 학습 스크립트 (통합 간소화 버전)')
    
    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None, help='학습에 사용할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true', help='데이터 수집 여부')
    
    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=INITIAL_BALANCE, help='초기 자본금')
    parser.add_argument('--multi_asset', action='store_true', help='다중 자산 환경 사용 여부')
    
    # 모델 관련 인자(기본 MLP)
    parser.add_argument('--hidden_dim', type=int, default=HIDDEN_DIM, help='은닉층 차원')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'cnn', 'lstm', 'mamba', 'tinytransformer'], 
                       help='모델 타입 (--use_* 옵션 대신 사용 가능)')
    # CNN
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용 여부')
    # LSTM
    parser.add_argument('--use_lstm', action='store_true', help='LSTM 모델 사용 여부')
    # Mamba
    parser.add_argument('--use_mamba', action='store_true', help='Mamba 모델 사용 여부')
    # TinyTransformer
    parser.add_argument('--use_tinytransformer', action='store_true', help='TinyTransformer 모델 사용 여부')
    # 로드 경로
    parser.add_argument('--load_model', type=str, default=None, help='로드할 모델 경로')
    
    # 학습 관련 인자
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--num_episodes', type=int, default=NUM_EPISODES, help='학습할 총 에피소드 수')
    parser.add_argument('--log_interval', type=int, default=EVALUATE_INTERVAL, help='로그 출력 간격')
    parser.add_argument('--save_interval', type=int, default=SAVE_MODEL_INTERVAL, help='모델 저장 간격')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_PER_EPISODE, help='에피소드당 최대 스텝 수')
    parser.add_argument('--buffer_type', type=str, default='standard', choices=['standard', 'prioritized'], help='리플레이 버퍼 타입')

    
    # 스텝별 로깅 관련 인자
    parser.add_argument('--step_log_interval', type=int, default=100, help='스텝별 로그 출력 간격')
    
    # 로그 레벨
    parser.add_argument("--log_level", choices=['minimal', 'normal', 'detailed'], default='normal', help="로그 레벨")
    
    return parser.parse_args()

def create_training_environment(results, symbols, args):
    """표준 SAC 학습용 환경을 생성"""
    LOGGER.info("표준 SAC 학습용 환경 생성 중...")
    
    if args.multi_asset:
        # 다중 자산의 경우 기존 방식 유지 (일단)
        LOGGER.error("❌ 다중 자산 환경은 아직 구현되지 않았습니다.")
        LOGGER.info("단일 자산 모드로 실행하세요: --multi_asset 옵션 제거")
        return None
        
    else:
        # 단일 자산 기본 환경 생성
        symbol = symbols[0]
        LOGGER.info(f"단일 자산 트레이딩 환경 생성 중: {symbol}")
        
        if symbol not in results:
            LOGGER.error(f"{symbol} 데이터 처리 결과가 없습니다.")
            return None
        
        if 'train' in results[symbol] and 'featured_data' in results[symbol]:
            normalized_data = results[symbol]['train']
            original_data = results[symbol]['featured_data']
        else:
            LOGGER.error(f"{symbol} 훈련 데이터가 없습니다.")
            return None
        
        # 기본 환경 생성
        base_env = TradingEnvironment(
            data=normalized_data,
            raw_data=original_data,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            symbol=symbol,
            log_level=args.log_level,
            train_data=True
        )
        
        LOGGER.info(f"✅ 표준 SAC 학습 환경 생성 완료")
        LOGGER.info(f"   └─ 데이터 길이: {len(normalized_data)}")
        LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
        LOGGER.info(f"   └─ 초기 잔고: ${args.initial_balance:,.0f}")
        
        return base_env

def create_agent(env, args):
    """SAC 에이전트 생성 (동적 설정 적용 버전)"""
    LOGGER.info("SAC 에이전트 생성 중...")

    # 행동 차원 결정
    if args.multi_asset:
        action_dim = len(env.envs)
    else:
        action_dim = 1

    # --model_type 인자가 있으면 우선 사용
    if args.model_type:
        model_type = args.model_type.lower()
        args.use_cnn = (model_type == 'cnn')
        args.use_lstm = (model_type == 'lstm')
        args.use_mamba = (model_type == 'mamba')
        args.use_tinytransformer = (model_type == 'tinytransformer')
        LOGGER.info(f"📋 --model_type 인자로 {model_type.upper()} 모델 지정됨")
    else:
        # 상호 배타적 검증 추가
        model_flags = [args.use_cnn, args.use_lstm, args.use_mamba, args.use_tinytransformer]
        if sum(model_flags) > 1:
            LOGGER.error("❌ CNN, LSTM, Mamba, TinyTransformer 중 하나만 선택할 수 있습니다.")
            return None

        # --use_* 옵션으로 모델 타입 결정
        if args.use_tinytransformer:
            model_type = 'tinytransformer'
            LOGGER.info("🤖 TinyTransformer 모델을 사용합니다.")
        elif args.use_mamba:
            model_type = 'mamba'
            LOGGER.info("🐍 Mamba 모델을 사용합니다.")
        elif args.use_lstm:
            model_type = 'lstm'
            LOGGER.info("🧠 LSTM 모델을 사용합니다.")
        elif args.use_cnn:
            model_type = 'cnn'
            LOGGER.info("🖼️ CNN 모델을 사용합니다.")
        else:
            model_type = 'mlp'
            LOGGER.info("📊 기본 MLP 모델을 사용합니다.")

    # 🔥 심볼 정보 (동적 설정에 필요)
    symbol = None
    if hasattr(args, 'symbols') and args.symbols and len(args.symbols) == 1:
        symbol = args.symbols[0]
    elif hasattr(env, 'symbol'):
        symbol = env.symbol

    LOGGER.info(f"🔧 동적 모델 설정:")
    LOGGER.info(f"   └─ 타입: {model_type.upper()}")
    LOGGER.info(f"   └─ 심볼: {symbol or 'Multi/Unknown'}")
    LOGGER.info(f"   └─ 설정 소스: ea_teb_config.MODEL_DEFAULTS")

    # 🤖 TinyTransformer 전용 최적화 설정 적용
    tt_config = {}  # 기본값 초기화
    
    if args.use_tinytransformer:
        try:
            from src.config.tinytransformer_config import get_optimized_config
            tt_config = get_optimized_config()
            
            LOGGER.info("🤖 TinyTransformer 최적화 설정 적용 중...")
            LOGGER.info(f"   └─ Batch Size: {tt_config['batch_size']}")
            LOGGER.info(f"   └─ Actor LR: {tt_config['actor_lr']}")
            LOGGER.info(f"   └─ Critic LR: {tt_config['critic_lr']}")
            LOGGER.info(f"   └─ Dropout: {tt_config['dropout_rate']}")
            LOGGER.info(f"   └─ Max Steps/Episode: {tt_config['max_steps_per_episode']}")
            
            # args에 TinyTransformer 설정 적용
            args.batch_size = tt_config['batch_size']
            args.max_steps_per_episode = tt_config['max_steps_per_episode']
            args.replay_buffer_size = tt_config['replay_buffer_size']
            args.num_episodes = tt_config['num_episodes']
            args.evaluate_interval = tt_config['evaluate_interval']
            args.save_model_interval = tt_config['save_model_interval']
            
        except ImportError as e:
            LOGGER.warning(f"⚠️ TinyTransformer 설정 로드 실패: {e}")
            LOGGER.info("기본 설정을 사용합니다.")
            tt_config = {
                'actor_lr': 8e-5,
                'critic_lr': 1e-4,
                'alpha_lr': 8e-5,
                'dropout_rate': 0.15,
                'alpha_init': 0.15,
                'gradient_clip_norm': 0.5,
            }

    # 🔥 에이전트 생성 파라미터 (TinyTransformer 최적화 적용)
    agent_kwargs = {
        'state_dim': None,
        'action_dim': action_dim,
        'hidden_dim': args.hidden_dim,
        'input_shape': (args.window_size, env.feature_dim if not args.multi_asset else list(env.envs.values())[0].feature_dim),
        'use_cnn': args.use_cnn,
        'use_lstm': args.use_lstm,
        'use_mamba': args.use_mamba,
        'use_tinytransformer': args.use_tinytransformer,
        'model_type': model_type,
        'symbol': symbol,  # 🆕 심볼 전달 (동적 설정용)
        
        # 🔥 모델별 최적화된 설정 적용
        'actor_lr': tt_config['actor_lr'] if args.use_tinytransformer else LEARNING_RATE_ACTOR,     
        'critic_lr': tt_config['critic_lr'] if args.use_tinytransformer else LEARNING_RATE_CRITIC,   
        'alpha_lr': tt_config['alpha_lr'] if args.use_tinytransformer else LEARNING_RATE_ALPHA,     
        'alpha_init': tt_config['alpha_init'] if args.use_tinytransformer else ALPHA_INIT,            
        'dropout_rate': tt_config['dropout_rate'] if args.use_tinytransformer else 0.1,                 
        'gradient_clip_norm': tt_config['gradient_clip_norm'] if args.use_tinytransformer else GRADIENT_CLIP, 
        
        # LSTM 전용 파라미터
        'lstm_hidden_dim': 128,              
        'num_lstm_layers': 2,                
        'lstm_dropout': 0.2,                 
              
        # 🆕 안정성 파라미터 (SAC 표준)
        'use_gradient_clipping': True,
        'adaptive_alpha': True,
        
        # 기타 파라미터
        'training_symbols': args.symbols if args.symbols else TARGET_SYMBOLS,
    }

    # 🔥 에이전트 생성 (내부에서 동적 설정 적용됨)
    agent = SACAgent(**agent_kwargs)

    # 모델을 디바이스로 이동
    agent.actor = agent.actor.to(DEVICE)
    agent.critic = agent.critic.to(DEVICE)
    agent.critic_target = agent.critic_target.to(DEVICE)
    
    # 모델 타입 정보를 에이전트에 저장 (저장 시 사용)
    agent.model_type = model_type
    agent.training_symbols = args.symbols if args.symbols else TARGET_SYMBOLS

    # 모델 로드 (선택적)
    if args.load_model:
        LOGGER.info(f"모델 로드 중: {args.load_model}")
        try:
            agent.load_model(args.load_model)
            # 로드 후에도 명시적으로 다시 이동 (GPU 로드 시 필요함)
            agent.actor = agent.actor.to(DEVICE)
            agent.critic = agent.critic.to(DEVICE)
            agent.critic_target = agent.critic_target.to(DEVICE)
            LOGGER.info("✅ 모델 로드 성공")
        except Exception as e:
            LOGGER.error(f"❌ 모델 로드 실패: {e}")
            LOGGER.info("✅ 새 모델로 학습을 시작합니다.")
    
    # 🔥 최종 설정 확인 로그 (동적 적용 결과)
    LOGGER.info(f"✅ SAC 에이전트 생성 완료")
    LOGGER.info(f"   └─ 모델: {model_type.upper()}")
    LOGGER.info(f"   └─ 행동 차원: {action_dim}")
    LOGGER.info(f"   └─ 은닉층: {args.hidden_dim}")
    LOGGER.info(f"   └─ 장치: {DEVICE}")
    LOGGER.info(f"   └─ 최종 Actor LR: {agent.actor_lr:.6f}")
    LOGGER.info(f"   └─ 최종 Critic LR: {agent.critic_lr:.6f}")
    LOGGER.info(f"   └─ 최종 Dropout: {agent.dropout_rate}")
    LOGGER.info(f"   └─ 버퍼 크기: {len(agent.replay_buffer) if hasattr(agent.replay_buffer, '__len__') else 'N/A'}")
    
    return agent

def validate_training_environment(train_env, args):
    """학습 환경 검증 및 데이터 손실 위험 사전 점검"""
    LOGGER.info("🔍 학습 환경 검증 중...")
    
    # 순차적 환경인지 확인
    if not hasattr(train_env, 'episode_manager'):
        LOGGER.warning("⚠️ 순차적 환경이 아닙니다. 데이터 활용률이 낮을 수 있습니다.")
        return True
    
    # 커버리지 분석
    episode_manager = train_env.episode_manager
    
    if hasattr(episode_manager, 'get_coverage_summary'):
        coverage = episode_manager.get_coverage_summary()
        
        LOGGER.info("📊 환경 검증 결과:")
        LOGGER.info(f"   └─ 총 에피소드 수: {coverage['total_episodes']}")
        LOGGER.info(f"   └─ 데이터 커버리지: {coverage['unique_coverage_pct']:.1f}%")
        LOGGER.info(f"   └─ 평균 에피소드 길이: {coverage['average_episode_length']:.1f}")
        
        # 경고 및 권장사항
        if coverage['unique_coverage_pct'] < 95:
            LOGGER.warning(f"⚠️ 데이터 커버리지가 낮습니다: {coverage['unique_coverage_pct']:.1f}%")
        
        if coverage['total_episodes'] > args.num_episodes:
            LOGGER.warning(f"⚠️ 계획된 에피소드 수({coverage['total_episodes']})가 학습 에피소드 수({args.num_episodes})보다 많습니다.")
            LOGGER.info("💡 권장사항: --num_episodes 값을 늘려서 전체 데이터를 활용하세요.")
        
        # 에피소드 타입 분포 표시
        if 'episode_types' in coverage:
            LOGGER.info("📊 에피소드 타입 분포:")
            for ep_type, count in coverage['episode_types'].items():
                percentage = (count / coverage['total_episodes']) * 100
                LOGGER.info(f"   └─ {ep_type}: {count}개 ({percentage:.1f}%)")
        
        LOGGER.info("✅ 환경 검증 완료")
        return True
        
    else:
        LOGGER.info("✅ 기본 순차적 환경 사용")
        return True

class StepLossTracker:
    """스텝별 loss 추적을 위한 클래스"""
    
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.alpha_values = []
        self.entropy_values = []
        
    def add_stats(self, stats):
        """통계 추가"""
        if stats is not None:
            self.actor_losses.append(stats.get('actor_loss', 0.0))
            self.critic_losses.append(stats.get('critic_loss', 0.0))
            self.alpha_losses.append(stats.get('alpha_loss', 0.0))
            self.alpha_values.append(stats.get('alpha', 0.0))
            self.entropy_values.append(stats.get('entropy', 0.0))
            
            # 윈도우 크기 유지
            if len(self.actor_losses) > self.window_size:
                self.actor_losses.pop(0)
                self.critic_losses.pop(0)
                self.alpha_losses.pop(0)
                self.alpha_values.pop(0)
                self.entropy_values.pop(0)
    
    def get_averages(self):
        """평균값 반환"""
        if not self.actor_losses:
            return {
                'avg_actor_loss': 0.0,
                'avg_critic_loss': 0.0,
                'avg_alpha_loss': 0.0,
                'avg_alpha': 0.0,
                'avg_entropy': 0.0,
                'num_samples': 0
            }
        
        return {
            'avg_actor_loss': np.mean(self.actor_losses),
            'avg_critic_loss': np.mean(self.critic_losses),
            'avg_alpha_loss': np.mean(self.alpha_losses),
            'avg_alpha': np.mean(self.alpha_values),
            'avg_entropy': np.mean(self.entropy_values),
            'num_samples': len(self.actor_losses)
        }
    
    def check_convergence(self, episode, min_episodes=100):
        """얼리스탑 수렴 조건 체크"""
        if episode < min_episodes:
            return False, "최소 에피소드 미달성"
        
        if len(self.actor_losses) < 20:
            return False, "데이터 부족"
        
        # 최근 20개 샘플 기준
        recent_alpha = np.mean(self.alpha_values[-20:]) if self.alpha_values else 1.0
        recent_entropy = np.mean(self.entropy_values[-20:]) if self.entropy_values else 1.0
        recent_actor_loss = np.mean(np.abs(self.actor_losses[-20:]))
        
        # 수렴 조건 체크
        alpha_converged = recent_alpha < 0.001
        entropy_converged = 0.06 < recent_entropy < 0.08
        loss_converged = recent_actor_loss < 0.001
        
        all_converged = alpha_converged and entropy_converged and loss_converged
        
        if all_converged:
            reason = f"수렴 감지 - Alpha:{recent_alpha:.6f}, Entropy:{recent_entropy:.3f}, Loss:{recent_actor_loss:.6f}"
            return True, reason
        
        return False, "수렴 조건 미달성"

def log_step_progress(episode, step, total_episodes, max_episode_steps, 
                     loss_tracker, timer, args, agent):
    """스텝별 진행상황 로깅"""
    
    # 진행률 계산
    episode_progress = (step / max_episode_steps) * 100
    total_progress = ((episode - 1) * 100 + episode_progress) / total_episodes
    
    # 평균 loss 가져오기
    averages = loss_tracker.get_averages()
    
    # 시간 정보
    elapsed_time = timer.get_training_time()
    eta = timer.get_eta(episode, total_episodes)
    
    LOGGER.info("=" * 50)
    LOGGER.info(f"✅ STEP PROGRESS | Episode {episode}/{total_episodes} | Step {step}/{max_episode_steps}")
    LOGGER.info("=" * 50)
    LOGGER.info(f"⏰ 진행률:")
    LOGGER.info(f"   └─ 에피소드 진행률: {episode_progress:.1f}%")
    LOGGER.info(f"   └─ 전체 진행률: {total_progress:.1f}%")
    LOGGER.info(f"   └─ 완료된 에피소드: {episode-1}/{total_episodes}")
    
    LOGGER.info(f"⏰ 시간 정보:")
    LOGGER.info(f"   └─ 경과 시간: {timer.format_time(elapsed_time)}")
    LOGGER.info(f"   └─ 예상 남은 시간: {timer.format_time(eta)}")
    
    LOGGER.info(f"✅ 평균 Loss (최근 {averages['num_samples']}스텝):")
    LOGGER.info(f"   └─ Actor Loss: {averages['avg_actor_loss']:.6f}")
    LOGGER.info(f"   └─ Critic Loss: {averages['avg_critic_loss']:.6f}")
    LOGGER.info(f"   └─ Alpha Loss: {averages['avg_alpha_loss']:.6f}")
    LOGGER.info(f"   └─ Alpha: {averages['avg_alpha']:.6f}")
    LOGGER.info(f"   └─ Entropy: {averages['avg_entropy']:.6f}")
    
    # 디버깅 정보
    LOGGER.info(f"🔍 디버깅 정보:")
    LOGGER.info(f"   └─ 버퍼 크기: {len(agent.replay_buffer):,}")
    LOGGER.info(f"   └─ 배치 크기: {args.batch_size}")
    LOGGER.info(f"   └─ 업데이트 카운터: {agent.update_counter}")
    LOGGER.info(f"   └─ Loss 샘플 수: {averages['num_samples']}")
    
    LOGGER.info("=" * 50)

def train_agent_sac(agent, train_env, args, timer):
    """표준 SAC 알고리즘 학습"""
    LOGGER.info("표준 SAC 학습 시작...")
    global episode_actions_history
    
    episode_rewards = []
    portfolio_values = []
    shares_history = []
    
    # 스텝별 loss 추적기 초기화
    loss_tracker = StepLossTracker(window_size=args.step_log_interval)
    global_step_count = 0
    
    for episode in range(args.num_episodes):
        episode_actions = []
        timer.start_episode()
        
        # 표준 환경 리셋 (랜덤 시작점)
        state = train_env.reset()
        
        episode_reward = 0
        steps = 0
        done = False
        
        LOGGER.info(f"✅ Episode {episode+1} 시작")
        LOGGER.info(f"   └─ 현재 스텝: {train_env.current_step}")
        LOGGER.info(f"   └─ 최대 스텝: {args.max_steps}")
            
        # 단일 에피소드 진행 루프
        while not done and steps < args.max_steps:
            # 행동 선택
            action = agent.select_action(state, evaluate=False)
            episode_actions.append(action)
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = train_env.step(action)
            
            # 경험을 리플레이 버퍼에 추가
            agent.add_experience(state, action, reward, next_state, done)
                     
            # 네트워크 업데이트
            if len(agent.replay_buffer) > args.batch_size:
                stats = agent.update_parameters(args.batch_size)
                loss_tracker.add_stats(stats)
                
                # 긴급 중단 체크 (Alpha 완전 소멸)
                if stats.get('alpha', 1.0) < 1e-6:
                    LOGGER.warning("🚨 Alpha 완전 소멸, 긴급 중단!")
                    done = True
                    break
            else:
                # 업데이트하지 않는 경우 None 추가
                loss_tracker.add_stats(None)
            
            episode_reward += reward
            steps += 1
            global_step_count += 1
            state = next_state
            
            # 스텝별 진행상황 로깅
            if args.step_log_interval > 0 and steps % args.step_log_interval == 0:
                log_step_progress(
                    episode + 1, steps, args.num_episodes, args.max_steps, 
                    loss_tracker, timer, args, agent
                )
                
                # 얼리스탑 체크 
                should_stop, reason = loss_tracker.check_convergence(episode + 1)
                if should_stop:
                    LOGGER.info(f"🛑 얼리스탑 발동: {reason}")
                    done = True
                    break
        
        # 에피소드 완료 후 행동 기록 추가
        episode_actions_history.append(episode_actions)
        
        episode_time = timer.end_episode()
        episode_rewards.append(episode_reward)
        
        # 적극적인 메모리 관리
        if episode % 5 == 0:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                torch.cuda.reset_peak_memory_stats()

        # 매 에피소드마다 가벼운 정리
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # VRAM 사용량 모니터링 (50 에피소드마다)
        if episode % 50 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            LOGGER.info(f"   └─ VRAM 사용량: 할당 {allocated:.2f}GB, 캐시 {reserved:.2f}GB")
            
        portfolio_values.append(info.get('portfolio_value', args.initial_balance))
        shares_history.append(info.get('shares_held', 0))
        
        # 에피소드 종료 시 최종 평균 loss 출력
        final_averages = loss_tracker.get_averages()
        LOGGER.info(f"✅ Episode {episode+1} 완료:")
        LOGGER.info(f"   └─ 에피소드 보상: {episode_reward:.4f}")
        LOGGER.info(f"   └─ 실행 스텝: {steps}/{args.max_steps}")
        LOGGER.info(f"   └─ 포트폴리오: ${info.get('portfolio_value', args.initial_balance):.2f}")
        
        # 리스크 정보 출력
        if hasattr(train_env, 'get_risk_metrics'):
            risk_metrics = train_env.get_risk_metrics()
            if 'max_drawdown_pct' in risk_metrics:
                LOGGER.info(f"   └─ 최대 낙폭: {risk_metrics['max_drawdown_pct']:.2f}%")
        
        LOGGER.info(f"   └─ 평균 Actor Loss: {final_averages['avg_actor_loss']:.6f}")
        LOGGER.info(f"   └─ 평균 Critic Loss: {final_averages['avg_critic_loss']:.6f}")
        LOGGER.info(f"   └─ 평균 Alpha: {final_averages['avg_alpha']:.6f}")
        LOGGER.info(f"   └─ 에피소드 시간: {episode_time:.2f}초")
        
        # 에피소드 완료 시 얼리스탑 체크
        should_stop, reason = loss_tracker.check_convergence(episode + 1)
        if should_stop:
            LOGGER.info(f"🛑 에피소드 완료 후 얼리스탑 발동: {reason}")
            break
            
        # 보상 분석 (10 에피소드마다)
        if episode % 10 == 0 and len(episode_rewards) >= 10:
            recent_rewards = episode_rewards[-10:]
            max_reward = max(recent_rewards)
            min_reward = min(recent_rewards)
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            LOGGER.info(f"✅ 최근 10 에피소드 분석:")
            LOGGER.info(f"   └─ 보상 범위: [{min_reward:.4f} ~ {max_reward:.4f}]")
            LOGGER.info(f"   └─ 평균 보상: {avg_reward:.4f}")
            
            # 극단적 보상 감지
            threshold = 100
            if abs(max_reward) > threshold or abs(min_reward) > threshold:
                LOGGER.warning(f"⚠️ 극단적 보상 감지! Episode {episode+1}")
                LOGGER.warning(f"   └─ 최대: {max_reward:.4f}, 최소: {min_reward:.4f}")
                
        # 주기적 평가 및 로깅
        if (episode + 1) % args.log_interval == 0:
            log_training_progress(
                episode + 1, args, episode_rewards, portfolio_values, 
                info, agent, timer, shares_history
            )
            
            # 주기적 모델 저장
            if (episode + 1) % args.save_interval == 0:
                model_name = f"checkpoint_episode_{episode + 1}"
                saved_path = agent.save_model(prefix=model_name)
                LOGGER.info(f"🔄 체크포인트 저장: {saved_path}")
    
    LOGGER.info("=" * 50)
    LOGGER.info("🎉 표준 SAC 학습 완료!")
    LOGGER.info("=" * 50)
    
    return episode_rewards, portfolio_values, shares_history

def analyze_recent_actions(episode_actions_history, num_recent=10):
    """최근 에피소드들의 행동 패턴 간단 분석"""
    if not episode_actions_history:
        return None
    
    # 최근 N개 에피소드 선택
    recent_episodes = episode_actions_history[-num_recent:] if len(episode_actions_history) >= num_recent else episode_actions_history
    
    # 모든 행동 합치기
    all_actions = []
    for ep_actions in recent_episodes:
        all_actions.extend(ep_actions)
    
    if not all_actions:
        return None
    
    # 행동 분류
    buy_threshold = 0.1
    sell_threshold = -0.1
    
    buy_count = sum(1 for a in all_actions if a > buy_threshold)
    sell_count = sum(1 for a in all_actions if a < sell_threshold)
    hold_count = len(all_actions) - buy_count - sell_count
    
    total = len(all_actions)
    buy_ratio = buy_count / total * 100
    sell_ratio = sell_count / total * 100
    hold_ratio = hold_count / total * 100
    
    # 평균 행동 강도
    avg_action = np.mean(all_actions)
    action_intensity = np.mean(np.abs(all_actions))
    
    return {
        'buy_ratio': buy_ratio,
        'sell_ratio': sell_ratio,
        'hold_ratio': hold_ratio,
        'avg_action': avg_action,
        'intensity': action_intensity,
        'episodes_analyzed': len(recent_episodes)
    }

def log_training_progress(episode, args, episode_rewards, portfolio_values, 
                         info, agent, timer, shares_history):
    """학습 진행 상황 로깅"""
    
    # 훈련 성능 계산
    recent_rewards = episode_rewards[-args.log_interval:] if len(episode_rewards) >= args.log_interval else episode_rewards
    recent_portfolios = portfolio_values[-args.log_interval:] if len(portfolio_values) >= args.log_interval else portfolio_values
    recent_shares = shares_history[-args.log_interval:] if len(shares_history) >= args.log_interval else shares_history
    
    avg_reward = np.mean(recent_rewards)
    avg_portfolio = np.mean(recent_portfolios)
    total_return = (avg_portfolio - args.initial_balance) / args.initial_balance * 100
    
    # 안전한 주식 지표 계산
    current_shares = info.get('shares_held', 0)  
    
    if recent_shares:
        avg_shares = np.mean(recent_shares)
        min_shares = np.min(recent_shares)
        max_shares = np.max(recent_shares)
        
        # 포지션 타입 결정
        if abs(current_shares) < 0.001:
            position_type = "현금 포지션"
        elif current_shares > avg_shares * 1.2:
            position_type = "평균 대비 높음"
        elif current_shares < avg_shares * 0.8:
            position_type = "평균 대비 낮음"
        else:
            position_type = "평균 수준"
    else:
        avg_shares = 0
        min_shares = 0
        max_shares = 0
        position_type = "데이터 부족"
    
    # 시간 정보
    elapsed_time = timer.get_training_time()
    avg_episode_time = timer.get_avg_episode_time()
    eta = timer.get_eta(episode, args.num_episodes)
    progress = episode / args.num_episodes * 100
    current_balance = info.get('balance', 0)  

    # 시간 정보
    LOGGER.info(f"⏱시간 정보:")
    LOGGER.info(f"   └─ 경과 시간: {timer.format_time(elapsed_time)}, 평균 에피소드 시간: {avg_episode_time:.2f}초, 예상 남은 시간: {timer.format_time(eta)}")
    LOGGER.info(f"훈련 성능 (최근 {len(recent_rewards)}개 에피소드):")
    
    # 학습 통계
    if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
        LOGGER.info(f"학습 통계:")
        LOGGER.info(f"   └─ Actor Loss: {agent.actor_losses[-1]:.6f}")
        if hasattr(agent, 'critic_losses') and len(agent.critic_losses) > 0:
            LOGGER.info(f"   └─ Critic Loss: {agent.critic_losses[-1]:.6f}")
        if hasattr(agent, 'alpha'):
            LOGGER.info(f"   └─ Alpha: {agent.alpha.item():.6f}")
        if hasattr(agent, 'replay_buffer'):
            LOGGER.info(f"   └─ 버퍼 크기: {len(agent.replay_buffer):,}")
            
    LOGGER.info(f"   └─ 최근 {len(recent_rewards)}개 에피소드 평균 수익률: {total_return:.2f}%")
    LOGGER.info(f"   └─ 최근 {len(recent_rewards)}개 에피소드 평균 보상: {avg_reward:.4f}")
    LOGGER.info(f"   └─ 현재 현금: ${current_balance:,.2f}")
    
    # 통합된 주식 정보 
    LOGGER.info(f"   └─ 현재 보유 주식: {current_shares:.4f}")
    if recent_shares:
        LOGGER.info(f"   └─ 최근 {len(recent_shares)}회 평균: {avg_shares:.4f} (범위: {min_shares:.2f}~{max_shares:.2f})")
        LOGGER.info(f"   └─ 포지션 상태: {position_type}")
    else:
        LOGGER.info(f"   └─ 보유량 히스토리: 데이터 부족")
    
    # 행동 패턴 정보 
    global episode_actions_history
    action_pattern = analyze_recent_actions(episode_actions_history, args.log_interval)
    
    if action_pattern:
        LOGGER.info(f"✅ 행동 패턴 (최근 {action_pattern['episodes_analyzed']}개 에피소드):")
        LOGGER.info(f"   └─ 매수 {action_pattern['buy_ratio']:.1f}% | 매도 {action_pattern['sell_ratio']:.1f}% | 홀드 {action_pattern['hold_ratio']:.1f}%")
        LOGGER.info(f"   └─ 평균 행동값: {action_pattern['avg_action']:+.3f} | 행동 강도: {action_pattern['intensity']:.3f}")
        
        # 지배적 행동 표시
        if action_pattern['buy_ratio'] > 40:
            LOGGER.info(f"   └─✅ 성향: 적극적 매수 성향")
        elif action_pattern['sell_ratio'] > 40:
            LOGGER.info(f"   └─✅ 성향: 적극적 매도 성향")
        elif action_pattern['hold_ratio'] > 60:
            LOGGER.info(f"   └─✅ 성향: 보수적 홀드 성향")
        else:
            LOGGER.info(f"   └─✅ 성향: 균형적 거래 성향")

    LOGGER.info("=" * 80)

def main():
    """메인 함수 (통합 간소화 버전)"""
    timer = TrainingTimer()
    timer.start_training()
    
    print('=' * 50)
    LOGGER.info('SAC 모델 학습 시작 (통합 간소화 버전)')
    
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"학습 대상 심볼: {symbols}")
    LOGGER.info(f"학습 설정:")
    LOGGER.info(f"   └─ 에피소드 수: {args.num_episodes:,}")
    LOGGER.info(f"   └─ 배치 크기: {args.batch_size}")
    LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
    LOGGER.info(f"   └─ 초기 자본금: ${args.initial_balance:,.2f}")
    LOGGER.info(f"   └─ 최대 에피소드 스텝: {args.max_steps}")
    
    # SAC 학습 설정 로그
    LOGGER.info(f"✅ SAC 학습 설정:")
    LOGGER.info(f"   └─ 랜덤 시작점: 활성화")
    LOGGER.info(f"   └─ 리플레이 버퍼: 랜덤 샘플링")
    LOGGER.info(f"   └─ 로그 레벨: {args.log_level}")
    
    # 데이터 수집
    LOGGER.info("데이터 수집 중...")
    collector = DataCollector(symbols=symbols)
    
    if args.collect_data:
        LOGGER.info("새로운 데이터 수집 중...")
        data = collector.load_and_save()
    else:
        LOGGER.info("저장된 데이터 로드 중...")
        data = collector.load_all_data()
        
        if not data:
            LOGGER.warning("저장된 데이터가 없어 새로 수집합니다.")
            data = collector.load_and_save()
    
    if not data:
        LOGGER.error("데이터 수집 실패")
        return
    
    LOGGER.info(f"데이터 수집 완료: {len(data)}개 심볼")
    
    # 데이터 전처리
    LOGGER.info("데이터 전처리 중...")
    processor = DataProcessor(window_size=args.window_size)
    results = processor.process_all_symbols(data)
    
    if not results:
        LOGGER.error("데이터 전처리 실패")
        return
    
    LOGGER.info(f"데이터 전처리 완료: {len(results)}개 심볼")
    
    # 개선된 환경 생성 (훈련용만)
    train_env = create_training_environment(results, symbols, args)
    
    if train_env is None:
        LOGGER.error("훈련 환경 생성 실패")
        return
    
    # 환경 검증 (데이터 손실 위험 사전 점검)
    validation_success = validate_training_environment(train_env, args)
    if not validation_success:
        LOGGER.error("환경 검증 실패")
        return
    
    # 에이전트 생성
    agent = create_agent(train_env, args)

    if agent is None:
        LOGGER.error("에이전트 생성 실패")
        return

    # 학습 실행
        LOGGER.info("=" * 60)
    LOGGER.info("표준 SAC 학습 시작")
    LOGGER.info("=" * 60)
    
    episode_rewards, portfolio_values, shares_history = train_agent_sac(agent, train_env, args, timer)

    # 최종 모델 저장
    final_model_path = agent.save_model(
        save_dir="models",
        prefix='',
        model_type=getattr(agent, 'model_type', 'mlp'),
        symbol=symbols[0] if len(symbols) == 1 else None,
        symbols=symbols if len(symbols) > 1 else None
    )
    
    # 최종 환경 상태 정보 수집 (전체 학습 기간 평균)
    final_env_state = train_env.reset()  # 환경 리셋해서 최종 상태 가져오기
    final_info = train_env._get_info()  # 최종 환경 정보

    LOGGER.info(f"최종 모델 저장 완료: {final_model_path}")
    
    # 최종 결과 출력
    total_time = timer.get_training_time()
    final_portfolio = portfolio_values[-1] if portfolio_values else args.initial_balance
    final_return = (final_portfolio - args.initial_balance) / args.initial_balance * 100
    
    # 전체 에피소드 통계 계산
    if episode_rewards:
        total_episodes = len(episode_rewards)
        avg_total_reward = np.mean(episode_rewards)
        avg_total_portfolio = np.mean(portfolio_values) if portfolio_values else args.initial_balance
        total_avg_return = (avg_total_portfolio - args.initial_balance) / args.initial_balance * 100
        max_portfolio = np.max(portfolio_values) if portfolio_values else args.initial_balance
        min_portfolio = np.min(portfolio_values) if portfolio_values else args.initial_balance
        max_return = (max_portfolio - args.initial_balance) / args.initial_balance * 100
        min_return = (min_portfolio - args.initial_balance) / args.initial_balance * 100
    else:
        total_episodes = 0
        avg_total_reward = 0
        total_avg_return = 0
        max_return = 0
        min_return = 0
    
    LOGGER.info("=" * 80)
    LOGGER.info(f"🎉 학습 완료 - 최종 결과")
    LOGGER.info("=" * 80)
    LOGGER.info(f"⏱ 학습 시간:")
    LOGGER.info(f"   └─ 총 학습 시간: {timer.format_time(total_time)}")
    LOGGER.info(f"   └─ 평균 에피소드 시간: {timer.get_avg_episode_time():.2f}초")
    LOGGER.info(f"📊 전체 {total_episodes}개 에피소드 성과:")
    LOGGER.info(f"   └─ 최종 포트폴리오 가치: ${final_portfolio:,.2f}")
    LOGGER.info(f"   └─ 최종 수익률: {final_return:+.2f}%")
    LOGGER.info(f"   └─ 전체 기간 평균 수익률: {total_avg_return:+.2f}%")
    LOGGER.info(f"   └─ 전체 기간 평균 보상: {avg_total_reward:.4f}")
    LOGGER.info(f"   └─ 최고 수익률: {max_return:+.2f}%")
    LOGGER.info(f"   └─ 최저 수익률: {min_return:+.2f}%")
    
    # 전체 에피소드의 평균 학습 통계
    if hasattr(agent, 'actor_losses') and len(agent.actor_losses) > 0:
        LOGGER.info(f"🧠 전체 학습 통계 (평균):")
        LOGGER.info(f"   └─ 전체 기간 평균 Actor Loss: {np.mean(agent.actor_losses):.6f}")
        if hasattr(agent, 'critic_losses') and len(agent.critic_losses) > 0:
            LOGGER.info(f"   └─ 전체 기간 평균 Critic Loss: {np.mean(agent.critic_losses):.6f}")
        if hasattr(agent, 'alpha'):
            LOGGER.info(f"   └─ 최종 Alpha: {agent.alpha.item():.6f}")
        if hasattr(agent, 'replay_buffer'):
            LOGGER.info(f"   └─ 최종 버퍼 크기: {len(agent.replay_buffer):,}")
    
    # 전체 기간 행동 패턴 분석
    global episode_actions_history
    if episode_actions_history:
        total_action_pattern = analyze_recent_actions(episode_actions_history, len(episode_actions_history))
        if total_action_pattern:
            LOGGER.info(f"⚡ 전체 {total_action_pattern['episodes_analyzed']}개 에피소드 행동 패턴:")
            LOGGER.info(f"   └─ 매수 {total_action_pattern['buy_ratio']:.1f}% | 매도 {total_action_pattern['sell_ratio']:.1f}% | 홀드 {total_action_pattern['hold_ratio']:.1f}%")
            LOGGER.info(f"   └─ 전체 기간 평균 행동값: {total_action_pattern['avg_action']:+.3f}")
            LOGGER.info(f"   └─ 전체 기간 행동 강도: {total_action_pattern['intensity']:.3f}")
            
            # 전체적인 성향 분석
            if total_action_pattern['buy_ratio'] > 40:
                LOGGER.info(f"   └─ 🎯 전체 성향: 적극적 매수 전략")
            elif total_action_pattern['sell_ratio'] > 40:
                LOGGER.info(f"   └─ 🎯 전체 성향: 적극적 매도 전략")
            elif total_action_pattern['hold_ratio'] > 60:
                LOGGER.info(f"   └─ 🎯 전체 성향: 보수적 홀드 전략")
            else:
                LOGGER.info(f"   └─ 🎯 전체 성향: 균형적 거래 전략")
    
    # 전체 기간 평균 포지션 정보
    if shares_history and len(shares_history) > 1:
        avg_shares = np.mean(shares_history)
        max_shares = np.max(shares_history)
        min_shares = np.min(shares_history)
        
        LOGGER.info(f"💰 전체 기간 포지션 통계:")
        LOGGER.info(f"   └─ 전체 기간 평균 주식 보유량: {avg_shares:.4f}")
        LOGGER.info(f"   └─ 최대 주식 보유량: {max_shares:.4f}")
        LOGGER.info(f"   └─ 최소 주식 보유량: {min_shares:.4f}")
        LOGGER.info(f"   └─ 최종 현금: ${final_info.get('balance', 0):,.2f}")
        LOGGER.info(f"   └─ 최종 주식 보유량: {final_info.get('shares_held', 0):.4f}")
        LOGGER.info(f"   └─ 최종 주식 비율: {final_info.get('stock_ratio', 0):.1%}")
        LOGGER.info(f"   └─ 최종 현금 비율: {final_info.get('cash_ratio', 0):.1%}")
    
    LOGGER.info(f"💾 모델 정보:")
    LOGGER.info(f"   └─ 저장 경로: {final_model_path}")
    LOGGER.info(f"   └─ 학습된 에피소드: {len(episode_rewards):,}개")
    
    # 간단한 벤치마크 비교 (Buy & Hold)
    if len(episode_rewards) > 0:
        LOGGER.info("")
        LOGGER.info(f"✅ 성과 비교:")
        LOGGER.info(f"    └─ SAC 모델 수익률: {final_return:.2f}%")
        
        # Buy & Hold 수익률 추정 (첫 번째와 마지막 포트폴리오 값 기준)
        if len(portfolio_values) >= 2:
            buy_hold_return = ((portfolio_values[-1] / portfolio_values[0]) - 1) * 100
            LOGGER.info(f"   └─ Buy & Hold 추정: {buy_hold_return:.2f}%")
            
            if final_return > buy_hold_return:
                outperformance = final_return - buy_hold_return
                LOGGER.info(f"   └─ ✅ SAC 모델이 {outperformance:.2f}%p 더 우수")
            else:
                underperformance = buy_hold_return - final_return
                LOGGER.info(f"   └─ ❌ Buy & Hold가 {underperformance:.2f}%p 더 우수")

if __name__ == "__main__":
    main()