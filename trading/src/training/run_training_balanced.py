#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
균형잡힌 SAC 모델 학습 스크립트 (매수 편향 문제 해결)
"""
import os
import argparse
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import sys
from pathlib import Path
import time
import json

# 프로젝트 루트 디렉토리를 path에 추가
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.models.sac_agent import SACAgent
from src.environment.trading_env_balanced import create_balanced_environment_from_results
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.config.ea_teb_config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER,
    WINDOW_SIZE,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT,
    BATCH_SIZE,
    NUM_EPISODES,
)
from src.utils.utils import create_directory


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='균형잡힌 SAC 모델 학습')

    # 데이터 관련
    parser.add_argument('--symbols', nargs='+', default=None, help='학습할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true', help='새로운 데이터 수집 여부')

    # 환경 관련
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=INITIAL_BALANCE, help='초기 자본금')
    parser.add_argument('--transaction_fee_percent', type=float, default=TRANSACTION_FEE_PERCENT, help='거래 수수료율')
    
    # 균형잡힌 환경 전용 설정
    parser.add_argument('--balanced_initialization', action='store_true', default=True, 
                        help='균형잡힌 초기화 (50% 현금, 50% 주식)')
    parser.add_argument('--disable_balanced_init', action='store_true', 
                        help='균형잡힌 초기화 비활성화')

    # 학습 관련
    parser.add_argument('--num_episodes', type=int, default=100, help='학습 에피소드 수 (균형잡힌 기본값)')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='배치 크기')
    parser.add_argument('--max_steps_per_episode', type=int, default=1000, help='에피소드당 최대 스텝')

    # 모델 관련
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용')
    parser.add_argument('--use_lstm', action='store_true', help='LSTM 모델 사용')
    parser.add_argument('--use_mamba', action='store_true', help='Mamba 모델 사용')
    parser.add_argument('--use_tinytransformer', action='store_true', help='TinyTransformer 모델 사용')

    # 학습률 및 하이퍼파라미터 (균형잡힌 학습에 최적화)
    parser.add_argument('--actor_lr', type=float, default=3e-5, help='Actor 학습률 (균형잡힌 기본값)')
    parser.add_argument('--critic_lr', type=float, default=3e-4, help='Critic 학습률')
    parser.add_argument('--alpha_lr', type=float, default=3e-5, help='Alpha 학습률')

    # 결과 저장
    parser.add_argument('--save_dir', type=str, default='models_balanced', help='모델 저장 디렉토리')
    parser.add_argument('--log_level', type=str, default='normal', 
                        choices=['minimal', 'normal', 'detailed'], help='로그 레벨')

    return parser.parse_args()


def analyze_action_balance(episode_actions_history, num_recent=5):
    """행동 균형 분석"""
    if len(episode_actions_history) < num_recent:
        return "분석에 충분한 데이터가 없습니다."

    recent_episodes = episode_actions_history[-num_recent:]
    all_actions = []
    
    for actions in recent_episodes:
        all_actions.extend(actions)
    
    if not all_actions:
        return "행동 데이터가 없습니다."
    
    actions_array = np.array(all_actions)
    
    # 행동 분포 분석 (시그모이드 변환 후)
    # action -> target_ratio 변환: 1 / (1 + exp(-action * 2))
    target_ratios = 1 / (1 + np.exp(-actions_array * 2))
    
    # 매수/매도 판단 (50% 기준)
    buy_actions = np.sum(target_ratios > 0.6)  # 60% 이상 주식 -> 매수 성향
    sell_actions = np.sum(target_ratios < 0.4)  # 40% 미만 주식 -> 매도 성향
    hold_actions = np.sum((target_ratios >= 0.4) & (target_ratios <= 0.6))  # 중립
    total_actions = len(actions_array)
    
    buy_ratio = buy_actions / total_actions * 100
    sell_ratio = sell_actions / total_actions * 100
    hold_ratio = hold_actions / total_actions * 100
    
    # 균형 점수 계산
    balance_score = 100 - abs(buy_ratio - sell_ratio)
    
    analysis = f"""
🎯 최근 {num_recent}개 에피소드 행동 균형 분석:
   └─ 매수 성향: {buy_ratio:.1f}% | 매도 성향: {sell_ratio:.1f}% | 중립: {hold_ratio:.1f}%
   └─ 평균 행동값: {np.mean(actions_array):+.3f}
   └─ 평균 목표 비율: {np.mean(target_ratios):.3f}
   └─ 행동 다양성: {np.std(actions_array):.3f}
   └─ 균형 점수: {balance_score:.1f}/100
   └─ 균형 상태: {'🎯 균형잡힘' if balance_score > 70 else '⚠️ 편향됨' if balance_score > 40 else '❌ 심각한 편향'}
    """
    
    return analysis


def main():
    """메인 함수"""
    print('=' * 80)
    LOGGER.info('🎯 균형잡힌 SAC 모델 학습 시작')

    # 인자 파싱
    args = parse_args()

    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS[:1]  # 기본적으로 첫 번째 심볼만
    symbol = symbols[0] if isinstance(symbols, list) else symbols

    LOGGER.info(f"🎯 균형잡힌 학습 설정:")
    LOGGER.info(f"   └─ 심볼: {symbol}")
    LOGGER.info(f"   └─ 에피소드 수: {args.num_episodes}")
    LOGGER.info(f"   └─ 배치 크기: {args.batch_size}")
    LOGGER.info(f"   └─ 균형잡힌 초기화: {args.balanced_initialization and not args.disable_balanced_init}")
    LOGGER.info(f"   └─ Actor LR: {args.actor_lr}")

    # 결과 디렉토리 설정
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"balanced_{symbol}_{timestamp}"
    create_directory(save_dir)

    LOGGER.info(f"💾 모델 저장 경로: {save_dir}")

    try:
        # 데이터 수집
        LOGGER.info("📊 데이터 수집 중...")
        collector = DataCollector(symbols=[symbol])
        data = collector.load_all_data()
        
        if not data:
            LOGGER.info("🔄 새로운 데이터 수집 중...")
            data = collector.load_and_save()

        # 데이터 전처리
        LOGGER.info("⚙️ 데이터 전처리 중...")
        processor = DataProcessor(window_size=args.window_size)
        results = processor.process_all_symbols(data)

        # 균형잡힌 환경 생성
        LOGGER.info(f"🏗️ 균형잡힌 학습 환경 생성: {symbol}")
        
        balanced_init = args.balanced_initialization and not args.disable_balanced_init
        
        env = create_balanced_environment_from_results(
            results=results,
            symbol=symbol,
            data_type='train',
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            transaction_fee_percent=args.transaction_fee_percent,
            balanced_initialization=balanced_init,
            log_level=args.log_level
        )

        LOGGER.info(f"✅ 균형잡힌 환경 생성 완료 (균형 초기화: {balanced_init})")

        # 에이전트 생성
        LOGGER.info("🤖 균형잡힌 SAC 에이전트 생성 중...")
        
        obs = env.reset()
        input_shape = obs['market_data'].shape if isinstance(obs, dict) else None
        state_dim = None if input_shape else len(obs)

        agent = SACAgent(
            state_dim=state_dim,
            action_dim=1,
            input_shape=input_shape,
            use_cnn=args.use_cnn,
            use_lstm=args.use_lstm,
            use_mamba=args.use_mamba,
            use_tinytransformer=args.use_tinytransformer,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            alpha_lr=args.alpha_lr,
            alpha_init=0.1,  # 낮은 초기 엔트로피
            gamma=0.99,
            tau=0.005,
            device=DEVICE
        )

        # 균형잡힌 학습 실행
        LOGGER.info("🚀 균형잡힌 학습 시작!")
        
        episode_rewards = []
        episode_actions_history = []
        
        for episode in range(args.num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_actions = []
            step = 0

            while not done and step < args.max_steps_per_episode:
                action = agent.select_action(state, evaluate=False)
                episode_actions.append(action)

                next_state, reward, done, info = env.step(action)
                agent.add_experience(state, action, reward, next_state, done)

                if len(agent.replay_buffer) > args.batch_size:
                    agent.update_parameters(args.batch_size)

                state = next_state
                episode_reward += reward if isinstance(reward, (int, float)) else reward
                step += 1

            episode_rewards.append(episode_reward)
            episode_actions_history.append(episode_actions)

            # 진행 상황 로깅
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                LOGGER.info(f"🎯 에피소드 {episode+1}/{args.num_episodes} - 평균 보상: {avg_reward:.4f}")
                
                if episode >= 20:
                    balance_analysis = analyze_action_balance(episode_actions_history, num_recent=10)
                    LOGGER.info(balance_analysis)

        # 모델 저장
        model_type = 'TinyTransformer' if args.use_tinytransformer else 'LSTM' if args.use_lstm else 'Mamba' if args.use_mamba else 'CNN' if args.use_cnn else 'MLP'
        
        saved_path = agent.save_model(
            save_dir=save_dir,
            prefix=f"balanced_{model_type}",
            symbol=symbol
        )

        # 최종 균형 분석
        if episode_actions_history:
            final_analysis = analyze_action_balance(episode_actions_history, num_recent=min(20, len(episode_actions_history)))
            LOGGER.info("🏁 최종 균형잡힌 학습 결과:")
            LOGGER.info(final_analysis)

        LOGGER.info("=" * 80)
        LOGGER.info(f"🎉 균형잡힌 학습 완료! - {symbol}")
        LOGGER.info(f"📁 저장 경로: {saved_path}")
        LOGGER.info(f"🤖 모델 타입: {model_type}")
        LOGGER.info(f"🎯 균형잡힌 학습: 활성화")
        LOGGER.info("=" * 80)

        return saved_path

    except Exception as e:
        LOGGER.error(f"❌ 균형잡힌 학습 중 오류: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())


if __name__ == "__main__":
    main() 