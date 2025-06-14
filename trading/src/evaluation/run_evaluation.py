"""
SAC 모델 평가 실행 스크립트 (수정된 버전)
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import argparse
import torch
import numpy as np
from pathlib import Path

from src.config.ea_teb_config import (
    DEVICE,
    WINDOW_SIZE,
    TARGET_SYMBOLS,
    INITIAL_BALANCE,
    LOGGER
)
from src.data_collection.data_collector import DataCollector
from src.preprocessing.data_processor import DataProcessor
from src.environment.trading_env import TradingEnvironment, create_environment_from_results
from src.models.sac_agent import SACAgent
from src.evaluation.evaluator import Evaluator
from src.utils.utils import create_directory, get_timestamp

def parse_args():
    """
    명령줄 인자 파싱
    
    Returns:
        파싱된 인자
    """
    parser = argparse.ArgumentParser(description='SAC 모델 평가 스크립트')
    
    # 데이터 관련 인자
    parser.add_argument('--symbols', nargs='+', default=None, help='평가에 사용할 주식 심볼 목록')
    parser.add_argument('--collect_data', action='store_true', help='데이터 수집 여부')
    
    # 환경 관련 인자
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE, help='관측 윈도우 크기')
    parser.add_argument('--initial_balance', type=float, default=INITIAL_BALANCE, help='초기 자본금')
    parser.add_argument('--multi_asset', action='store_true', help='다중 자산 환경 사용 여부')
    
    # 모델 관련 인자
    parser.add_argument('--model_path', type=str, required=True, help='로드할 모델 경로')
    parser.add_argument('--use_cnn', action='store_true', help='CNN 모델 사용 여부')
    parser.add_argument('--use_lstm', action='store_true', help='LSTM 모델 사용 여부')
    parser.add_argument('--use_mamba', action='store_true', help='Mamba 모델 사용 여부')
    parser.add_argument('--use_tinytransformer', action='store_true', help='TinyTransformer 모델 사용 여부')
    
    # 평가 관련 인자
    parser.add_argument('--num_episodes', type=int, default=1, help='평가할 에피소드 수')
    parser.add_argument('--render', action='store_true', help='환경 렌더링 여부')
    parser.add_argument('--result_prefix', type=str, default='', help='결과 파일 접두사')
    parser.add_argument(
        '--data_type',
        type=str,
        default='valid',
        choices=['train', 'valid', 'test'],
        help='평가에 사용할 데이터 유형 (기본값: valid)'
    )
    
    # 리플레이 버퍼 관련 인자
    parser.add_argument("--buffer_type", type=str, choices=["random", "sequential"], default="sequential", help="리플레이 버퍼 타입 선택 (random 또는 sequential)")

    return parser.parse_args()

def load_model(model_path, env, args):
    """
    모델 로드 (LSTM/CNN/MLP 지원)

    Args:
        model_path: 모델 경로
        env: 환경 객체
        args: 명령줄 인자

    Returns:
        로드된 SAC 에이전트 또는 None
    """
    LOGGER.info(f"🤖 모델 로드 중: {model_path}")

    try:
        # 설정 파일 로드
        config_path = os.path.join(model_path, "config.pth")
        if not os.path.exists(config_path):
            LOGGER.error(f"❌ 모델 설정 파일이 없습니다: {config_path}")
            return None

        config = torch.load(config_path, map_location=DEVICE)
        LOGGER.info("✅ 모델 설정 로드 성공")

        # 저장된 설정값
        saved_state_dim = config.get('state_dim')
        saved_action_dim = config.get('action_dim', 1)
        saved_hidden_dim = config.get('hidden_dim', 256)
        saved_use_cnn = config.get('use_cnn', False)
        saved_use_lstm = config.get('use_lstm', False)
        saved_use_mamba = config.get('use_mamba', False)
        saved_use_tinytransformer = config.get('use_tinytransformer', False)
        saved_input_shape = config.get('input_shape')

        LOGGER.info("💾 저장된 모델 설정:")
        LOGGER.info(f"   └ 상태 차원: {saved_state_dim}")
        LOGGER.info(f"   └ 행동 차원: {saved_action_dim}")
        LOGGER.info(f"   └ 은닉층: {saved_hidden_dim}")
        LOGGER.info(f"   └ CNN 사용: {saved_use_cnn}")
        LOGGER.info(f"   └ LSTM 사용: {saved_use_lstm}")
        LOGGER.info(f"   └ Mamba 사용: {saved_use_mamba}")
        LOGGER.info(f"   └ TinyTransformer 사용: {saved_use_tinytransformer}")

        # 현재 환경에서 상태 차원 계산
        market_shape = env.observation_space['market_data'].shape
        portfolio_shape = env.observation_space['portfolio_state'].shape
        actual_state_dim = market_shape[0] * market_shape[1] + portfolio_shape[0]
        input_shape = (market_shape[0], market_shape[1])

        LOGGER.info("📏 환경 상태 정보:")
        LOGGER.info(f"   └ 마켓 데이터: {market_shape}")
        LOGGER.info(f"   └ 포트폴리오: {portfolio_shape}")
        LOGGER.info(f"   └ 계산된 상태 차원: {actual_state_dim}")

        # 모델 타입 충돌 방지
        model_flags = [args.use_cnn, args.use_lstm, args.use_mamba, args.use_tinytransformer]
        if sum(model_flags) > 1:
            LOGGER.error("❌ CNN, LSTM, Mamba, TinyTransformer 중 하나만 선택할 수 있습니다.")
            return None

        # 우선순위: TinyTransformer > Mamba > LSTM > CNN > MLP
        final_use_tinytransformer = args.use_tinytransformer or saved_use_tinytransformer
        final_use_mamba = args.use_mamba or saved_use_mamba
        final_use_lstm = args.use_lstm or saved_use_lstm
        final_use_cnn = args.use_cnn or saved_use_cnn

        if final_use_tinytransformer:
            LOGGER.info("🔧 TinyTransformer 모델 생성 중...")
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=saved_input_shape if saved_input_shape else input_shape,
                use_tinytransformer=True,
                device=DEVICE
            )

        elif final_use_mamba:
            LOGGER.info("🔧 Mamba 모델 생성 중...")
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=saved_input_shape if saved_input_shape else input_shape,
                use_mamba=True,
                device=DEVICE
            )

        elif final_use_lstm:
            LOGGER.info("🔧 LSTM 모델 생성 중...")
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=saved_input_shape if saved_input_shape else input_shape,
                use_lstm=True,
                device=DEVICE
            )

        elif final_use_cnn:
            LOGGER.info("🔧 CNN 모델 생성 중...")
            agent = SACAgent(
                state_dim=None,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                input_shape=saved_input_shape if saved_input_shape else input_shape,
                use_cnn=True,
                device=DEVICE
            )

        else:
            LOGGER.info("🔧 MLP 모델 생성 중...")
            final_state_dim = saved_state_dim if saved_state_dim else actual_state_dim

            if saved_state_dim and saved_state_dim != actual_state_dim:
                LOGGER.warning("⚠️ 상태 차원 불일치:")
                LOGGER.warning(f"   └ 저장된 모델: {saved_state_dim}")
                LOGGER.warning(f"   └ 현재 환경: {actual_state_dim}")
                LOGGER.warning("   └ 저장된 차원을 사용합니다.")

            agent = SACAgent(
                state_dim=final_state_dim,
                action_dim=saved_action_dim,
                hidden_dim=saved_hidden_dim,
                device=DEVICE
            )

        # 모델 로드
        try:
            agent.load_model(model_path)
            LOGGER.info("✅ 모델 로드 완료")
            
            # 🚨 모델 로드 후 명시적으로 디바이스 이동
            agent.actor = agent.actor.to(DEVICE)
            agent.critic = agent.critic.to(DEVICE)
            agent.critic_target = agent.critic_target.to(DEVICE)
            
            # ✅ 평가 모드 설정 (중요!)
            agent.actor.eval()
            agent.critic.eval()
            agent.critic_target.eval()
            
            LOGGER.info(f"✅ 모델이 {DEVICE}로 이동 및 평가 모드 설정 완료")
            
        except Exception as e:
            LOGGER.warning(f"⚠️ 표준 모델 로드 실패: {e}")
            if hasattr(agent, 'load_model_with_resize'):
                try:
                    agent.load_model_with_resize(model_path)
                    LOGGER.info("✅ 크기 조정 방식으로 모델 로드 성공")
                    
                    # 🚨 크기 조정 로드 후에도 디바이스 이동 및 평가 모드 설정
                    agent.actor = agent.actor.to(DEVICE)
                    agent.critic = agent.critic.to(DEVICE)
                    agent.critic_target = agent.critic_target.to(DEVICE)
                    
                    # ✅ 평가 모드 설정 (중요!)
                    agent.actor.eval()
                    agent.critic.eval()
                    agent.critic_target.eval()
                    
                    LOGGER.info(f"✅ 크기 조정된 모델이 {DEVICE}로 이동 및 평가 모드 설정 완료")
                    
                except Exception as e2:
                    LOGGER.error(f"❌ 크기 조정 방식도 실패: {e2}")
                    return None
            else:
                return None

        return agent

    except Exception as e:
        LOGGER.error(f"❌ 모델 로드 중 오류: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return None

def main():
    """
    메인 함수
    """
    print('=' * 80)
    LOGGER.info('🎯 SAC 모델 평가 시작 (검증 데이터 사용 권장)')
    
    # 인자 파싱
    args = parse_args()
    
    # 심볼 목록 설정
    symbols = args.symbols if args.symbols else TARGET_SYMBOLS
    
    LOGGER.info(f"📈 평가 설정:")
    LOGGER.info(f"   └─ 대상 심볼: {symbols}")
    LOGGER.info(f"   └─ 데이터 유형: {args.data_type}")
    LOGGER.info(f"   └─ 에피소드 수: {args.num_episodes}")
    LOGGER.info(f"   └─ 윈도우 크기: {args.window_size}")
    LOGGER.info(f"   └─ 초기 자본금: ${args.initial_balance:,.2f}")
    LOGGER.info(f"   └─ 다중 자산: {'예' if args.multi_asset else '아니오'}")
    LOGGER.info(f"   └─ CNN 사용: {'예' if args.use_cnn else '아니오'}")
    
    # 다중 자산 환경 체크
    if args.multi_asset:
        LOGGER.error("❌ 다중 자산 환경은 현재 지원되지 않습니다.")
        LOGGER.info("💡 단일 자산 환경을 사용해주세요.")
        return
    
    # 데이터 유형에 따른 안내 메시지
    if args.data_type == 'test':
        LOGGER.warning("⚠️  테스트 데이터를 사용합니다. 이는 최종 성능 측정용으로만 사용하세요!")
        LOGGER.info("💡 일반적인 모델 평가에는 --data_type valid를 권장합니다.")
    elif args.data_type == 'train':
        LOGGER.warning("⚠️  훈련 데이터를 사용합니다. 이는 과적합 확인용입니다.")
        LOGGER.info("💡 실제 성능 평가에는 --data_type valid를 권장합니다.")
    else:
        LOGGER.info("✅ 검증 데이터를 사용합니다. (권장 설정)")
    
    # 데이터 수집
    LOGGER.info("📊 데이터 수집 중...")
    collector = DataCollector(symbols=symbols)
    
    if args.collect_data:
        LOGGER.info("🔄 새로운 데이터 수집 중...")
        data = collector.load_and_save()
    else:
        LOGGER.info("💾 저장된 데이터 로드 중...")
        data = collector.load_all_data()
        
        if not data:
            LOGGER.warning("⚠️  저장된 데이터가 없어 새로 수집합니다.")
            data = collector.load_and_save()
    
    if not data:
        LOGGER.error("❌ 데이터 수집 실패")
        return
    
    LOGGER.info(f"✅ 데이터 수집 완료: {len(data)}개 심볼")
    
    # 데이터 전처리
    LOGGER.info("⚙️  데이터 전처리 중...")
    processor = DataProcessor(window_size=args.window_size)
    results = processor.process_all_symbols(data)
    
    if not results:
        LOGGER.error("❌ 데이터 전처리 실패")
        return
    
    LOGGER.info(f"✅ 데이터 전처리 완료: {len(results)}개 심볼")

    # 데이터 분할 정보 로깅
    symbol = symbols[0]  # 첫 번째 심볼 사용
    if symbol in results:
        train_len = len(results[symbol].get("train", []))
        valid_len = len(results[symbol].get("valid", []))
        test_len = len(results[symbol].get("test", []))
        total_len = train_len + valid_len + test_len

        LOGGER.info(f"📊 {symbol} 데이터 분할 정보:")
        LOGGER.info(f"   └─ 전체: {total_len} 행")
        LOGGER.info(f"   └─ 훈련: {train_len} 행")
        LOGGER.info(f"   └─ 검증: {valid_len} 행")
        LOGGER.info(f"   └─ 테스트: {test_len} 행")
    else:
        LOGGER.warning(f"❌ {symbol} 데이터가 results에 존재하지 않습니다.")

    
    # 평가 환경 생성 - create_environment_from_results 함수 사용
    LOGGER.info(f"🏗️  {args.data_type} 환경 생성 중...")
    try:
        env = create_environment_from_results(
            results=results,
            symbol=symbol,
            data_type=args.data_type,
            window_size=args.window_size,
            initial_balance=args.initial_balance
        )
        
        if env is None:
            LOGGER.error("❌ 평가 환경 생성 실패")
            return
            
        LOGGER.info(f"✅ {args.data_type} 환경 생성 완료")
        LOGGER.info(f"   └─ 심볼: {env.symbol}")
        LOGGER.info(f"   └─ 데이터 길이: {env.data_length}")
        LOGGER.info(f"   └─ 특성 차원: {env.feature_dim}")
        LOGGER.info(f"   └─ 윈도우 크기: {env.window_size}")
        
    except Exception as e:
        LOGGER.error(f"❌ 환경 생성 실패: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())
        return
    
    # 모델 로드
    agent = load_model(args.model_path, env, args)
    
    if agent is None:
        LOGGER.error("❌ 모델 로드 실패")
        return
    
    # 평가기 생성
    LOGGER.info("🎯 평가기 생성 중...")
    try:
        # Evaluator 생성 시 data_type 인자 전달 시도
        try:
            evaluator = Evaluator(agent=agent, env=env, data_type=args.data_type)
            LOGGER.info("✅ 평가기 생성 완료")
        except TypeError:
            # data_type 인자를 받지 않는 경우
            evaluator = Evaluator(agent=agent, env=env)
            LOGGER.info("✅ 평가기 생성 완료 (data_type 인자 없이)")
    except Exception as e:
        LOGGER.error(f"❌ 평가기 생성 실패: {str(e)}")
        return
    
    # 평가 실행
    LOGGER.info(f"🚀 {args.data_type} 데이터로 평가 시작: {args.num_episodes}개 에피소드")
    try:
        # ✅ 평가 전 행동값 테스트
        LOGGER.info("🔍 모델 행동값 테스트 중...")
        test_obs = env.reset()
        test_actions = []
        test_ratios = []
        
        for i in range(10):  # 10번 테스트
            test_action = agent.select_action(test_obs, evaluate=True)
            test_actions.append(test_action)
            
            # 현재 포트폴리오 비율 계산
            current_price = env._get_current_price()
            stock_value = env.shares_held * current_price
            total_portfolio = env.balance + stock_value
            current_ratio = stock_value / total_portfolio if total_portfolio > 0 else 0.0
            test_ratios.append(current_ratio)
            
            LOGGER.info(f"   테스트 {i+1:2d}: 행동값 = {test_action:+.6f}, 현재비율 = {current_ratio:.4f}, 차이 = {abs(test_action - current_ratio):.6f}")
        
        action_std = np.std(test_actions) if len(test_actions) > 1 else 0.0
        ratio_diffs = [abs(act - ratio) for act, ratio in zip(test_actions, test_ratios)]
        avg_diff = np.mean(ratio_diffs)
        
        LOGGER.info(f"🎯 행동 분석:")
        LOGGER.info(f"   └─ 행동값 표준편차: {action_std:.6f}")
        LOGGER.info(f"   └─ 평균 비율 차이: {avg_diff:.6f}")
        LOGGER.info(f"   └─ 거래 임계값: {0.0005 if env.log_level == 'minimal' else 0.001:.6f}")
        
        if action_std < 1e-6:
            LOGGER.warning("⚠️ 행동값이 거의 고정되어 있습니다!")
        
        if avg_diff < 0.0005:
            LOGGER.warning("⚠️ 비율 차이가 매우 작아 거래가 거의 발생하지 않을 수 있습니다!")
        else:
            LOGGER.info("✅ 행동값에 적절한 변동성이 있습니다.")
        
        # 실제 평가 실행
        results_eval = evaluator.evaluate(num_episodes=args.num_episodes, render=args.render)
        LOGGER.info("✅ 평가 완료!")
        
        # 결과 저장
        prefix = f"{args.result_prefix}_" if args.result_prefix else ""
        prefix += f"{args.data_type}_"  # 데이터 유형을 prefix에 추가
        result_dir = evaluator.save_results(results_eval, prefix=prefix)
        
        # 결과 출력
        LOGGER.info("=" * 80)
        LOGGER.info(f"🎉 평가 결과 ({args.data_type} 데이터)")
        LOGGER.info("=" * 80)
        LOGGER.info(f"📁 결과 저장 경로: {result_dir}")
        LOGGER.info(f"💰 총 수익률: {results_eval['total_return']:.2f}%")
        LOGGER.info(f"📊 샤프 비율: {results_eval['sharpe_ratio']:.2f}")
        LOGGER.info(f"📉 최대 낙폭: {results_eval['max_drawdown']:.2f}%")
             
        # 추가 성능 지표 (있는 경우)
        if 'annualized_return' in results_eval:
            LOGGER.info(f"📈 연간 수익률: {results_eval['annualized_return']:.2f}%")
        if 'volatility' in results_eval:
            LOGGER.info(f"📊 변동성: {results_eval['volatility']:.2f}%")
        if 'total_trades' in results_eval:
            LOGGER.info(f"🔄 총 거래 횟수: {results_eval['total_trades']}")
        if 'win_rate' in results_eval:
            LOGGER.info(f"🎯 승률: {results_eval['win_rate']:.2f}%")
        
        LOGGER.info("=" * 80)
        LOGGER.info("💡 팁: 다른 데이터셋으로 평가하려면 --data_type 옵션을 사용하세요")
        LOGGER.info("   └─ --data_type train  : 훈련 데이터로 평가 (과적합 확인용)")
        LOGGER.info("   └─ --data_type valid : 검증 데이터로 평가 (기본값, 권장)")
        LOGGER.info("   └─ --data_type test  : 테스트 데이터로 평가 (최종 성능 측정용)")
        
    except Exception as e:
        LOGGER.error(f"❌ 평가 실행 중 오류 발생: {str(e)}")
        import traceback
        LOGGER.error(traceback.format_exc())

if __name__ == "__main__":
    main()