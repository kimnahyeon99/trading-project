"""
순차적 에피소드 데이터 관리 클래스 (수정된 버전)
전체 데이터를 순차적으로 활용하여 학습 효율성 극대화
"""
import numpy as np
from typing import Tuple, Dict, Any
import logging

from src.config.ea_teb_config import (
    WINDOW_SIZE,
    MAX_STEPS_PER_EPISODE,
    overlap_ratio
)


class SequentialEpisodeManager:
    """
    순차적 에피소드 데이터 관리자
    - 전체 데이터를 겹치지 않게 순차적으로 분할
    - 에피소드별로 다른 시간 구간 학습
    - 100% 데이터 활용률 달성
    """
    
    def __init__(
        self,
        data_length: int, 
        window_size: int = WINDOW_SIZE, 
        max_steps: int = MAX_STEPS_PER_EPISODE,
        min_steps: int = 100,  # 최소 에피소드 길이
        overlap_ratio: float = overlap_ratio,  # 에피소드 간 겹침 비율 (0.0 = 겹침 없음, 0.1 = 10% 겹침)
        adaptive_length: bool = True,  # 적응형 길이 사용
        logger: logging.Logger = None
    ):
        """
        순차적 에피소드 관리자 초기화
        
        Args:
            data_length: 전체 데이터 길이
            window_size: 관측 윈도우 크기
            max_steps: 에피소드당 최대 스텝 수
            min_steps: 최소 에피소드 길이
            overlap_ratio: 에피소드 간 겹침 비율 (0.0~0.5 권장)
            adaptive_length: 적응형 길이 사용 여부
            logger: 로깅용 로거
        """
        self.data_length = data_length
        self.window_size = WINDOW_SIZE
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.overlap_ratio = max(0.0, min(overlap_ratio, 0.5))
        self.adaptive_length = adaptive_length
        self.logger = logger or logging.getLogger(__name__)
        
        # 사용 가능한 실제 데이터 길이
        self.usable_length = data_length - window_size
        
        # 에피소드 계획 생성
        self.episode_plan = self._create_adaptive_episode_plan()
        
        # 현재 상태 추적
        self.current_cycle = 0
        self.current_episode_in_cycle = 0
        
        self._log_initialization()
    
    def _create_adaptive_episode_plan(self) -> list:
        """적응형 에피소드 계획 생성"""
        plan = []
        current_pos = 0
        episode_id = 0
        
        while current_pos < self.usable_length:
            # 남은 데이터 길이 계산
            remaining_data = self.usable_length - current_pos
            
            if self.adaptive_length:
                # 적응형 길이 계산
                if remaining_data <= self.max_steps:
                    # 남은 데이터가 max_steps 이하면 모두 사용
                    episode_length = remaining_data
                elif remaining_data <= self.max_steps * 1.5:
                    # 1.5배 이하면 절반으로 나누어 두 에피소드로
                    episode_length = remaining_data // 2
                else:
                    # 충분히 크면 max_steps 사용
                    episode_length = self.max_steps
            else:
                # 고정 길이 사용
                episode_length = min(self.max_steps, remaining_data)
            
            # 최소 길이 보장
            if episode_length < self.min_steps and remaining_data > self.min_steps:
                episode_length = self.min_steps
            
            # 에피소드 정보 생성
            end_pos = min(current_pos + episode_length, self.usable_length)
            actual_length = end_pos - current_pos
            
            if actual_length >= self.min_steps:  # 최소 길이 이상인 경우만 추가
                episode_info = {
                    'episode_id': episode_id,
                    'start_index': current_pos,
                    'end_index': end_pos,
                    'planned_length': actual_length,
                    'actual_length': actual_length,  # 호환성을 위해 추가
                    'coverage_start_pct': (current_pos / self.usable_length) * 100,
                    'coverage_end_pct': (end_pos / self.usable_length) * 100,
                    'cycle_number': 0,  # 기본값
                    'episode_in_cycle': episode_id,
                    'episode_type': self._classify_episode_type({'planned_length': actual_length}),
                    'is_last_in_cycle': False  # 나중에 업데이트
                }
                plan.append(episode_info)
                episode_id += 1
            
            # 다음 에피소드 시작점 계산 (겹침 고려)
            if self.overlap_ratio > 0:
                step_size = int(actual_length * (1 - self.overlap_ratio))
                current_pos += max(step_size, self.min_steps)
            else:
                current_pos = end_pos
            
            # 무한루프 방지
            if current_pos >= self.usable_length or actual_length == 0:
                break
        
        # 마지막 에피소드 표시
        if plan:
            plan[-1]['is_last_in_cycle'] = True
        
        return plan
    
    def _classify_episode_type(self, episode_plan: Dict) -> str:
        """에피소드 타입 분류"""
        length = episode_plan['planned_length']
        
        if length >= self.max_steps * 0.9:
            return "full"  # 거의 최대 길이
        elif length <= self.min_steps * 1.1:
            return "minimal"  # 최소 길이
        elif length >= self.max_steps * 0.5:
            return "standard"  # 표준 길이
        else:
            return "short"  # 짧은 길이
    
    def get_episode_info(self, episode_num: int) -> Tuple[int, Dict[str, Any]]:
        """
        에피소드 정보 반환
        
        Args:
            episode_num: 에피소드 번호 (0부터 시작)
            
        Returns:
            (시작_인덱스, 메타정보) 튜플
        """
        if not self.episode_plan:
            # 빈 계획인 경우 기본값 반환
            return 0, {
                'start_index': 0,
                'end_index': min(self.max_steps, self.usable_length),
                'actual_length': min(self.max_steps, self.usable_length),
                'cycle_number': 0,
                'episode_in_cycle': 0,
                'episode_type': 'default',
                'is_last_in_cycle': True,
                'coverage_start_pct': 0.0,
                'coverage_end_pct': 100.0
            }
        
        # 에피소드 번호를 계획 범위 내로 조정
        episode_index = episode_num % len(self.episode_plan)
        episode_info = self.episode_plan[episode_index].copy()
        
        # 사이클 정보 업데이트
        cycle_num = episode_num // len(self.episode_plan)
        episode_info['cycle_number'] = cycle_num
        
        return episode_info['start_index'], episode_info
    
    def get_coverage_summary(self) -> Dict[str, Any]:
        """전체 데이터 커버리지 요약 정보"""
        if not self.episode_plan:
            return {}
        
        total_planned_steps = sum(ep['planned_length'] for ep in self.episode_plan)
        
        # 겹침 계산
        total_overlap = 0
        for i in range(len(self.episode_plan) - 1):
            curr_end = self.episode_plan[i]['end_index']
            next_start = self.episode_plan[i + 1]['start_index']
            if curr_end > next_start:
                total_overlap += curr_end - next_start
        
        unique_covered = total_planned_steps - total_overlap
        
        # 에피소드 타입별 통계
        episode_types = {}
        for ep in self.episode_plan:
            ep_type = ep.get('episode_type', self._classify_episode_type(ep))
            episode_types[ep_type] = episode_types.get(ep_type, 0) + 1
        
        return {
            'total_data_length': self.data_length,
            'usable_length': self.usable_length,
            'total_episodes': len(self.episode_plan),
            'total_planned_steps': total_planned_steps,
            'unique_covered_steps': unique_covered,
            'total_overlap_steps': total_overlap,
            'unique_coverage_pct': (unique_covered / self.usable_length) * 100,
            'total_coverage_pct': (total_planned_steps / self.usable_length) * 100,
            'average_episode_length': total_planned_steps / len(self.episode_plan),
            'episode_types': episode_types,
            'adaptive_length_enabled': self.adaptive_length,
            'min_episode_length': min(ep['planned_length'] for ep in self.episode_plan),
            'max_episode_length': max(ep['planned_length'] for ep in self.episode_plan)
        }
    
    def _log_initialization(self):
        """초기화 정보 로깅"""
        summary = self.get_coverage_summary()
        
        self.logger.info("=" * 70)
        self.logger.info("🔄 적응형 순차적 에피소드 관리자 초기화")
        self.logger.info("=" * 70)
        self.logger.info(f"📊 전체 데이터 길이: {self.data_length:,}")
        self.logger.info(f"🎯 사용 가능 길이: {self.usable_length:,}")
        self.logger.info(f"📈 총 에피소드 수: {summary['total_episodes']}")
        self.logger.info(f"🔧 적응형 길이: {'활성화' if self.adaptive_length else '비활성화'}")
        self.logger.info(f"📏 에피소드 길이 범위: {summary['min_episode_length']}~{summary['max_episode_length']}")
        self.logger.info(f"📏 평균 에피소드 길이: {summary['average_episode_length']:.1f}")
        self.logger.info(f"🔗 겹침 비율: {self.overlap_ratio:.1%}")
        self.logger.info(f"🎯 고유 커버리지: {summary['unique_coverage_pct']:.1f}%")
        
        # 에피소드 타입 분포
        if summary['episode_types']:
            self.logger.info(f"📊 에피소드 타입 분포:")
            for ep_type, count in summary['episode_types'].items():
                self.logger.info(f"   └─ {ep_type}: {count}개")
        
        self.logger.info("=" * 70)
        
        # 처음 몇 개 에피소드 계획 미리보기
        self.logger.info("📋 에피소드 계획 미리보기 (처음 5개):")
        for i in range(min(5, len(self.episode_plan))):
            ep = self.episode_plan[i]
            ep_type = ep.get('episode_type', 'unknown')
            self.logger.info(
                f"   Episode {i:2d}: [{ep['start_index']:5d}~{ep['end_index']:5d}] "
                f"({ep['planned_length']:3d} steps, {ep_type})"
            )
        
        if len(self.episode_plan) > 5:
            self.logger.info(f"   ... 총 {len(self.episode_plan)}개 에피소드")
        self.logger.info("=" * 70)


class SequentialTradingEnvironment:
    """
    순차적 데이터 활용을 위한 TradingEnvironment 확장 (수정됨)
    """
    
    def __init__(self, base_env, episode_manager: SequentialEpisodeManager):
        """
        개선된 순차적 거래 환경 초기화
        
        Args:
            base_env: 기존 TradingEnvironment 인스턴스
            episode_manager: 순차적 에피소드 관리자
        """
        self.base_env = base_env
        self.episode_manager = episode_manager
        self.episode_meta_info = {}
        
        # 디버깅을 위한 로거
        self.logger = getattr(base_env, 'logger', None) or logging.getLogger(__name__)
        
    def reset(self, episode_num: int = None):
        """
        환경 재설정 (개선된 순차적 시작점)
        
        Args:
            episode_num: 에피소드 번호 (필수)
        """
        if episode_num is None:
            self.logger.warning("⚠️ episode_num이 None입니다. 0으로 설정합니다.")
            episode_num = 0
        
        # 디버깅 로그
        self.logger.debug(f"🔄 Episode {episode_num} reset 시작")
        
        # 적응형 시작점 가져오기
        start_index, self.episode_meta_info = self.episode_manager.get_episode_info(episode_num)
        
        # 디버깅 로그
        self.logger.debug(f"📊 Episode {episode_num}: 시작점 {start_index}, 범위 [{self.episode_meta_info['start_index']}~{self.episode_meta_info['end_index']}]")
        
        # 기본 환경 초기화 (시작점 설정)
        self.base_env.current_step = start_index
        self.base_env.balance = self.base_env.initial_balance
        self.base_env.shares_held = 0
        self.base_env.cost_basis = 0
        self.base_env.total_shares_purchased = 0
        self.base_env.total_shares_sold = 0
        self.base_env.total_sales_value = 0
        self.base_env.total_commission = 0
        self.base_env.trade_executed = False
        self.base_env.position = "홀드"
        self.base_env.previous_shares_held = 0
        self.base_env.invalid_sell_penalty = False
        
        # 히스토리 초기화
        self.base_env.states_history = []
        self.base_env.actions_history = []
        self.base_env.rewards_history = []
        self.base_env.portfolio_values_history = []
        
        # 분석용 히스토리 초기화
        if hasattr(self.base_env, 'trade_effects_history'):
            self.base_env.trade_effects_history = []
            self.base_env.market_effects_history = []
            self.base_env.combined_effects_history = []
        
        return self.base_env._get_observation()
    
    def step(self, action):
        """
        환경에서 한 스텝 진행 (개선된 종료 조건)
        """
        # 기본 스텝 실행
        obs, reward, done, info = self.base_env.step(action)
        
        # 개선된 종료 조건 체크
        current_step = self.base_env.current_step
        start_index = self.episode_meta_info['start_index']
        end_index = self.episode_meta_info['end_index']
        planned_length = self.episode_meta_info['actual_length']
        
        steps_in_episode = current_step - start_index
        
        # 다중 종료 조건 (우선순위대로)
        natural_end = (current_step >= end_index)  # 계획된 종료점 도달
        planned_end = (steps_in_episode >= planned_length)  # 계획된 길이 도달
        safety_end = (steps_in_episode >= planned_length * 1.2)  # 안전장치 (20% 여유)
        
        should_end = natural_end or planned_end or done
        
        if should_end and not done:
            done = True
            termination_reasons = []
            if natural_end:
                termination_reasons.append("자연 종료점 도달")
            if planned_end:
                termination_reasons.append("계획된 길이 달성")
            if safety_end:
                termination_reasons.append("안전장치 작동")
            
            self.logger.debug(f"✅ Episode 종료: {', '.join(termination_reasons)} (step {current_step}, episode_steps {steps_in_episode})")
        
        # 메타 정보 업데이트
        info.update({
            'episode_meta': self.episode_meta_info,
            'sequential_manager': 'improved',
            'data_coverage': f"{self.episode_meta_info['coverage_start_pct']:.1f}%~{self.episode_meta_info['coverage_end_pct']:.1f}%",
            'current_step': current_step,
            'episode_start': start_index,
            'episode_end': end_index,
            'steps_in_episode': steps_in_episode,
            'episode_progress': f"{steps_in_episode}/{planned_length}",
            'episode_type': self.episode_meta_info.get('episode_type', 'unknown')
        })
        
        return obs, reward, done, info
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """리스크 지표 반환 (base_env에서 가져옴)"""
        if hasattr(self.base_env, 'get_risk_metrics'):
            return self.base_env.get_risk_metrics()
        else:
            # 기본값 반환
            return {
                'max_drawdown_pct': 0.0,
                'max_daily_loss_pct': 0.0,
                'peak_portfolio_value': getattr(self.base_env, 'initial_balance', 100000.0),
                'daily_start_portfolio': getattr(self.base_env, 'initial_balance', 100000.0)
            }
    
    def _check_risk_limits(self, current_portfolio_value: float) -> Tuple[bool, bool]:
        """리스크 한도 체크 (base_env에 위임)"""
        if hasattr(self.base_env, '_check_risk_limits'):
            return self.base_env._check_risk_limits(current_portfolio_value)
        else:
            return False, False
        
    # 나머지 메서드들은 base_env에 위임
    def __getattr__(self, name):
        return getattr(self.base_env, name)


def create_sequential_training_setup(base_env, overlap_ratio: float = 0.0, logger=None):
    """
    순차적 학습 설정 생성 헬퍼 함수
    
    Args:
        base_env: 기본 TradingEnvironment
        overlap_ratio: 에피소드 간 겹침 비율
        logger: 로거
        
    Returns:
        (SequentialTradingEnvironment, SequentialEpisodeManager)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 순차적 에피소드 관리자 생성
    episode_manager = SequentialEpisodeManager(
        data_length=base_env.data_length,
        window_size=base_env.window_size,
        max_steps=MAX_STEPS_PER_EPISODE,
        min_steps=100,
        overlap_ratio=overlap_ratio,
        adaptive_length=True,
        logger=logger
    )
    
    # 순차적 환경 생성
    sequential_env = SequentialTradingEnvironment(base_env, episode_manager)
    
    return sequential_env, episode_manager


def create_improved_sequential_training_setup(base_env, overlap_ratio: float = 0.1, 
                                            adaptive_length: bool = True, logger=None):
    """
    개선된 순차적 학습 설정 생성 헬퍼 함수 (호환성 함수)
    
    Args:
        base_env: 기본 TradingEnvironment
        overlap_ratio: 에피소드 간 겹침 비율
        adaptive_length: 적응형 길이 사용 여부
        logger: 로거
        
    Returns:
        (SequentialTradingEnvironment, SequentialEpisodeManager)
    """
    # 기본 함수와 동일하게 처리 (호환성 유지)
    return create_sequential_training_setup(base_env, overlap_ratio, logger)


# 디버깅을 위한 테스트 함수
def test_sequential_episodes(sequential_env, num_test_episodes=5):
    """순차적 에피소드 테스트"""
    print("="*60)
    print("🧪 순차적 에피소드 테스트")
    print("="*60)
    
    for episode in range(num_test_episodes):
        print(f"\n📋 Episode {episode} 테스트:")
        
        # 리셋
        obs = sequential_env.reset(episode_num=episode)
        print(f"   리셋 완료: current_step = {sequential_env.base_env.current_step}")
        
        # 메타 정보 확인
        meta = sequential_env.episode_meta_info
        print(f"   데이터 범위: [{meta['start_index']}~{meta['end_index']}]")
        print(f"   실제 길이: {meta['actual_length']} steps")
        print(f"   커버리지: {meta['coverage_start_pct']:.1f}%~{meta['coverage_end_pct']:.1f}%")
        
        # 몇 스텝 실행 테스트
        for step in range(5):
            obs, reward, done, info = sequential_env.step(0.0)  # 홀드 액션
            if done:
                print(f"   Episode {episode} - Step {step+1}에서 종료됨")
                break
        else:
            print(f"   Episode {episode} - 5 steps 정상 실행")
    
    print("\n✅ 테스트 완료")