"""
균형잡힌 트레이딩 환경 모듈 (매수 편향 문제 해결)
"""
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional, Any
import gym
from gym import spaces
from src.environment.trading_env import TradingEnvironment

from src.config.ea_teb_config import (
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    LOGGER,
)


class BalancedTradingEnvironment(TradingEnvironment):
    """
    균형잡힌 트레이딩 환경 클래스 (매수 편향 문제 해결)
    """

    def __init__(self, *args, **kwargs):
        """초기화"""
        # 균형잡힌 초기 상태 설정 (부모 클래스에 전달하지 않음)
        self.balanced_initialization = kwargs.pop('balanced_initialization', True)
        
        super().__init__(*args, **kwargs)
        
        LOGGER.info(f"🎯 균형잡힌 트레이딩 환경 초기화 (균형 시작: {self.balanced_initialization})")

    def reset(self) -> Dict[str, np.ndarray]:
        """🔄 환경 초기화 (균형잡힌 시작점)"""
        if self.log_level != 'minimal':
            print(f"\n🔄 균형잡힌 환경 리셋 시작 - {self.symbol}")
        
        # 기본 초기화
        self._init_state_variables()
        
        # 균형잡힌 초기 포지션 설정
        if self.balanced_initialization:
            current_price = self._get_current_price()
            
            # 50% 현금, 50% 주식으로 시작
            target_stock_value = self.initial_balance * 0.5
            shares_to_buy = target_stock_value / current_price
            cost = shares_to_buy * current_price
            
            self.balance = self.initial_balance - cost
            self.shares_held = shares_to_buy
            self.cost_basis = current_price
            
            if self.log_level != 'minimal':
                print(f"💰 균형잡힌 시작: 현금 ${self.balance:.2f}, 주식 ${shares_to_buy:.3f}주")
        
        return self._get_observation()

    def _execute_trade_action(self, action: float) -> None:
        """개선된 거래 행동 실행 (균형잡힌 해석)"""
        self.trade_executed = False
        self.previous_shares_held = self.shares_held
        self.invalid_sell_penalty = False
        
        # 🆕 개선된 행동 해석: action을 상대적 변화로 해석
        current_ratio = self._get_current_stock_ratio()
        
        # action을 [-1, 1]에서 [0, 1]로 매핑 (시그모이드 변환)
        target_ratio = 1 / (1 + np.exp(-action * 2))  # 시그모이드로 [0, 1] 매핑
        
        # 또는 더 직관적인 선형 매핑도 가능
        # target_ratio = (action + 1) / 2  # [-1, 1] -> [0, 1]
        
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        ratio_diff = target_ratio - current_ratio
        
        # 더 엄격한 임계값 (더 선별적인 거래)
        threshold = 0.02  # 2% 이상 차이날 때만 거래
        
        if abs(ratio_diff) > threshold:
            if self.log_level == 'detailed':
                print(f"   🎯 균형잡힌 거래: action={action:.3f} -> 목표비율 {target_ratio:.3f}, 현재 {current_ratio:.3f}")
            
            if ratio_diff > 0:
                self._execute_buy_by_ratio(ratio_diff, current_price, portfolio_value)
            else:
                self._execute_sell_by_ratio(-ratio_diff, current_price, portfolio_value)
        else:
            if self.log_level == 'detailed':
                print(f"   💤 거래 없음: 비율 차이 {abs(ratio_diff):.4f} < 임계값 {threshold:.4f}")

    def _calculate_reward(self, prev_portfolio_value: float, portfolio_after_trade: float, 
                         current_portfolio_value: float) -> Union[float, Dict[str, float]]:
        """🎁 균형잡힌 보상 계산 (매수 편향 제거)"""
        if prev_portfolio_value <= 0:
            prev_portfolio_value = max(self.balance, 1.0)
        
        # 기본 포트폴리오 수익률 (가장 중요한 요소)
        if prev_portfolio_value > 0:
            portfolio_return = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            portfolio_return = 0
        
        # 🆕 균형잡힌 포지션 보상 (매수 편향 제거)
        current_stock_ratio = self._get_current_stock_ratio()
        
        # 이상적인 균형: 40-60% 사이가 좋음 (더 넓은 범위)
        if 0.4 <= current_stock_ratio <= 0.6:
            balance_bonus = 0.02  # 작은 보너스
        elif 0.2 <= current_stock_ratio <= 0.8:
            balance_bonus = 0.01  # 매우 작은 보너스
        elif current_stock_ratio < 0.1 or current_stock_ratio > 0.9:
            balance_bonus = -0.05  # 극단적 포지션에 패널티
        else:
            balance_bonus = 0  # 중립
        
        # 🆕 거래 빈도 조절 (과도한 거래 방지)
        trade_penalty = 0
        if self.trade_executed:
            # 거래할 때마다 작은 비용 (현실적인 슬리피지 반영)
            trade_penalty = -0.001
        
        # 🆕 변동성 기반 보상 조정
        volatility_adjustment = 0
        if hasattr(self, 'portfolio_values_history') and len(self.portfolio_values_history) > 10:
            recent_values = self.portfolio_values_history[-10:]
            volatility = np.std(recent_values) / np.mean(recent_values) if np.mean(recent_values) > 0 else 0
            
            # 적당한 변동성은 좋지만, 너무 높으면 패널티
            if volatility > 0.05:  # 5% 이상 변동성
                volatility_adjustment = -0.02
        
        # 최종 보상 계산 (포트폴리오 수익률이 가장 중요)
        base_reward = np.tanh(portfolio_return * 100)  # 포트폴리오 수익률 기반
        
        total_reward = base_reward + balance_bonus + trade_penalty + volatility_adjustment
        
        # 패널티 적용
        if self.invalid_sell_penalty:
            total_reward -= 0.1  # 더 가벼운 패널티
        
        # 클리핑
        total_reward = np.clip(total_reward, -1.0, 1.0)
        
        if self.log_level == 'detailed':
            print(f"   📊 보상 구성: 기본={base_reward:.4f}, 균형={balance_bonus:.4f}, "
                  f"거래={trade_penalty:.4f}, 변동성={volatility_adjustment:.4f}")
        
        return total_reward


def create_balanced_environment_from_results(results: Dict[str, Dict[str, Any]], symbol: str, 
                                           data_type: str = 'test', **kwargs) -> BalancedTradingEnvironment:
    """
    균형잡힌 환경 생성 헬퍼 함수
    """
    if symbol not in results:
        raise ValueError(f"심볼 {symbol}을 결과에서 찾을 수 없습니다")
    
    result = results[symbol]
    
    if data_type not in result:
        raise ValueError(f"데이터 타입 {data_type}을 심볼 {symbol}에서 찾을 수 없습니다")
    
    normalized_data = result[data_type]
    featured_data = result['featured_data']
    
    if data_type == 'train':
        raw_data = featured_data
    elif data_type == 'valid':
        raw_data = featured_data
    else:  # test
        raw_data = featured_data
    
    # 환경 생성
    env = BalancedTradingEnvironment(
        data=normalized_data,
        raw_data=raw_data,
        symbol=symbol,
        train_data=(data_type == 'train'),
        **kwargs
    )
    
    LOGGER.info(f"✅ 균형잡힌 {symbol} 환경 생성 완료 ({data_type} 데이터)")
    return env


# 기존 환경과의 호환성을 위한 래퍼
def create_environment_from_results_balanced(results: Dict[str, Dict[str, Any]], symbol: str, 
                                           data_type: str = 'test', use_balanced: bool = True, 
                                           **kwargs):
    """
    기존 코드와 호환되는 환경 생성 함수
    """
    if use_balanced:
        return create_balanced_environment_from_results(results, symbol, data_type, **kwargs)
    else:
        # 기존 환경 사용
        from src.environment.trading_env import create_environment_from_results
        return create_environment_from_results(results, symbol, data_type, **kwargs) 