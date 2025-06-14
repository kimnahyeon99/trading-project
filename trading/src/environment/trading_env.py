"""
강화학습을 위한 트레이딩 환경 모듈 (효율적인 로깅 버전)
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
from src.data_collection.data_collector import DataCollector
from decimal import Decimal, ROUND_DOWN, getcontext

# 저장 경로 폴더 생성
save_dir = './results'
os.makedirs(save_dir, exist_ok=True)

from src.config.ea_teb_config import (
    INITIAL_BALANCE,
    MAX_TRADING_UNITS,
    TRANSACTION_FEE_PERCENT,
    WINDOW_SIZE,
    LOGGER,
)

class TradingEnvironment:
    """
    강화학습을 위한 트레이딩 환경 클래스 (효율적인 로깅 버전)
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        raw_data: pd.DataFrame = None,
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT,
        symbol: str = None,
        train_data: bool = True,
        reward_mode: str = 'combined',
        detailed_logging: bool = True,
        log_level: str = 'normal'
    ):
        """
        TradingEnvironment 클래스 초기화
        """
        # 📊 기본 데이터 설정
        self.data = data
        if raw_data is not None:
            self.raw_data = raw_data
        else:
            LOGGER.warning("⚠️ raw_data가 제공되지 않았습니다. 정규화된 데이터를 사용합니다.")
            self.raw_data = data
        
        print(f'📈 정규화된 데이터 형태: {data.shape}')
        print(f'📊 원본 데이터 형태: {self.raw_data.shape}')
        
        # ⏰ 타임스탬프 초기화
        if hasattr(self.raw_data.index, 'values'):
            self.timestamps = self.raw_data.index.values
        else:
            self.timestamps = np.array([None] * len(self.raw_data))

        # 🎯 환경 설정 파라미터
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.max_trading_units = max_trading_units
        self.transaction_fee_percent = transaction_fee_percent
        self.symbol = symbol if symbol else "UNKNOWN"
        self.train_data = train_data
        self.reward_mode = reward_mode
        self.detailed_logging = detailed_logging
        self.log_level = log_level  # 로그 레벨 추가
        
        # 📏 데이터 관련 변수
        self.feature_dim = data.shape[1]
        self.data_length = len(data)
        
        # 💰 환경 상태 변수 초기화
        self._init_state_variables()
        
        # 🎮 행동 공간: [-1.0, 1.0] 범위의 연속적인 값
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # 👀 관측 공간: 가격 데이터 + 포트폴리오 상태
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.window_size, self.feature_dim), dtype=np.float32
            ),
            'portfolio_state': spaces.Box(
                low=0, high=np.inf, shape=(2,), dtype=np.float32
            )
        })
        
        # 🔍 원본 데이터 검증
        if 'close' not in self.raw_data.columns:
            LOGGER.warning("⚠️ raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
        
        # ✅ 유효성 검사
        valid_modes = ['combined', 'trade_only', 'market_only', 'separated']
        if reward_mode not in valid_modes:
            raise ValueError(f"reward_mode는 {valid_modes} 중 하나여야 합니다")
        
    def _init_state_variables(self):
        """환경 상태 변수 초기화"""
        # 랜덤 시작점 설정 (window_size 이후부터 시작 가능)
        max_start = self.data_length - self.window_size - 1
        if max_start > self.window_size:
            self.current_step = np.random.randint(self.window_size, max_start)
        else:
            self.current_step = self.window_size
            
        self.balance = self.initial_balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_purchased = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission = 0
        self.position = "홀드"
        self.trade_executed = False 
        self.previous_shares_held = 0
        self.invalid_sell_penalty = False
        
        # 📈 에피소드 히스토리
        self.states_history = []
        self.actions_history = []
        self.rewards_history = []
        self.portfolio_values_history = []
        self.trade_effects_history = []
        self.market_effects_history = []
        self.combined_effects_history = []
    
    def reset(self) -> Dict[str, np.ndarray]:
        """🔄 환경 초기화 (랜덤 시작점)"""
        if self.log_level != 'minimal':
            print(f"\n🔄 환경 리셋 시작 - {self.symbol}")
            print(f"💰 초기 잔고: ${self.initial_balance:.2f}")
        
        self._init_state_variables()
        
        if self.log_level != 'minimal':
            print(f"📍 시작점: {self.current_step} (랜덤)")
            
        return self._get_observation()

    def _get_current_price(self) -> float:
        """💵 현재 주가 반환 (최적화된 로깅)"""
        if self.current_step >= len(self.raw_data):
            last_valid_index = len(self.raw_data) - 1
            if self.log_level == 'detailed':
                LOGGER.warning(f"⚠️ 인덱스 범위 초과: {self.current_step} >= {len(self.raw_data)}")
            
            if 'close' in self.raw_data.columns:
                price = float(self.raw_data.iloc[last_valid_index]['close'])
            else:
                price = float(self.raw_data.iloc[last_valid_index][-1])
        else:
            if 'close' in self.raw_data.columns:
                price = float(self.raw_data.iloc[self.current_step]['close'])
            else:
                price = float(self.raw_data.iloc[self.current_step][-1])
        
        # 🚨 가격이 0 이하인 경우 처리
        if price <= 0:
            if self.log_level == 'detailed':
                LOGGER.warning(f"⚠️ 현재 가격이 0 이하입니다: {price}, 최소값으로 조정")
            price = 0.01
            
        return price

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """👀 현재 관측값 반환"""
        # 📊 윈도우 크기만큼의 정규화된 가격 데이터 가져오기
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # 🔧 데이터가 충분하지 않은 경우 패딩 처리
        if start_idx == 0 and end_idx - start_idx < self.window_size:
            market_data = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
            actual_data = self.data.iloc[start_idx:end_idx].values
            market_data[-len(actual_data):] = actual_data
        else:
            market_data = self.data.iloc[start_idx:end_idx].values
            if len(market_data) < self.window_size:
                padding = np.zeros((self.window_size - len(market_data), self.feature_dim), dtype=np.float32)
                market_data = np.vstack([padding, market_data])
        
        # 💎 포트폴리오 상태 계산
        portfolio_value = self._get_portfolio_value()
        stock_value = self.shares_held * self._get_current_price()
        
        portfolio_state = np.array([
            self.balance / portfolio_value,  # 현금 비율
            stock_value / portfolio_value    # 주식 비율
        ], dtype=np.float32)
        
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio_state': portfolio_state
        }
        
        self.states_history.append(observation)
        return observation

    def _get_portfolio_value(self) -> float:
        """💎 현재 포트폴리오 가치 계산"""
        current_price = self._get_current_price()
        stock_value = self.shares_held * current_price
        total_value = self.balance + stock_value
        
        if total_value <= 0:
            return max(self.balance, 1.0)
            
        return total_value

    def _calculate_reward(self, prev_portfolio_value: float, portfolio_after_trade: float, 
                         current_portfolio_value: float) -> Union[float, Dict[str, float]]:
        """🎁 개선된 보상 계산 (실시간 트레이딩 최적화)"""
        if prev_portfolio_value <= 0:
            prev_portfolio_value = max(self.balance, 1.0)
        
        # 각 효과 계산 (정규화)
        if prev_portfolio_value > 0:
            trade_effect = (portfolio_after_trade - prev_portfolio_value) / prev_portfolio_value
        else:
            trade_effect = 0
            
        if portfolio_after_trade > 0:
            market_effect = (current_portfolio_value - portfolio_after_trade) / portfolio_after_trade
        else:
            market_effect = 0
            
        if prev_portfolio_value > 0:
            total_effect = (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        else:
            total_effect = 0
        
        # ✅ 실시간 트레이딩 최적화된 보상
        if self.reward_mode == 'trade_only':
            # 거래 효과만 고려 (더 보수적인 스케일링)
            reward = np.tanh(trade_effect * 50)  # [-1, 1] 범위로 부드럽게 제한
        elif self.reward_mode == 'market_only':
            # 시장 효과만 고려
            reward = np.tanh(market_effect * 30)
        elif self.reward_mode == 'separated':
            trade_reward = np.tanh(trade_effect * 50)
            market_reward = np.tanh(market_effect * 30)
            reward = {
                'trade': trade_reward,
                'market': market_reward,
                'total': trade_reward + market_reward
            }
        else:  # 'combined' - 실시간 트레이딩 최적화
            # ✅ 포트폴리오 총 수익률 기반 (더 부드러운 보상)
            base_reward = np.tanh(total_effect * 20)  # 기본 수익률 보상
            
            # ✅ 포지션 다양성 보상 (무거래 방지)
            current_stock_ratio = self._get_current_stock_ratio()
            # 적절한 포지션 유지 보상 (0.2~0.8 사이가 좋음)
            if 0.2 <= current_stock_ratio <= 0.8:
                position_bonus = 0.1
            elif current_stock_ratio == 0 or current_stock_ratio == 1:
                position_bonus = -0.2  # 극단적 포지션 패널티
            else:
                position_bonus = 0
            
            # ✅ 변동성 조정 (과도한 거래 방지)
            if abs(trade_effect) > 0.05:  # 5% 이상 변화
                volatility_penalty = -0.05
            else:
                volatility_penalty = 0
            
            reward = base_reward + position_bonus + volatility_penalty
        
        # ✅ 개선된 패널티 시스템 (더 부드럽게)
        if self.invalid_sell_penalty:
            penalty = -0.3  # -1.0에서 -0.3으로 완화
            if isinstance(reward, dict):
                reward['trade'] += penalty
                reward['total'] += penalty
            else:
                reward += penalty
        
        # ✅ 최종 보상 클리핑 (극단값 방지)
        if isinstance(reward, dict):
            for key in reward:
                reward[key] = np.clip(reward[key], -2.0, 2.0)
        else:
            reward = np.clip(reward, -2.0, 2.0)
        
        return reward

    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], Union[float, Dict[str, float]], bool, Dict[str, Any]]:
        """
        환경에서 한 스텝 진행 (효율적인 로깅 버전)
        """
        # 🚀 Step 시작 로깅
        if self.log_level != 'minimal':
            print(f"\n✅ Step {self.current_step}")
            if self.log_level == 'detailed':
                print("=" * 60)
        
        # 초기화
        self.invalid_sell_penalty = False
        self.actions_history.append(action)
        
        # 📊 거래 전 상태
        current_price_before = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()
        
        # 시간 정보 (필요한 경우만)
        if self.current_step < len(self.raw_data):
            before_timestamp = self.raw_data.index[self.current_step]
        else:
            before_timestamp = "범위 초과"
        
        if self.log_level != 'minimal':
            action_value = action[0] if isinstance(action, np.ndarray) else action
            print(f"⏰ {before_timestamp} | ✅ 거래 전 가격 : ${current_price_before:.2f} | ✅ 행동 값 : {action_value:.3f}")
            print(f"✅  포트폴리오: ${prev_portfolio_value:.2f} (${self.balance:.2f} + {self.shares_held:.3f}주)")
        
        # 거래 실행
        self._execute_trade_action(action)
        
        # 거래 후 포트폴리오 가치 계산
        portfolio_after_trade = self.balance + self.shares_held * current_price_before
        
        # 다음 스텝으로 이동
        self.current_step += 1
        
        # 조기 종료 조건 확인
        if self.current_step >= len(self.raw_data):
            if self.log_level == 'detailed':
                LOGGER.warning(f"⚠️ 데이터 범위 초과로 에피소드 조기 종료")
            done = True
            current_portfolio_value = portfolio_after_trade
            current_price_after = current_price_before
            after_timestamp = before_timestamp
        else:
            # 시장 변동 후 포트폴리오 가치
            current_price_after = self._get_current_price()
            current_portfolio_value = self._get_portfolio_value()
            after_timestamp = self.raw_data.index[self.current_step]
            done = self.current_step >= self.data_length - 1
        
        # 가격 및 포트폴리오 변화 로깅
        if self.log_level != 'minimal':
            price_change = current_price_after - current_price_before
            portfolio_change = current_portfolio_value - prev_portfolio_value
            
            print(f"✅ 가격 변화: ${price_change:+.2f} | ✅ 포트폴리오 변화: ${portfolio_change:+.2f}")
            
            if self.log_level == 'detailed':
                print(f"✅ 가격: ${current_price_before:.2f} → ${current_price_after:.2f}")
                print(f"✅ 포트폴리오: ${prev_portfolio_value:.2f} → ${current_portfolio_value:.2f}")
        
        # 보상 계산
        reward = self._calculate_reward(prev_portfolio_value, portfolio_after_trade, current_portfolio_value)
        
        # 보상 로깅
        if self.log_level != 'minimal':
            if isinstance(reward, dict):
                print(f"✅ 보상 - 거래:{reward['trade']:.4f} | 시장:{reward['market']:.4f} | 총:{reward['total']:.4f}")
            else:
                print(f"✅ 보상: {reward:.4f}")
        
        # 히스토리 업데이트
        self.portfolio_values_history.append(current_portfolio_value)
        
        if isinstance(reward, dict):
            self.rewards_history.append(reward['total'])
            self.trade_effects_history.append(reward['trade'])
            self.market_effects_history.append(reward['market'])
        else:
            self.rewards_history.append(reward)
            trade_effect = (portfolio_after_trade - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            market_effect = (current_portfolio_value - portfolio_after_trade) / portfolio_after_trade if portfolio_after_trade > 0 else 0
            self.trade_effects_history.append(trade_effect)
            self.market_effects_history.append(market_effect)
        
        # 관측값 및 추가 정보
        observation = self._get_observation()
        info = self._get_info()
        
        # 시간 정보를 info에 추가
        info.update({
            'reward_mode': self.reward_mode,
            'current_timestamp': after_timestamp,
            'previous_timestamp': before_timestamp,
            'price_before': current_price_before,
            'price_after': current_price_after,
            'portfolio_before': prev_portfolio_value,
            'portfolio_after_trade': portfolio_after_trade,
            'portfolio_after_market': current_portfolio_value,
            # 🆕 포지션 변화 정보 추가 (엔트로피 조정용)
            'position_change': abs(info['stock_ratio'] - (self.previous_shares_held * current_price_before / prev_portfolio_value) if prev_portfolio_value > 0 else 0),
            'target_position_ratio': info['stock_ratio'],  # 현재가 목표가 됨
        })
        
        if self.log_level == 'detailed':
            print("=" * 60)
        
        return observation, reward, done, info

    def _get_info(self) -> Dict[str, Any]:
        """추가 정보 반환"""
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        # 🆕 포트폴리오 비율 정보 추가
        stock_ratio = self._get_current_stock_ratio()
        cash_ratio = 1.0 - stock_ratio
        
        # 타임스탬프 정보 추가
        if self.current_step < len(self.raw_data):
            current_timestamp = self.raw_data.index[self.current_step]
        else:
            current_timestamp = "범위 초과"
        
        timestamps_info = {
            'current': current_timestamp,
            'total_length': len(self.raw_data)
        }
        
        total_return = (portfolio_value - self.initial_balance) / self.initial_balance
        position = self.position
        
        return {
            # 기존 필드들...
            'step': self.current_step,
            'timestamp': current_timestamp,
            'timestamps': timestamps_info,
            'balance': self.balance,
            'shares_held': self.shares_held,
            'position': position, 
            "previous_shares_held": self.previous_shares_held,
            'current_price': current_price,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'cost_basis': self.cost_basis,
            'total_shares_purchased': self.total_shares_purchased,
            'total_shares_sold': self.total_shares_sold,
            'total_sales_value': self.total_sales_value,
            'total_commission': self.total_commission,
            'trade_executed': self.trade_executed,
            'stock_ratio': stock_ratio,
            'cash_ratio': cash_ratio,
        }
    
    def _execute_trade_action(self, action: float) -> None:
        """거래 행동 실행 (개선된 버전)"""
        self.trade_executed = False
        self.previous_shares_held = self.shares_held
        self.invalid_sell_penalty = False
        
        target_ratio = action
        current_ratio = self._get_current_stock_ratio()
        
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        ratio_diff = target_ratio - current_ratio
        
        # 임계값을 더 낮춰서 evaluation에서도 더 민감한 거래 실행
        threshold = 0.0005 if self.log_level == 'minimal' else 0.001  # evaluation: 0.05%, training: 0.1%
        
        if abs(ratio_diff) > threshold:
            if self.log_level == 'detailed':
                print(f"   🎯 거래 결정: 목표비율 {target_ratio:.4f}, 현재비율 {current_ratio:.4f}, 차이 {ratio_diff:.4f}")
            
            if ratio_diff > 0:
                self._execute_buy_by_ratio(ratio_diff, current_price, portfolio_value)
            else:
                self._execute_sell_by_ratio(-ratio_diff, current_price, portfolio_value)
        else:
            if self.log_level == 'detailed':
                print(f"   💤 거래 없음: 비율 차이 {abs(ratio_diff):.6f} < 임계값 {threshold:.6f}")

    def _execute_buy_by_ratio(self, target_increase_ratio: float, current_price: float, portfolio_value: float) -> None:
        """비율 기반 매수 실행"""
        # 목표 매수 금액 계산
        target_buy_amount = portfolio_value * target_increase_ratio
        
        # 거래 비용 고려한 실제 필요 금액
        total_cost_needed = target_buy_amount * (1 + self.transaction_fee_percent)
        
        if self.balance < total_cost_needed:
            # 잔고 부족 시 가능한 최대 금액으로 조정
            available_for_stocks = self.balance / (1 + self.transaction_fee_percent)
            if available_for_stocks > 0:
                target_buy_amount = available_for_stocks
                total_cost_needed = self.balance
            else:
                if self.log_level == 'detailed':
                    print(f"   ❌ 잔고 부족: 필요 ${total_cost_needed:.2f}, 보유 ${self.balance:.2f}")
                return
        
        # 매수할 주식 수 계산
        shares_to_buy = target_buy_amount / current_price
        
        if shares_to_buy > 0:
            # 거래 실행
            actual_cost = shares_to_buy * current_price
            commission = actual_cost * self.transaction_fee_percent
            total_cost = actual_cost + commission
            
            self.balance -= total_cost
            self.shares_held += shares_to_buy
            self.total_shares_purchased += shares_to_buy
            self.total_commission += commission
            
            # 평균 매수 단가 업데이트
            if self.shares_held > 0:
                self.cost_basis = ((self.cost_basis * (self.shares_held - shares_to_buy)) + actual_cost) / self.shares_held
            
            self.trade_executed = True
            self.position = "매수"
            
            if self.log_level == 'detailed':
                print(f"   ✅ 매수 성공: {shares_to_buy:.3f}주 @ ${current_price:.2f} (목표금액: ${target_buy_amount:.2f})")

    def _execute_sell_by_ratio(self, target_decrease_ratio: float, current_price: float, portfolio_value: float) -> None:
        """비율 기반 매도 실행"""
        # 목표 매도 금액 계산
        target_sell_amount = portfolio_value * target_decrease_ratio
        
        # 매도할 주식 수 계산
        shares_to_sell = min(target_sell_amount / current_price, self.shares_held)
        
        if shares_to_sell <= 0:
            if self.log_level == 'detailed':
                print(f"   ❌ 매도할 주식이 없음")
            return
        
        # 거래 실행
        gross_sell_value = shares_to_sell * current_price
        commission = gross_sell_value * self.transaction_fee_percent
        net_value = gross_sell_value - commission
        
        self.balance += net_value
        self.shares_held -= shares_to_sell
        self.total_shares_sold += shares_to_sell
        self.total_sales_value += gross_sell_value
        self.total_commission += commission
        
        self.trade_executed = True
        self.position = "매도"
        
        if self.log_level == 'detailed':
            print(f"   ✅ 매도 성공: {shares_to_sell:.3f}주 @ ${current_price:.2f} (목표금액: ${target_sell_amount:.2f})")

    def _get_current_stock_ratio(self) -> float:
        """현재 포트폴리오에서 주식이 차지하는 비율 계산"""
        current_price = self._get_current_price()
        stock_value = self.shares_held * current_price
        total_portfolio = self.balance + stock_value
        
        if total_portfolio <= 0:
            return 0.0
        
        return stock_value / total_portfolio

    def render(self, mode: str = 'human') -> None:
        """환경 시각화"""
        info = self._get_info()
        
        print(f"✅ Step: {info['step']}")
        print(f"✅ 잔고: ${info['balance']:.2f}")
        print(f"✅ 보유량: {info['shares_held']}")
        print(f"✅ 현재가: ${info['current_price']:.2f}")
        print(f"✅ 포트폴리오 가치: ${info['portfolio_value']:.2f}")
        print(f"✅ 총 수익률: {info['total_return'] * 100:.2f}%")
        print(f"✅ 총 수수료: ${info['total_commission']:.2f}")
        # 🆕 비율 정보 추가
        print(f"✅ 주식 비율: {info['stock_ratio']:.1%}")
        print(f"✅ 현금 비율: {info['cash_ratio']:.1%}")
        print("-" * 50)
    
    def get_episode_data(self) -> Dict[str, List]:
        """에피소드 데이터 반환"""
        return {
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'portfolio_values': self.portfolio_values_history
        }
    
    def get_final_portfolio_value(self) -> float:
        """최종 포트폴리오 가치 반환"""
        return self._get_portfolio_value()
    
    def get_total_reward(self) -> float:
        """총 보상 반환"""
        return sum(self.rewards_history)
    
    def get_reward_analysis(self) -> Dict[str, Any]:
        """보상 분석 결과 반환"""
        if not self.trade_effects_history:
            return {}
        
        trade_effects = np.array(self.trade_effects_history)
        market_effects = np.array(self.market_effects_history)
        
        return {
            'trade_effects': {
                'mean': np.mean(trade_effects),
                'std': np.std(trade_effects),
                'sum': np.sum(trade_effects),
                'positive_ratio': np.mean(trade_effects > 0)
            },
            'market_effects': {
                'mean': np.mean(market_effects),
                'std': np.std(market_effects),
                'sum': np.sum(market_effects),
                'positive_ratio': np.mean(market_effects > 0)
            },
            'correlation': np.corrcoef(trade_effects, market_effects)[0, 1] if len(trade_effects) > 1 else 0
        }

    def set_log_level(self, level: str):
        """로그 레벨 변경"""
        if level in ['minimal', 'normal', 'detailed']:
            self.log_level = level
            if level != 'minimal':
                print(f"로그 레벨 변경: {level}")


def create_environment_from_results(results: Dict[str, Dict[str, Any]], symbol: str, data_type: str = 'test', 
                                  log_level: str = 'normal', **kwargs) -> TradingEnvironment:
    """
    DataProcessor 결과로부터 환경 생성하는 헬퍼 함수
    """
    if symbol not in results:
        raise ValueError(f"심볼 {symbol}을 결과에서 찾을 수 없습니다")
    
    result = results[symbol]
    
    if data_type not in result:
        raise ValueError(f"데이터 타입 {data_type}을 심볼 {symbol}에서 찾을 수 없습니다")
    
    normalized_data = result[data_type]
    featured_data = result['featured_data']
    
    if data_type == 'train':
        raw_data = featured_data.iloc[:len(normalized_data)]
    elif data_type == 'valid':
        train_len = len(result['train'])
        raw_data = featured_data.iloc[train_len:train_len + len(normalized_data)]
    else:  # test
        train_len = len(result['train'])
        valid_len = len(result['valid'])
        raw_data = featured_data.iloc[train_len + valid_len:train_len + valid_len + len(normalized_data)]
    
    print(f"✅ 환경 생성: {symbol} - {data_type} (로그 레벨: {log_level})")
    print(f"✅ 정규화 데이터: {normalized_data.shape}")
    print(f"✅ 원본 데이터: {raw_data.shape}")
    
    env = TradingEnvironment(
        data=normalized_data,
        raw_data=raw_data,
        symbol=symbol,
        train_data=(data_type == 'train'),
        log_level=log_level,
        **kwargs
    )
    return env    

class MultiAssetTradingEnvironment:
    """🏢 다중 자산 트레이딩 환경 클래스"""
    
    def __init__(
        self,
        results: Dict[str, Dict[str, Any]],
        symbols: List[str],
        data_type: str = 'test',
        window_size: int = WINDOW_SIZE,
        initial_balance: float = INITIAL_BALANCE,
        max_trading_units: int = MAX_TRADING_UNITS,
        transaction_fee_percent: float = TRANSACTION_FEE_PERCENT,
        reward_mode: str = 'combined',
        log_level: str = 'normal'
    ):
        """MultiAssetTradingEnvironment 클래스 초기화"""
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.data_type = data_type
        self.initial_balance = initial_balance
        
        print(f"🏢 다중 자산 환경 초기화: {self.n_assets}개 자산")
        
        # 🏭 개별 환경 생성
        self.envs = {}
        for symbol in symbols:
            print(f"📊 {symbol} 환경 생성 중...")
            self.envs[symbol] = create_environment_from_results(
                results=results,
                symbol=symbol,
                data_type=data_type,
                window_size=window_size,
                initial_balance=initial_balance / self.n_assets,
                max_trading_units=max_trading_units,
                transaction_fee_percent=transaction_fee_percent,
                reward_mode=reward_mode,
                log_level=log_level
            )
        
        # 🎮 행동 공간: 각 자산에 대한 연속 행동
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # 👀 관측 공간
        self.observation_space = spaces.Dict({
            symbol: env.observation_space for symbol, env in self.envs.items()
        })
        
        LOGGER.info(f"✅ 다중 자산 환경 초기화 완료: {self.n_assets}개 자산, {data_type} 데이터")

    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """🔄 환경 초기화"""
        print(f"🔄 다중 자산 환경 리셋")
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
        return observations
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """🚀 환경에서 한 스텝 진행"""
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        print(f"\n🚀 다중 자산 Step 진행")
        
        for symbol, env in self.envs.items():
            action = actions.get(symbol, 0.0)
            print(f"📈 {symbol} 처리 중...")
            obs, rew, done, info = env.step(action)
            
            observations[symbol] = obs
            rewards[symbol] = rew
            dones[symbol] = done
            infos[symbol] = info
        
        # 전체 보상은 각 자산의 보상 평균
        total_reward = sum(rewards.values()) / self.n_assets
        
        # 모든 자산의 에피소드가 종료되면 전체 에피소드 종료
        done = all(dones.values())
        
        # 전체 포트폴리오 가치 계산
        total_portfolio_value = sum(info['portfolio_value'] for info in infos.values())
        trade_executed_any = any(info.get('trade_executed', False) for info in infos.values())
        
        infos['total'] = {
            'portfolio_value': total_portfolio_value,
            'total_return': (total_portfolio_value - self.initial_balance) / self.initial_balance,
            'trade_executed': trade_executed_any
        }

        return observations, total_reward, done, infos
    
    def set_log_level(self, level: str):
        """모든 환경의 로그 레벨 변경"""
        for env in self.envs.values():
            env.set_log_level(level)
    
    def render(self, mode: str = 'human') -> None:
        """🖥️ 환경 시각화"""
        total_portfolio_value = 0
        
        print("=" * 50)
        print("✅ 다중 자산 트레이딩 환경 상태")
        print("=" * 50)
        
        for symbol, env in self.envs.items():
            info = env._get_info()
            total_portfolio_value += info['portfolio_value']
            
            print(f"✅ 자산: {symbol}")
            print(f"  ✅ 가격: ${info['current_price']:.2f}")
            print(f"  ✅ 보유량: {info['shares_held']:.3f}")
            print(f"  ✅ 포트폴리오 가치: ${info['portfolio_value']:.2f}")
            print(f"  ✅ 수익률: {info['total_return'] * 100:.2f}%")
            print("-" * 50)
        
        total_return = (total_portfolio_value - self.initial_balance) / self.initial_balance
        print(f"✅ 총 포트폴리오 가치: ${total_portfolio_value:.2f}")
        print(f"✅ 총 수익률: {total_return * 100:.2f}%")
        print("=" * 50)
    
    def get_final_portfolio_value(self) -> float:
        """최종 포트폴리오 가치 반환"""
        return sum(env.get_final_portfolio_value() for env in self.envs.values())
    
    def get_total_reward(self) -> float:
        """총 보상 반환"""
        return sum(env.get_total_reward() for env in self.envs.values()) / self.n_assets


def create_flexible_environment(results, symbol, reward_mode='combined', log_level='normal', 
                              detailed_logging=True, data_type='train'):
    """
    🔧 유연한 보상 시스템을 가진 환경 생성
    """
    print(f" 유연한 환경 생성: {symbol} - {reward_mode} (로그: {log_level})")
    env = create_environment_from_results(
        results=results,
        symbol=symbol,
        data_type=data_type,
        reward_mode=reward_mode,
        detailed_logging=detailed_logging,
        log_level=log_level
    )
    return env


if __name__ == "__main__":
    # 🧪 모듈 테스트 코드
    import matplotlib.pyplot as plt
    from src.data_collection.data_collector import DataCollector
    from src.preprocessing.data_processor import DataProcessor
    from src.config.config import config
    import argparse
    
    parser = argparse.ArgumentParser(description="테스트할 심볼")
    parser.add_argument("--symbols", nargs="+", help="테스트할 심볼 리스트", default=config.trading_symbols)
    parser.add_argument("--log_level", choices=['minimal', 'normal', 'detailed'], default='normal', help="로그 레벨")
    args = parser.parse_args()
    
    symbols = args.symbols
    TARGET_SYMBOL = symbols[0] if symbols else config.trading_symbols
    print(f'📈 테스트 심볼들: {symbols}')
    print(f'🎯 주요 테스트 심볼: {TARGET_SYMBOL}')
    print(f'📝 로그 레벨: {args.log_level}')
    
    # 📊 데이터 수집 및 전처리
    print("📊 데이터 수집 시작...")
    collector = DataCollector(symbols=symbols)
    data = collector.load_all_data()
    
    try:
        if data:
            print("🔄 데이터 전처리 시작...")
            processor = DataProcessor()
            results = processor.process_all_symbols(data)
            
            if TARGET_SYMBOL in results:
                print("=" * 60)
                print(f"🧪 효율적인 로깅 TradingEnvironment 테스트 시작 - {TARGET_SYMBOL}")
                print("=" * 60)
                
                # 🎯 다양한 로그 레벨로 테스트
                for log_level in ['minimal', 'normal', 'detailed']:
                    print(f"\n🔄 로그 레벨 테스트: {log_level}")
                    print("-" * 40)
                    
                    env = create_flexible_environment(
                        results=results,
                        symbol=TARGET_SYMBOL,
                        reward_mode='combined',
                        log_level=log_level,
                        detailed_logging=True,
                        data_type='train'
                    )
                    
                    obs = env.reset()
                    
                    # 🚀 3스텝 실행
                    for i in range(3):
                        action = np.random.uniform(-1.0, 1.0)
                        obs, reward, done, info = env.step(action)
                        
                        if done:
                            break
                    
                    print(f"✅ {log_level} 레벨 테스트 완료")
                
                print("\n" + "=" * 60)
                print("✅ 모든 테스트 완료!")
                print("=" * 60)
                
            else:
                print(f"❌ {TARGET_SYMBOL} 데이터를 찾을 수 없습니다.")
                print(f"✅ 사용 가능한 심볼: {list(results.keys())}")
        else:
            print("❌ 데이터를 로드할 수 없습니다.")
            
    except Exception as e:
        print(f"🚨 테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()