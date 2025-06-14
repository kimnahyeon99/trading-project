"""
강화학습을 위한 트레이딩 환경 모듈 (로그 최적화 버전)
debug_level 파라미터 추가: 'INFO' (필수 로그), 'DEBUG' (상세 로그)
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
    LOGGER
)

class TradingEnvironment:
    """
    강화학습을 위한 트레이딩 환경 클래스 (로그 최적화 버전)
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
        detailed_logging: bool = False,  # 기본값을 False로 변경
        debug_level: str = 'INFO'  # 새로 추가: INFO, DEBUG
    ):
        """
        TradingEnvironment 클래스 초기화
        """
        # 📊 기본 데이터 설정
        self.data = data
        if raw_data is not None:
            self.raw_data = raw_data
        else:
            LOGGER.warning("raw_data가 제공되지 않았습니다. 정규화된 데이터를 사용합니다.")
            self.raw_data = data
        
        # 로그 레벨 설정
        self.debug_level = debug_level
        self.detailed_logging = detailed_logging
        
        # 초기화 로그 (필수 정보만)
        LOGGER.info(f"환경 초기화: {symbol} | 데이터: {data.shape} | 모드: {reward_mode}")
        
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
            LOGGER.warning("raw_data에 'close' 컬럼이 없습니다. 마지막 컬럼을 종가로 사용합니다.")
        
        # ✅ 유효성 검사
        valid_modes = ['combined', 'trade_only', 'market_only', 'separated']
        if reward_mode not in valid_modes:
            raise ValueError(f"reward_mode는 {valid_modes} 중 하나여야 합니다")
        
    def _init_state_variables(self):
        """💰 환경 상태 변수 초기화"""
        self.current_step = 0
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
        """
        🔄 환경 초기화 (로그 간소화)
        """
        if self.debug_level == 'DEBUG':
            LOGGER.debug(f"환경 리셋: {self.symbol} | 초기잔고: ${self.initial_balance:.2f}")
        
        self._init_state_variables()
        return self._get_observation()

    def _get_current_price(self) -> float:
        """
        💵 현재 주가 반환 (로그 최적화)
        """
        # 🔍 인덱스 유효성 검사
        if self.current_step >= len(self.raw_data):
            last_valid_index = len(self.raw_data) - 1
            if self.debug_level == 'DEBUG':
                LOGGER.warning(f"인덱스 범위 초과: {self.current_step} >= {len(self.raw_data)}")
            
            if 'close' in self.raw_data.columns:
                price = float(self.raw_data.iloc[last_valid_index]['close'])
            else:
                price = float(self.raw_data.iloc[last_valid_index][-1])
        else:
            # 📊 정상적인 경우
            if 'close' in self.raw_data.columns:
                price = float(self.raw_data.iloc[self.current_step]['close'])
            else:
                price = float(self.raw_data.iloc[self.current_step][-1])
        
        # 🚨 가격이 0 이하인 경우 처리
        if price <= 0:
            LOGGER.warning(f"현재 가격이 0 이하입니다: {price}")
            price = 0.01
            
        return price

    def _execute_trade_action(self, action: float) -> None:
        """
        🔄 거래 행동 실행 (로그 간소화)
        """
        current_price = self._get_current_price()
        
        if current_price <= 0:
            LOGGER.warning(f"현재 가격이 0 이하입니다: {current_price}")
            return
        
        action_value = action[0] if isinstance(action, np.ndarray) else action
        
        # 🎮 기본 상태 설정
        self.trade_executed = False
        self.position = "홀드"
        self.invalid_sell_penalty = False
        
        # 💪 최소 행동 임계값 처리
        min_action_threshold = 0.1
        if 0 < abs(action_value) < min_action_threshold:
            action_value = min_action_threshold * (1 if action_value > 0 else -1)
        
        getcontext().prec = 10
        
        # 거래 전 상태 (DEBUG 레벨에서만 상세 로그)
        if self.debug_level == 'DEBUG':
            old_portfolio = self.balance + self.shares_held * current_price
            LOGGER.debug(f"거래실행: Step {self.current_step} | 행동: {action_value:.4f} | "
                        f"가격: ${current_price:.2f} | 포트폴리오: ${old_portfolio:.2f}")
        
        if action_value > 0:  # 💰 매수
            self._execute_buy(action_value, current_price)
        elif action_value < 0:  # 💸 매도
            self._execute_sell(action_value, current_price)
        
        # 거래 결과 로그 (간소화)
        if self.trade_executed:
            new_portfolio = self.balance + self.shares_held * current_price
            if self.debug_level == 'DEBUG':
                LOGGER.debug(f"거래완료: {self.position} | 잔고: ${self.balance:.2f} | "
                            f"보유량: {self.shares_held:.4f} | 포트폴리오: ${new_portfolio:.2f}")

    def _execute_buy(self, action_value: float, current_price: float) -> None:
        """💰 매수 실행 (로그 간소화)"""
        max_affordable = self.balance / (current_price * (1 + self.transaction_fee_percent))
        shares_to_buy = min(max_affordable, self.max_trading_units * action_value)
        
        if shares_to_buy > 0:
            shares_to_buy = float(Decimal(str(shares_to_buy)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
            
            buy_cost = shares_to_buy * current_price
            commission = buy_cost * self.transaction_fee_percent
            total_cost = buy_cost + commission
            
            if self.balance >= total_cost:
                # ✅ 매수 실행
                self.balance -= total_cost
                self.shares_held += shares_to_buy
                self.total_shares_purchased += shares_to_buy
                self.total_commission += commission
                
                self.trade_executed = True
                self.position = "매수"
                
                # 📈 평균 매수 단가 업데이트
                if self.shares_held > 0:
                    self.cost_basis = ((self.cost_basis * (self.shares_held - shares_to_buy)) + buy_cost) / self.shares_held
                
                if self.debug_level == 'DEBUG':
                    LOGGER.debug(f"매수성공: {shares_to_buy:.4f}주 @ ${current_price:.2f} | 비용: ${total_cost:.2f}")
            else:
                if self.debug_level == 'DEBUG':
                    LOGGER.debug(f"매수실패: 잔고부족 (필요: ${total_cost:.2f}, 보유: ${self.balance:.2f})")

    def _execute_sell(self, action_value: float, current_price: float) -> None:
        """💸 매도 실행 (로그 간소화)"""
        shares_to_sell = min(self.shares_held, self.max_trading_units * abs(action_value))
        
        if shares_to_sell > 0:
            shares_to_sell = float(Decimal(str(shares_to_sell)).quantize(Decimal('0.0001'), rounding=ROUND_DOWN))
            
            if shares_to_sell <= 0:
                return
            
            gross_sell_value = shares_to_sell * current_price
            commission = gross_sell_value * self.transaction_fee_percent
            net_value = gross_sell_value - commission
            
            # ✅ 매도 실행
            self.balance += net_value
            self.shares_held -= shares_to_sell
            self.total_shares_sold += shares_to_sell
            self.total_sales_value += gross_sell_value
            self.total_commission += commission
            
            self.trade_executed = True
            self.position = "매도"
            
            if self.debug_level == 'DEBUG':
                LOGGER.debug(f"매도성공: {shares_to_sell:.4f}주 @ ${current_price:.2f} | 수익: ${net_value:.2f}")
        else:
            if self.debug_level == 'DEBUG':
                LOGGER.debug("매도실패: 보유량 부족")

    def _get_portfolio_value(self) -> float:
        """
        💎 현재 포트폴리오 가치 계산 (로그 최적화)
        """
        current_price = self._get_current_price()
        stock_value = self.shares_held * current_price
        total_value = self.balance + stock_value
        
        # 🚨 포트폴리오 가치가 너무 작으면 최소 가치 보장
        if total_value <= 0:
            return max(self.balance, 1.0)
            
        return total_value

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        👀 현재 관측값 반환
        """
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
        
        # 📊 관측값 딕셔너리 생성
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio_state': portfolio_state
        }
        
        self.states_history.append(observation)
        return observation

    def _calculate_reward(self, prev_portfolio_value: float, portfolio_after_trade: float, 
                         current_portfolio_value: float) -> Union[float, Dict[str, float]]:
        """
        🎁 보상 계산 (로그 간소화)
        """
        # 🔍 안전한 이전 포트폴리오 값 설정
        if prev_portfolio_value <= 0:
            prev_portfolio_value = max(self.balance, 1.0)
        
        # 📊 각 효과 계산
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
        
        # 보상 모드에 따른 계산 클리핑
        if self.reward_mode == 'trade_only':
            reward = np.clip(trade_effect * 100, -2, 2)
        elif self.reward_mode == 'market_only':
            reward = np.clip(market_effect * 50, -5, 5)
        elif self.reward_mode == 'separated':
            trade_reward = np.clip(trade_effect * 100, -2, 2)
            market_reward = np.clip(market_effect * 50, -5, 5)
            reward = {
                'trade': trade_reward,
                'market': market_reward,
                'total': trade_reward + market_reward
            }
        else:  # 'combined'
            reward = np.clip(total_effect * 50, -5, 5)
        
        # 🚨 잘못된 매도 페널티
        if self.invalid_sell_penalty:
            penalty = -1.0
            if isinstance(reward, dict):
                reward['trade'] += penalty
                reward['total'] += penalty
            else:
                reward += penalty
            if self.debug_level == 'DEBUG':
                LOGGER.debug(f"잘못된 매도 페널티: {penalty}")
        
        # 📊 상세 로깅 (DEBUG 레벨에서만)
        if self.debug_level == 'DEBUG':
            if isinstance(reward, dict):
                LOGGER.debug(f"보상계산: 거래={reward['trade']:.6f}, 시장={reward['market']:.6f}, 총={reward['total']:.6f}")
            else:
                LOGGER.debug(f"보상계산 ({self.reward_mode}): {reward:.6f}")
        
        return reward

    def step(self, action: float) -> Tuple[Dict[str, np.ndarray], Union[float, Dict[str, float]], bool, Dict[str, Any]]:
        """
        🚀 환경에서 한 스텝 진행 (로그 대폭 간소화)
        """
        # 🔄 초기화
        self.invalid_sell_penalty = False
        self.actions_history.append(action)
        
        # 📊 1. 거래 전 상태 저장
        current_price_before = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value()
        
        # 🔄 2. 거래 실행
        self._execute_trade_action(action)
        
        # 💰 3. 거래 후 포트폴리오 가치 계산
        portfolio_after_trade = self.balance + self.shares_held * current_price_before
        
        # ⏰ 4. 다음 스텝으로 이동
        self.current_step += 1
        
        # 🚨 5. 조기 종료 조건 확인
        if self.current_step >= len(self.raw_data):
            if self.debug_level == 'DEBUG':
                LOGGER.warning(f"데이터 범위 초과로 에피소드 조기 종료: step {self.current_step}")
            done = True
            current_portfolio_value = self.balance + self.shares_held * current_price_before
            current_price_after = current_price_before
        else:
            # 📈 6. 시장 변동 후 포트폴리오 가치
            current_price_after = self._get_current_price()
            current_portfolio_value = self._get_portfolio_value()
            
            # ✅ 7. 기존 종료 조건 확인
            done = self.current_step >= self.data_length - 1
        
        # 🎁 8. 보상 계산
        reward = self._calculate_reward(prev_portfolio_value, portfolio_after_trade, current_portfolio_value)
        
        # 📈 9. 히스토리 업데이트
        self.portfolio_values_history.append(current_portfolio_value)
        
        if isinstance(reward, dict):
            self.rewards_history.append(reward['total'])
            self.trade_effects_history.append(reward['trade'])
            self.market_effects_history.append(reward['market'])
        else:
            self.rewards_history.append(reward)
            # 간단한 효과 계산해서 저장
            trade_effect = (portfolio_after_trade - prev_portfolio_value) / prev_portfolio_value if prev_portfolio_value > 0 else 0
            market_effect = (current_portfolio_value - portfolio_after_trade) / portfolio_after_trade if portfolio_after_trade > 0 else 0
            self.trade_effects_history.append(trade_effect)
            self.market_effects_history.append(market_effect)
        
        # 👀 10. 관측값 및 추가 정보
        observation = self._get_observation()
        info = self._get_info()
        
        # 📊 시간 정보를 info에 추가
        if self.current_step < len(self.raw_data):
            current_timestamp = self.raw_data.index[self.current_step]
        else:
            current_timestamp = "범위초과"
            
        info.update({
            'reward_mode': self.reward_mode,
            'current_timestamp': current_timestamp,
            'price_before': current_price_before,
            'price_after': current_price_after,
            'portfolio_before': prev_portfolio_value,
            'portfolio_after_trade': portfolio_after_trade,
            'portfolio_after_market': current_portfolio_value
        })
        
        # 간단한 진행 로그 (주요 마일스톤에서만)
        if self.debug_level == 'INFO' and (self.current_step % 1000 == 0 or done):
            LOGGER.info(f"Step {self.current_step}: 포트폴리오=${current_portfolio_value:.2f}, "
                       f"보상={reward if not isinstance(reward, dict) else reward['total']:.4f}, "
                       f"거래={self.position}")
        
        return observation, reward, done, info

    def _get_info(self) -> Dict[str, Any]:
        """
        📊 추가 정보 반환
        """
        current_price = self._get_current_price()
        portfolio_value = self._get_portfolio_value()
        
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance) if self.initial_balance > 0 else 0
        
        # 📊 포지션 결정
        if self.trade_executed:
            position = self.position
        else:
            if self.shares_held < self.previous_shares_held:
                position = "매도"
            elif self.shares_held > self.previous_shares_held:
                position = "매수"
            else:
                position = "홀드"
        
        self.previous_shares_held = self.shares_held
        
        # ⏰ 현재 타임스탬프 정보 가져오기
        current_timestamp = None
        if self.current_step < len(self.timestamps):
            current_timestamp = self.timestamps[self.current_step]
        
        timestamps_info = None
        if self.current_step < len(self.raw_data):
            timestamps_info = self.raw_data.index[self.current_step]
        
        return {
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
        }
    
    def render(self, mode: str = 'human') -> None:
        """
        🖥️ 환경 시각화 (로그 간소화)
        """
        if self.debug_level != 'DEBUG':
            return  # DEBUG 레벨에서만 출력
            
        info = self._get_info()
        
        LOGGER.debug(f"Step: {info['step']} | 잔고: ${info['balance']:.2f} | "
                    f"보유량: {info['shares_held']:.4f} | 현재가: ${info['current_price']:.2f} | "
                    f"포트폴리오: ${info['portfolio_value']:.2f} | 수익률: {info['total_return'] * 100:.2f}%")
    
    def get_episode_data(self) -> Dict[str, List]:
        """
        📈 에피소드 데이터 반환
        """
        return {
            'actions': self.actions_history,
            'rewards': self.rewards_history,
            'portfolio_values': self.portfolio_values_history
        }
    
    def get_final_portfolio_value(self) -> float:
        """
        💎 최종 포트폴리오 가치 반환
        """
        return self._get_portfolio_value()
    
    def get_total_reward(self) -> float:
        """
        🎁 총 보상 반환
        """
        return sum(self.rewards_history)
    
    def get_reward_analysis(self) -> Dict[str, Any]:
        """
        📊 보상 분석 결과 반환
        """
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

def create_environment_from_results(results: Dict[str, Dict[str, Any]], symbol: str, data_type: str = 'test', **kwargs) -> TradingEnvironment:
    """
    🏭 DataProcessor 결과로부터 환경 생성하는 헬퍼 함수
    
    Args:
        results: DataProcessor.process_all_symbols()의 결과
        symbol: 주식 심볼
        data_type: 'train', 'valid', 'test' 중 하나
        **kwargs: TradingEnvironment 추가 인자
        
    Returns:
        TradingEnvironment 인스턴스
    """
    if symbol not in results:
        raise ValueError(f"심볼 {symbol}을 결과에서 찾을 수 없습니다")
    
    result = results[symbol]
    
    # 📊 정규화된 데이터 (에이전트 관측용)
    if data_type not in result:
        raise ValueError(f"데이터 타입 {data_type}을 심볼 {symbol}에서 찾을 수 없습니다")
    
    normalized_data = result[data_type]
    
    # 📈 원본 데이터 (실제 거래용) - featured_data에서 해당 구간 추출
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
    
    # 기본 로그는 간소화
    LOGGER.info(f"환경생성: {symbol}-{data_type} | 정규화:{normalized_data.shape} | 원본:{raw_data.shape}")
    
    # ✅ 환경 생성 (기본값으로 detailed_logging=False)
    env = TradingEnvironment(
        data=normalized_data,
        raw_data=raw_data,
        symbol=symbol,
        train_data=(data_type == 'train'),
        detailed_logging=kwargs.get('detailed_logging', False),  # 기본값 False
        debug_level=kwargs.get('debug_level', 'INFO'),  # 기본값 INFO
        **{k: v for k, v in kwargs.items() if k not in ['detailed_logging', 'debug_level']}
    )
    return env    

class MultiAssetTradingEnvironment:
    """🏢 다중 자산 트레이딩 환경 클래스 (로그 최적화 버전)"""
    
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
        detailed_logging: bool = False,  # 기본값 False로 변경
        debug_level: str = 'INFO'
    ):
        """
        MultiAssetTradingEnvironment 클래스 초기화
        """
        self.symbols = symbols
        self.n_assets = len(symbols)
        self.data_type = data_type
        self.initial_balance = initial_balance
        
        LOGGER.info(f"다중자산환경 초기화: {self.n_assets}개 자산 | {data_type} 데이터")
        
        # 🏭 개별 환경 생성
        self.envs = {}
        for symbol in symbols:
            self.envs[symbol] = create_environment_from_results(
                results=results,
                symbol=symbol,
                data_type=data_type,
                window_size=window_size,
                initial_balance=initial_balance / self.n_assets,  # 자산별 균등 배분
                max_trading_units=max_trading_units,
                transaction_fee_percent=transaction_fee_percent,
                reward_mode=reward_mode,
                detailed_logging=detailed_logging,
                debug_level=debug_level
            )
        
        # 🎮 행동 공간: 각 자산에 대한 연속 행동
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_assets,), dtype=np.float32
        )
        
        # 👀 관측 공간
        self.observation_space = spaces.Dict({
            symbol: env.observation_space for symbol, env in self.envs.items()
        })
        
        LOGGER.info(f"다중자산환경 초기화 완료: {self.n_assets}개 자산")

    def reset(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        🔄 환경 초기화
        """
        LOGGER.info("다중자산환경 리셋")
        observations = {}
        for symbol, env in self.envs.items():
            observations[symbol] = env.reset()
        
        return observations
    
    def step(self, actions: Dict[str, float]) -> Tuple[Dict[str, Dict[str, np.ndarray]], float, bool, Dict[str, Any]]:
        """
        🚀 환경에서 한 스텝 진행 (로그 간소화)
        
        Args:
            actions: 심볼을 키로 하고 행동을 값으로 하는 딕셔너리
            
        Returns:
            (관측값, 보상, 종료 여부, 추가 정보) 튜플
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        # 📊 각 자산에 대한 행동 실행
        for symbol, env in self.envs.items():
            action = actions.get(symbol, 0.0)  # 행동이 없는 경우 홀드
            obs, rew, done, info = env.step(action)
            
            observations[symbol] = obs
            rewards[symbol] = rew
            dones[symbol] = done
            infos[symbol] = info
        
        # 🎁 전체 보상은 각 자산의 보상 평균
        total_reward = sum(rewards.values()) / self.n_assets
        
        # 🏁 모든 자산의 에피소드가 종료되면 전체 에피소드 종료
        done = all(dones.values())
        
        # 💎 전체 포트폴리오 가치 계산
        total_portfolio_value = sum(info['portfolio_value'] for info in infos.values())
        trade_executed_any = any(info.get('trade_executed', False) for info in infos.values())
        
        # 📊 추가 정보에 전체 포트폴리오 가치 포함
        infos['total'] = {
            'portfolio_value': total_portfolio_value,
            'total_return': (total_portfolio_value - self.initial_balance) / self.initial_balance,
            'trade_executed': trade_executed_any
        }

        return observations, total_reward, done, infos
    
    def render(self, mode: str = 'human') -> None:
        """
        🖥️ 환경 시각화 (로그 간소화)
        """
        total_portfolio_value = sum(env.get_final_portfolio_value() for env in self.envs.values())
        total_return = (total_portfolio_value - self.initial_balance) / self.initial_balance
        
        LOGGER.info(f"다중자산상태: 총포트폴리오=${total_portfolio_value:.2f} | 수익률={total_return*100:.2f}%")
    
    def get_final_portfolio_value(self) -> float:
        """
        💎 최종 포트폴리오 가치 반환
        """
        return sum(env.get_final_portfolio_value() for env in self.envs.values())
    
    def get_total_reward(self) -> float:
        """
        🎁 총 보상 반환
        """
        return sum(env.get_total_reward() for env in self.envs.values()) / self.n_assets


def create_flexible_environment(results, symbol, reward_mode='combined', detailed_logging=False, data_type='train', debug_level='INFO'):
    """
    🔧 유연한 보상 시스템을 가진 환경 생성 (로그 최적화)
    
    Args:
        results: DataProcessor 결과
        symbol: 주식 심볼
        reward_mode: 'combined', 'trade_only', 'market_only', 'separated'
        detailed_logging: 상세 로깅 여부 (기본값 False로 변경)
        data_type: 'train', 'valid', 'test'
        debug_level: 'INFO', 'DEBUG'
    """
    LOGGER.info(f"유연환경생성: {symbol} | 모드:{reward_mode} | 레벨:{debug_level}")
    env = create_environment_from_results(
        results=results,
        symbol=symbol,
        data_type=data_type,
        reward_mode=reward_mode,
        detailed_logging=detailed_logging,
        debug_level=debug_level
    )
    return env


if __name__ == "__main__":
    # 🧪 모듈 테스트 코드 (로그 간소화)
    import matplotlib.pyplot as plt
    from src.data_collection.data_collector import DataCollector
    from src.preprocessing.data_processor import DataProcessor
    from src.config.config import config
    import argparse
    
    parser = argparse.ArgumentParser(description="테스트할 심볼")
    parser.add_argument("--symbols", nargs="+", help="테스트할 심볼 리스트", default=config.trading_symbols)
    parser.add_argument("--debug", action="store_true", help="디버그 모드 활성화")
    args = parser.parse_args()
    
    # 📊 심볼 리스트 할당
    symbols = args.symbols
    TARGET_SYMBOL = symbols[0] if symbols else config.trading_symbols
    debug_level = 'DEBUG' if args.debug else 'INFO'
    
    LOGGER.info(f'테스트 시작: {TARGET_SYMBOL} | 디버그레벨: {debug_level}')
    
    # 📊 데이터 수집 및 전처리
    collector = DataCollector(symbols=symbols)
    data = collector.load_all_data()
    
    try:
        if data:
            processor = DataProcessor()
            results = processor.process_all_symbols(data)
            
            if TARGET_SYMBOL in results:
                LOGGER.info(f"최적화된 TradingEnvironment 테스트 시작: {TARGET_SYMBOL}")
                
                # 🎯 간단한 테스트 (로그 최소화)
                env = create_flexible_environment(
                    results=results,
                    symbol=TARGET_SYMBOL,
                    reward_mode='combined',
                    detailed_logging=False,
                    data_type='train',
                    debug_level=debug_level
                )
                
                LOGGER.info(f"환경 생성 완료: {env.symbol}")
                
                # 🧪 5스텝만 간단히 테스트
                obs = env.reset()
                total_reward = 0
                
                for i in range(5):
                    action = np.random.uniform(-1.0, 1.0)
                    obs, reward, done, info = env.step(action)
                    
                    reward_val = reward if not isinstance(reward, dict) else reward['total']
                    total_reward += reward_val
                    
                    if debug_level == 'DEBUG':
                        LOGGER.debug(f"Step {i+1}: 행동={action:.4f}, 보상={reward_val:.6f}")
                    
                    if done:
                        break
                
                LOGGER.info(f"테스트 완료: 총보상={total_reward:.6f}")
                
            else:
                LOGGER.error(f"{TARGET_SYMBOL} 데이터를 찾을 수 없습니다.")
        else:
            LOGGER.error("데이터를 로드할 수 없습니다.")
            
    except Exception as e:
        LOGGER.error(f"테스트 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()