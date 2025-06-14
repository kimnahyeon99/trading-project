import numpy as np
import pandas as pd
import time
import threading
import queue
import os
import json
import torch
import pytz
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple, Callable

from src.models.sac_agent import SACAgent
from src.preprocessing.data_processor import DataProcessor
from src.utils.logger import Logger
from src.config.config import Config
from src.trading.api_connector import APIConnector
from src.trading.order_manager import OrderManager
from src.trading.position_manager import PositionManager
from src.trading.risk_manager import RiskManager

class LiveTrader:
    """
    실시간 트레이딩 모듈: 학습된 SAC 에이전트를 사용하여 실시간 트레이딩을 수행합니다.
    백테스팅과 동일한 데이터 전처리 파이프라인을 사용합니다.
    장 마감 시 자동 종료 기능 추가.
    """
    
    def __init__(
        self,
        agent: SACAgent, 
        api_connector: APIConnector,
        config: Config,
        logger: Optional[Logger] = None,
        data_processor: Optional[DataProcessor] = None,
        risk_manager: Optional[RiskManager] = None,
    ):
        """
        LiveTrader 클래스 초기화
        
        Args:
            agent: 학습된 SAC 에이전트
            api_connector: API 커넥터 인스턴스
            config: 설정 객체
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
            data_processor: 데이터 전처리 인스턴스 (백테스팅과 동일한 것 사용)
            risk_manager: 리스크 관리자 인스턴스 (옵션)
        """
        self.agent = agent
        self.api = api_connector
        self.config = config
        self.logger = logger
        
        # 백테스팅과 동일한 데이터 전처리 파이프라인 사용
        self.data_processor = data_processor if data_processor else DataProcessor(
            window_size=config.window_size
        )
        
        # API 연결 확인
        if not self.api.connected:
            self.api.connect()
            
        # 계정 정보 조회
        self.account_info = self.api.get_account_info()
        
        # 주문 관리자 및 포지션 관리자 초기화
        self.order_manager = OrderManager(self.api, logger=self.logger)
        self.position_manager = PositionManager(self.api, logger=self.logger)
        
        # 리스크 관리자 설정
        if risk_manager is None:
            self.risk_manager = RiskManager(
                max_position_size=config.max_position_size,
                max_drawdown=config.max_drawdown,
                max_trade_amount=config.max_trade_amount,
                max_daily_loss=config.max_daily_loss,
                logger=self.logger
            )
        else:
            self.risk_manager = risk_manager
            
        # 실시간 데이터 스트림 설정
        self.data_queue = queue.Queue()
        self.state_dict = {}
        self.running = False
        self.data_thread = None
        self.trading_thread = None
        self.market_monitor_thread = None  # 장 시간 모니터링 스레드 추가
        
        # 트레이딩 설정
        self.trading_symbols = config.trading_symbols
        self.trading_interval = config.trading_interval
        self.window_size = config.window_size
        
        # 각 심볼별 원본 데이터 저장 (특성 추출용)
        self.raw_data_buffer = {symbol: pd.DataFrame() for symbol in self.trading_symbols}
        
        # 트레이딩 통계
        self.trading_stats = {
            "start_time": None,
            "trades": [],
            "successful_trades": 0,
            "failed_trades": 0,
            "initial_balance": 0,
            "current_balance": 0,
            "pnl": 0,
            "daily_start_balance": 0,  # 일일 시작 잔고 추가
            "session_start_positions": {}  # 세션 시작 시 포지션 정보
        }
        
        # 한국 시간대 설정
        self.korea_tz = pytz.timezone('Asia/Seoul')
        
        # 장 마감 자동 종료 설정 (기본값)
        self.auto_stop_on_market_close = True
        self.market_close_time_kst = "05:00"  # 한국시간 새벽 5시
        self.market_close_checked = False  # 당일 장 마감 체크 완료 여부
        
        if self.logger:
            self.logger.info("LiveTrader 초기화 완료")
            self.logger.info(f"🕐 장 마감 자동 종료 설정: {self.market_close_time_kst} KST")
    
    def _safe_dict_get(self, data, key, default=None):
        """안전한 딕셔너리 접근 - 타입 체크 포함"""
        if isinstance(data, dict):
            return data.get(key, default)
        else:
            if self.logger:
                self.logger.warning(f"Expected dict but got {type(data)} for key '{key}': {data}")
            return default

    def _safe_float(self, value, default=0.0):
        """안전한 float 변환"""
        try:
            if value is None:
                return default
            return float(value)
        except (ValueError, TypeError):
            if self.logger:
                self.logger.warning(f"Cannot convert to float: {value}, using default: {default}")
            return default
    
    def start(self) -> bool:
        """실시간 트레이딩 시작"""
        if self.running:
            if self.logger:
                self.logger.warning("이미 실시간 트레이딩이 실행 중입니다.")
            return False
            
        try:
            # API 연결 확인
            if not self.api.connected:
                self.api.connect()
                
            if not self.api.connected:
                if self.logger:
                    self.logger.error("API 서버에 연결할 수 없어 트레이딩을 시작할 수 없습니다.")
                return False
                
            # 초기 계정 정보 업데이트
            self.account_info = self.api.get_account_info()
            
            # 포지션 및 주문 정보 로드
            self.position_manager.update_all_positions()
            self.order_manager.update_open_orders()
            
            # 초기 데이터 로드
            self._load_initial_data()
            
            # 트레이딩 통계 초기화
            current_time = datetime.now()
            self.trading_stats["start_time"] = current_time
            self.trading_stats["initial_balance"] = self._safe_float(self._safe_dict_get(self.account_info, "cash", 0))
            self.trading_stats["current_balance"] = self.trading_stats["initial_balance"]
            self.trading_stats["daily_start_balance"] = self.trading_stats["initial_balance"]
            
            # 세션 시작 시 포지션 정보 저장 (포트폴리오 변화율 계산용)
            self.trading_stats["session_start_positions"] = self.position_manager.get_all_positions()
            
            # 장 마감 체크 초기화
            self.market_close_checked = False
            
            # 스레드 시작
            self.running = True
            self.data_thread = threading.Thread(target=self._data_stream_worker)
            self.data_thread.daemon = True
            self.data_thread.start()
            
            self.trading_thread = threading.Thread(target=self._trading_worker)
            self.trading_thread.daemon = True
            self.trading_thread.start()
            
            # 장 시간 모니터링 스레드 시작 (자동 종료가 활성화된 경우만)
            if self.auto_stop_on_market_close:
                self.market_monitor_thread = threading.Thread(target=self._market_time_monitor)
                self.market_monitor_thread.daemon = True
                self.market_monitor_thread.start()
            
            if self.logger:
                self.logger.info("실시간 트레이딩을 시작합니다.")
                self.logger.info(f"트레이딩 심볼: {', '.join(self.trading_symbols)}")
                self.logger.info(f"트레이딩 간격: {self.trading_interval}초")
                if self.auto_stop_on_market_close:
                    self.logger.info(f"🕐 장 마감 자동 종료: {self.market_close_time_kst} KST")
                else:
                    self.logger.info("🕐 장 마감 자동 종료: 비활성화")
                
            return True
            
        except Exception as e:
            self.running = False
            if self.logger:
                self.logger.error(f"실시간 트레이딩 시작 중 오류 발생: {e}")
            return False
    
    def stop(self, reason: str = "수동 종료") -> bool:
        """실시간 트레이딩 중지"""
        if not self.running:
            if self.logger:
                self.logger.warning("실시간 트레이딩이 실행 중이 아닙니다.")
            return False
            
        try:
            # 종료 플래그 설정
            self.running = False
            
            if self.logger:
                self.logger.info(f"🛑 실시간 트레이딩 중지 중... ({reason})")
            
            # 스레드 종료 대기
            if self.data_thread and self.data_thread.is_alive():
                self.data_thread.join(timeout=5.0)
                
            if self.trading_thread and self.trading_thread.is_alive():
                self.trading_thread.join(timeout=5.0)
                
            if self.market_monitor_thread and self.market_monitor_thread.is_alive():
                self.market_monitor_thread.join(timeout=5.0)
            
            # 최종 계정 정보 업데이트
            self.account_info = self.api.get_account_info()
            
            # 트레이딩 통계 업데이트
            self.trading_stats["current_balance"] = self._safe_float(self._safe_dict_get(self.account_info, "cash", 0))
            self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
            
            # 최종 포트폴리오 성과 출력
            self._print_final_portfolio_performance(reason)
            
            if self.logger:
                self.logger.info(f"✅ 실시간 트레이딩이 중지되었습니다. ({reason})")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"실시간 트레이딩 중지 중 오류 발생: {e}")
            return False
    
    def _market_time_monitor(self) -> None:
        """장 시간 모니터링 워커 스레드 - 장 마감 시 자동 종료"""
        while self.running:
            try:
                # 현재 한국 시간
                korea_now = datetime.now(self.korea_tz)
                current_time_str = korea_now.strftime("%H:%M")
                
                # 날짜가 바뀌면 장 마감 체크 초기화
                if korea_now.hour == 0 and korea_now.minute == 0:
                    self.market_close_checked = False
                    # 새로운 거래일 시작 - 일일 시작 잔고 업데이트
                    self.account_info = self.api.get_account_info()
                    self.trading_stats["daily_start_balance"] = self._safe_float(self._safe_dict_get(self.account_info, "portfolio_value", 0))
                    
                    if self.logger:
                        self.logger.info(f"📅 새로운 거래일 시작: {korea_now.strftime('%Y-%m-%d')}")
                        self.logger.info(f"💰 일일 시작 포트폴리오: ${self.trading_stats['daily_start_balance']:,.2f}")
                
                # 장 마감 시간 체크 (한국시간 05:00)
                if (current_time_str == self.market_close_time_kst and 
                    not self.market_close_checked and 
                    self.auto_stop_on_market_close):
                    
                    self.market_close_checked = True
                    
                    if self.logger:
                        self.logger.info("🔔" + "=" * 60 + "🔔")
                        self.logger.info(f"🕐 장 마감 시간 도달: {korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST")
                        self.logger.info("🔔" + "=" * 60 + "🔔")
                    
                    # 장 마감 시 모델 저장 콜백 호출
                    if hasattr(self, 'on_market_close_callback') and self.on_market_close_callback:
                        try:
                            self.on_market_close_callback()
                            if self.logger:
                                self.logger.info("💾 장 마감 시 모델 저장 완료")
                        except Exception as e:
                            if self.logger:
                                self.logger.error(f"❌ 장 마감 시 모델 저장 실패: {e}")
                    
                    # 트레이딩 자동 종료
                    self.stop(reason="장 마감 자동 종료")
                    break
                
                # 1분마다 체크
                time.sleep(60)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"장 시간 모니터링 중 오류 발생: {e}")
                time.sleep(60)  # 오류 발생 시 1분 대기 후 재시도
    
    def _print_final_portfolio_performance(self, reason: str) -> None:
        """최종 포트폴리오 성과 출력"""
        try:
            if self.logger:
                self.logger.info("📊" + "=" * 80 + "📊")
                self.logger.info(f"🏁 TRADING SESSION ENDED - {reason}")
                self.logger.info("📊" + "=" * 80 + "📊")
                
                # 현재 계정 정보
                current_account = self.api.get_account_info()
                current_portfolio_value = self._safe_float(self._safe_dict_get(current_account, "portfolio_value", 0))
                current_cash = self._safe_float(self._safe_dict_get(current_account, "cash", 0))    
                current_equity = self._safe_float(self._safe_dict_get(current_account, "equity", current_portfolio_value))
                
                # 세션 시작 대비 성과
                session_start_balance = self.trading_stats["initial_balance"]
                session_pnl = current_portfolio_value - session_start_balance
                session_return_pct = (session_pnl / session_start_balance * 100) if session_start_balance > 0 else 0
                
                # 일일 성과 (당일 시작 대비)
                daily_start_balance = self.trading_stats.get("daily_start_balance", session_start_balance)
                daily_pnl = current_portfolio_value - daily_start_balance
                daily_return_pct = (daily_pnl / daily_start_balance * 100) if daily_start_balance > 0 else 0
                
                # 현재 포지션 정보
                current_positions = self.position_manager.get_all_positions()
                total_position_value = sum(abs(float(pos.get("market_value", 0))) for pos in current_positions)
                
                # 세션 정보
                start_time = self.trading_stats["start_time"]
                end_time = datetime.now()
                session_duration = end_time - start_time
                total_trades = len(self.trading_stats["trades"])
                successful_trades = self.trading_stats["successful_trades"]
                failed_trades = self.trading_stats["failed_trades"]
                
                # 성과 출력
                self.logger.info("💰 PORTFOLIO PERFORMANCE:")
                self.logger.info(f"   📈 Current Portfolio Value: ${current_portfolio_value:,.2f}")
                self.logger.info(f"   💵 Current Cash: ${current_cash:,.2f}")
                self.logger.info(f"   🏢 Current Positions Value: ${total_position_value:,.2f}")
                self.logger.info("")
                
                # 세션 성과
                session_emoji = "📈" if session_pnl >= 0 else "📉"
                self.logger.info("🔄 SESSION PERFORMANCE:")
                self.logger.info(f"   💰 Session Start: ${session_start_balance:,.2f}")
                self.logger.info(f"   {session_emoji} Session P&L: ${session_pnl:+,.2f} ({session_return_pct:+.2f}%)")
                self.logger.info("")
                
                # 일일 성과
                daily_emoji = "📈" if daily_pnl >= 0 else "📉"
                self.logger.info("📅 DAILY PERFORMANCE:")
                self.logger.info(f"   🌅 Daily Start: ${daily_start_balance:,.2f}")
                self.logger.info(f"   {daily_emoji} Daily P&L: ${daily_pnl:+,.2f} ({daily_return_pct:+.2f}%)")
                self.logger.info("")
                
                # 포지션별 성과
                if current_positions:
                    self.logger.info("🏢 FINAL POSITIONS:")
                    for pos in current_positions:
                        symbol = pos.get("symbol", "Unknown")
                        qty = float(pos.get("qty", 0))
                        if abs(qty) > 0.001:  # 의미있는 포지션만 표시
                            avg_cost = float(pos.get("avg_entry_price", 0))
                            current_price = float(pos.get("current_price", 0))
                            market_value = float(pos.get("market_value", 0))
                            unrealized_pl = float(pos.get("unrealized_pl", 0))
                            
                            pos_emoji = "📈" if unrealized_pl >= 0 else "📉"
                            self.logger.info(f"   {symbol}:")
                            self.logger.info(f"     Shares: {qty:+.4f}")
                            self.logger.info(f"     Avg Cost: ${avg_cost:.2f}")
                            self.logger.info(f"     Current: ${current_price:.2f}")
                            self.logger.info(f"     Market Value: ${market_value:,.2f}")
                            self.logger.info(f"     {pos_emoji} Unrealized P&L: ${unrealized_pl:+,.2f}")
                else:
                    self.logger.info("🏢 FINAL POSITIONS: No positions")
                
                self.logger.info("")
                
                # 트레이딩 통계
                success_rate = (successful_trades / total_trades * 100) if total_trades > 0 else 0
                self.logger.info("📊 TRADING STATISTICS:")
                self.logger.info(f"   🔢 Total Trades: {total_trades}")
                self.logger.info(f"   ✅ Successful: {successful_trades}")
                self.logger.info(f"   ❌ Failed: {failed_trades}")
                self.logger.info(f"   📈 Success Rate: {success_rate:.1f}%")
                self.logger.info(f"   ⏰ Session Duration: {str(session_duration).split('.')[0]}")
                
                # 모델 타입
                model_type = "CNN" if getattr(self.agent, 'use_cnn', False) else \
                            "LSTM" if getattr(self.agent, 'use_lstm', False) else "MLP"
                self.logger.info(f"   🧠 Model Type: {model_type}")
                
                # 종료 시간
                korea_now = datetime.now(self.korea_tz)
                self.logger.info(f"   🕐 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ({korea_now.strftime('%Y-%m-%d %H:%M:%S')} KST)")
                
                self.logger.info("📊" + "=" * 80 + "📊")
                
                # 성과 요약 한 줄
                if reason == "장 마감 자동 종료":
                    self.logger.info(f"🏁 Market Closed Auto-Stop | Daily: {daily_return_pct:+.2f}% | Session: {session_return_pct:+.2f}% | Trades: {total_trades}")
                else:
                    self.logger.info(f"🏁 Manual Stop | Daily: {daily_return_pct:+.2f}% | Session: {session_return_pct:+.2f}% | Trades: {total_trades}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ 최종 성과 출력 중 오류: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """현재 트레이딩 상태 조회"""
        # 계정 정보 업데이트
        self.account_info = self.api.get_account_info()
        
        # 포지션 업데이트
        positions = self.position_manager.update_all_positions()
        
        # 주문 업데이트
        open_orders = self.order_manager.update_open_orders()
        
        # 트레이딩 통계 업데이트
        self.trading_stats["current_balance"] = self._safe_float(self._safe_dict_get(self.account_info, "cash", 0))
        self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
        
        # 한국 시간 정보 추가
        korea_now = datetime.now(self.korea_tz)
        
        return {
            "running": self.running,
            "account": self.account_info,
            "positions": positions,
            "open_orders": open_orders,
            "trading_stats": self.trading_stats,
            "market_close_settings": {
                "auto_stop_enabled": self.auto_stop_on_market_close,
                "market_close_time_kst": self.market_close_time_kst,
                "current_korea_time": korea_now.strftime("%Y-%m-%d %H:%M:%S"),
                "market_close_checked": self.market_close_checked
            },
            "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def execute_trade(self, symbol: str, action: float) -> Dict[str, Any]:
        """
        트레이딩 행동 실행 (분할 매도 지원)
        """
        try:
            # 계정 및 포지션 정보 업데이트
            self.account_info = self.api.get_account_info()
            current_position = self.position_manager.get_position(symbol)
            
            # 현재 상태 가져오기
            if symbol not in self.state_dict:
                if self.logger:
                    self.logger.warning(f"{symbol}의 상태 정보가 없습니다.")
                return {"success": False, "error": "상태 정보 없음"}
            
            current_state = self.state_dict[symbol]
            
            # 사용 가능한 현금 확인
            available_cash = self._safe_float(self._safe_dict_get(self.account_info, "cash", 0))
            current_price = self._get_current_price(symbol)
            
            if current_price <= 0:
                if self.logger:
                    self.logger.error(f"{symbol}의 현재 가격을 얻을 수 없습니다.")
                return {"success": False, "error": "현재 가격을 얻을 수 없습니다."}
            
            # 현재 보유 수량 - 안전한 접근
            current_position_qty = self._safe_float(self._safe_dict_get(current_position, "qty", 0))
            
            # 🔥 백테스팅과 동일한 비율 기반 로직 적용
            # action 값을 목표 주식 비율로 해석 (0.0 = 전액 현금, 1.0 = 전액 주식)
            target_stock_ratio = max(0.0, min(1.0, action))  # [0, 1] 범위로 클램핑
            
            # 현재 주식 비율 계산
            stock_value = current_position_qty * current_price
            total_portfolio = available_cash + stock_value
            current_stock_ratio = stock_value / total_portfolio if total_portfolio > 0 else 0.0
            
            # 비율 차이 계산
            ratio_diff = target_stock_ratio - current_stock_ratio
            
            # 거래 임계값 (백테스팅과 동일)
            threshold = 0.001  # 0.1% 차이부터 거래
            
            if abs(ratio_diff) <= threshold:
                if self.logger:
                    self.logger.info(f"{symbol} 거래 건너뜀: 비율 차이 {abs(ratio_diff):.4f} < 임계값 {threshold}")
                return {
                    "success": True,
                    "action": "no_trade",
                    "reason": f"비율 차이 너무 작음 ({abs(ratio_diff):.4f})",
                    "symbol": symbol,
                    "target_ratio": target_stock_ratio,
                    "current_ratio": current_stock_ratio,
                    "price": current_price,
                    "fee": 0,
                    "quantity": 0  # 명시적으로 quantity를 0으로 설정
                }
            
            # 거래 방향 결정
            side = "buy" if ratio_diff > 0 else "sell"
            position_size = abs(ratio_diff)
            
            # ===== 비율 기반 거래 로직 =====
            if side == "buy":
                # 매수의 경우: 목표 비율에 맞춰 매수할 금액 계산
                target_stock_value = total_portfolio * target_stock_ratio
                current_stock_value = current_position_qty * current_price
                additional_stock_needed = target_stock_value - current_stock_value
                
                if additional_stock_needed <= 0:
                    quantity = 0
                else:
                    # 매수할 주식 수 계산
                    quantity = additional_stock_needed / current_price
                    
                    # 수수료 고려하여 필요한 현금 확인
                    required_cash = additional_stock_needed * 1.001  # 0.1% 수수료 고려
                    if available_cash < required_cash:
                        # 현금 부족 시 가능한 최대 매수
                        available_for_stocks = available_cash / 1.001
                        quantity = available_for_stocks / current_price
                
                if self.logger:
                    self.logger.info(f"{symbol} 매수 계산: 목표비율 {target_stock_ratio:.3f}, 현재비율 {current_stock_ratio:.3f}")
                    self.logger.info(f"   └─ 추가 필요 주식가치: ${additional_stock_needed:.2f}, 수량: {quantity:.4f}주")
                
                # 매수 수량이 0이면 최소 수량으로 시도
                if quantity <= 0:
                    # 최소 주문 수량 계산 (예: $10)
                    min_order_amount = 10.0
                    if available_cash >= min_order_amount:
                        quantity = min_order_amount / current_price
                        if self.logger:
                            self.logger.info(f"{symbol} 최소 주문 수량으로 매수 시도: {quantity:.4f}주")
                    else:
                        if self.logger:
                            self.logger.info(f"{symbol} 매수 건너뜀: 현금 부족 (필요: ${min_order_amount}, 보유: ${available_cash:.2f})")
                        return {
                            "success": True,
                            "action": "no_trade",
                            "reason": "현금 부족",
                            "symbol": symbol,
                            "side": side,
                            "quantity": 0,
                            "price": current_price,
                            "fee": 0
                        }
                
            else:  # 매도의 경우
                if current_position_qty <= 0:
                    # 보유 주식이 없으면 공매도이므로 거래 차단
                    if self.logger:
                        self.logger.info(f"{symbol} 공매도 시도 차단: 보유 주식 없음 (현재: {current_position_qty})")
                    return {
                        "success": True, 
                        "action": "no_trade", 
                        "reason": "공매도 차단 - 보유 주식 없음",
                        "symbol": symbol,
                        "target_ratio": target_stock_ratio,
                        "current_ratio": current_stock_ratio,
                        "price": current_price,
                        "fee": 0
                    }
                
                # 매도의 경우: 목표 비율에 맞춰 매도할 금액 계산
                target_stock_value = total_portfolio * target_stock_ratio
                current_stock_value = current_position_qty * current_price
                excess_stock_value = current_stock_value - target_stock_value
                
                if excess_stock_value <= 0:
                    quantity = 0
                else:
                    # 매도할 주식 수 계산
                    quantity = excess_stock_value / current_price
                    # 보유 수량을 초과하지 않도록 제한
                    quantity = min(quantity, current_position_qty)
                
                if self.logger:
                    self.logger.info(f"{symbol} 매도 계산: 목표비율 {target_stock_ratio:.3f}, 현재비율 {current_stock_ratio:.3f}")
                    self.logger.info(f"   └─ 초과 주식가치: ${excess_stock_value:.2f}, 수량: {quantity:.4f}주")
            
            # 거래 수량이 0이면 거래 실행하지 않음
            if quantity <= 0:
                if self.logger:
                    self.logger.info(f"{symbol} {side} 거래 건너뜀: 수량이 0 이하입니다.")
                return {
                    "success": True, 
                    "action": "no_trade", 
                    "reason": "수량이 0 이하입니다.",
                    "symbol": symbol,
                    "target_ratio": target_stock_ratio,
                    "current_ratio": current_stock_ratio,
                    "price": current_price,
                    "fee": 0,
                    "quantity": 0  # 명시적으로 quantity를 0으로 설정
                }
            
            # 시장가 주문 실행
            order_result = self.api.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            # 모델 타입 미리 확인 (한 번만 계산)
            model_type = "CNN" if getattr(self.agent, 'use_cnn', False) else \
                        "LSTM" if getattr(self.agent, 'use_lstm', False) else "MLP"
            
            # 거래 실행 결과 처리
            if order_result.get("success", True):
                # 거래 정보 생성
                trade_info = {
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "order_id": order_result.get("id", ""),
                    "status": "success",
                    "model_type": model_type
                }
                
                # 성공한 거래 통계 업데이트
                self.trading_stats["successful_trades"] += 1
                self.trading_stats["trades"].append(trade_info)
                
                # 거래 실행 로그 출력
                self._log_trade_execution(trade_info)
                
                # 포지션 정보 업데이트
                try:
                    position = self.position_manager.get_position(symbol)
                    if position:
                        if self.logger:
                            self.logger.info(f"{symbol} 포지션 정보 업데이트 완료")
                    else:
                        if self.logger:
                            self.logger.debug(f"{symbol} 포지션 없음 (청산 완료)")
                except Exception as e:
                    if self.logger:
                        self.logger.debug(f"{symbol} 포지션 조회 실패 (청산 완료): {str(e)}")
                
                return {
                    "success": True,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "target_ratio": target_stock_ratio,
                    "current_ratio": current_stock_ratio,
                    "order_id": order_result.get("id", ""),
                    "model_type": model_type,
                    "fee": order_result.get("fee", 0)
                }
            else:
                # 실패한 거래 처리
                self.trading_stats["failed_trades"] += 1
                if self.logger:
                    self.logger.error("=" * 60)
                    self.logger.error(f"❌ 트레이드 실패")
                    self.logger.error(f"📈 종목: {symbol}")
                    self.logger.error(f"🧭 방향: {side.upper()}")
                    self.logger.error(f"🔢 수량: {quantity:.4f} 주")
                    self.logger.error(f"💵 가격: ${current_price:.2f}")
                    self.logger.error(f"🚨 오류: {order_result.get('error', 'Unknown error')}")
                    self.logger.error("=" * 60)
                        
                return {
                    "success": False,
                    "symbol": symbol,
                    "side": side,
                    "quantity": quantity,
                    "price": current_price,
                    "error": order_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 거래 실행 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def save_trading_stats(self, filepath: str) -> bool:
        """트레이딩 통계 저장"""
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 통계 업데이트
            self.account_info = self.api.get_account_info()
            self.trading_stats["current_balance"] = self._safe_float(self._safe_dict_get(self.account_info, "cash", 0))
            self.trading_stats["pnl"] = self.trading_stats["current_balance"] - self.trading_stats["initial_balance"]
            
            # 저장할 데이터 준비
            save_data = {
                "trading_stats": self.trading_stats,
                "account_info": self.account_info,
                "positions": self.position_manager.get_all_positions(),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # JSON 파일로 저장
            with open(filepath, 'w') as f:
                json.dump(save_data, f, indent=4, default=str)
                
            if self.logger:
                self.logger.info(f"트레이딩 통계를 {filepath}에 저장했습니다.")
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 저장 중 오류 발생: {e}")
            return False
    
    def _data_stream_worker(self) -> None:
        """실시간 데이터 스트림 처리 워커 스레드"""
        while self.running:
            try:
                # 각 심볼별 최신 데이터 가져오기
                for symbol in self.trading_symbols:
                    # TradingView에서 최신 데이터 가져오기
                    market_data = self.api.get_market_data(
                        symbol=symbol,
                        limit=1  # 최신 데이터 1개만
                    )
                    
                    if market_data.empty:
                        if self.logger:
                            self.logger.warning(f"{symbol} 데이터를 가져올 수 없습니다.")
                        continue
                    
                    # 원본 데이터 버퍼 업데이트
                    if self.raw_data_buffer[symbol].empty:
                        # 처음이면 window_size만큼의 이전 데이터 로드
                        historic_data = self.api.get_market_data(
                            symbol=symbol,
                            limit=self.window_size + 50  # 특성 추출을 위해 여유분 확보
                        )
                        self.raw_data_buffer[symbol] = historic_data
                    else:
                        # 기존 데이터에 새 데이터 추가
                        self.raw_data_buffer[symbol] = pd.concat([
                            self.raw_data_buffer[symbol], 
                            market_data
                        ]).drop_duplicates().sort_index()
                        
                        # 메모리 절약을 위해 오래된 데이터 제거 (window_size + 200개만 유지)
                        if len(self.raw_data_buffer[symbol]) > self.window_size + 200:
                            self.raw_data_buffer[symbol] = self.raw_data_buffer[symbol].iloc[-(self.window_size + 200):]
                    
                    # 백테스팅과 동일한 전처리 파이프라인 적용
                    try:
                        # 1. 데이터 전처리 (결측치 처리, 이상치 제거)
                        processed_data = self.data_processor.preprocess_data(self.raw_data_buffer[symbol])
                        
                        # 2. 특성 추출 (기술적 지표 계산)
                        featured_data = self.data_processor.extract_features(processed_data)
                        
                        # 3. 특성 정규화 (학습 시와 동일한 스케일러 사용)
                        normalized_data = self.data_processor.normalize_features(
                            featured_data, 
                            symbol, 
                            is_training=False  # 기존 스케일러 사용
                        )
                        
                        # 4. 백테스팅과 동일한 형태의 상태 생성
                        state = self._create_trading_state(normalized_data, symbol)
                        
                        # 상태 업데이트
                        self.state_dict[symbol] = state
                        
                        # 데이터 큐에 추가
                        self.data_queue.put({
                            "symbol": symbol,
                            "state": state,
                            "timestamp": datetime.now()
                        })
                        
                        if self.logger:
                            self.logger.debug(f"{symbol} 데이터 처리 완료")
                            
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"{symbol} 데이터 전처리 중 오류 발생: {e}")
                        continue
                
                # 다음 데이터 수집까지 대기
                time.sleep(60)  # 1분 간격으로 데이터 수집
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"데이터 스트림 처리 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _trading_worker(self) -> None:
        """트레이딩 로직 실행 워커 스레드"""
        last_trade_time = {symbol: datetime.min for symbol in self.trading_symbols}
        
        while self.running:
            try:
                # 데이터 큐에서 메시지 가져오기
                try:
                    message = self.data_queue.get(timeout=1.0)
                    symbol = message["symbol"]
                    state = message["state"]
                    timestamp = message["timestamp"]
                except queue.Empty:
                    continue
                
                # 트레이딩 간격 확인
                current_time = datetime.now()
                time_since_last_trade = current_time - last_trade_time[symbol]
                
                if time_since_last_trade.total_seconds() < self.trading_interval:
                    continue  # 트레이딩 간격이 지나지 않았으면 건너뜀
                
                # 리스크 체크
                current_balance = self._safe_float(self._safe_dict_get(self.account_info, "portfolio_value", 0))
                risk_check_passed = True
                for symbol in self.trading_symbols:
                    try:
                        current_price = self._get_current_price(symbol)
                        if current_price > 0:
                            risk_check = self.risk_manager.check_risk_limits(
                                current_balance=current_balance,
                                symbol=symbol,
                                current_price=current_price
                            )
                            
                            if self.logger:
                                self.logger.info(f"📊 {symbol} 리스크 체크 결과:")
                                self.logger.info(f"  💰 현재가: ${current_price:.2f}")
                                self.logger.info(f"  📉 현재 낙폭: {risk_check.get('current_drawdown', 0):.2f}%")
                                self.logger.info(f"  ⚠️ 최대 낙폭 한도: {risk_check.get('max_drawdown_limit', 0):.2f}%")
                                self.logger.info(f"  ✅ 거래 허용: {risk_check.get('trade_allowed', True)}")
                            
                            if not risk_check.get('trade_allowed', True):
                                risk_check_passed = False
                                if self.logger:
                                    self.logger.warning(f"⚠️ {symbol} 리스크 한도 초과!")
                                    for warning in risk_check.get('warnings', []):
                                        self.logger.warning(f"   └─ {warning}")
                    except Exception as e:
                        if self.logger:
                            self.logger.error(f"❌ {symbol} 리스크 체크 실패: {e}")
                
                # 리스크 체크 실패 시 거래 중단
                if not risk_check_passed:
                    if self.logger:
                        self.logger.warning("⚠️ 리스크 한도 초과로 거래 중단")
                    continue
                
                # 에이전트로부터 행동 선택 (백테스팅과 동일한 형태의 상태 사용)
                action = self.agent.select_action(state, evaluate=True)
                
                if self.logger:
                    self.logger.debug(f"{symbol} 에이전트 행동: {action:.4f}, 현재 포지션: {self._safe_float(self._safe_dict_get(self.position_manager.get_position(symbol), 'qty', 0))}")
                
                # 행동 임계값 확인 (너무 작은 행동은 무시)
                if abs(action) < 0.02:  # 임계값을 0.05에서 0.02로 낮춤
                    if self.logger:
                        self.logger.debug(f"{symbol} 행동 임계값 미달: {action:.4f} (임계값: 0.02)")
                    continue
                
                # ✅ 거래 실행
                trade_result = self.execute_trade(symbol, action)
                
                # ✅ 거래 결과 안전한 검증
                if self._safe_dict_get(trade_result, "success", False):
                    last_trade_time[symbol] = current_time
                    
                    # 리스크 관리자에 거래 기록 (안전한 키 접근)
                    side = self._safe_dict_get(trade_result, "side")
                    quantity = self._safe_dict_get(trade_result, "quantity")
                    price = self._safe_dict_get(trade_result, "price")
                    
                    # ✅ no_trade 액션 제외 (실제 거래만 기록)
                    action_type = self._safe_dict_get(trade_result, "action")
                    if action_type != "no_trade" and all([side, quantity is not None, price is not None]):
                        try:
                            self.risk_manager.record_trade(
                                symbol=symbol,
                                side=side,
                                quantity=float(quantity),
                                price=float(price)
                            )
                        except (ValueError, TypeError) as e:
                            if self.logger:
                                self.logger.warning(f"{symbol} 리스크 관리자 거래 기록 실패: {e}")
                    elif action_type == "no_trade":
                        if self.logger:
                            reason = self._safe_dict_get(trade_result, "reason", "Unknown")
                            self.logger.debug(f"{symbol} 거래 건너뜀: {reason}")
                    else:
                        if self.logger:
                            self.logger.warning(f"{symbol} 거래 결과에서 필수 정보 누락: {trade_result}")
                else:
                    # 거래 실패 시 로깅
                    error_msg = self._safe_dict_get(trade_result, "error", "Unknown error")
                    if self.logger:
                        self.logger.warning(f"{symbol} 거래 실패: {error_msg}")
                                
                # 트레이딩 로직 주기 대기
                time.sleep(1)
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"트레이딩 로직 실행 중 오류 발생: {e}")
                time.sleep(5)  # 오류 발생 시 5초 대기 후 재시도
    
    def _load_initial_data(self) -> None:
        """초기 과거 데이터 로드"""
        try:
            for symbol in self.trading_symbols:
                # window_size + 여유분만큼의 과거 데이터 로드
                historic_data = self.api.get_market_data(
                    symbol=symbol,
                    limit=self.window_size + 50  # 특성 추출을 위해 여유분 확보
                )
                
                if historic_data.empty:
                    if self.logger:
                        self.logger.warning(f"{symbol} 초기 데이터를 로드할 수 없습니다.")
                    continue
                
                # 원본 데이터 버퍼 초기화
                self.raw_data_buffer[symbol] = historic_data
                
                # 백테스팅과 동일한 전처리 파이프라인 적용
                try:
                    processed_data = self.data_processor.preprocess_data(historic_data)
                    featured_data = self.data_processor.extract_features(processed_data)
                    normalized_data = self.data_processor.normalize_features(
                        featured_data, 
                        symbol, 
                        is_training=False
                    )
                    
                    # 상태 생성
                    state = self._create_trading_state(normalized_data, symbol)
                    self.state_dict[symbol] = state
                    
                    if self.logger:
                        self.logger.info(f"{symbol} 초기 데이터 로드 및 전처리 완료: {len(historic_data)} 행")
                        
                except Exception as e:
                    if self.logger:
                        self.logger.error(f"{symbol} 초기 데이터 전처리 중 오류: {e}")
                    
        except Exception as e:
            if self.logger:
                self.logger.error(f"초기 데이터 로드 중 오류 발생: {e}")
    
    def _create_trading_state(self, normalized_data: pd.DataFrame, symbol: str) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        백테스팅과 동일한 형태의 상태 생성 (LSTM/CNN 지원)
        
        Args:
            normalized_data: 정규화된 특성 데이터
            symbol: 심볼
            
        Returns:
            TradingEnvironment과 동일한 형태의 상태
        """
        try:
            # 최근 window_size 개의 데이터만 사용
            if len(normalized_data) < self.window_size:
                # 데이터가 부족한 경우 패딩
                padding_size = self.window_size - len(normalized_data)
                padding = pd.DataFrame(
                    np.zeros((padding_size, normalized_data.shape[1])),
                    columns=normalized_data.columns
                )
                market_data = pd.concat([padding, normalized_data], ignore_index=True)
            else:
                market_data = normalized_data.iloc[-self.window_size:]
            
            # market_data를 numpy 배열로 변환
            market_data_array = market_data.values.astype(np.float32)
            
            # 포트폴리오 상태 계산
            current_position = self.position_manager.get_position(symbol)
            account_info = self.api.get_account_info()
            
            cash = self._safe_float(self._safe_dict_get(account_info, "cash", 0))
            portfolio_value = self._safe_float(self._safe_dict_get(account_info, "portfolio_value", cash))
            stock_value = abs(float(current_position.get("market_value", 0)))
            
            # 포트폴리오 가치가 0이 아닌지 확인
            if portfolio_value <= 0:
                portfolio_value = max(cash, 1.0)
            
            portfolio_state = np.array([
                cash / portfolio_value,        # 현금 비율
                stock_value / portfolio_value  # 주식 비율
            ], dtype=np.float32)
            
            # 에이전트의 모델 타입에 따라 상태 형식 결정
            if hasattr(self.agent, 'use_cnn') and self.agent.use_cnn:
                # CNN 모델을 위한 2D 텐서 형태로 변환
                # (time_steps, features) -> (1, time_steps, features) for batch dimension
                market_tensor = torch.FloatTensor(market_data_array).unsqueeze(0)  # (1, window_size, n_features)
                portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0)  # (1, 2)
                
                return {
                    'market_data': market_tensor,
                    'portfolio_state': portfolio_tensor
                }
                
            elif hasattr(self.agent, 'use_lstm') and getattr(self.agent, 'use_lstm', False):
                # LSTM 모델을 위한 시퀀스 형태로 변환
                # (time_steps, features) -> (1, time_steps, features) for batch dimension
                market_tensor = torch.FloatTensor(market_data_array).unsqueeze(0)  # (1, window_size, n_features)
                portfolio_tensor = torch.FloatTensor(portfolio_state).unsqueeze(0)  # (1, 2)
                
                return {
                    'market_data': market_tensor,
                    'portfolio_state': portfolio_tensor
                }
            else:
                # MLP 모델을 위한 플래튼된 형태
                # (time_steps, features) -> (time_steps * features,) flattened
                market_flattened = market_data_array.flatten()
                
                return {
                    'market_data': market_flattened,
                    'portfolio_state': portfolio_state
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 상태 생성 중 오류: {e}")
            
            # 오류 발생 시 기본 상태 반환
            n_features = 40  # 대략적인 특성 수
            
            if hasattr(self.agent, 'use_cnn') and self.agent.use_cnn:
                return {
                    'market_data': torch.zeros((1, self.window_size, n_features), dtype=torch.float32),
                    'portfolio_state': torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                }
            elif hasattr(self.agent, 'use_lstm') and getattr(self.agent, 'use_lstm', False):
                return {
                    'market_data': torch.zeros((1, self.window_size, n_features), dtype=torch.float32),
                    'portfolio_state': torch.tensor([[1.0, 0.0]], dtype=torch.float32)
                }
            else:
                return {
                    'market_data': np.zeros((self.window_size * n_features,), dtype=np.float32),
                    'portfolio_state': np.array([1.0, 0.0], dtype=np.float32)
                }

    
    def _get_current_price(self, symbol: str) -> float:
        """현재 가격 조회"""
        try:
            # 최신 시장 데이터 조회
            market_data = self.api.get_market_data(
                symbol=symbol,
                limit=1
            )
            
            if market_data.empty:
                return 0.0
            
            # 종가 반환
            return float(market_data.iloc[-1]["close"])
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 현재 가격 조회 중 오류 발생: {e}")
            return 0.0
    
    
    def _log_trade_execution(self, trade_info):
        """거래 실행 시 깔끔한 로그"""
        try:
            symbol = self._safe_dict_get(trade_info, 'symbol', 'Unknown')
            side = self._safe_dict_get(trade_info, 'side', 'Unknown')
            quantity = self._safe_float(self._safe_dict_get(trade_info, 'quantity', 0))
            price = self._safe_float(self._safe_dict_get(trade_info, 'price', 0))
            amount = quantity * price
            
            # 거래 시간 (한국시간도 포함)
            trade_time = datetime.now().strftime('%H:%M:%S')
            korea_time = datetime.now(self.korea_tz).strftime('%H:%M:%S')
            
            # 거래 방향 이모지
            side_emoji = "🟢 BUY" if side.lower() == 'buy' else "🔴 SELL"
            model_type = self._safe_dict_get(trade_info, 'model_type', 'Unknown')
            
            if self.logger:
                self.logger.info("🔥" + "=" * 60 + "🔥")
                self.logger.info(f"⚡ TRADE EXECUTED at {trade_time} ({korea_time} KST) ({model_type} Model)")
                self.logger.info("🔥" + "=" * 60 + "🔥")
                self.logger.info(f"📈 Symbol: {symbol}")
                self.logger.info(f"{side_emoji}: {quantity:.4f} shares @ ${price:.2f}")
                self.logger.info(f"💰 Trade Amount: ${amount:,.2f}")
                
                # 거래 후 즉시 포트폴리오 상태 조회
                time.sleep(0.3)  # API 업데이트 대기
                try:
                    account_info = self.api.get_account_info()
                    
                    self.logger.info("📊 PORTFOLIO AFTER TRADE:")
                    cash = self._safe_float(self._safe_dict_get(account_info, 'cash', 0))
                    portfolio_value = self._safe_float(self._safe_dict_get(account_info, 'portfolio_value', 0))
                    
                    self.logger.info(f"💵 Cash: ${cash:,.2f}")
                    self.logger.info(f"📈 Portfolio Value: ${portfolio_value:,.2f}")
                    
                    # 포지션 정보
                    position = self.position_manager.get_position(symbol)
                    if position:
                        qty = self._safe_float(self._safe_dict_get(position, 'qty', 0))
                        
                        if abs(qty) > 0.001:
                            # avg_entry_price를 여러 키로 시도
                            avg_cost = (self._safe_float(self._safe_dict_get(position, 'avg_entry_price', 0)) or
                                    self._safe_float(self._safe_dict_get(position, 'avg_cost', 0)) or
                                    self._safe_float(self._safe_dict_get(position, 'average_entry_price', 0)))
                            
                            market_value = self._safe_float(self._safe_dict_get(position, 'market_value', 0))
                            unrealized_pl = (self._safe_float(self._safe_dict_get(position, 'unrealized_pnl', 0)) or
                                            self._safe_float(self._safe_dict_get(position, 'unrealized_pl', 0)))
                            
                            self.logger.info(f"🏢 {symbol} Position:")
                            self.logger.info(f"   Shares: {qty:+.4f}")
                            self.logger.info(f"   Avg Cost: ${avg_cost:.2f}")
                            self.logger.info(f"   Market Value: ${market_value:,.2f}")
                            
                            pnl_emoji = "📈" if unrealized_pl >= 0 else "📉"
                            self.logger.info(f"   {pnl_emoji} Unrealized P&L: ${unrealized_pl:+,.2f}")
                    
                    # 총 수익률 (세션 기준)
                    initial_balance = self._safe_float(self._safe_dict_get(self.trading_stats, 'initial_balance', 0))
                    if initial_balance > 0:
                        total_return = ((portfolio_value - initial_balance) / initial_balance) * 100
                        pnl_amount = portfolio_value - initial_balance
                        
                        return_emoji = "📈" if total_return >= 0 else "📉"
                        self.logger.info(f"{return_emoji} Session Return: {total_return:+.2f}% (${pnl_amount:+,.2f})")
                    
                    # 일일 수익률
                    daily_start_balance = self._safe_float(self._safe_dict_get(self.trading_stats, "daily_start_balance", initial_balance))
                    if daily_start_balance > 0:
                        daily_return = ((portfolio_value - daily_start_balance) / daily_start_balance) * 100
                        daily_pnl = portfolio_value - daily_start_balance
                        
                        daily_emoji = "📈" if daily_return >= 0 else "📉"
                        self.logger.info(f"{daily_emoji} Daily Return: {daily_return:+.2f}% (${daily_pnl:+,.2f})")
                    
                    # 거래 통계
                    total_trades = len(self._safe_dict_get(self.trading_stats, 'trades', []))
                    successful_trades = self._safe_dict_get(self.trading_stats, 'successful_trades', 0)
                    failed_trades = self._safe_dict_get(self.trading_stats, 'failed_trades', 0)
                    
                    self.logger.info(f"🔢 Session Stats: {total_trades} trades (✅{successful_trades} ❌{failed_trades})")
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Portfolio update failed: {e}")
                
                self.logger.info("🔥" + "=" * 60 + "🔥")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Trade logging failed: {e}")