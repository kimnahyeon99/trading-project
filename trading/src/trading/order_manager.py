import pandas as pd
from typing import Dict, List, Any, Optional
import time

from src.utils.logger import Logger
from src.trading.api_connector import APIConnector


class OrderManager:
    """
    주문 관리를 담당하는 매니저 클래스
    """
    
    def __init__(self, api_connector: APIConnector, logger: Optional[Logger] = None):
        """
        OrderManager 클래스 초기화
        
        Args:
            api_connector: API 커넥터 인스턴스
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.api = api_connector
        self.logger = logger
        self.open_orders = {}  # 주문 ID별 미체결 주문 캐시
        
        # 초기 미체결 주문 정보 로드
        self.update_open_orders()
        
    def update_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        미체결 주문 정보 업데이트
        
        Args:
            symbol: 특정 심볼만 업데이트 (옵션)
            
        Returns:
            업데이트된 미체결 주문 정보 딕셔너리
        """
        try:
            # API에서 미체결 주문 정보 조회
            open_orders_list = self.api.get_open_orders(symbol)
            
            # 미체결 주문 정보 캐시 업데이트
            self.open_orders = {}
            for order in open_orders_list:
                order_id = order.get("order_id", "")
                if order_id:
                    self.open_orders[order_id] = order
            
            if self.logger:
                symbol_text = f" ({symbol})" if symbol else ""
                self.logger.info(f"미체결 주문 정보 업데이트 완료{symbol_text}: {len(self.open_orders)}개")
                
            return self.open_orders
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"미체결 주문 정보 업데이트 중 오류 발생: {e}")
            return self.open_orders
    
    def update_order(self, order_id: str) -> Dict[str, Any]:
        """
        특정 주문의 상태 업데이트
        
        Args:
            order_id: 주문 ID
            
        Returns:
            업데이트된 주문 정보
        """
        try:
            # API에서 주문 상태 조회
            order_status = self.api.get_order_status(order_id)
            
            # 주문 상태 캐시 업데이트
            if order_status:
                # 주문이 체결 완료 또는 취소됐으면 캐시에서 제거
                if order_status.get("status") in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                    if order_id in self.open_orders:
                        del self.open_orders[order_id]
                else:
                    # 아직 체결 중이면 캐시 업데이트
                    self.open_orders[order_id] = order_status
            
            if self.logger:
                self.logger.info(f"주문 {order_id} 상태 업데이트 완료: {order_status.get('status', 'Unknown')}")
                
            return order_status
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"주문 {order_id} 상태 업데이트 중 오류 발생: {e}")
            
            # 캐시에서 조회하거나 빈 딕셔너리 반환
            return self.open_orders.get(order_id, {})
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        특정 주문 정보 조회
        
        Args:
            order_id: 주문 ID
            
        Returns:
            주문 정보
        """
        # 캐시에서 조회
        if order_id in self.open_orders:
            return self.open_orders[order_id]
        
        # 없으면 API에서 조회
        return self.update_order(order_id)
    
    def get_open_orders_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """
        특정 심볼의 미체결 주문 목록 조회
        
        Args:
            symbol: 심볼/티커
            
        Returns:
            해당 심볼의 미체결 주문 목록
        """
        return [order for order in self.open_orders.values() if order.get("symbol") == symbol]
    
    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        특정 주문 취소
        
        Args:
            order_id: 주문 ID
            
        Returns:
            취소 결과 정보
        """
        try:
            # 주문 취소 전에 주문 정보 확인
            order_info = self.get_order(order_id)
            
            if not order_info:
                if self.logger:
                    self.logger.warning(f"주문 {order_id}이(가) 존재하지 않아 취소할 수 없습니다.")
                return {"success": False, "message": "주문이 존재하지 않습니다."}
            
            # 이미 체결된 주문인지 확인
            if order_info.get("status") in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                if self.logger:
                    self.logger.warning(f"주문 {order_id}은(는) 이미 {order_info.get('status')} 상태입니다.")
                return {"success": False, "message": f"주문이 이미 {order_info.get('status')} 상태입니다."}
            
            # API를 통한 주문 취소
            cancel_result = self.api.cancel_order(order_id)
            
            if cancel_result.get("success", False):
                # 주문 캐시에서 제거
                if order_id in self.open_orders:
                    del self.open_orders[order_id]
                
                if self.logger:
                    self.logger.info(f"주문 {order_id} 취소 성공")
                    
                return {
                    "success": True,
                    "order_id": order_id,
                    "symbol": order_info.get("symbol", ""),
                    "message": "주문이 성공적으로 취소되었습니다."
                }
            else:
                if self.logger:
                    self.logger.error(f"주문 {order_id} 취소 실패: {cancel_result.get('error', 'Unknown error')}")
                    
                return {
                    "success": False,
                    "order_id": order_id,
                    "error": cancel_result.get("error", "Unknown error")
                }
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"주문 {order_id} 취소 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        모든 미체결 주문 취소
        
        Args:
            symbol: 특정 심볼만 취소 (옵션)
            
        Returns:
            취소 결과 정보
        """
        results = {
            "success": True,
            "canceled_orders": [],
            "failed_orders": []
        }
        
        # 미체결 주문 정보 업데이트
        self.update_open_orders(symbol)
        
        # 취소할 주문 목록 준비
        orders_to_cancel = self.open_orders.values()
        if symbol:
            orders_to_cancel = [order for order in orders_to_cancel if order.get("symbol") == symbol]
        
        # 각 주문 취소
        for order in orders_to_cancel:
            order_id = order.get("order_id", "")
            if not order_id:
                continue
                
            result = self.cancel_order(order_id)
            
            if result.get("success", False):
                results["canceled_orders"].append({
                    "order_id": order_id,
                    "symbol": order.get("symbol", ""),
                    "side": order.get("side", ""),
                    "type": order.get("type", ""),
                    "price": order.get("price", 0),
                    "quantity": order.get("quantity", 0)
                })
            else:
                results["failed_orders"].append({
                    "order_id": order_id,
                    "symbol": order.get("symbol", ""),
                    "error": result.get("error", "Unknown error")
                })
                results["success"] = False
        
        symbol_text = f" ({symbol})" if symbol else ""
        if self.logger:
            self.logger.info(f"미체결 주문 전체 취소 결과{symbol_text}: 성공 {len(results['canceled_orders'])}개, 실패 {len(results['failed_orders'])}개")
            
        return results
    
    def place_market_order(self, symbol: str, side: str, quantity: float) -> Dict[str, Any]:
        """
        시장가 주문 실행
        
        Args:
            symbol: 심볼/티커
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            
        Returns:
            주문 결과 정보
        """
        try:
            # API를 통한 시장가 주문 실행
            order_result = self.api.place_market_order(
                symbol=symbol,
                side=side,
                quantity=quantity
            )
            
            if order_result.get("success", True):
                # 주문 ID 추출
                order_id = order_result.get("order_id", "")
                
                if order_id:
                    # 일부 거래소는 시장가 주문이 즉시 체결될 수 있어
                    # 미체결 주문 목록에 추가하지 않을 수 있음
                    time.sleep(1)  # 주문 처리 대기
                    
                    # 주문 상태 업데이트
                    self.update_order(order_id)
                    
                if self.logger:
                    self.logger.info(f"{symbol} {side.upper()} 시장가 주문 성공: {quantity} 수량")
                    
                return order_result
            else:
                if self.logger:
                    self.logger.error(f"{symbol} {side.upper()} 시장가 주문 실패: {order_result.get('error', 'Unknown error')}")
                return order_result
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} {side.upper()} 시장가 주문 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def place_limit_order(self, symbol: str, side: str, quantity: float, price: float) -> Dict[str, Any]:
        """
        지정가 주문 실행
        
        Args:
            symbol: 심볼/티커
            side: 매수/매도 ('buy' 또는 'sell')
            quantity: 수량
            price: 가격
            
        Returns:
            주문 결과 정보
        """
        try:
            # API를 통한 지정가 주문 실행
            order_result = self.api.place_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=price
            )
            
            if order_result.get("success", True):
                # 주문 ID 추출
                order_id = order_result.get("order_id", "")
                
                if order_id:
                    # 주문 상태 업데이트
                    time.sleep(1)  # 주문 처리 대기
                    order_status = self.update_order(order_id)
                    
                    # 주문을 캐시에 추가
                    self.open_orders[order_id] = order_status
                    
                if self.logger:
                    self.logger.info(f"{symbol} {side.upper()} 지정가 주문 성공: {quantity} 수량, 가격: {price}")
                    
                return order_result
            else:
                if self.logger:
                    self.logger.error(f"{symbol} {side.upper()} 지정가 주문 실패: {order_result.get('error', 'Unknown error')}")
                return order_result
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} {side.upper()} 지정가 주문 중 오류 발생: {e}")
            return {"success": False, "error": str(e)}
    
    def wait_for_order_fill(self, order_id: str, timeout: int = 60, check_interval: int = 5) -> Dict[str, Any]:
        """
        주문 체결을 기다림
        
        Args:
            order_id: 주문 ID
            timeout: 최대 대기 시간(초)
            check_interval: 상태 확인 간격(초)
            
        Returns:
            최종 주문 상태 정보
        """
        try:
            start_time = time.time()
            last_status = None
            
            while time.time() - start_time < timeout:
                # 주문 상태 조회
                order_status = self.update_order(order_id)
                status = order_status.get("status", "")
                
                # 상태가 변경되면 로그 기록
                if status != last_status:
                    if self.logger:
                        self.logger.info(f"주문 {order_id} 상태: {status}")
                    last_status = status
                
                # 체결 완료 또는 취소/거부된 경우
                if status in ["FILLED", "CANCELED", "REJECTED", "EXPIRED"]:
                    return order_status
                
                # 일정 시간 대기
                time.sleep(check_interval)
            
            # 타임아웃 발생
            if self.logger:
                self.logger.warning(f"주문 {order_id} 체결 대기 타임아웃: {timeout}초")
                
            # 마지막 상태 반환
            return self.update_order(order_id)
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"주문 {order_id} 체결 대기 중 오류 발생: {e}")
            return {"error": str(e)} 