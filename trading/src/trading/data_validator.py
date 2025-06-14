"""
실시간 데이터 검증 시스템
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging

class RealTimeDataValidator:
    """
    실시간으로 들어오는 데이터의 품질을 검증하는 클래스
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.previous_data = {}  # 심볼별 이전 데이터 저장
        self.anomaly_counts = {}  # 심볼별 이상 데이터 카운트
        
    def validate_market_data(self, symbol: str, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        시장 데이터 검증
        
        Args:
            symbol: 심볼
            data: 시장 데이터
            
        Returns:
            (검증 통과 여부, 경고 메시지 리스트)
        """
        warnings = []
        
        try:
            # 1. 기본 데이터 구조 검증
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                warnings.append(f"필수 컬럼 누락: {missing_columns}")
                return False, warnings
            
            # 2. 데이터 타입 검증
            for col in required_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    warnings.append(f"잘못된 데이터 타입: {col}")
                    return False, warnings
            
            # 3. 결측치 검증
            null_counts = data[required_columns].isnull().sum()
            if null_counts.any():
                warnings.append(f"결측치 발견: {null_counts.to_dict()}")
            
            # 4. 가격 논리 검증
            latest_row = data.iloc[-1]
            if not (latest_row['low'] <= latest_row['open'] <= latest_row['high'] and
                    latest_row['low'] <= latest_row['close'] <= latest_row['high']):
                warnings.append("가격 논리 오류: low <= open,close <= high 조건 위반")
                return False, warnings
            
            # 5. 이상치 검증
            price_change_threshold = 0.2  # 20% 이상 변동 시 이상치로 간주
            if symbol in self.previous_data:
                prev_close = self.previous_data[symbol]['close']
                current_close = latest_row['close']
                price_change = abs(current_close - prev_close) / prev_close
                
                if price_change > price_change_threshold:
                    warnings.append(f"급격한 가격 변동 감지: {price_change:.2%}")
            
            # 6. 거래량 검증
            if latest_row['volume'] < 0:
                warnings.append("음수 거래량 감지")
                return False, warnings
            
            # 이전 데이터 업데이트
            self.previous_data[symbol] = latest_row.to_dict()
            
            # 경고가 있지만 치명적이지 않은 경우
            is_valid = len([w for w in warnings if any(keyword in w for keyword in ['누락', '타입', '논리', '음수'])]) == 0
            
            if warnings and self.logger:
                self.logger.warning(f"{symbol} 데이터 검증 경고: {warnings}")
            
            return is_valid, warnings
            
        except Exception as e:
            error_msg = f"데이터 검증 중 오류: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return False, [error_msg]
    
    def validate_feature_data(self, symbol: str, features: np.ndarray) -> Tuple[bool, List[str]]:
        """
        전처리된 특성 데이터 검증
        
        Args:
            symbol: 심볼
            features: 특성 배열
            
        Returns:
            (검증 통과 여부, 경고 메시지 리스트)
        """
        warnings = []
        
        try:
            # 1. NaN/Inf 검증
            if np.isnan(features).any():
                warnings.append("NaN 값 발견")
                return False, warnings
            
            if np.isinf(features).any():
                warnings.append("무한대 값 발견")
                return False, warnings
            
            # 2. 특성 범위 검증 (정규화된 데이터는 대체로 -3 ~ 3 범위 내에 있어야 함)
            extreme_values = np.abs(features) > 5
            if extreme_values.any():
                extreme_count = extreme_values.sum()
                warnings.append(f"극단값 {extreme_count}개 발견 (절댓값 > 5)")
            
            # 3. 영벡터 검증
            if np.allclose(features, 0):
                warnings.append("모든 특성이 0인 영벡터 감지")
                return False, warnings
            
            return True, warnings
            
        except Exception as e:
            error_msg = f"특성 데이터 검증 중 오류: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            return False, [error_msg]
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        데이터 품질 보고서 생성
        
        Returns:
            데이터 품질 보고서
        """
        return {
            "validated_symbols": list(self.previous_data.keys()),
            "anomaly_counts": self.anomaly_counts.copy(),
            "last_validation_time": datetime.now().isoformat(),
            "total_symbols": len(self.previous_data)
        }