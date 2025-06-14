# -*- coding: utf-8 -*-
"""
최적화된 실시간 트레이딩 설정 파일 - 딜레이 최소화

기존 config.py의 설정을 상속받되, 실시간 성능에 최적화된 값들로 override
"""

from src.config.config import *

# ========================================
# 🚀 최적화된 실시간 트레이딩 설정
# ========================================

# 📊 데이터 수집 최적화
OPTIMIZED_DATA_COLLECTION = {
    'interval': 10,              # 10초마다 데이터 수집 (기존: 60초)
    'use_websocket': True,       # WebSocket 우선 사용
    'cache_enabled': True,       # 가격 캐싱 활성화
    'parallel_collection': True, # 병렬 데이터 수집
    'buffer_size': 50,          # 적은 버퍼 크기로 빠른 응답
}

# ⚡ 거래 실행 최적화
OPTIMIZED_TRADING = {
    'min_interval': 15,          # 최소 15초 거래 간격 (기존: 60초)
    'max_interval': 90,          # 최대 90초 거래 간격 (기존: 60초)
    'action_threshold': 0.005,   # 더 민감한 행동 임계값 (기존: 0.02)
    'urgent_threshold': 0.003,   # 긴급 거래 임계값
    'use_market_orders': True,   # 시장가 주문으로 빠른 실행
    'adaptive_intervals': True,  # 적응형 거래 간격 활성화
}

# 🎯 변동성 기반 우선순위
VOLATILITY_CONFIG = {
    'base_threshold': 0.012,     # 기본 변동성 임계값 1.2%
    'urgent_multiplier': 1.5,    # 긴급 거래를 위한 배수
    'calm_multiplier': 0.7,      # 안정적인 상황의 배수
    'calculation_window': 10,    # 변동성 계산 윈도우
}

# 📈 리스크 관리 최적화
OPTIMIZED_RISK = {
    'max_position_size': 0.02,   # 2%로 약간 증가 (빠른 대응 위해)
    'emergency_threshold': 0.03, # 3% 응급 손절
    'daily_loss_limit': 0.05,    # 5% 일일 손실 한도
    'rapid_exit_enabled': True,  # 빠른 청산 활성화
}

# 🔄 적응형 시스템 설정
ADAPTIVE_CONFIG = {
    'adjustment_interval': 30,   # 30초마다 간격 조정
    'volatility_memory': 20,     # 20개 변동성 히스토리 유지
    'success_rate_weight': 0.3,  # 성공률 가중치
    'volatility_weight': 0.7,    # 변동성 가중치
    'max_adjustment': 0.1,       # 최대 10% 조정
}

# 📊 성능 모니터링
PERFORMANCE_TRACKING = {
    'latency_tracking': True,    # 지연 시간 추적
    'processing_metrics': True,  # 처리 시간 메트릭
    'real_time_reporting': True, # 실시간 성능 리포팅
    'alert_threshold': 1.0,      # 1초 초과 시 알림
}

# ========================================
# 🎛️ 최적화된 설정 클래스
# ========================================

class OptimizedConfig(Config):
    """최적화된 실시간 트레이딩 설정"""
    
    # 📊 데이터 관련 설정 오버라이드
    data_collection_interval = OPTIMIZED_DATA_COLLECTION['interval']
    use_websocket_priority = OPTIMIZED_DATA_COLLECTION['use_websocket']
    cache_enabled = OPTIMIZED_DATA_COLLECTION['cache_enabled']
    
    # ⚡ 거래 관련 설정 오버라이드
    min_trading_interval = OPTIMIZED_TRADING['min_interval']
    max_trading_interval = OPTIMIZED_TRADING['max_interval']
    action_threshold = OPTIMIZED_TRADING['action_threshold']
    urgent_threshold = OPTIMIZED_TRADING['urgent_threshold']
    
    # 🎯 변동성 설정
    volatility_threshold = VOLATILITY_CONFIG['base_threshold']
    volatility_urgent_multiplier = VOLATILITY_CONFIG['urgent_multiplier']
    volatility_calm_multiplier = VOLATILITY_CONFIG['calm_multiplier']
    
    # 📈 리스크 설정 업데이트
    max_position_size = OPTIMIZED_RISK['max_position_size']
    emergency_threshold = OPTIMIZED_RISK['emergency_threshold']
    
    # 🔄 적응형 설정
    adaptive_adjustment_interval = ADAPTIVE_CONFIG['adjustment_interval']
    volatility_memory_size = ADAPTIVE_CONFIG['volatility_memory']
    
    # 📊 성능 설정
    enable_performance_tracking = PERFORMANCE_TRACKING['latency_tracking']
    enable_real_time_alerts = PERFORMANCE_TRACKING['real_time_reporting']

# ========================================
# 🎚️ 동적 설정 조정 함수들
# ========================================

def get_optimized_config_for_symbol(symbol: str) -> Dict[str, Any]:
    """심볼별 최적화된 설정 반환"""
    base_config = {
        'data_interval': OPTIMIZED_DATA_COLLECTION['interval'],
        'min_trade_interval': OPTIMIZED_TRADING['min_interval'],
        'max_trade_interval': OPTIMIZED_TRADING['max_interval'],
        'volatility_threshold': VOLATILITY_CONFIG['base_threshold'],
    }
    
    # 심볼별 특수 설정
    symbol_specific = {
        'TSLA': {
            'data_interval': 8,           # Tesla는 더 빠른 수집
            'min_trade_interval': 12,     # 더 빠른 거래
            'volatility_threshold': 0.015, # 더 높은 변동성 임계값
        },
        'AAPL': {
            'data_interval': 12,          # Apple은 적당한 속도
            'min_trade_interval': 18,
            'volatility_threshold': 0.010,
        },
        'NVDA': {
            'data_interval': 8,           # NVIDIA도 빠른 수집
            'min_trade_interval': 15,
            'volatility_threshold': 0.018,
        }
    }
    
    if symbol in symbol_specific:
        base_config.update(symbol_specific[symbol])
    
    return base_config

def adjust_config_for_market_condition(market_volatility: float) -> Dict[str, Any]:
    """시장 상황에 따른 동적 설정 조정"""
    if market_volatility > 0.025:  # 고변동성
        return {
            'data_interval': 5,        # 5초마다 수집
            'min_trade_interval': 10,  # 10초 최소 간격
            'action_threshold': 0.003, # 더 민감한 반응
            'mode': 'high_volatility'
        }
    elif market_volatility > 0.015:  # 중간 변동성
        return {
            'data_interval': 8,
            'min_trade_interval': 15,
            'action_threshold': 0.005,
            'mode': 'medium_volatility'
        }
    else:  # 저변동성
        return {
            'data_interval': 15,
            'min_trade_interval': 25,
            'action_threshold': 0.008,
            'mode': 'low_volatility'
        }

def get_emergency_config() -> Dict[str, Any]:
    """응급 상황용 초고속 설정"""
    return {
        'data_interval': 2,        # 2초마다 수집
        'min_trade_interval': 5,   # 5초 최소 간격
        'action_threshold': 0.001, # 매우 민감
        'use_limit_orders': False, # 시장가만 사용
        'risk_checks_minimal': True, # 최소한의 리스크 체크
        'mode': 'emergency'
    }

# ========================================
# 📋 설정 검증 및 안전장치
# ========================================

def validate_optimized_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """최적화된 설정 검증"""
    errors = []
    
    # 최소 안전 간격 확인
    if config.get('min_trade_interval', 0) < 5:
        errors.append("최소 거래 간격이 너무 짧습니다 (5초 이상 권장)")
    
    # 데이터 수집 간격 확인
    if config.get('data_interval', 0) < 2:
        errors.append("데이터 수집 간격이 너무 짧습니다 (2초 이상 권장)")
    
    # 행동 임계값 확인
    if config.get('action_threshold', 0) < 0.001:
        errors.append("행동 임계값이 너무 낮습니다 (0.001 이상 권장)")
    
    # 변동성 임계값 확인
    if config.get('volatility_threshold', 0) > 0.05:
        errors.append("변동성 임계값이 너무 높습니다 (5% 이하 권장)")
    
    return len(errors) == 0, errors

# ========================================
# 🎯 성능 프로파일 설정
# ========================================

PERFORMANCE_PROFILES = {
    'ULTRA_FAST': {
        'name': '초고속 모드',
        'data_interval': 3,
        'min_trade_interval': 8,
        'action_threshold': 0.002,
        'description': '최대한 빠른 반응, 높은 리스크'
    },
    'FAST': {
        'name': '고속 모드',
        'data_interval': 8,
        'min_trade_interval': 15,
        'action_threshold': 0.005,
        'description': '빠른 반응, 중간 리스크'
    },
    'BALANCED': {
        'name': '균형 모드',
        'data_interval': 15,
        'min_trade_interval': 30,
        'action_threshold': 0.008,
        'description': '균형잡힌 속도와 안정성'
    },
    'CONSERVATIVE': {
        'name': '보수적 모드',
        'data_interval': 30,
        'min_trade_interval': 60,
        'action_threshold': 0.015,
        'description': '안정적이고 보수적인 거래'
    }
}

def get_performance_profile(profile_name: str) -> Dict[str, Any]:
    """성능 프로파일 반환"""
    return PERFORMANCE_PROFILES.get(profile_name, PERFORMANCE_PROFILES['BALANCED'])

# ========================================
# 🎛️ 글로벌 최적화 설정
# ========================================

# 기본 최적화 설정 인스턴스
optimized_config = OptimizedConfig()

# 빠른 접근을 위한 상수들
FAST_DATA_INTERVAL = OPTIMIZED_DATA_COLLECTION['interval']
FAST_TRADE_INTERVAL = OPTIMIZED_TRADING['min_interval']
FAST_ACTION_THRESHOLD = OPTIMIZED_TRADING['action_threshold']
FAST_VOLATILITY_THRESHOLD = VOLATILITY_CONFIG['base_threshold']

if __name__ == "__main__":
    print("🚀 최적화된 실시간 트레이딩 설정")
    print("=" * 50)
    print(f"📊 데이터 수집 간격: {FAST_DATA_INTERVAL}초")
    print(f"⚡ 최소 거래 간격: {FAST_TRADE_INTERVAL}초")
    print(f"🎯 행동 임계값: {FAST_ACTION_THRESHOLD}")
    print(f"📈 변동성 임계값: {FAST_VOLATILITY_THRESHOLD}")
    print("=" * 50)
    
    # 프로파일 출력
    print("\n📋 사용 가능한 성능 프로파일:")
    for name, profile in PERFORMANCE_PROFILES.items():
        print(f"  {name}: {profile['name']} - {profile['description']}") 