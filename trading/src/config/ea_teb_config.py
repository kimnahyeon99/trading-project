"""
백테스팅/학습 전용 설정 파일 (통합 간소화 버전)
SAC 트레이딩 시스템 - 학습, 평가, 백테스트용
config.py를 상속하여 고유 기능만 추가
"""
import os
import re

# config.py에서 모든 기본 설정 import
from src.config.config import *

# ========================================
# ea_teb_config.py 고유 설정만 정의
# ========================================

# 모델 패턴 정의 (ea_teb_config.py 고유)
MODEL_PATTERNS = {
    'mlp': r'final_sac_model_([A-Z]+)_(\d{8}_\d{6})',
    'cnn': r'final_cnn_sac_model_(\d{8}_\d{6})',
    'lstm': r'final_lstm_sac_model_(\d{8}_\d{6})',
    'transformer': r'final_transformer_sac_model_(\d{8}_\d{6})'
}

# MySQL 설정 (백테스트 결과 저장용) - ea_teb_config.py 고유
MYSQL_HOST = os.getenv("MYSQL_HOST", "192.168.40.199")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "trading")
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "mysecretpassword")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
SAVE_TO_DATABASE = True
SKIP_DB_ON_ERROR = True

# ========================================
# 모델별 최적화된 기본값 설정 (개선 버전)
# ========================================

# 모델별 기본 하이퍼파라미터 (트레이딩 최적화됨)
MODEL_DEFAULTS = {
    'mlp': {
        'learning_rate_factor': 1.0,        # 기본 학습률 유지
        'dropout_rate': 0.1,                # 가벼운 정규화
        'alpha_init': ALPHA_INIT,           # 기본 엔트로피 계수
        'gradient_clip': GRADIENT_CLIP,     # 기본 그래디언트 클리핑
        'buffer_type': 'sequential',        # 시계열 특성 활용
        'sequence_length': 32,              # 적당한 시퀀스 길이
        
        # MLP 전용 최적화
        'hidden_layers': 3,                 # 은닉층 개수
        'activation': 'relu',               # 활성화 함수
        'batch_norm': False,                # MLP는 배치정규화 생략
        'weight_decay': 1e-4,               # 가벼운 L2 정규화
    },
    'cnn': {
        'learning_rate_factor': 0.25,       # 더욱 보수적인 학습률 (기존 0.3 → 0.25)
        'dropout_rate': 0.2,                # 조금 더 강한 정규화 (0.15 → 0.2)
        'alpha_init': ALPHA_INIT * 0.8,     # 더 낮은 초기 엔트로피 (안정성)
        'gradient_clip': GRADIENT_CLIP * 0.5, # 더 강한 그래디언트 클리핑
        'buffer_type': 'sequential',        # 시계열 패턴 학습
        'sequence_length': 64,              # 긴 시퀀스로 패턴 포착
        
        # CNN 전용 최적화
        'conv_layers': 3,                   # 컨볼루션 층 개수
        'kernel_sizes': [3, 5, 7],          # 다양한 커널 크기
        'pooling_type': 'adaptive',         # 적응형 풀링
        'batch_norm': True,                 # 배치 정규화 필수
        'weight_decay': 1e-5,               # 더 강한 L2 정규화
        'early_stopping_patience': 50,     # 조기 종료 인내심
    },
    'lstm': {
        'learning_rate_factor': 0.6,        # 더 보수적으로 조정 (0.8 → 0.6)
        'dropout_rate': 0.25,               # 더 강한 드롭아웃 (과적합 방지)
        'alpha_init': ALPHA_INIT * 1.2,     # 더 높은 초기 엔트로피 (탐색)
        'gradient_clip': GRADIENT_CLIP,     # 기본 그래디언트 클리핑
        'buffer_type': 'sequential',        # 순차적 경험 필수
        'sequence_length': 48,              # 더 긴 시퀀스 (32 → 48)
        
        # LSTM 전용 최적화
        'lstm_layers': 2,                   # LSTM 층 개수
        'lstm_hidden_dim': 128,             # LSTM 은닉 차원
        'bidirectional': False,             # 단방향 (실시간 트레이딩 호환)
        'lstm_dropout': 0.15,               # LSTM 내부 드롭아웃
        'forget_bias': 1.0,                 # 망각 게이트 편향
        'weight_decay': 2e-5,               # 중간 정도 L2 정규화
        'early_stopping_patience': 75,     # 더 긴 인내심 (LSTM은 느리게 학습)
    }
}

# 🆕 성능 기반 동적 조정 설정
PERFORMANCE_ADAPTIVE_SETTINGS = {
    'enable_adaptive_lr': True,             # 성능 기반 학습률 조정
    'performance_window': 50,               # 성능 평가 윈도우
    'lr_boost_threshold': 0.1,              # 학습률 증가 임계값 (좋은 성능)
    'lr_reduce_threshold': -0.05,           # 학습률 감소 임계값 (나쁜 성능)
    'max_lr_multiplier': 2.0,               # 최대 학습률 배수
    'min_lr_multiplier': 0.1,               # 최소 학습률 배수
}

# 🆕 리스크 관리 기반 설정
RISK_MANAGEMENT_SETTINGS = {
    'conservative_mode': False,             # 보수적 모드 (리스크 최소화)
    'aggressive_mode': False,               # 공격적 모드 (수익 최대화)
    
    # 보수적 모드 설정 (conservative_mode=True 시 적용)
    'conservative_lr_factor': 0.5,          # 학습률 절반
    'conservative_dropout_boost': 1.5,      # 드롭아웃 1.5배
    'conservative_alpha_factor': 0.7,       # 더 낮은 엔트로피
    
    # 공격적 모드 설정 (aggressive_mode=True 시 적용)
    'aggressive_lr_factor': 1.5,            # 학습률 1.5배
    'aggressive_dropout_factor': 0.8,       # 드롭아웃 20% 감소
    'aggressive_alpha_factor': 1.3,         # 더 높은 엔트로피
}

# 🆕 공통 고급 설정 (업데이트됨)
COMMON_ADVANCED_SETTINGS = {
    'use_gradient_clipping': True,
    'use_lr_scheduling': True, 
    'target_update_method': 'soft',
    'adaptive_alpha': True,
    'use_batch_norm': True,                 # 모델별로 오버라이드 가능
    'weight_decay': 1e-5,                   # 기본값 (모델별로 오버라이드)
    
    # 🆕 새로운 고급 기능
    'use_layer_norm': False,                # 레이어 정규화 (실험적)
    'use_spectral_norm': False,             # 스펙트럴 정규화 (안정성)
    'warmup_steps': 1000,                   # 학습률 워밍업
    'cosine_annealing': False,              # 코사인 어닐링 스케줄러
    'polyak_averaging': True,               # 폴리악 평균화
}



# ========================================
# 모델별 엔트로피 설정
# ========================================

MODEL_ENTROPY_CONFIGS = {
    'mlp': {
        'use_adaptive': False,
        'fixed_target': -1.0,
        'description': 'MLP 고정 엔트로피 (-1.0)'
    },
    'cnn': {
        'use_adaptive': True,
        'entropy_range': (-1.2, -0.5),
        'initial_target': -0.8,
        'description': 'CNN 적응적 엔트로피 (-1.2 ~ -0.5)'
    },
    'lstm': {
        'use_adaptive': True,
        'entropy_range': (-1.0, -0.8),
        'initial_target': -0.9,
        'description': 'LSTM 적응적 엔트로피 (-1.0 ~ -0.8)'
    }
}

# ========================================
# ea_teb_config.py 고유 함수들
# ========================================

def find_latest_model_by_pattern(model_type: str, symbol: str = None, base_dir: str = None) -> str:
    """
    패턴별로 최신 모델 경로 찾기

    Args:
        model_type: 모델 타입 ('mlp', 'cnn', 'lstm', 'transformer')
        symbol: 주식 심볼 (MLP 모델의 경우 필수)
        base_dir: 모델 검색 기본 디렉토리 (기본값: models/)

    Returns:
        str: 최신 모델의 전체 경로, 없으면 None
    """
    if base_dir is None:
        base_dir = os.path.join(ROOT_DIR, "models")

    if not os.path.exists(base_dir):
        print(f"❌ 모델 디렉토리가 존재하지 않습니다: {base_dir}")
        return None

    pattern = MODEL_PATTERNS.get(model_type.lower())
    if not pattern:
        print(f"❌ 지원하지 않는 모델 타입: {model_type}")
        return None

    # 디렉토리 내 모든 폴더 스캔
    model_dirs = []

    try:
        for item in os.listdir(base_dir):
            item_path = os.path.join(base_dir, item)
            if os.path.isdir(item_path):
                match = re.match(pattern, item)
                if match:
                    # MLP 모델의 경우 심볼 매칭 확인
                    if model_type.lower() == 'mlp':
                        if symbol and match.group(1) == symbol:
                            timestamp = match.group(2)
                            model_dirs.append((timestamp, item_path))
                    else:
                        # CNN, LSTM 등은 심볼 무관
                        timestamp = match.group(1)
                        model_dirs.append((timestamp, item_path))

        if not model_dirs:
            if model_type.lower() == 'mlp' and symbol:
                print(f"❌ {symbol}에 대한 {model_type.upper()} 모델을 찾을 수 없습니다.")
            else:
                print(f"❌ {model_type.upper()} 모델을 찾을 수 없습니다.")
            return None

        # 타임스탬프 기준으로 정렬하여 최신 모델 선택
        model_dirs.sort(key=lambda x: x[0], reverse=True)
        latest_model_path = model_dirs[0][1]

        # config.pth 파일 존재 여부 확인
        config_path = os.path.join(latest_model_path, 'config.pth')
        if os.path.exists(config_path):
            print(f"✅ 최신 {model_type.upper()} 모델 발견: {os.path.basename(latest_model_path)}")
            return latest_model_path
        else:
            print(f"⚠️  config.pth 파일이 없습니다: {latest_model_path}")
            return None

    except Exception as e:
        print(f"❌ 모델 검색 중 오류: {str(e)}")
        return None

def find_model_for_backtest(symbol: str = None, model_type: str = None, model_path: str = None) -> dict:
    """
    백테스트용 모델 검색

    Args:
        symbol: 주식 심볼 (선택사항)
        model_type: 모델 타입 (선택사항: 'mlp', 'cnn', 'lstm')
        model_path: 직접 지정한 모델 경로 (선택사항)

    Returns:
        dict: {'path': str, 'type': str, 'symbol': str} 또는 None
    """
    # 1. 직접 경로 지정된 경우
    if model_path:
        if os.path.exists(os.path.join(model_path, 'config.pth')):
            # 경로에서 모델 타입 추론
            model_name = os.path.basename(model_path)
            detected_type = 'mlp'  # 기본값
            detected_symbol = None

            for mtype, pattern in MODEL_PATTERNS.items():
                if re.match(pattern, model_name):
                    detected_type = mtype
                    if mtype == 'mlp':
                        match = re.match(pattern, model_name)
                        if match:
                            detected_symbol = match.group(1)
                    break

            return {
                'path': model_path,
                'type': detected_type,
                'symbol': detected_symbol or symbol
            }

    # 2. 자동 검색
    if model_type and symbol:
        # 특정 타입과 심볼로 검색
        latest_model = find_latest_model_by_pattern(model_type, symbol)
        if latest_model:
            return {
                'path': latest_model,
                'type': model_type,
                'symbol': symbol
            }
    elif model_type:
        # 타입만 지정 (CNN, LSTM 등)
        latest_model = find_latest_model_by_pattern(model_type)
        if latest_model:
            return {
                'path': latest_model,
                'type': model_type,
                'symbol': symbol
            }
    elif symbol:
        # 심볼만 지정 (MLP 우선 검색)
        for mtype in ['mlp', 'cnn', 'lstm']:
            latest_model = find_latest_model_by_pattern(mtype, symbol if mtype == 'mlp' else None)
            if latest_model:
                return {
                    'path': latest_model,
                    'type': mtype,
                    'symbol': symbol
                }

    return None

def list_available_models():
    """사용 가능한 모든 모델 목록 출력"""
    print("\n" + "=" * 60)
    print("✅ 사용 가능한 모델 목록")
    print("=" * 60)

    base_dir = os.path.join(ROOT_DIR, "models")
    if not os.path.exists(base_dir):
        print("❌ 모델 디렉토리가 존재하지 않습니다.")
        return

    models_found = []

    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and os.path.exists(os.path.join(item_path, 'config.pth')):
            # 모델 타입과 심볼 감지
            model_info = {'name': item, 'path': item_path, 'type': 'unknown', 'symbol': None}

            for mtype, pattern in MODEL_PATTERNS.items():
                match = re.match(pattern, item)
                if match:
                    model_info['type'] = mtype
                    if mtype == 'mlp' and len(match.groups()) >= 2:
                        model_info['symbol'] = match.group(1)
                    break

            models_found.append(model_info)

    # 타입별로 정렬
    models_found.sort(key=lambda x: (x['type'], x['symbol'] or '', x['name']))

    for model in models_found:
        symbol_info = f" ({model['symbol']})" if model['symbol'] else ""
        print(f"✅ {model['type'].upper()}{symbol_info}: {model['name']}")

    print(f"\n✅ 총 {len(models_found)}개 모델 발견")
    print("=" * 60 + "\n")

def get_model_config(model_type: str, symbol: str = None):
    """동적 모델 설정 반환 (공통 설정 포함)"""
    defaults = MODEL_DEFAULTS.get(model_type.lower(), MODEL_DEFAULTS['mlp'])
    
    config = {
        # 🔥 학습률 (모델별 차별화)
        'actor_lr': LEARNING_RATE_ACTOR * defaults['learning_rate_factor'],
        'critic_lr': LEARNING_RATE_CRITIC * defaults['learning_rate_factor'],
        'alpha_lr': LEARNING_RATE_ALPHA * defaults['learning_rate_factor'],
        
        # 🔥 모델별 설정
        'alpha_init': defaults['alpha_init'],
        'dropout_rate': defaults['dropout_rate'],
        'gradient_clip': defaults['gradient_clip'],
        'buffer_type': defaults['buffer_type'],
        'sequence_length': defaults['sequence_length'],
        'model_type': model_type.lower(),
        
        # 🆕 공통 고급 설정 추가 (LSTM, CNN에서 모두 도움)
        **COMMON_ADVANCED_SETTINGS
    }
    
    # 🆕 모델별 특수 설정
    if model_type.lower() == 'cnn':
        config.update({
            'use_batch_norm': True,
            'conv_layers': 3,
            'adaptive_pooling': True,
        })
    elif model_type.lower() == 'lstm':
        config.update({
            'bidirectional': False,
            'lstm_layers': 2,
            'lstm_hidden_dim': 128,
        })
    
    return config

def get_entropy_config_for_model(model_type: str, action_dim: int = 1):
    """
    모델 타입에 따른 엔트로피 설정 반환
    
    Args:
        model_type: 'mlp', 'cnn', 'lstm' 중 하나
        action_dim: 행동 차원 (기본값 1)
        
    Returns:
        dict: 엔트로피 설정 딕셔너리
    """
    config = MODEL_ENTROPY_CONFIGS.get(model_type.lower(), MODEL_ENTROPY_CONFIGS['mlp'])
    
    # action_dim에 따른 스케일링 적용
    if config['use_adaptive']:
        entropy_min, entropy_max = config['entropy_range']
        scaled_config = {
            'use_adaptive': True,
            'entropy_range': (entropy_min * action_dim, entropy_max * action_dim),
            'initial_target': config['initial_target'] * action_dim,
            'description': f"{config['description']} (action_dim={action_dim})"
        }
    else:
        scaled_config = {
            'use_adaptive': False,
            'fixed_target': config['fixed_target'] * action_dim,
            'description': f"{config['description']} (action_dim={action_dim})"
        }
    
    return scaled_config

def get_model_config(model_type: str, symbol: str = None, **overrides):
    """동적 모델 설정 반환 (리스크 관리 및 성능 적응 포함)"""
    defaults = MODEL_DEFAULTS.get(model_type.lower(), MODEL_DEFAULTS['mlp'])
    
    # 기본 설정 구성
    config = {
        # 🔥 학습률 (모델별 차별화)
        'actor_lr': LEARNING_RATE_ACTOR * defaults['learning_rate_factor'],
        'critic_lr': LEARNING_RATE_CRITIC * defaults['learning_rate_factor'],
        'alpha_lr': LEARNING_RATE_ALPHA * defaults['learning_rate_factor'],
        
        # 🔥 모델별 설정
        'alpha_init': defaults['alpha_init'],
        'dropout_rate': defaults['dropout_rate'],
        'gradient_clip': defaults['gradient_clip'],
        'buffer_type': defaults['buffer_type'],
        'sequence_length': defaults['sequence_length'],
        'model_type': model_type.lower(),
        
        # 🆕 공통 고급 설정 추가
        **COMMON_ADVANCED_SETTINGS
    }
    
    # 🆕 리스크 관리 모드 적용
    if RISK_MANAGEMENT_SETTINGS['conservative_mode']:
        config['actor_lr'] *= RISK_MANAGEMENT_SETTINGS['conservative_lr_factor']
        config['critic_lr'] *= RISK_MANAGEMENT_SETTINGS['conservative_lr_factor']
        config['dropout_rate'] *= RISK_MANAGEMENT_SETTINGS['conservative_dropout_boost']
        config['alpha_init'] *= RISK_MANAGEMENT_SETTINGS['conservative_alpha_factor']
        LOGGER.info("🛡️ 보수적 모드 활성화")
        
    elif RISK_MANAGEMENT_SETTINGS['aggressive_mode']:
        config['actor_lr'] *= RISK_MANAGEMENT_SETTINGS['aggressive_lr_factor']
        config['critic_lr'] *= RISK_MANAGEMENT_SETTINGS['aggressive_lr_factor']  
        config['dropout_rate'] *= RISK_MANAGEMENT_SETTINGS['aggressive_dropout_factor']
        config['alpha_init'] *= RISK_MANAGEMENT_SETTINGS['aggressive_alpha_factor']
        LOGGER.info("🚀 공격적 모드 활성화")
    
    # 🆕 모델별 특수 설정 (업데이트됨)
    if model_type.lower() == 'cnn':
        config.update({
            'conv_layers': defaults['conv_layers'],
            'kernel_sizes': defaults['kernel_sizes'],
            'pooling_type': defaults['pooling_type'],
            'batch_norm': defaults['batch_norm'],
            'weight_decay': defaults['weight_decay'],
            'early_stopping_patience': defaults['early_stopping_patience'],
        })
    elif model_type.lower() == 'lstm':
        config.update({
            'lstm_layers': defaults['lstm_layers'],
            'lstm_hidden_dim': defaults['lstm_hidden_dim'],
            'bidirectional': defaults['bidirectional'],
            'lstm_dropout': defaults['lstm_dropout'],
            'forget_bias': defaults['forget_bias'],
            'weight_decay': defaults['weight_decay'],
            'early_stopping_patience': defaults['early_stopping_patience'],
        })
    elif model_type.lower() == 'mlp':
        config.update({
            'hidden_layers': defaults['hidden_layers'],
            'activation': defaults['activation'],
            'batch_norm': defaults['batch_norm'],
            'weight_decay': defaults['weight_decay'],
        })
    
    # 🆕 사용자 오버라이드 적용
    config.update(overrides)
    
    return config

# 🆕 간편 설정 함수들
def enable_conservative_mode():
    """보수적 모드 활성화 (리스크 최소화)"""
    RISK_MANAGEMENT_SETTINGS['conservative_mode'] = True
    RISK_MANAGEMENT_SETTINGS['aggressive_mode'] = False
    LOGGER.info("🛡️ 보수적 모드 활성화됨")

def enable_aggressive_mode():
    """공격적 모드 활성화 (수익 최대화)"""
    RISK_MANAGEMENT_SETTINGS['aggressive_mode'] = True
    RISK_MANAGEMENT_SETTINGS['conservative_mode'] = False
    LOGGER.info("🚀 공격적 모드 활성화됨")

def reset_to_default_mode():
    """기본 모드로 리셋"""
    RISK_MANAGEMENT_SETTINGS['conservative_mode'] = False
    RISK_MANAGEMENT_SETTINGS['aggressive_mode'] = False
    LOGGER.info("⚖️ 기본 모드로 리셋됨")

def adjust_learning_rates(multiplier: float):
    """모든 모델의 학습률 일괄 조정"""
    for model_type in MODEL_DEFAULTS:
        MODEL_DEFAULTS[model_type]['learning_rate_factor'] *= multiplier
    LOGGER.info(f"📈 모든 학습률이 {multiplier}배 조정됨")

# ========================================
# Config 클래스 확장 
# ========================================

class EaTebConfig(Config):
    """
    config.Config를 상속받아 백테스팅/학습 전용 설정 추가 
    """
    
    # MySQL 설정 추가
    MYSQL_HOST = MYSQL_HOST
    MYSQL_DATABASE = MYSQL_DATABASE
    MYSQL_USER = MYSQL_USER
    MYSQL_PASSWORD = MYSQL_PASSWORD
    MYSQL_PORT = MYSQL_PORT
    SAVE_TO_DATABASE = SAVE_TO_DATABASE
    SKIP_DB_ON_ERROR = SKIP_DB_ON_ERROR
    
    # 모델 패턴 추가
    MODEL_PATTERNS = MODEL_PATTERNS
    MODEL_DEFAULTS = MODEL_DEFAULTS
    MODEL_ENTROPY_CONFIGS = MODEL_ENTROPY_CONFIGS
    
    # 헬퍼 메서드 추가
    @staticmethod
    def find_latest_model_by_pattern(model_type: str, symbol: str = None, base_dir: str = None):
        return find_latest_model_by_pattern(model_type, symbol, base_dir)
    
    @staticmethod
    def find_model_for_backtest(symbol: str = None, model_type: str = None, model_path: str = None):
        return find_model_for_backtest(symbol, model_type, model_path)
    
    @staticmethod
    def list_available_models():
        return list_available_models()

    @staticmethod
    def get_model_config(model_type: str, symbol: str = None):
        return get_model_config(model_type, symbol)
    
    @staticmethod
    def get_entropy_config_for_model(model_type: str, action_dim: int = 1):
        return get_entropy_config_for_model(model_type, action_dim)

# 글로벌 config 인스턴스 (확장된 버전)
config = EaTebConfig()

# ✅ 설정 확인 메시지 (간소화됨)
print("=" * 60)
print("✅ 백테스팅/학습 전용 설정 로드됨 (통합 간소화 버전)")
print("=" * 60)
print(f"✅ 대상 심볼: {TARGET_SYMBOLS}")
print(f"✅ 연산 장치: {DEVICE}")
print(f"✅ 모델 디렉토리: {MODELS_DIR}")
print(f"✅ MySQL 연결: {MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DATABASE}")
print(f"✅ 배치 크기: {BATCH_SIZE}")
print(f"✅ 학습 에피소드: {NUM_EPISODES}")
print(f"✅ 지원 모델 타입: {list(MODEL_PATTERNS.keys())}")
print("=" * 60)
print("🔧 모델별 기본 설정:")