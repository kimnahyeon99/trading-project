"""
SAC 트레이딩 시스템 설정 파일 (모델 호환 버전)
자동 생성됨 - 모델: C:\finalproject\models\final_sac_model_20250523_120624
테스트용 심볼: AAPL
"""
import os
import logging
import torch
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# 데이터 관련 설정
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

TARGET_SYMBOLS = ['AAPL','MSFT','GOOGL','GOOG','AMZN','NVDA','META','TSLA']
TRADING_SYMBOLS = ['TSLA']
trading_symbols = ['TSLA']

base_model_dir = r'models\{ticker}'
model_name_pattern = 'final_sac_model_{ticker}'

tickers = ['AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA']
ACTIVE_TRADING_SYMBOLS = ['TSLA']

# 모든 가능한 모델 경로 패턴들
possible_model_patterns = [
    r'models\{ticker}\final_sac_model_{ticker}',  # 기본 패턴
    r'models\final_sac_model_{ticker}',           # 단순 패턴
    r'final_sac_model_{ticker}',                  # 루트 패턴
]

def find_model_path(ticker):
    """여러 패턴으로 모델 경로 찾기"""
    for pattern in possible_model_patterns:
        model_path = pattern.format(ticker=ticker)
        config_path = os.path.join(model_path, 'config.pth')
        if os.path.exists(config_path):
            return model_path
    return None

# 자동으로 모든 모델 찾기 및 활성화
ACTIVE_TRADING_SYMBOLS = {}
for ticker in tickers:
    model_path = find_model_path(ticker)
    if model_path:
        ACTIVE_TRADING_SYMBOLS[ticker] = {
            'model_path': model_path,
            'enabled': True,
            'max_position_size': 0.01,   # 1%
            'trading_interval': 60,      # 1분
        }
        print(f"✅ {ticker} 모델 발견: {model_path}")
    else:
        print(f"⚠️  {ticker} 모델이 없어 거래 대상에서 제외됩니다.")

# TSLA 모델 경로 직접 설정
ACTIVE_TRADING_SYMBOLS['TSLA'] = {
    'model_path': 'models/TSLA/final_sac_model_TSLA',
    'enabled': True,
    'max_position_size': 0.01,   # 1%
    'trading_interval': 60,      # 1분
}

print(f"🎯 총 {len(ACTIVE_TRADING_SYMBOLS)}개 활성화된 모델")

# 매우 보수적인 리스크 관리 설정
GLOBAL_MAX_EXPOSURE = 0.3       # 전체 계좌의 5%까지만 투자 # 전체 계좌의 30%까지만 투자
MAX_DRAWDOWN = 0.15             # 최대 2% 낙폭 허용 # 5% 낙폭 허용 # 15% 낙폭 허용
MAX_DAILY_LOSS = 0.10           # 일일 최대 손실 1% # 일일 최대 손실 3% # 일일 최대 손실 10%
EMERGENCY_STOP_LOSS = 0.05      # 응급 손절 3% # 응급 손절 5%

# 개별 종목 기본 설정 (매우 보수적)
DEFAULT_SYMBOL_CONFIG = {
    'max_position_size': 0.03,      # 1%로 매우 낮게 # 3%로 매우 낮게
    'trading_interval': 60,         # 1분 간격 
    'min_trade_amount': 0.001,      # 최소 거래 금액 $10 # 최소 거래 금액 $0.001
    'max_trade_amount': 500.0,      # 최대 거래 금액 $100 # 최대 거래 금액 $500
}
MAX_POSITION_SIZE = DEFAULT_SYMBOL_CONFIG['max_position_size']
TRADING_INTERVAL = DEFAULT_SYMBOL_CONFIG['trading_interval']
MIN_TRADE_AMOUNT = DEFAULT_SYMBOL_CONFIG['min_trade_amount']
MAX_TRADE_AMOUNT = DEFAULT_SYMBOL_CONFIG['max_trade_amount']
# MySQL 데이터베이스 설정
MYSQL_HOST = "192.168.40.199"
MYSQL_USER = "root"
MYSQL_PASSWORD = "mysecretpassword"
MYSQL_DATABASE = "trading"

# TimescaleDB 데이터베이스 설정
DB_USER = "postgres"
DB_PASSWORD = "mysecretpassword"
DB_HOST = "192.168.40.193"
DB_PORT = 5432
DB_NAME = "trading"

DB_SCHEMA = "public"
DB_TABLE_PREFIX = "ticker_"
DB_POOL_SIZE = 5
DB_TIMEOUT = 30
DB_MAX_OVERFLOW = 10
DB_RETRY_COUNT = 3
DB_RETRY_DELAY = 5

# 데이터 수집 설정
DATA_START_DATE = "2024-01-01"
DATA_END_DATE = "2025-05-29"

# 데이터 전처리 설정
WINDOW_SIZE = 30
window_size = 30
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 트레이딩 환경 설정
INITIAL_BALANCE = 100000.0  # 페이퍼 트레이딩 초기 자본금
MAX_TRADING_UNITS = 10
TRANSACTION_FEE_PERCENT = 0.0

# SAC 모델 하이퍼파라미터 공통 설정 (모든 모델 타입에서 사용)
HIDDEN_DIM = 256
LEARNING_RATE_ACTOR = 1e-4
LEARNING_RATE_CRITIC = 1e-4
LEARNING_RATE_ALPHA = 1e-4
GAMMA = 0.90
TAU = 0.01
ALPHA_INIT = 0.2
REPLAY_BUFFER_SIZE = 2000000
TARGET_UPDATE_INTERVAL = 1
BATCH_SIZE = 512
WEIGHT_DECAY = 0.0  # 필요시 활성화
GRADIENT_CLIP = 1.0
HIDDEN_DIM = 256
# 학습 설정
BATCH_SIZE = 512
NUM_EPISODES = 1000
EVALUATE_INTERVAL = 10
SAVE_MODEL_INTERVAL = 50
MAX_STEPS_PER_EPISODE = 1000
overlap_ratio = 0.0

# 장치 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alpaca API 설정 (페이퍼 트레이딩)
API_KEY = os.getenv("APCA_API_KEY_ID", "PKDNLSCCUDJF3G9I9QYZ")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "UIcam0hQK3c6Ve9XDsJ7ZERIHdSepTOlJZZyVOL4")
BASE_URL = "https://paper-api.alpaca.markets"  # 페이퍼 트레이딩 URL
DATA_FEED = 'iex'  # 무료 데이터 피드
DEBUG = True       # 디버그 모드 활성화
MAX_RETRIES = 5

# 백테스트 설정 (하현팀 수정)
BACKTEST_START_DATE = "2024-01-01"
BACKTEST_END_DATE = "2025-05-29"

# 로깅 설정
def setup_logger(name, log_file, level=logging.INFO):
    """로거 설정 함수 (중복 핸들러 방지)"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    logger = logging.getLogger(name)

    # 이미 핸들러가 있는 경우 중복 설정 방지
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # 파일 핸들러
    handler = logging.FileHandler(log_file, encoding='utf-8')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 상위 로거로의 전파 방지 (선택사항)
    logger.propagate = False

    return logger

# 기본 로거 설정
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"sac_trading_{timestamp}.log"
LOGGER = setup_logger("sac_trading", LOG_FILE)

# 데이터베이스 로거 설정
DB_LOG_FILE = LOGS_DIR / f"db_connection_{timestamp}.log"
DB_LOGGER = setup_logger("db_connection", DB_LOG_FILE)

# 헬퍼 함수들
def get_db_connection_string():
    """데이터베이스 연결 문자열 반환"""
    return f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def get_active_symbol_config(symbol: str):
    """활성화된 심볼의 설정 반환"""
    return ACTIVE_TRADING_SYMBOLS.get(symbol, DEFAULT_SYMBOL_CONFIG)

def get_total_max_exposure():
    """전체 최대 노출도 반환"""
    return GLOBAL_MAX_EXPOSURE

# 데이터베이스 공통 쿼리 설정
DB_QUERIES = {
    "get_table_list": """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = '{schema}' AND table_name LIKE '{prefix}%';
    """,
    "get_date_range": """
        SELECT 
            MIN(timestamp) as start_date,
            MAX(timestamp) as end_date,
            COUNT(*) as rows
        FROM ticker_{table_name};
    """,
    "get_daily_data": """
        SELECT *
        FROM ticker_{table_name}
        WHERE timestamp BETWEEN '{start_date}' AND '{end_date}'
        ORDER BY timestamp ASC;
    """
}

class Config:
    # 디렉토리 설정
    ROOT_DIR = ROOT_DIR
    DATA_DIR = DATA_DIR
    LOGS_DIR = LOGS_DIR
    MODELS_DIR = MODELS_DIR
    RESULTS_DIR = RESULTS_DIR
    
    # 심볼 설정
    TARGET_SYMBOLS = TARGET_SYMBOLS
    TRADING_SYMBOLS = TRADING_SYMBOLS
    trading_symbols = trading_symbols
    ACTIVE_TRADING_SYMBOLS = ACTIVE_TRADING_SYMBOLS
    DEFAULT_SYMBOL_CONFIG = DEFAULT_SYMBOL_CONFIG
    
    # 리스크 관리 설정
    GLOBAL_MAX_EXPOSURE = GLOBAL_MAX_EXPOSURE
    MAX_DRAWDOWN = MAX_DRAWDOWN
    max_drawdown = MAX_DRAWDOWN 
    MAX_DAILY_LOSS = MAX_DAILY_LOSS
    max_daily_loss = MAX_DAILY_LOSS
    EMERGENCY_STOP_LOSS = EMERGENCY_STOP_LOSS
    emergency_stop_loss = EMERGENCY_STOP_LOSS
    
    # 개별 속성들 (트레이딩 스크립트에서 사용)
    max_position_size = DEFAULT_SYMBOL_CONFIG['max_position_size']
    max_trade_amount = DEFAULT_SYMBOL_CONFIG['max_trade_amount']
    trading_interval = DEFAULT_SYMBOL_CONFIG['trading_interval']
    
    # 데이터베이스 설정
    DB_USER = DB_USER
    DB_PASSWORD = DB_PASSWORD
    DB_HOST = DB_HOST
    DB_PORT = DB_PORT
    DB_NAME = DB_NAME
    DB_SCHEMA = DB_SCHEMA
    DB_TABLE_PREFIX = DB_TABLE_PREFIX
    DB_POOL_SIZE = DB_POOL_SIZE
    DB_TIMEOUT = DB_TIMEOUT
    DB_MAX_OVERFLOW = DB_MAX_OVERFLOW
    DB_RETRY_COUNT = DB_RETRY_COUNT
    DB_RETRY_DELAY = DB_RETRY_DELAY
    DB_QUERIES = DB_QUERIES
    
    # 데이터 설정
    DATA_START_DATE = DATA_START_DATE
    DATA_END_DATE = DATA_END_DATE
    WINDOW_SIZE = WINDOW_SIZE
    window_size = WINDOW_SIZE
    TRAIN_RATIO = TRAIN_RATIO
    VALID_RATIO = VALID_RATIO
    TEST_RATIO = TEST_RATIO
    
    # 트레이딩 환경 설정
    INITIAL_BALANCE = INITIAL_BALANCE
    MAX_TRADING_UNITS = MAX_TRADING_UNITS
    TRANSACTION_FEE_PERCENT = TRANSACTION_FEE_PERCENT
    
    # 모델 설정
    HIDDEN_DIM = HIDDEN_DIM
    LEARNING_RATE_ACTOR = LEARNING_RATE_ACTOR
    LEARNING_RATE_CRITIC = LEARNING_RATE_CRITIC
    LEARNING_RATE_ALPHA = LEARNING_RATE_ALPHA
    GAMMA = GAMMA
    TAU = TAU
    ALPHA_INIT = ALPHA_INIT
    TARGET_UPDATE_INTERVAL = TARGET_UPDATE_INTERVAL
    REPLAY_BUFFER_SIZE = REPLAY_BUFFER_SIZE
    BATCH_SIZE = BATCH_SIZE
    NUM_EPISODES = NUM_EPISODES
    EVALUATE_INTERVAL = EVALUATE_INTERVAL
    SAVE_MODEL_INTERVAL = SAVE_MODEL_INTERVAL
    MAX_STEPS_PER_EPISODE = MAX_STEPS_PER_EPISODE
    DEVICE = DEVICE
    
    # 로거 설정
    LOGGER = LOGGER
    DB_LOGGER = DB_LOGGER
    
    # 백테스트 설정
    BACKTEST_START_DATE = BACKTEST_START_DATE
    BACKTEST_END_DATE = BACKTEST_END_DATE
    
    # 알파카 API 설정
    API_KEY = API_KEY
    API_SECRET = API_SECRET
    BASE_URL = BASE_URL
    DATA_FEED = DATA_FEED
    MAX_RETRIES = MAX_RETRIES
    DEBUG = DEBUG
    
    # 헬퍼 메서드들
    @staticmethod
    def get_db_connection_string():
        return get_db_connection_string()
    
    @staticmethod
    def get_active_symbol_config(symbol: str):
        return get_active_symbol_config(symbol)
    
    @staticmethod
    def get_total_max_exposure():
        return get_total_max_exposure()

    @staticmethod
    def setup_logger(name, log_file, level=logging.INFO):
        return setup_logger(name, log_file, level)

# 글로벌 config 인스턴스
config = Config()

# 설정 확인 메시지
print("=" * 60)
print("🔧 모델 호환 설정 로드됨")
print("=" * 60)
print(f"📊 대상 심볼: {TARGET_SYMBOLS}")
print(f"💰 최대 포지션: {DEFAULT_SYMBOL_CONFIG['max_position_size']*100}%")
print(f"💵 최대 거래금액: ${DEFAULT_SYMBOL_CONFIG['max_trade_amount']}")
print(f"⏰ 거래 간격: {DEFAULT_SYMBOL_CONFIG['trading_interval']//60}분")
print(f"🛡️ 일일 최대 손실: {MAX_DAILY_LOSS*100}%")
print("=" * 60)
