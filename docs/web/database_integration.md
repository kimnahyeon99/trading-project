# SAC 트레이딩 시스템 데이터베이스 통합

SAC 트레이딩 시스템은 파일 기반과 데이터베이스 기반 데이터 관리를 모두 지원합니다. 이 문서는 MySQL 데이터베이스 통합의 설계, 구현 및 사용 방법에 대해 설명합니다.

## 데이터베이스 아키텍처

### 테이블 구조

#### 1. trading_stats 테이블
```sql
CREATE TABLE IF NOT EXISTS trading_stats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    portfolio_value DECIMAL(15, 2) NOT NULL,
    cash_balance DECIMAL(15, 2) NOT NULL,
    equity_value DECIMAL(15, 2) NOT NULL,
    daily_pnl DECIMAL(15, 2) DEFAULT 0,
    total_pnl DECIMAL(15, 2) DEFAULT 0
)
```

#### 2. trades 테이블
```sql
CREATE TABLE IF NOT EXISTS trades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp DATETIME NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    side ENUM('buy', 'sell') NOT NULL,
    quantity DECIMAL(15, 6) NOT NULL,
    price DECIMAL(15, 6) NOT NULL,
    fee DECIMAL(15, 6) DEFAULT 0,
    pnl DECIMAL(15, 2) DEFAULT NULL,
    model_id VARCHAR(50) DEFAULT NULL
)
```

#### 3. positions 테이블
```sql
CREATE TABLE IF NOT EXISTS positions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(15, 6) NOT NULL,
    avg_entry_price DECIMAL(15, 6) NOT NULL,
    current_price DECIMAL(15, 6) DEFAULT NULL,
    unrealized_pnl DECIMAL(15, 2) DEFAULT 0,
    timestamp DATETIME NOT NULL,
    UNIQUE KEY (symbol)
)
```

#### 4. market_data 테이블
```sql
CREATE TABLE IF NOT EXISTS market_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp DATETIME NOT NULL,
    open DECIMAL(15, 6) NOT NULL,
    high DECIMAL(15, 6) NOT NULL,
    low DECIMAL(15, 6) NOT NULL,
    close DECIMAL(15, 6) NOT NULL,
    volume DECIMAL(20, 6) DEFAULT 0,
    UNIQUE KEY (symbol, timestamp)
)
```

#### 5. models 테이블
```sql
CREATE TABLE IF NOT EXISTS models (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    file_path VARCHAR(255) NOT NULL,
    created_time DATETIME NOT NULL,
    modified_time DATETIME NOT NULL,
    file_size INT NOT NULL,
    description TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    UNIQUE KEY (model_id)
)
```

#### 6. backtest_results 테이블
```sql
CREATE TABLE IF NOT EXISTS backtest_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    model_id VARCHAR(50) NOT NULL,
    backtest_date DATETIME NOT NULL,
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    initial_balance DECIMAL(15, 2) NOT NULL,
    final_balance DECIMAL(15, 2) NOT NULL,
    total_return DECIMAL(10, 6) NOT NULL,
    annualized_return DECIMAL(10, 6) DEFAULT NULL,
    sharpe_ratio DECIMAL(10, 6) DEFAULT NULL,
    max_drawdown DECIMAL(10, 6) DEFAULT NULL,
    win_rate DECIMAL(10, 6) DEFAULT NULL,
    profit_factor DECIMAL(10, 6) DEFAULT NULL,
    total_trades INT DEFAULT 0,
    parameters JSON,
    UNIQUE KEY (model_id, backtest_date)
)
```

## 구현 클래스

### DatabaseManager 클래스
MySQL 데이터베이스 연결 및 쿼리 실행을 관리합니다.

```python
class DatabaseManager:
    def __init__(self, host, port, user, password, database, ...):
        # 데이터베이스 연결 설정
    
    def execute_query(self, query, params=None, fetch=True):
        # SQL 쿼리 실행
    
    def execute_many(self, query, params_list):
        # 여러 쿼리 일괄 실행
    
    def create_tables(self):
        # 필요한 테이블 생성
```

### DBDataManager 클래스
데이터베이스 기반 데이터 관리를 구현합니다.

```python
class DBDataManager:
    def __init__(self, db_manager, data_dir=None, logger=None):
        # 데이터베이스 기반 데이터 관리자 초기화
    
    def get_trading_stats(self, refresh=False):
        # 트레이딩 통계 조회
    
    def get_backtest_results(self, model_id=None, refresh=False):
        # 백테스트 결과 조회
        
    # 기타 데이터 접근 메서드
    
    def sync_file_to_db(self, file_type, file_path):
        # 파일 데이터를 데이터베이스에 동기화
```

### DataManagerFactory 클래스
적절한 데이터 관리자 인스턴스를 생성하는 팩토리 클래스입니다.

```python
class DataManagerFactory:
    @staticmethod
    def create_manager(manager_type='file', data_dir='./data', 
                      db_config=None, logger=None):
        # 데이터 관리자 생성
    
    @staticmethod
    def sync_file_to_db(file_data_manager, db_data_manager, 
                       sync_types=None):
        # 파일 데이터를 데이터베이스로 동기화
```

## 데이터베이스 모드 사용 방법

### 1. MySQL 서버 설정

```bash
# MySQL 설치 (Ubuntu)
sudo apt update
sudo apt install mysql-server

# 데이터베이스 생성
mysql -u root -p
CREATE DATABASE sac_trading CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
CREATE USER 'sac_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON sac_trading.* TO 'sac_user'@'localhost';
FLUSH PRIVILEGES;
```

### 2. 대시보드 실행 (데이터베이스 모드)

```bash
# 기본 데이터베이스 모드 실행
python src/dashboard/run_dashboard.py --db-mode --db-user sac_user --db-password your_password

# 파일 데이터를 데이터베이스로 동기화 후 실행
python src/dashboard/run_dashboard.py --db-mode --sync-to-db
```

### 3. 모든 실행 옵션

```bash
python src/dashboard/run_dashboard.py --help
```

| 옵션 | 설명 |
|------|------|
| `--db-mode` | 데이터베이스 모드 활성화 |
| `--db-host` | 데이터베이스 호스트 (기본값: localhost) |
| `--db-port` | 데이터베이스 포트 (기본값: 3306) |
| `--db-user` | 데이터베이스 사용자 (기본값: root) |
| `--db-password` | 데이터베이스 비밀번호 |
| `--db-name` | 데이터베이스 이름 (기본값: sac_trading) |
| `--sync-to-db` | 파일 데이터를 DB로 동기화 후 실행 |

## 파일 기반 vs 데이터베이스 기반 비교

| 특성 | 파일 기반 | 데이터베이스 기반 |
|------|----------|-----------------|
| 설정 복잡성 | 낮음 | 중간 |
| 데이터 접근 속도 | 느림 (캐시 사용) | 빠름 |
| 동시 접근 | 제한적 | 우수함 |
| 데이터 무결성 | 제한적 | 우수함 |
| 확장성 | 제한적 | 우수함 |
| 실시간 업데이트 | 어려움 | 쉬움 |
| 사용 사례 | 단위 테스트, 개발 | 프로덕션, 실시간 트레이딩 |

## 데이터 동기화 프로세스

파일 데이터를 데이터베이스로 동기화하는 과정은 다음과 같습니다:

1. 파일 기반 데이터 관리자 인스턴스 생성
2. 데이터베이스 기반 데이터 관리자 인스턴스 생성
3. 각 데이터 유형별 파일 읽기
   - 트레이딩 통계 파일(JSON)
   - 백테스트 결과 파일(JSON)
   - 모델 정보 파일(.pt)
   - 시장 데이터 파일(CSV)
4. 읽은 데이터를 데이터베이스에 삽입 또는 업데이트
5. 결과 로깅 및 보고

## 성능 최적화

### 연결 풀링
```python
# 연결 풀 초기화
self.connection_pool = pooling.MySQLConnectionPool(
    pool_name=self.pool_name,
    pool_size=self.pool_size,
    **self.config
)
```

### 일괄 처리
```python
# 여러 레코드 일괄 삽입
def execute_many(self, query, params_list):
    with self.get_connection() as connection:
        cursor = connection.cursor()
        try:
            cursor.executemany(query, params_list)
            connection.commit()
            return cursor.rowcount
        except mysql.connector.Error as e:
            connection.rollback()
            raise
        finally:
            cursor.close()
```

### 인덱싱
중요 열에 인덱스를 추가하여 검색 성능을 향상시킵니다:
```sql
CREATE INDEX idx_timestamp ON trading_stats(timestamp);
CREATE INDEX idx_symbol ON market_data(symbol);
CREATE INDEX idx_model_id ON backtest_results(model_id);
```

## 결론

SAC 트레이딩 시스템의 데이터베이스 통합을 통해 다음과 같은 이점을 얻을 수 있습니다:

1. **실시간 데이터 처리**: 대용량 데이터를 실시간으로 효율적으로 처리
2. **확장성**: 시스템 확장에 따른 데이터 관리 용이
3. **데이터 무결성**: 트랜잭션 처리를 통한 데이터 일관성 유지
4. **유연한 전환**: 개발/테스트 환경에서는 파일 기반, 운영 환경에서는 데이터베이스 기반으로 유연하게 전환 가능
5. **효율적인 쿼리**: 복잡한 분석 및 보고서 생성 효율화

단위 테스트 및 개발 환경에서는 간편한 파일 기반 방식을 사용하고, 실제 운영 환경에서는 안정성과 성능이 우수한 데이터베이스 기반 방식을 사용하는 접근법은 SAC 트레이딩 시스템의 모든 요구 사항을 효과적으로 충족시킵니다. 