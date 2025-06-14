"""
데이터베이스 연결 관리 모듈
"""
import os
from typing import Optional, Dict, Any, Union
import warnings

# MySQL 라이브러리 조건부 로드
try:
    import mysql.connector
    from mysql.connector import pooling
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False
    warnings.warn("mysql-connector-python 패키지가 설치되지 않았습니다. 데이터베이스 기능은 비활성화됩니다.")

from contextlib import contextmanager

from src.utils.logger import Logger


class DatabaseManager:
    """
    MySQL 데이터베이스 연결 관리 클래스
    """
    
    def __init__(
        self,
        host: str = '192.168.40.199',
        port: int = 3306,
        user: str = 'root',
        password: str = 'mysecretpassword',
        database: str = 'trading',
        pool_size: int = 5,
        pool_name: str = 'sac_pool',
        logger: Optional[Logger] = None
    ):
        """
        DatabaseManager 클래스 초기화
        
        Args:
            host: 데이터베이스 호스트
            port: 데이터베이스 포트
            user: 데이터베이스 사용자
            password: 데이터베이스 비밀번호
            database: 데이터베이스 이름
            pool_size: 연결 풀 크기
            pool_name: 연결 풀 이름
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.logger = logger
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database
        }
        self.pool_name = pool_name
        self.pool_size = pool_size
        self.connection_pool = None
        
        # MySQL이 없으면 초기화 건너뛰기
        if HAS_MYSQL:
            # 연결 풀 초기화
            self._initialize_pool()
        else:
            if self.logger:
                self.logger.warning("MySQL 라이브러리가 설치되지 않아 데이터베이스 연결이 비활성화됩니다.")
        
    def _initialize_pool(self) -> None:
        """
        데이터베이스 연결 풀 초기화
        """
        if not HAS_MYSQL:
            return
            
        try:
            self.connection_pool = pooling.MySQLConnectionPool(
                pool_name=self.pool_name,
                pool_size=self.pool_size,
                **self.config
            )
            
            if self.logger:
                self.logger.info(f"데이터베이스 연결 풀 초기화 완료: {self.config['host']}:{self.config['port']}/{self.config['database']}")
                
        except mysql.connector.Error as e:
            if self.logger:
                self.logger.error(f"데이터베이스 연결 풀 초기화 실패: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """
        데이터베이스 연결 획득 (컨텍스트 매니저)
        
        사용 예시:
        ```
        with db_manager.get_connection() as conn:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM trades")
            results = cursor.fetchall()
        ```
        """
        if not HAS_MYSQL:
            if self.logger:
                self.logger.error("MySQL 라이브러리가 설치되지 않아 데이터베이스 연결을 획득할 수 없습니다.")
            raise ImportError("MySQL 라이브러리가 설치되지 않았습니다.")
            
        connection = None
        try:
            connection = self.connection_pool.get_connection()
            yield connection
        except mysql.connector.Error as e:
            if self.logger:
                self.logger.error(f"데이터베이스 연결 획득 실패: {e}")
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    def execute_query(self, query: str, params: Optional[Union[tuple, dict]] = None, fetch: bool = True) -> Union[list, int]:
        """
        SQL 쿼리 실행
        
        Args:
            query: SQL 쿼리
            params: 쿼리 파라미터 (옵션)
            fetch: 결과 조회 여부
            
        Returns:
            조회 결과 또는 영향받은 행 수
        """
        if not HAS_MYSQL:
            if self.logger:
                self.logger.error("MySQL 라이브러리가 설치되지 않아 쿼리를 실행할 수 없습니다.")
            return [] if fetch else 0
            
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                cursor.execute(query, params)
                
                if fetch:
                    result = cursor.fetchall()
                    return result
                else:
                    connection.commit()
                    return cursor.rowcount
            except mysql.connector.Error as e:
                if self.logger:
                    self.logger.error(f"쿼리 실행 실패: {e}, 쿼리: {query}, 매개변수: {params}")
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def execute_many(self, query: str, params_list: list) -> int:
        """
        여러 쿼리 일괄 실행
        
        Args:
            query: SQL 쿼리
            params_list: 파라미터 목록
            
        Returns:
            영향받은 행 수
        """
        if not HAS_MYSQL:
            if self.logger:
                self.logger.error("MySQL 라이브러리가 설치되지 않아 일괄 쿼리를 실행할 수 없습니다.")
            return 0
            
        with self.get_connection() as connection:
            cursor = connection.cursor()
            try:
                cursor.executemany(query, params_list)
                connection.commit()
                return cursor.rowcount
            except mysql.connector.Error as e:
                if self.logger:
                    self.logger.error(f"일괄 쿼리 실행 실패: {e}, 쿼리: {query}")
                connection.rollback()
                raise
            finally:
                cursor.close()
    
    def create_tables(self) -> None:
        """
        필요한 테이블 생성
        """
        if not HAS_MYSQL:
            if self.logger:
                self.logger.warning("MySQL 라이브러리가 설치되지 않아 테이블을 생성할 수 없습니다.")
            return
            
        # 트레이딩 통계 테이블
        create_trading_stats_table = """
        CREATE TABLE IF NOT EXISTS trading_stats (
            id INT AUTO_INCREMENT PRIMARY KEY,
            timestamp DATETIME NOT NULL,
            portfolio_value DECIMAL(15, 2) NOT NULL,
            cash_balance DECIMAL(15, 2) NOT NULL,
            equity_value DECIMAL(15, 2) NOT NULL,
            daily_pnl DECIMAL(15, 2) DEFAULT 0,
            total_pnl DECIMAL(15, 2) DEFAULT 0
        )
        """
        
        # 거래 내역 테이블
        create_trades_table = """
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
        """
        
        # 포지션 테이블
        create_positions_table = """
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
        """
        
        # 시장 데이터 테이블
        create_market_data_table = """
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
        """
        
        # 모델 정보 테이블
        create_models_table = """
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
        """
        
        # 백테스트 결과 테이블
        create_backtest_results_table = """
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
        """
        
        try:
            self.execute_query(create_trading_stats_table, fetch=False)
            self.execute_query(create_trades_table, fetch=False)
            self.execute_query(create_positions_table, fetch=False)
            self.execute_query(create_market_data_table, fetch=False)
            self.execute_query(create_models_table, fetch=False)
            self.execute_query(create_backtest_results_table, fetch=False)
            
            if self.logger:
                self.logger.info("필요한 테이블이 모두 생성되었습니다.")
                
        except mysql.connector.Error as e:
            if self.logger:
                self.logger.error(f"테이블 생성 중 오류 발생: {e}")
            raise 