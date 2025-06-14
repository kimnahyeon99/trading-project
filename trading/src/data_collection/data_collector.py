"""
TimescaleDB에서 주식 데이터를 수집하는 모듈
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import sys
import os
import argparse
from collections import Counter

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import (
    DB_USER,
    DB_PASSWORD,
    DB_HOST,
    DB_PORT,
    DB_NAME,
    TARGET_SYMBOLS,
    DATA_DIR,
    LOGGER,
    # 하현 추가 코드 아래 시작날짜, 종료날짜 추가
    DATA_START_DATE,
    DATA_END_DATE
)
from src.utils.utils import create_directory, save_to_csv, load_from_csv

class DataCollector:
    """
    TimescaleDB에서 주식 데이터를 수집하는 클래스
    """
    
    def __init__(
        self,
        user: str = DB_USER,
        password: str = DB_PASSWORD,
        host: str = DB_HOST,
        port: Union[str, int] = DB_PORT,
        db_name: str = DB_NAME,
        symbols: List[str] = None,
        # 하현 추가 코드 아래 시작날짜, 종료날짜 추가
        start_date: str = DATA_START_DATE,
        end_date: str = DATA_END_DATE
    ):
        """
        DataCollector 클래스 초기화
        
        Args:
            user: 데이터베이스 사용자명
            password: 데이터베이스 비밀번호
            host: 데이터베이스 호스트
            port: 데이터베이스 포트
            db_name: 데이터베이스 이름
            symbols: 수집할 주식 심볼 리스트 (None인 경우 설정 파일의 기본값 사용)
        """
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.db_name = db_name
        self.symbols = symbols if symbols is not None else TARGET_SYMBOLS
        self.engine = None
        self.data_dir = DATA_DIR
        # 하현 추가 코드 아래 시작날짜, 종료날짜 추가
        self.start_date = start_date
        self.end_date = end_date    
        create_directory(self.data_dir)
        self._create_engine()
        LOGGER.info(f"DataCollector 초기화 완료: {len(self.symbols)}개 종목 대상")
        LOGGER.info(f"📅 데이터 수집 기간: {self.start_date} ~ {self.end_date}")

    def _create_engine(self) -> None:
        """
        SQLAlchemy 엔진 생성
        """
        try:
            self.engine = create_engine(
                f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db_name}"
            )
            LOGGER.info("DB 엔진 생성 성공")
        except SQLAlchemyError as e:
            LOGGER.error(f"DB 엔진 생성 실패: {str(e)}")
            raise
   
    def load_data(self, symbol: str) -> pd.DataFrame:
        
        """
        데이터베이스에서 특정 심볼의 주식 데이터 로드
        
        Args:
            symbol: 주식 심볼
            
        Returns:
            로드된 주식 데이터 데이터프레임
        """
                
        # 기존 쿼리 
        query = f"""
            SELECT *
            FROM ticker_{symbol.lower()}
            WHERE timestamp >= '{self.start_date}' 
            AND timestamp <= '{self.end_date}'
            ORDER BY timestamp ASC;
        """
        print(f'query :{query}')
        try:
            LOGGER.info(f"{symbol} 데이터 로드 시작...")
            df = pd.read_sql(query, self.engine)
            
            if 'timestamp' not in df.columns:
                LOGGER.warning(f"{symbol} 테이블에 'timestamp' 컬럼이 없습니다.")
                return pd.DataFrame()
            
            # 데이터 전처리
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            # 컬럼 제거
            df.drop(columns=['transactions', 'vwap'], inplace=True, errors='ignore')
            
            LOGGER.info(f"{symbol} 데이터 로드 완료: {len(df)} 행")
            return df
            
        except SQLAlchemyError as e:
            LOGGER.error(f"{symbol} 데이터 로드 중 DB 오류 발생: {str(e)}")
            return pd.DataFrame()
        except Exception as e:
            LOGGER.error(f"{symbol} 데이터 로드 실패: {str(e)}")
            return pd.DataFrame()
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        모든 심볼에 대한 데이터 로드

        Returns:
            심볼을 키로 하고 데이터프레임을 값으로 하는 딕셔너리
        """
        all_data = {}

        for symbol in self.symbols:
            try:
                # ✅ 중복 체크 먼저 실행
                dup_count = self.check_duplicate_timestamps(symbol)
                if dup_count > 0:
                    LOGGER.warning(f"[중복 경고] {symbol} 테이블에 중복된 timestamp가 {dup_count}개 존재합니다.")

                data = self.load_data(symbol)
                if not data.empty:
                    all_data[symbol] = data
            except Exception as e:
                LOGGER.error(f"{symbol} 처리 중 오류 발생: {str(e)}")

        LOGGER.info(f"총 {len(all_data)}/{len(self.symbols)} 종목 데이터 로드 완료")
        return all_data
    
    def save_data(self, data: Dict[str, pd.DataFrame], subdir: str = None) -> None:
        """
        수집된 데이터를 CSV 파일로 저장
        
        Args:
            data: 심볼을 키로 하고 데이터프레임을 값으로 하는 딕셔너리
            subdir: 저장할 하위 디렉토리 (None인 경우 날짜 기반으로 생성)
        """
        if subdir is None:
            subdir = datetime.now().strftime("%Y%m%d")
        
        save_dir = self.data_dir / subdir
        create_directory(save_dir)
        
        for symbol, df in data.items():
            file_path = save_dir / f"{symbol}.csv"
            save_to_csv(df, file_path)
        
        LOGGER.info(f"모든 데이터 저장 완료: {save_dir}")
    
    def check_tables(self) -> List[str]:
        """
        데이터베이스에 존재하는 테이블 확인
        
        Returns:
            테이블 이름 리스트
        """
        try:
            query = """
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' AND table_name ~ '(aapl|msft|googl|goog|amzn|nvda|meta|tsla)';
            """
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                tables = [row[0] for row in result]
            
            LOGGER.info(f"총 {len(tables)}개의 테이블 확인됨")
            return tables
        
        except SQLAlchemyError as e:
            LOGGER.error(f"테이블 확인 중 오류 발생: {str(e)}")
            return []
    
    def get_data_range(self, symbol: str) -> Dict[str, datetime]:
        """
        특정 심볼의 데이터 기간 조회
        
        Args:
            symbol: 주식 심볼
            
        Returns:
            시작일과 종료일이 포함된 딕셔너리
        """
        query = f"""
            SELECT 
                MIN(timestamp) as start_date,
                MAX(timestamp) as end_date
            FROM ticker_{symbol.lower()};
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                row = result.fetchone()
                
            if row and row[0] and row[1]:
                range_info = {
                    'start_date': row[0],
                    'end_date': row[1],
                    'days': (row[1] - row[0]).days
                }
                LOGGER.info(f"{symbol} 데이터 기간: {range_info['start_date']} ~ {range_info['end_date']} ({range_info['days']}일)")
                return range_info
            else:
                LOGGER.warning(f"{symbol} 테이블에 데이터가 없거나 날짜 정보가 없습니다.")
                return {'start_date': None, 'end_date': None, 'days': 0}
                
        except SQLAlchemyError as e:
            LOGGER.error(f"{symbol} 데이터 기간 조회 중 오류 발생: {str(e)}")
            return {'start_date': None, 'end_date': None, 'days': 0}
    
    def load_and_save(self) -> Dict[str, pd.DataFrame]:
        """
        데이터 로드 및 저장을 한 번에 수행
        
        Returns:
            로드된 데이터 딕셔너리
        """
        data = self.load_all_data()
        if data:
            self.save_data(data)
        return data


    def check_duplicate_timestamps(self, symbol: str) -> int:
        """
        특정 심볼 테이블에서 timestamp 중복 개수 확인

        Returns:
            중복 timestamp 개수
        """
        query = f"""
            SELECT COUNT(*) - COUNT(DISTINCT timestamp) AS duplicate_count
            FROM ticker_{symbol.lower()};
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(query))
                duplicate_count = result.scalar()
            LOGGER.info(f"{symbol} 중복 timestamp 개수: {duplicate_count}")
            return duplicate_count
        except SQLAlchemyError as e:
            LOGGER.error(f"{symbol} 중복 timestamp 체크 중 오류: {str(e)}")
            return -1

if __name__ == "__main__":
    from src.config.config import config
    
    parser = argparse.ArgumentParser(description="심볼별 데이터 수집")
    parser.add_argument("--symbols", nargs="+", help="수집할 심볼 리스트", default=config.trading_symbols)
    args = parser.parse_args()
    
    # DataCollector에 심볼 리스트 전달
    collector = DataCollector(
        # user='postgres',
        # password='mysecretpassword',
        # host='192.168.40.193',
        # port=5432,
        # db_name='mydb',
        symbols=args.symbols  # 여기가 핵심!
    )
    
    print(f"수집 대상 심볼: {collector.symbols}")
    
    # 데이터베이스 테이블 확인
    tables = collector.check_tables()
    print(f"사용 가능한 테이블: {tables}")
    
    # 데이터 로드 및 저장
    data = collector.load_and_save()
    print(f"로드된 데이터 종목: {list(data.keys())}")
    
    #하현 중복코드 확인

    from collections import Counter
    print('--------------------------------------------------------------------------------------------------------')
    for symbol, df in data.items():
        print(f"\n[연속된 종가 중복 체크 (30회 이상): {symbol}]")

        df = df.reset_index(drop=True)  # 인덱스를 숫자로 초기화
        close_series = df['close']

        start_idx = None
        current_value = None
        count = 0

        for i in range(len(close_series)):
            if i == 0:
                current_value = close_series[i]
                start_idx = i
                count = 1
                continue

            if close_series[i] == current_value:
                count += 1
            else:
                if count >= 30:
                    end_idx = i - 1
                    print(f"연속된 'close' 값 {current_value}이 {count}번 반복됨 → 시작 인덱스: {start_idx}, 종료 인덱스: {end_idx}")
                # reset
                current_value = close_series[i]
                start_idx = i
                count = 1

        # 마지막 구간도 검사
        if count >= 30:
            end_idx = len(close_series) - 1
            print(f"연속된 'close' 값 {current_value}이 {count}번 반복됨 → 시작 인덱스: {start_idx}, 종료 인덱스: {end_idx}")

        print("확인 완료.")
    print('--------------------------------------------------------------------------------------------------------')



    
    # 첫 번째 종목의 데이터 샘플 출력
    if data:
        symbol = list(data.keys())[0]
        print(f"\n{symbol} 데이터 샘플:")
        print(data[symbol].head())