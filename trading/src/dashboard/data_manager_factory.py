"""
대시보드 데이터 관리자 팩토리 모듈
"""
from typing import Optional, Dict, Any, Union, Literal
import os
import warnings

from src.utils.logger import Logger
from src.utils.database import DatabaseManager, HAS_MYSQL
from src.dashboard.data_manager_file import FileDataManager


# MYSQL 라이브러리가 설치된 경우에만 DBDataManager 임포트
if HAS_MYSQL:
    from src.dashboard.data_manager_db import DBDataManager
else:
    warnings.warn("MySQL이 설치되지 않았습니다. 데이터베이스 기반 데이터 관리자를 사용할 수 없습니다.")
    # 테스트용 더미 클래스 정의
    class DBDataManager:
        def __init__(self, *args, **kwargs):
            pass


class DataManagerFactory:
    """
    데이터 관리자 생성 팩토리 클래스
    단위 테스트에서는 파일 기반 DataManager를,
    운영 환경에서는 MySQL 기반 DBDataManager를 선택적으로 생성
    """
    
    @classmethod
    def create_manager(
        cls,                                                    # 클래스 메서드의 클래스 참조
        manager_type: str,                                      # 관리자 유형 ('file' 또는 'db')
        data_dir: str,                                          # 데이터 디렉토리 경로
        db_manager: Optional['DatabaseManager'] = None,          # 데이터베이스 설정 정보 (호스트, 포트, 사용자 등)
        logger: Optional[Logger] = None                         # 로깅을 위한 Logger 인스턴스
    ) -> Union['FileDataManager', 'DBDataManager']:                   
        # DataManager 또는 DBDataManager 인스턴스 반환
        """
        데이터 관리자 생성 팩토리 메서드
        
        Args:
            manager_type: 관리자 유형 ('file' 또는 'db')
            data_dir: 데이터 디렉토리 경로
            db_config: 데이터베이스 구성 정보 (DB 모드일 때 필요)
            logger: 로거 객체
            
        Returns:
            DataManager 또는 DBDataManager 인스턴스
        """
        # 데이터베이스 구성 유효성 검사
        if manager_type == 'db':
            if not db_manager:
                raise ValueError("db_manager 구성이 제공되지 않았습니다.")
            
            return DBDataManager(
                db_manager=db_manager,
                data_dir=data_dir,
                logger=logger
            )
        elif manager_type == 'file':
            return FileDataManager(
                data_dir=data_dir,
                logger=logger
            )
            
        else:
            raise ValueError(f"지원되지 않는 관리자 유형입니다: {manager_type}")
    @staticmethod
    def sync_file_to_db(
        file_data_manager: 'FileDataManager',
        db_data_manager: 'DBDataManager',
        sync_types: Optional[list] = None
    ) -> Dict[str, int]:
        """
        파일 데이터를 데이터베이스에 동기화
        
        Args:
            file_data_manager: 파일 기반 데이터 관리자
            db_data_manager: 데이터베이스 기반 데이터 관리자
            sync_types: 동기화할 데이터 유형 목록 (기본값: 모두)
            
        Returns:
            각 유형별 동기화된 파일 수
        """
        # MySQL이 없으면 동기화 건너뛰기
        if not HAS_MYSQL:
            warnings.warn("MySQL이 설치되지 않아 파일을 데이터베이스로 동기화할 수 없습니다.")
            return {t: 0 for t in sync_types} if sync_types else {}
            
        if sync_types is None:
            sync_types = ['trading_stats', 'backtest_results', 'models', 'market_data']
        
        result = {t: 0 for t in sync_types}
        
        # 데이터 디렉토리
        data_dir = file_data_manager.data_dir
        
        # 1. 트레이딩 통계 동기화
        if 'trading_stats' in sync_types:
            stats_dir = os.path.join(data_dir, 'live_trading')
            
            if os.path.exists(stats_dir):
                stats_files = [f for f in os.listdir(stats_dir) if f.endswith('.json')]
                
                for file_name in stats_files:
                    file_path = os.path.join(stats_dir, file_name)
                    if db_data_manager.sync_file_to_db('trading_stats', file_path):
                        result['trading_stats'] += 1
        
        # 2. 백테스트 결과 동기화
        if 'backtest_results' in sync_types:
            backtest_dir = os.path.join(data_dir, 'backtesting')
            
            if os.path.exists(backtest_dir):
                backtest_files = [f for f in os.listdir(backtest_dir) if f.endswith('.json')]
                
                for file_name in backtest_files:
                    file_path = os.path.join(backtest_dir, file_name)
                    if db_data_manager.sync_file_to_db('backtest_results', file_path):
                        result['backtest_results'] += 1
        
        # 3. 모델 정보 동기화
        if 'models' in sync_types:
            models_dir = os.path.join(os.path.dirname(data_dir), 'models')
            
            if os.path.exists(models_dir):
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                
                for file_name in model_files:
                    file_path = os.path.join(models_dir, file_name)
                    if db_data_manager.sync_file_to_db('models', file_path):
                        result['models'] += 1
        
        # 4. 시장 데이터 동기화
        if 'market_data' in sync_types:
            market_data_dir = os.path.join(os.path.dirname(data_dir), 'data')
            
            if os.path.exists(market_data_dir):
                market_data_files = [f for f in os.listdir(market_data_dir) if f.endswith('.csv')]
                
                for file_name in market_data_files:
                    file_path = os.path.join(market_data_dir, file_name)
                    if db_data_manager.sync_file_to_db('market_data', file_path):
                        result['market_data'] += 1
        
        return result 