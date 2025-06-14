"""
SAC 트레이딩 대시보드 패키지
"""
import warnings
from src.utils.database import HAS_MYSQL

# DashboardApp 및 DataManager 가져오기
from src.dashboard.dashboard_app import DashboardApp
from src.dashboard.data_manager_file import FileDataManager
from src.dashboard.data_manager_factory import DataManagerFactory

# MySQL이 설치된 경우에만 DBDataManager 가져오기
if HAS_MYSQL:
    from src.dashboard.data_manager_db import DBDataManager
else:
    warnings.warn("MySQL이 설치되지 않았습니다. 데이터베이스 기반 대시보드 기능은 제한됩니다.")

__all__ = ['DashboardApp', 'DataManager', 'DataManagerFactory']
if HAS_MYSQL:
    __all__.append('DBDataManager') 