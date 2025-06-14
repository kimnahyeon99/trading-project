"""
대시보드 데이터 관리 모듈
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import time

from src.utils.logger import Logger


class FileDataManager:
    """
    대시보드에서 사용하는 데이터를 관리하는 클래스
    """
    
    def __init__(
        self, 
        data_dir: str, 
        logger: Optional[Logger] = None
    ):
        """
        DataManager 클래스 초기화
        
        Args:
            data_dir: 데이터 디렉토리 경로
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.data_dir = data_dir
        self.logger = logger
        
        # 데이터 캐시
        self.trading_stats_cache = {}
        self.backtest_results_cache = {}
        self.model_info_cache = {}
        self.market_data_cache = {}
        
        # 캐시 마지막 업데이트 시간
        self.last_update = {
            'trading_stats': datetime.min,
            'backtest_results': datetime.min,
            'model_info': datetime.min,
            'market_data': datetime.min
        }
        
        # 캐시 유효 기간 (초)
        self.cache_ttl = {
            'trading_stats': 60,  # 1분
            'backtest_results': 3600,  # 1시간
            'model_info': 3600,  # 1시간
            'market_data': 300  # 5분
        }
        
        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"FileDataManager 초기화 완료: {data_dir}")
    
    def get_trading_stats(self, refresh: bool = False) -> Dict[str, Any]:
        """
        트레이딩 통계 조회
        
        Args:
            refresh: 캐시 갱신 여부
            
        Returns:
            트레이딩 통계 정보
        """
        current_time = datetime.now()
        
        # 캐시 만료 또는 강제 갱신 확인
        cache_age = (current_time - self.last_update['trading_stats']).total_seconds()
        if refresh or cache_age > self.cache_ttl['trading_stats'] or not self.trading_stats_cache:
            try:
                # 트레이딩 통계 파일 조회
                stats_dir = os.path.join(self.data_dir, 'live_trading')
                
                if not os.path.exists(stats_dir):
                    if self.logger:
                        self.logger.warning(f"트레이딩 통계 디렉토리가 없습니다: {stats_dir}")
                    return {}
                
                # 가장 최근 통계 파일 찾기
                stats_files = [f for f in os.listdir(stats_dir) if f.endswith('.json')]
                
                if not stats_files:
                    if self.logger:
                        self.logger.warning("트레이딩 통계 파일이 없습니다.")
                    return {}
                
                # 파일 이름에서 타임스탬프 추출하여 정렬
                latest_file = sorted(stats_files, key=lambda x: x.split('_')[-1].split('.')[0] if '_' in x else '0')[-1]
                stats_path = os.path.join(stats_dir, latest_file)
                
                # 최신 통계 파일 로드
                with open(stats_path, 'r') as f:
                    stats_data = json.load(f)
                
                # 캐시 업데이트
                self.trading_stats_cache = stats_data
                self.last_update['trading_stats'] = current_time
                
                if self.logger:
                    self.logger.info(f"트레이딩 통계 로드 완료: {stats_path}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"트레이딩 통계 로드 중 오류 발생: {e}")
                # 오류 발생 시 기존 캐시 유지
        
        return self.trading_stats_cache
    
    def get_backtest_results(self, model_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        """
        백테스트 결과 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            refresh: 캐시 갱신 여부
            
        Returns:
            백테스트 결과 정보
        """
        current_time = datetime.now()
        cache_key = model_id if model_id else 'all'
        
        # 캐시 만료 또는 강제 갱신 확인
        if cache_key not in self.backtest_results_cache:
            refresh = True
        else:
            cache_age = (current_time - self.last_update['backtest_results']).total_seconds()
            if refresh or cache_age > self.cache_ttl['backtest_results']:
                refresh = True
        
        if refresh:
            try:
                # 백테스트 결과 디렉토리
                backtest_dir = os.path.join(self.data_dir, 'backtesting')
                
                if not os.path.exists(backtest_dir):
                    if self.logger:
                        self.logger.warning(f"백테스트 결과 디렉토리가 없습니다: {backtest_dir}")
                    return {}
                
                # 결과 파일 조회
                if model_id:
                    # 특정 모델 결과만 조회
                    result_files = [f for f in os.listdir(backtest_dir) 
                                   if f.endswith('.json') and model_id in f]
                    if self.logger:
                        self.logger.info(f"모델 ID '{model_id}'에 대한 백테스트 파일: {result_files}")
                else:
                    # 모든 결과 조회
                    result_files = [f for f in os.listdir(backtest_dir) if f.endswith('.json')]
                    if self.logger:
                        self.logger.info(f"모든 백테스트 파일: {result_files}")
                
                if not result_files:
                    if self.logger:
                        self.logger.warning(f"백테스트 결과 파일이 없습니다: {model_id if model_id else '모든 모델'}")
                    return {}
                
                # 결과 로드
                results = {}
                for file_name in result_files:
                    file_path = os.path.join(backtest_dir, file_name)
                    with open(file_path, 'r') as f:
                        result_data = json.load(f)
                    
                    # 결과 파일에서 모델 ID 추출 수정
                    # 파일 이름에서 추출 시도하되, JSON 데이터에 model_id 필드가 있으면 그것을 우선 사용
                    if 'model_id' in result_data:
                        model_id_from_file = result_data['model_id']
                    else:
                        # 파일 이름 형식: backtest_model_sac_v1_20240510.json
                        parts = file_name.split('_')
                        if len(parts) >= 3 and parts[0] == 'backtest' and parts[1] == 'model':
                            # "model_" 이후, 날짜 부분 이전까지 추출
                            date_part = next((p for p in parts if p.startswith('202')), None)
                            if date_part:
                                date_index = parts.index(date_part)
                                model_id_from_file = '_'.join(parts[2:date_index])
                            else:
                                model_id_from_file = '_'.join(parts[2:]).split('.')[0]
                        else:
                            model_id_from_file = file_name.split('.')[0]
                    
                    if self.logger:
                        self.logger.info(f"파일 '{file_name}'에서 추출한 모델 ID: {model_id_from_file}")
                    
                    results[model_id_from_file] = result_data
                
                # 캐시 업데이트
                self.backtest_results_cache[cache_key] = results
                self.last_update['backtest_results'] = current_time
                
                if self.logger:
                    self.logger.info(f"백테스트 결과 로드 완료: {len(results)}개")
                    self.logger.info(f"로드된 모델 ID 목록: {list(results.keys())}")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"백테스트 결과 로드 중 오류 발생: {e}")
                # 오류 발생 시 기존 캐시 유지 또는 빈 딕셔너리 반환
                if cache_key not in self.backtest_results_cache:
                    self.backtest_results_cache[cache_key] = {}
        
        return self.backtest_results_cache.get(cache_key, {})
    
    def get_model_info(self, refresh: bool = False) -> Dict[str, Any]:
        """
        학습된 모델 정보 조회
        
        Args:
            refresh: 캐시 갱신 여부
            
        Returns:
            모델 정보 목록
        """
        current_time = datetime.now()
        
        # 캐시 만료 또는 강제 갱신 확인
        cache_age = (current_time - self.last_update['model_info']).total_seconds()
        if refresh or cache_age > self.cache_ttl['model_info'] or not self.model_info_cache:
            try:
                # 모델 디렉토리
                models_dir = os.path.join(os.path.dirname(self.data_dir), 'models')
                
                if not os.path.exists(models_dir):
                    if self.logger:
                        self.logger.warning(f"모델 디렉토리가 없습니다: {models_dir}")
                    return {}
                
                # 모델 파일 조회
                model_files = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
                
                if not model_files:
                    if self.logger:
                        self.logger.warning("학습된 모델 파일이 없습니다.")
                    return {}
                
                # 모델 정보 수집
                model_info = {}
                for file_name in model_files:
                    model_path = os.path.join(models_dir, file_name)
                    
                    # 파일 이름에서 모델 ID 추출
                    model_id = file_name.split('.')[0]
                    
                    # 파일 정보 추출
                    file_stats = os.stat(model_path)
                    created_time = datetime.fromtimestamp(file_stats.st_ctime)
                    modified_time = datetime.fromtimestamp(file_stats.st_mtime)
                    file_size = file_stats.st_size
                    
                    # 모델 정보 저장
                    model_info[model_id] = {
                        'model_id': model_id,
                        'file_name': file_name,
                        'file_path': model_path,
                        'created_time': created_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'modified_time': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'file_size': file_size,
                        'file_size_mb': round(file_size / (1024 * 1024), 2)
                    }
                
                # 캐시 업데이트
                self.model_info_cache = model_info
                self.last_update['model_info'] = current_time
                
                if self.logger:
                    self.logger.info(f"모델 정보 로드 완료: {len(model_info)}개")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"모델 정보 로드 중 오류 발생: {e}")
                # 오류 발생 시 기존 캐시 유지
        
        return self.model_info_cache
    
    def get_market_data(self, symbol: str, interval: str = '1d', limit: int = 100, refresh: bool = False) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol: 심볼/티커
            interval: 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: 가져올 데이터 개수
            refresh: 캐시 갱신 여부
            
        Returns:
            시장 데이터 DataFrame
        """
        current_time = datetime.now()
        cache_key = f"{symbol}_{interval}_{limit}"
        
        # 캐시 만료 또는 강제 갱신 확인
        if cache_key not in self.market_data_cache:
            refresh = True
        else:
            cache_age = (current_time - self.last_update['market_data']).total_seconds()
            if refresh or cache_age > self.cache_ttl['market_data']:
                refresh = True
        
        if refresh:
            try:
                # 시장 데이터 파일 조회
                market_data_dir = os.path.join(os.path.dirname(self.data_dir), 'data')
                
                if not os.path.exists(market_data_dir):
                    if self.logger:
                        self.logger.warning(f"시장 데이터 디렉토리가 없습니다: {market_data_dir}")
                    return pd.DataFrame()
                
                # 심볼에 해당하는 데이터 파일 찾기
                data_file = None
                for file_name in os.listdir(market_data_dir):
                    if file_name.startswith(f"{symbol}_") and file_name.endswith('.csv'):
                        data_file = os.path.join(market_data_dir, file_name)
                        break
                
                if not data_file:
                    if self.logger:
                        self.logger.warning(f"{symbol} 시장 데이터 파일이 없습니다.")
                    return pd.DataFrame()
                
                # 데이터 로드
                df = pd.read_csv(data_file)
                
                # 날짜 열을 datetime으로 변환
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                elif 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                # 최신 데이터만 선택
                if limit > 0 and len(df) > limit:
                    df = df.tail(limit)
                
                # 캐시 업데이트
                self.market_data_cache[cache_key] = df
                self.last_update['market_data'] = current_time
                
                if self.logger:
                    self.logger.info(f"{symbol} 시장 데이터 로드 완료: {len(df)} 행")
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"{symbol} 시장 데이터 로드 중 오류 발생: {e}")
                # 오류 발생 시 기존 캐시 유지 또는 빈 DataFrame 반환
                if cache_key not in self.market_data_cache:
                    self.market_data_cache[cache_key] = pd.DataFrame()
        
        return self.market_data_cache.get(cache_key, pd.DataFrame())
    
    def get_performance_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 성능 지표 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            
        Returns:
            성능 지표 정보
        """
        # 백테스트 결과 조회
        backtest_results = self.get_backtest_results(model_id)
        
        # 성능 지표 추출
        metrics = {}
        
        for model_id, result in backtest_results.items():
            if 'metrics' in result:
                metrics[model_id] = result['metrics']
            elif 'performance' in result:
                metrics[model_id] = result['performance']
            else:
                # 기본 지표 구조 생성
                metrics[model_id] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'profit_factor': result.get('profit_factor', 0)
                }
        
        return metrics
    
    def clear_cache(self, cache_type: Optional[str] = None) -> None:
        """
        캐시 초기화
        
        Args:
            cache_type: 캐시 유형 (옵션, 특정 캐시만 초기화)
        """
        if cache_type is None or cache_type == 'trading_stats':
            self.trading_stats_cache = {}
            self.last_update['trading_stats'] = datetime.min
            
        if cache_type is None or cache_type == 'backtest_results':
            self.backtest_results_cache = {}
            self.last_update['backtest_results'] = datetime.min
            
        if cache_type is None or cache_type == 'model_info':
            self.model_info_cache = {}
            self.last_update['model_info'] = datetime.min
            
        if cache_type is None or cache_type == 'market_data':
            self.market_data_cache = {}
            self.last_update['market_data'] = datetime.min
            
        if self.logger:
            cache_type_str = f"{cache_type} " if cache_type else ""
            self.logger.info(f"{cache_type_str}캐시가 초기화되었습니다.") 