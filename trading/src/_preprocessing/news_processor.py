"""
뉴스 감성분석 데이터 처리 모듈
"""
import pandas as pd
import numpy as np
from pymongo import MongoClient
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import DATA_START_DATE, LOGGER

logger = logging.getLogger(__name__)

class NewsProcessor:
    def __init__(self):
        """
        NewsProcessor 초기화
        """
        self.client = MongoClient("mongodb://192.168.40.192:27017/")
        self.db = self.client['polygon']
        self.collection = self.db['bulletin']
        
        # 시작 날짜 설정 (UTC로 변환)
        self.start_date = pd.to_datetime(DATA_START_DATE)
        if self.start_date.tz is None:
            self.start_date = self.start_date.tz_localize('UTC')
        
        # 감성 점수 매핑
        self.sentiment_mapping = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        
        # 발행사별 가중치
        self.publisher_weights = {
            'Bloomberg': 1.2,
            'Reuters': 1.2,
            'CNBC': 1.1,
            'Wall Street Journal': 1.1,
            'Financial Times': 1.1,
            'default': 1.0
        }
        
        LOGGER.info("📰 뉴스 감성분석 프로세서 초기화 완료")
        LOGGER.info(f"   📅 시작 날짜: {self.start_date.strftime('%Y-%m-%d')}")
    
    def _convert_sentiment_to_score(self, sentiment: str) -> float:
        """
        감성 문자열을 수치 점수로 변환
        """
        return self.sentiment_mapping.get(sentiment.lower(), 0.0)
    
    def _ensure_utc(self, dt: Union[datetime, pd.Timestamp]) -> pd.Timestamp:
        """
        datetime을 UTC로 변환하는 헬퍼 함수
        """
        if isinstance(dt, (datetime, pd.Timestamp)):
            dt = pd.to_datetime(dt)
            if dt.tz is None:
                return dt.tz_localize('UTC')
            return dt.tz_convert('UTC')
        return pd.to_datetime(dt).tz_localize('UTC')
    
    def get_news_sentiment(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        특정 심볼의 뉴스 감성 데이터 조회
        
        Args:
            symbol: 주식 심볼
            start_date: 시작 날짜
            end_date: 종료 날짜
            
        Returns:
            뉴스 감성 데이터가 포함된 DataFrame
        """
        # 날짜를 UTC로 변환
        start_date = self._ensure_utc(start_date)
        end_date = self._ensure_utc(end_date)
        
        # MongoDB 쿼리
        query = {
            'name': symbol,
            'published_date_kst': {
                '$gte': start_date,
                '$lte': end_date
            }
        }
        
        # 데이터 조회
        news_data = list(self.collection.find(query))
        
        if not news_data:
            return pd.DataFrame()
        
        # DataFrame 생성
        df = pd.DataFrame(news_data)
        
        # 날짜 컬럼을 UTC로 변환
        df['published_date_kst'] = pd.to_datetime(df['published_date_kst'])
        if df['published_date_kst'].dt.tz is None:
            df['published_date_kst'] = df['published_date_kst'].dt.tz_localize('UTC')
        else:
            df['published_date_kst'] = df['published_date_kst'].dt.tz_convert('UTC')
        
        # 감성 점수 변환
        df['sentiment_score'] = df['sentiment'].apply(self._convert_sentiment_to_score)
        
        # 발행사 가중치 적용
        df['publisher_weight'] = df['publisher'].apply(
            lambda x: self.publisher_weights.get(x, self.publisher_weights['default'])
        )
        
        # 시간 경과에 따른 가중치 계산
        df['time_weight'] = np.exp(-0.1 * (end_date - df['published_date_kst']).dt.total_seconds() / (24 * 3600))
        
        # 최종 가중 감성 점수 계산
        df['weighted_sentiment'] = df['sentiment_score'] * df['publisher_weight'] * df['time_weight']
        
        LOGGER.info(f"📰 {symbol} 뉴스 데이터 {len(df)}개 로드 완료 (기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')})")
        return df
    
    def get_daily_sentiment(self, symbol: str, date: datetime) -> float:
        """
        특정 날짜의 일별 감성 점수 계산
        
        Args:
            symbol: 주식 심볼
            date: 날짜
            
        Returns:
            일별 가중 감성 점수
        """
        # 날짜를 UTC로 변환
        date = self._ensure_utc(date)
        
        # 시작일 이전이면 0 반환
        if date < self.start_date:
            return 0.0
        
        # 해당 날짜의 뉴스 데이터 조회
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        df = self.get_news_sentiment(symbol, start_of_day, end_of_day)
        
        if df.empty:
            return 0.0
        
        # 일별 가중 평균 감성 점수 계산
        daily_sentiment = df['weighted_sentiment'].mean()
        
        return daily_sentiment
    
    def add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        뉴스 감성 분석 피처 추가
        
        Args:
            df: 원본 데이터프레임
            symbol: 주식 심볼
            
        Returns:
            감성 분석 피처가 추가된 데이터프레임
        """
        # 날짜 인덱스를 UTC로 변환
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        else:
            df.index = df.index.tz_convert('UTC')
        
        # 시작일 이전 데이터 처리
        before_start_mask = df.index < self.start_date
        if before_start_mask.any():
            LOGGER.warning(f"⚠️  {before_start_mask.sum()}개 데이터가 시작일({self.start_date.strftime('%Y-%m-%d')}) 이전입니다.")
        
        # 일별 감성 점수 계산
        sentiment_scores = []
        total_days = len(df.index)
        days_with_news = 0
        
        for date in df.index:
            if date < self.start_date:
                sentiment_scores.append(0.0)
            else:
                score = self.get_daily_sentiment(symbol, date)
                sentiment_scores.append(score)
                if score != 0.0:
                    days_with_news += 1
        
        # 감성 점수 추가
        df['news_sentiment'] = sentiment_scores
        
        # 결측치 처리 (가중치를 고려한 forward fill)
        missing_mask = df['news_sentiment'].isna()
        if missing_mask.any():
            LOGGER.info(f"📊 결측치 {missing_mask.sum()}개 발견 - 가중치 기반 forward fill 적용")
            
            def weighted_forward_fill(series):
                """가중치를 고려한 forward fill"""
                result = series.copy()
                last_valid_idx = None
                last_valid_value = None
                
                for idx in range(len(series)):
                    if pd.notna(series.iloc[idx]):
                        last_valid_idx = idx
                        last_valid_value = series.iloc[idx]
                    elif last_valid_value is not None:
                        # 시간 차이에 따른 가중치 계산
                        time_diff = (series.index[idx] - series.index[last_valid_idx]).total_seconds() / (24 * 3600)
                        weight = np.exp(-0.1 * time_diff)  # 시간 경과에 따른 감소
                        result.iloc[idx] = last_valid_value * weight
                
                return result
            
            df['news_sentiment'] = weighted_forward_fill(df['news_sentiment'])
        
        # 감성 기반 파생 지표 추가
        df['sentiment_ma5'] = df['news_sentiment'].rolling(window=5).mean()
        df['sentiment_ma10'] = df['news_sentiment'].rolling(window=10).mean()
        df['sentiment_std5'] = df['news_sentiment'].rolling(window=5).std()
        df['sentiment_change'] = df['news_sentiment'].pct_change()
        df['sentiment_momentum'] = df['news_sentiment'] - df['news_sentiment'].shift(5)
        
        LOGGER.info(f"✅ {symbol} 뉴스 감성분석 피처 추가 완료 (뉴스 데이터 있는 날: {days_with_news}/{total_days}일)")
        return df 