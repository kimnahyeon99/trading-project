"""
최적화된 데이터 전처리 모듈 - 기본 OHLCV + 핵심 지표만 생성
feature_selector가 중복 제거를 담당하도록 역할 분담
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from sklearn.preprocessing import RobustScaler, StandardScaler
import talib as ta
from datetime import datetime
from pathlib import Path
from src.preprocessing.feature_selector import FeatureSelector
from src.preprocessing.news_processor import NewsProcessor
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)
from src.config.config import (
    WINDOW_SIZE,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    DATA_DIR,
    LOGGER
)
from src.utils.utils import create_directory, save_to_csv, load_from_csv

class DataProcessor:
    """
    주식 데이터 전처리 클래스 - 효율적인 피처 생성 + 스마트한 중복 제거
    """
    
    def __init__(self, window_size: int = WINDOW_SIZE, enable_feature_selection: bool = True, 
                 correlation_threshold: float = 0.85, variance_threshold: float = 0.0001
                #  ,enable_news_sentiment: bool = True
                 ):
        """
        DataProcessor 클래스 초기화
        
        Args:
            window_size: 관측 윈도우 크기
            enable_feature_selection: 피처 선택 활성화 여부
            correlation_threshold: 상관관계 임계값 (85% 이상이면 중복으로 간주)
            variance_threshold: 분산 임계값 (0.0001 이하면 변동성 없는 피처로 간주)
            enable_news_sentiment: 뉴스 감성분석 활성화 여부
        """
        self.window_size = window_size
        self.scalers = {}
        self.normalized_data_columns = {}
        self.enable_feature_selection = enable_feature_selection
        # self.enable_news_sentiment = enable_news_sentiment
        
        # 필수 보존 컬럼들 (절대 제거되면 안 되는 컬럼들)
        self.preserve_columns = ['open', 'high', 'low', 'close', 'volume']
        
        self.feature_selector = FeatureSelector(
            correlation_threshold=correlation_threshold,
            variance_threshold=variance_threshold,
            preserve_columns=self.preserve_columns,
            remove_duplicates_only=True
        ) if enable_feature_selection else None
        
        # 뉴스 감성분석 프로세서 초기화
        # self.news_processor = NewsProcessor() if enable_news_sentiment else None
        
        LOGGER.info(f"📊 효율적인 DataProcessor 초기화 완료")
        LOGGER.info(f"   🔧 윈도우 크기: {window_size}")
        LOGGER.info(f"   🎯 피처 선택: {'활성화' if enable_feature_selection else '비활성화'}")
        LOGGER.info(f"   📏 상관관계 임계값: {correlation_threshold}")
        LOGGER.info(f"   📐 분산 임계값: {variance_threshold}")
        LOGGER.info(f"   🔒 필수 보존 컬럼: {self.preserve_columns}")
        # LOGGER.info(f"   📰 뉴스 감성분석: {'활성화' if enable_news_sentiment else '비활성화'}")
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        주식 데이터 전처리 (결측치 처리, 이상치 제거 등)
        """
        if data.empty:
            LOGGER.warning("빈 데이터프레임이 입력되었습니다.")
            return data

        # ✅ 복사 후 정렬 (순서 중요)
        data = data.copy()
        data = data.sort_index()  # 시간 오름차순 정렬

        # ✅ 메타 컬럼 제거
        meta_cols = ['symbol', 'ticker', 'asset', 'exchange']
        data.drop(columns=[col for col in meta_cols if col in data.columns], inplace=True)

        LOGGER.info(f"전처리 시작 - 컬럼 목록: {list(data.columns)}")

        # ✅ 결측치 채우기
        data.fillna(method='ffill', inplace=True)
        data.fillna(method='bfill', inplace=True)

        # ✅ 중복 인덱스 제거
        data = data[~data.index.duplicated(keep='first')]

        # ✅ 필수 컬럼 존재 여부 및 처리
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                upper_col = col.upper()
                if upper_col in data.columns:
                    data[col] = data.pop(upper_col)
                    LOGGER.warning(f"{col} 컬럼을 {upper_col}에서 가져와 생성했습니다.")
                else:
                    LOGGER.warning(f"데이터에 필수 컬럼 '{col}'이 없습니다.")

        # ✅ 데이터 타입 정리
        for col in required_cols:
            if col in data.columns:
                try:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    LOGGER.debug(f"{col} 컬럼을 float64 타입으로 변환했습니다")
                except Exception as e:
                    LOGGER.error(f"{col} 컬럼 변환 오류: {e}")

        # ✅ 0 이하 값 처리
        for col in ['open', 'high', 'low', 'close']:
            if col in data.columns:
                mask = data[col] <= 0
                if mask.any():
                    LOGGER.warning(f"{col} 열에 0 이하 값 {mask.sum()}개 → 이전 값으로 대체")
                    data.loc[mask, col] = data[col].shift(1)[mask]

        # ✅ 마지막 결측치 점검
        if data.isna().any().any():
            LOGGER.warning(f"남은 결측치 {data.isna().sum().sum()}개 존재 → 추가 처리")
            data.fillna(method='ffill', inplace=True)
            data.fillna(method='bfill', inplace=True)
            for col in data.columns:
                if data[col].isna().any():
                    if data[col].dtype.kind in 'iufc':
                        data[col].fillna(data[col].mean(), inplace=True)
                    else:
                        data[col].fillna(0, inplace=True)

        LOGGER.info(f"데이터 전처리 완료: {len(data)} 행, {len(data.columns)} 컬럼")
        return data
        
    def extract_features(self, data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        핵심 기술적 지표만 추가 - 효율성 중심 접근
        기본 OHLCV + 핵심 지표만 생성하여 feature_selector가 최적화하도록 함
        
        Args:
            data: 전처리된 주식 데이터 데이터프레임
            symbol: 주식 심볼 (뉴스 감성분석용)
            
        Returns:
            핵심 특성이 추가된 데이터프레임
        """
        if data.empty:
            LOGGER.warning("빈 데이터프레임이 입력되었습니다.")
            return data
        
        df = data.copy()
        
        # 기본 가격 데이터 확인
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            LOGGER.error("필수 OHLCV 컬럼이 누락되었습니다.")
            return df
        
        open_prices = df['open'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        close_prices = df['close'].values
        volumes = df['volume'].values
        
        LOGGER.info("핵심 기술적 지표 계산 시작...")
        
        try:
            # =============================================================================
            # 📊 1. 이동평균선 (다양한 기간) - 트렌드 파악용
            # =============================================================================
            LOGGER.debug("이동평균선 계산 중...")
            df['sma_5'] = self._safe_ta(ta.SMA, close_prices, timeperiod=5)
            df['sma_10'] = self._safe_ta(ta.SMA, close_prices, timeperiod=10)
            df['sma_20'] = self._safe_ta(ta.SMA, close_prices, timeperiod=20)
            df['sma_50'] = self._safe_ta(ta.SMA, close_prices, timeperiod=50)
            df['sma_200'] = self._safe_ta(ta.SMA, close_prices, timeperiod=200)
            
            # 지수이동평균
            df['ema_12'] = self._safe_ta(ta.EMA, close_prices, timeperiod=12)
            df['ema_26'] = self._safe_ta(ta.EMA, close_prices, timeperiod=26)
            df['ema_50'] = self._safe_ta(ta.EMA, close_prices, timeperiod=50)
            
            # =============================================================================
            # 📈 2. 모멘텀 지표 - 매매 신호용
            # =============================================================================
            LOGGER.debug("모멘텀 지표 계산 중...")
            
            # MACD
            macd, macd_signal, macd_hist = self._safe_ta(ta.MACD, close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            
            # RSI
            df['rsi_14'] = self._safe_ta(ta.RSI, close_prices, timeperiod=14)
            df['rsi_30'] = self._safe_ta(ta.RSI, close_prices, timeperiod=30)
            
            # Stochastic
            slowk, slowd = self._safe_ta(ta.STOCH, high_prices, low_prices, close_prices, 
                                         fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
            
            # Williams %R
            df['williams_r'] = self._safe_ta(ta.WILLR, high_prices, low_prices, close_prices, timeperiod=14)
            
            # =============================================================================
            # 📊 3. 변동성 지표 - 리스크 관리용
            # =============================================================================
            LOGGER.debug("변동성 지표 계산 중...")
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._safe_ta(ta.BBANDS, close_prices, timeperiod=20, nbdevup=2, nbdevdn=2)
            df['bb_upper'] = bb_upper
            df['bb_middle'] = bb_middle
            df['bb_lower'] = bb_lower
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle
            df['bb_position'] = (close_prices - bb_lower) / (bb_upper - bb_lower)
            
            # ATR (Average True Range)
            df['atr_14'] = self._safe_ta(ta.ATR, high_prices, low_prices, close_prices, timeperiod=14)
            df['atr_30'] = self._safe_ta(ta.ATR, high_prices, low_prices, close_prices, timeperiod=30)
            
            # =============================================================================
            # 🎯 4. 추세 지표 - 방향성 파악용
            # =============================================================================
            LOGGER.debug("추세 지표 계산 중...")
            
            # ADX (Average Directional Index)
            df['adx'] = self._safe_ta(ta.ADX, high_prices, low_prices, close_prices, timeperiod=14)
            df['di_plus'] = self._safe_ta(ta.PLUS_DI, high_prices, low_prices, close_prices, timeperiod=14)
            df['di_minus'] = self._safe_ta(ta.MINUS_DI, high_prices, low_prices, close_prices, timeperiod=14)
            
            # CCI (Commodity Channel Index)
            df['cci_14'] = self._safe_ta(ta.CCI, high_prices, low_prices, close_prices, timeperiod=14)
            
            # =============================================================================
            # 💰 5. 거래량 지표 - 강도 측정용
            # =============================================================================
            LOGGER.debug("거래량 지표 계산 중...")
            
            # OBV (On Balance Volume)
            df['obv'] = self._safe_ta(ta.OBV, close_prices, volumes)
            
            # MFI (Money Flow Index)
            df['mfi_14'] = self._safe_ta(ta.MFI, high_prices, low_prices, close_prices, volumes, timeperiod=14)
            
            # Volume moving averages
            df['volume_sma_10'] = self._safe_ta(ta.SMA, volumes, timeperiod=10)
            df['volume_sma_30'] = self._safe_ta(ta.SMA, volumes, timeperiod=30)
            
            # =============================================================================
            # 🔧 6. 기본 파생 지표 - 수학적 변환
            # =============================================================================
            LOGGER.debug("기본 파생 지표 계산 중...")
            
            # 수익률 지표
            df['returns_1d'] = df['close'].pct_change(1)
            df['returns_5d'] = df['close'].pct_change(5)
            df['returns_10d'] = df['close'].pct_change(10)
            
            # 가격 비율
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # 거래량 변화율
            df['volume_change'] = df['volume'].pct_change(1)
            df['volume_ratio_10d'] = df['volume'] / df['volume_sma_10']
            
            # 변동성 지표
            df['price_volatility'] = (df['high'] - df['low']) / df['close']
            
            # =============================================================================
            # 🕐 7. 시간 기반 패턴 (기본적인 순환 패턴)
            # =============================================================================
            LOGGER.debug("시간 패턴 계산 중...")
            
            # 시간 순환 패턴 (if timestamp available)
            if hasattr(df.index, 'hour'):
                df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            else:
                df['hour_sin'] = 0
                df['hour_cos'] = 1
                
            if hasattr(df.index, 'dayofweek'):
                df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
                df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
            else:
                df['day_sin'] = 0
                df['day_cos'] = 1
            
            # =============================================================================
            # 🎯 8. 트레이딩 신호 지표 (간단한 조합)
            # =============================================================================
            LOGGER.debug("트레이딩 신호 계산 중...")
            
            # 이동평균 교차
            df['sma_5_above_20'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['sma_20_above_50'] = (df['sma_20'] > df['sma_50']).astype(int)
            
            # 가격 위치 지표
            df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
            df['price_above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
            
            # RSI 레벨
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            
            # 뉴스 감성분석 피처 추가
            # if self.enable_news_sentiment and self.news_processor is not None and symbol is not None:
            #     LOGGER.info(f"📰 {symbol} 뉴스 감성분석 피처 추가 중... (기간: {df.index[0].strftime('%Y-%m-%d')} ~ {df.index[-1].strftime('%Y-%m-%d')})")
            #     df = self.news_processor.add_sentiment_features(df, symbol)
            #     LOGGER.info(f"✅ {symbol} 뉴스 감성분석 피처 추가 완료")
            
        except Exception as e:
            LOGGER.error(f"기술적 지표 계산 중 오류 발생: {str(e)}")
            # 기본 지표라도 추가
            df['returns_1d'] = df['close'].pct_change(1)
            df['price_volatility'] = (df['high'] - df['low']) / df['close']
        
        # 이상치 및 무한대 값 처리
        df = self._handle_outliers_and_infinities(df)
        
        # NaN 값 처리
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        df.fillna(0, inplace=True)
        
        # 🔒 **중요**: close 컬럼이 마지막에 위치하도록 보장
        df = self._ensure_close_column_last(df)
        
        initial_features = len(df.columns) - 5  # OHLCV 제외
        LOGGER.info(f"핵심 기술적 지표 추가 완료: {initial_features}개 파생 지표 생성")
        LOGGER.info(f"총 컬럼 수: {len(df.columns)}개 (OHLCV + {initial_features} 파생지표)")
        
        return df
    
    def _ensure_close_column_last(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        close 컬럼이 마지막에 위치하도록 보장
        trading_env에서 [-1] 인덱스로 접근할 수 있도록 함
        """
        if 'close' in df.columns:
            # close 컬럼을 제외한 모든 컬럼
            other_columns = [col for col in df.columns if col != 'close']
            # close를 마지막에 배치
            reordered_columns = other_columns + ['close']
            df = df[reordered_columns]
            LOGGER.debug("close 컬럼을 마지막 위치로 이동 완료")
        else:
            LOGGER.warning("close 컬럼을 찾을 수 없습니다.")
        
        return df
    
    def _safe_ta(self, ta_func, *args, **kwargs):
        """
        TA-Lib 함수를 안전하게 호출하는 헬퍼 함수
        """
        try:
            # 입력 배열 타입 확인 및 강제 변환
            new_args = []
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if arg.dtype != np.float64:
                        arg = arg.astype(np.float64)
                    
                    # NaN 값 처리
                    if np.isnan(arg).any():
                        mean_val = np.nanmean(arg) if np.any(~np.isnan(arg)) else 0
                        arg = np.nan_to_num(arg, nan=mean_val)
                new_args.append(arg)
            
            result = ta_func(*new_args, **kwargs)
            return result
        except Exception as e:
            LOGGER.warning(f"기술적 지표 계산 실패 ({ta_func.__name__}): {str(e)}")
            # 다중 출력 함수인지 확인
            if ta_func.__name__ in ['BBANDS', 'MACD', 'STOCH', 'AROON']:
                shapes = [len(args[0])] * 3
                return [np.full(shape, np.nan) for shape in shapes]
            else:
                return np.full(len(args[0]), np.nan)
    
    def _handle_outliers_and_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        이상치 및 무한대 값 처리 (보수적 접근)
        """
        # 무한대 값 처리
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # 필수 컬럼은 제외하고 이상치 처리
        protected_cols = self.preserve_columns
        
        for col in df.columns:
            if col in protected_cols:
                continue  # 필수 컬럼은 이상치 처리 제외
                
            if df[col].dtype == np.float64 or df[col].dtype == np.int64:
                # 결측치가 너무 많은 컬럼은 건너뜀
                if df[col].isna().sum() > len(df) * 0.5:
                    continue
                    
                # 99.5%/0.5% 경계값으로 클리핑
                q_low = df[col].quantile(0.005)
                q_high = df[col].quantile(0.995)
                df[col] = df[col].clip(lower=q_low, upper=q_high)
        
        # 남은 결측치 처리
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        # 검증
        nan_counts = df.isna().sum().sum()
        if nan_counts > 0:
            LOGGER.warning(f"결측치 {nan_counts}개가 여전히 존재합니다")
        
        numeric_data = df.select_dtypes(include=[np.number])
        inf_counts = np.isinf(numeric_data.values).sum()
        if inf_counts > 0:
            LOGGER.warning(f"무한대 값 {inf_counts}개가 여전히 존재합니다")
            df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def normalize_features(self, data: pd.DataFrame, symbol: str, is_training: bool = True) -> pd.DataFrame:
        """
        특성 정규화 (필수 컬럼 보존)
        """
        if data.empty:
            return data
        
        df = data.copy()
        
        # 정규화에서 제외할 컬럼 (필수 보존 + 날짜)
        exclude_cols = self.preserve_columns.copy()
        if 'date' in df.columns:
            exclude_cols.append('date')
        
        # 정규화할 컬럼 (파생 지표들만)
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]
        
        if not cols_to_normalize:
            LOGGER.warning("정규화할 컬럼이 없습니다.")
            return df
        
        if is_training:
            # 학습 데이터인 경우 새 스케일러 생성
            if symbol not in self.scalers:
                self.scalers[symbol] = {}
            
            scaler = RobustScaler()
            self.scalers[symbol]['scaler'] = scaler
            normalized_data = scaler.fit_transform(df[cols_to_normalize])
        else:
            # 테스트 데이터인 경우 기존 스케일러 사용
            if symbol not in self.scalers or 'scaler' not in self.scalers[symbol]:
                LOGGER.warning(f"{symbol}에 대한 스케일러가 없습니다. 새로 생성합니다.")
                scaler = RobustScaler()
                if symbol not in self.scalers:
                    self.scalers[symbol] = {}
                self.scalers[symbol]['scaler'] = scaler
                normalized_data = scaler.fit_transform(df[cols_to_normalize])
            else:
                scaler = self.scalers[symbol]['scaler']
                normalized_data = scaler.transform(df[cols_to_normalize])
        
        # 정규화된 데이터로 데이터프레임 생성
        normalized_df = df[exclude_cols].copy()  # 필수 컬럼 먼저 복사
        
        # 정규화된 파생 지표들 추가
        normalized_features_df = pd.DataFrame(
            normalized_data, 
            index=df.index, 
            columns=cols_to_normalize
        )
        
        # 합치기 (필수 컬럼 + 정규화된 파생 지표)
        result_df = pd.concat([normalized_df, normalized_features_df], axis=1)
        
        # 🔒 close가 마지막 위치에 있는지 재확인
        result_df = self._ensure_close_column_last(result_df)
        
        LOGGER.debug(f"정규화 완료: {len(exclude_cols)}개 필수 컬럼 보존, {len(cols_to_normalize)}개 컬럼 정규화")
        
        return result_df

    def split_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """데이터 분할"""
        if data.empty:
            LOGGER.warning("빈 데이터가 입력되었습니다.")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
        # 데이터 분할 인덱스 계산
        train_idx = int(len(data) * TRAIN_RATIO)
        valid_idx = int(len(data) * (TRAIN_RATIO + VALID_RATIO))
    
        # 시간 순서대로 분할 (미래 데이터 누수 방지)
        train = data.iloc[:train_idx].copy()
        valid = data.iloc[train_idx:valid_idx].copy()
        test = data.iloc[valid_idx:].copy()
    
        LOGGER.info(f"데이터 분할 완료: 학습 {len(train)}개, 검증 {len(valid)}개, 테스트 {len(test)}개")
        return train, valid, test
    
    def process_symbol_data(self, data: pd.DataFrame, symbol: str, use_windows: bool = False) -> Dict[str, Any]:
        """
        단일 심볼 데이터에 대한 전체 전처리 과정 수행 (효율적인 피처 선택 포함)
        """
        LOGGER.info(f"{symbol} 데이터 전처리 시작")
        
        # 1. 데이터 전처리
        processed_data = self.preprocess_data(data)
        if processed_data.empty:
            LOGGER.error(f"{symbol} 데이터 전처리 실패")
            return {}
        
        # 2. 핵심 특성 추출 (기본 OHLCV + 핵심 지표만)
        featured_data = self.extract_features(processed_data, symbol)
        
        # 3. 스마트한 피처 선택 (활성화된 경우만)
        if self.enable_feature_selection and self.feature_selector is not None:
            initial_count = len(featured_data.columns)
            LOGGER.info(f"{symbol} 스마트 피처 선택 시작: {initial_count}개 피처")
            featured_data = self.feature_selector.fit_transform(featured_data, target_col='close')
            final_count = len(featured_data.columns)
            removed_count = initial_count - final_count
            LOGGER.info(f"{symbol} 피처 선택 완료: {removed_count}개 중복 제거, {final_count}개 최종 피처")
        
        # 4. 특성 정규화
        normalized_data = self.normalize_features(featured_data, symbol, is_training=True)
        
        # 5. 데이터 분할
        train, valid, test = self.split_data(normalized_data)
        
        # 결과 반환
        result = {
            'processed_data': processed_data,
            'featured_data': featured_data,
            'normalized_data': normalized_data,
            'train': train,
            'valid': valid,
            'test': test
        }
        
        # 선택된 피처 정보 추가
        if self.feature_selector is not None and hasattr(self.feature_selector, 'selected_features'):
            result['selected_features'] = self.feature_selector.selected_features
            result['removed_features_info'] = getattr(self.feature_selector, 'removed_features_info', {})
        
        LOGGER.info(f"{symbol} 효율적인 데이터 전처리 완료")
        return result
    
    def process_all_symbols(self, raw_data_dict: Dict[str, pd.DataFrame], use_windows: bool = False) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        모든 심볼 데이터 처리 (효율적인 피처 선택 포함)
        """
        # 피처 선택이 비활성화되어 있으면 자동으로 활성화
        if not self.enable_feature_selection:
            LOGGER.info("⚠️  피처 선택이 비활성화되어 있어 자동으로 활성화합니다.")
            self.enable_feature_selection = True
            self.feature_selector = FeatureSelector(
                correlation_threshold=0.85, 
                variance_threshold=0.0001,
                preserve_columns=self.preserve_columns,
                remove_duplicates_only=True
            )
        
        results = {}
        first_symbol = True

        LOGGER.info(f"🔄 {len(raw_data_dict)}개 심볼 효율적인 데이터 처리 시작...")
        
        for i, (symbol, raw_data) in enumerate(raw_data_dict.items(), 1):
            LOGGER.info(f"[{i}/{len(raw_data_dict)}] {symbol} 처리 중...")

            # 1. 전처리 및 핵심 특성 추출
            preprocessed = self.preprocess_data(raw_data)
            featured = self.extract_features(preprocessed, symbol)
            
            # 2. 스마트한 피처 선택
            if first_symbol:
                initial_features = len(featured.columns)
                LOGGER.info(f"   📋 스마트 피처 선택 중: {initial_features}개 피처에서...")
                featured = self.feature_selector.fit_transform(featured, target_col='close')
                final_features = len(featured.columns)
                removed_features = initial_features - final_features
                first_symbol = False
                LOGGER.info(f"   ✨ 피처 선택 완료: {removed_features}개 중복 제거, {final_features}개 최종 피처")
            else:
                LOGGER.info(f"   🔄 동일한 피처 기준 적용...")
                featured = self.feature_selector.transform(featured, target_col='close')

            # 3. 데이터 분할 및 정규화
            train_data = featured.iloc[:int(len(featured)*TRAIN_RATIO)].copy()
            valid_data = featured.iloc[int(len(featured)*TRAIN_RATIO):int(len(featured)*(TRAIN_RATIO + VALID_RATIO))].copy()
            test_data = featured.iloc[int(len(featured)*(TRAIN_RATIO + VALID_RATIO)):].copy()

            train_data_norm = self.normalize_features(train_data, symbol=symbol, is_training=True)
            valid_data_norm = self.normalize_features(valid_data, symbol=symbol, is_training=False)
            test_data_norm = self.normalize_features(test_data, symbol=symbol, is_training=False)

            results[symbol] = {
                'processed_data': preprocessed,
                'featured_data': featured,
                'train': train_data_norm,
                'valid': valid_data_norm,
                'test': test_data_norm
            }

            LOGGER.info(f"   ✅ 완료: train {len(train_data)}, valid {len(valid_data)}, test {len(test_data)}")

        # 최종 피처 요약 출력
        if self.feature_selector and hasattr(self.feature_selector, 'selected_features'):
            LOGGER.info(f"\n🎯 최종 효율적인 피처셋 요약:")
            LOGGER.info("=" * 60)
            
            removed_info = getattr(self.feature_selector, 'removed_features_info', {})
            
            LOGGER.info(f"📊 제거 요약:")
            total_removed = 0
            for category, removed_list in removed_info.items():
                if removed_list:
                    total_removed += len(removed_list)
                    LOGGER.info(f"   - {category.replace('_', ' ').title()}: {len(removed_list)}개")
            
            LOGGER.info(f"\n📈 최종 유효 피처들 ({len(self.feature_selector.selected_features)}개):")
            
            # 필수 보존 피처들
            preserved_features = [f for f in self.feature_selector.selected_features if f in self.preserve_columns]
            derived_features = [f for f in self.feature_selector.selected_features if f not in self.preserve_columns]
            
            LOGGER.info(f"\n🔒 필수 보존 피처 ({len(preserved_features)}개):")
            for i, feature in enumerate(preserved_features, 1):
                LOGGER.info(f"{i:2d}. {feature}")
            
            LOGGER.info(f"\n⚡ 선별된 파생 피처 ({len(derived_features)}개):")
            for i, feature in enumerate(derived_features, 1):
                LOGGER.info(f"{i:2d}. {feature}")
            
            LOGGER.info("=" * 60)

        return results
    
    def save_processed_data(self, results: Dict[str, Dict[str, Any]], base_dir: Union[str, Path] = None) -> None:
        """
        전처리된 데이터를 파일로 저장
        """
        if base_dir is None:
            base_dir = DATA_DIR / "processed"
        
        create_directory(base_dir)
        
        for symbol, result in results.items():
            # 데이터프레임만 저장
            for key in ['processed_data', 'featured_data', 'normalized_data', 'train', 'valid', 'test']:
                if key in result and isinstance(result[key], pd.DataFrame):
                    save_dir = base_dir / symbol
                    create_directory(save_dir)
                    file_path = save_dir / f"{key}.csv"
                    save_to_csv(result[key], file_path)
        
        LOGGER.info(f"효율적으로 전처리된 데이터 저장 완료: {base_dir}")


if __name__ == "__main__":
    import argparse
    from src.data_collection.data_collector import DataCollector
    from src.config.config import config
    
    # 1. 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="효율적인 데이터 전처리")
    parser.add_argument("--symbols", nargs="+", help="전처리할 심볼 리스트", default=config.trading_symbols)
    parser.add_argument("--correlation_threshold", type=float, default=0.85, help="상관관계 임계값")
    parser.add_argument("--variance_threshold", type=float, default=0.0001, help="분산 임계값")
    args = parser.parse_args()
    
    # 2. 심볼 리스트 할당
    symbols = args.symbols
    
    # 데이터 수집
    LOGGER.info("📊 데이터 수집 시작...")
    collector = DataCollector(symbols=symbols)
    data = collector.load_all_data()
    
    # 효율적인 데이터 전처리
    processor = DataProcessor(
        window_size=config.WINDOW_SIZE, 
        enable_feature_selection=True,
        # enable_news_sentiment=True,
        correlation_threshold=args.correlation_threshold,
        variance_threshold=args.variance_threshold
    )
    
    LOGGER.info("🚀 효율적인 피처 생성 + 스마트한 중복 제거 시작...")
    results = processor.process_all_symbols(data, use_windows=False)
    
    # 결과 저장
    processor.save_processed_data(results)
    
    # 첫 번째 종목의 결과 확인
    if results:
        symbol = list(results.keys())[0]
        LOGGER.info(f"\n📊 {symbol} 효율적인 전처리 결과:")
        LOGGER.info(f"   원본 데이터 크기: {data[symbol].shape}")
        LOGGER.info(f"   전처리 데이터 크기: {results[symbol]['processed_data'].shape}")
        LOGGER.info(f"   특성 추출 데이터 크기: {results[symbol]['featured_data'].shape}")
        LOGGER.info(f"   학습 데이터 크기: {results[symbol]['train'].shape}")
        LOGGER.info(f"   검증 데이터 크기: {results[symbol]['valid'].shape}")
        LOGGER.info(f"   테스트 데이터 크기: {results[symbol]['test'].shape}")

        LOGGER.info(f"\n✅ 총 {len(results)}개 심볼 효율적인 처리 완료!")
        LOGGER.info(f"💾 데이터 저장 위치: {config.DATA_DIR}/processed/")
        
        # 효율적인 피처 선택 결과 요약
        if processor.feature_selector and hasattr(processor.feature_selector, 'selected_features'):
            removed_info = getattr(processor.feature_selector, 'removed_features_info', {})
            total_removed = sum(len(removed_list) for removed_list in removed_info.values() if removed_list)
            
            LOGGER.info(f"\n🎯 효율적인 피처 관리 결과:")
            original_count = len(processor.feature_selector.selected_features) + total_removed
            LOGGER.info(f"   원본 피처: {original_count}개 (OHLCV + 파생지표)")
            LOGGER.info(f"   제거된 피처: {total_removed}개 (중복/무용)")
            LOGGER.info(f"   최종 유효 피처: {len(processor.feature_selector.selected_features)}개")
            LOGGER.info(f"   효율성 개선: {total_removed/original_count:.1%} 중복 제거")
            
            # close 컬럼이 마지막에 있는지 확인
            final_columns = list(results[symbol]['train'].columns)
            if final_columns[-1] == 'close':
                LOGGER.info(f"   ✅ close 컬럼 마지막 위치 보장됨")
            else:
                LOGGER.warning(f"   ⚠️ close 컬럼이 마지막 위치에 없음: {final_columns[-1]}")
        
        LOGGER.info(f"\n🚀 SAC 강화학습에 최적화된 효율적인 피처셋 준비 완료!")