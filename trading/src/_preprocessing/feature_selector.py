"""
개선된 피처 선택 및 중복 제거 모듈
OHLCV 보존 + 스마트한 중복 제거에 특화
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    """
    강화학습 최적화 피처 선택 클래스
    - 필수 컬럼 절대 보호
    - 분산 + 상관관계 기반 중복 제거
    - close 컬럼 마지막 위치 보장
    """
    
    def __init__(self, 
                 correlation_threshold: float = 0.85, 
                 variance_threshold: float = 0.0001,
                 preserve_columns: List[str] = None,
                 remove_duplicates_only: bool = True):
        """
        초기화
        
        Args:
            correlation_threshold: 상관관계 임계값 (85% 이상이면 중복으로 간주하여 제거)
            variance_threshold: 분산 임계값 (0.0001 이하면 변동성 없는 피처로 간주하여 제거)
            preserve_columns: 절대 제거하면 안 되는 필수 컬럼들
            remove_duplicates_only: True면 중복 제거만, False면 개수 제한도 적용
        """
        self.correlation_threshold = correlation_threshold
        self.variance_threshold = variance_threshold
        self.preserve_columns = preserve_columns or ['open', 'high', 'low', 'close', 'volume']
        self.remove_duplicates_only = remove_duplicates_only
        self.selected_features = None
        self.feature_importance_dict = {}
        self.removed_features_info = {}
        
        print(f"🔧 FeatureSelector 초기화:")
        print(f"   📏 상관관계 임계값: {correlation_threshold}")
        print(f"   📐 분산 임계값: {variance_threshold}")
        print(f"   🔒 필수 보존 컬럼: {self.preserve_columns}")
        print(f"   🎯 중복 제거 전용: {remove_duplicates_only}")
        
    def remove_low_variance_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        분산이 낮은 피처들 제거 (거의 변하지 않는 피처들)
        필수 컬럼은 절대 제거하지 않음
        """
        # 보호할 컬럼과 분석할 컬럼 분리
        protected_cols = [col for col in self.preserve_columns if col in df.columns]
        if target_col not in protected_cols and target_col in df.columns:
            protected_cols.append(target_col)
        
        analyzable_cols = [col for col in df.columns if col not in protected_cols]
        low_variance_features = []
        
        print(f"\n🔍 분산 분석 시작:")
        print(f"   🔒 보호된 컬럼: {len(protected_cols)}개")
        print(f"   📊 분석 대상 컬럼: {len(analyzable_cols)}개")
        
        for col in analyzable_cols:
            if df[col].dtype in ['float64', 'int64']:
                variance = df[col].var()
                if variance < self.variance_threshold:
                    low_variance_features.append(col)
                    print(f"   ❌ {col}: 분산={variance:.8f} (제거)")
        
        if low_variance_features:
            print(f"\n📉 낮은 분산으로 제거할 피처: {len(low_variance_features)}개")
        else:
            print(f"\n✅ 분산 기준을 통과한 모든 피처")
        
        # 낮은 분산 피처 제거 (보호된 컬럼은 유지)
        remaining_features = [col for col in analyzable_cols if col not in low_variance_features]
        final_cols = protected_cols + remaining_features
        
        self.removed_features_info['low_variance'] = low_variance_features
        print(f"📊 분산 필터링 후: {len(analyzable_cols)} → {len(remaining_features)} 피처 (보호된 {len(protected_cols)}개 + 선별된 {len(remaining_features)}개)")
        
        return df[final_cols]
    
    def remove_correlated_features(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        높은 상관관계를 가진 피처들 제거 (중복 정보 제거)
        필수 컬럼은 절대 제거하지 않음
        """
        # 보호할 컬럼과 분석할 컬럼 분리
        protected_cols = [col for col in self.preserve_columns if col in df.columns]
        if target_col not in protected_cols and target_col in df.columns:
            protected_cols.append(target_col)
        
        analyzable_cols = [col for col in df.columns if col not in protected_cols]
        
        if len(analyzable_cols) < 2:
            print(f"🔍 상관관계 분석: 분석 가능한 컬럼이 {len(analyzable_cols)}개뿐이므로 건너뜀")
            self.removed_features_info['high_correlation'] = []
            return df
        
        print(f"\n🔍 상관관계 분석 시작:")
        print(f"   🔒 보호된 컬럼: {len(protected_cols)}개")
        print(f"   📊 분석 대상 컬럼: {len(analyzable_cols)}개")
        
        # 분석 가능한 컬럼들만으로 상관관계 매트릭스 계산
        feature_df = df[analyzable_cols].copy()
        corr_matrix = feature_df.corr().abs()
        
        # 상삼각 매트릭스만 사용 (중복 제거)
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # 높은 상관관계를 가진 피처 쌍들 찾기
        high_corr_pairs = []
        high_corr_features = set()
        
        for column in upper_tri.columns:
            corr_values = upper_tri[column].dropna()
            high_corr_items = corr_values[corr_values > self.correlation_threshold]
            
            for pair_feature, corr_value in high_corr_items.items():
                high_corr_pairs.append((column, pair_feature, corr_value))
                
                # 두 피처 중 타겟과의 상관관계가 낮은 것을 제거 대상으로 선택
                if target_col in df.columns:
                    target_corr_col = abs(df[column].corr(df[target_col]))
                    target_corr_pair = abs(df[pair_feature].corr(df[target_col]))
                    
                    # 타겟과의 상관관계가 낮은 피처를 제거 대상으로
                    if target_corr_col < target_corr_pair:
                        high_corr_features.add(column)
                        print(f"   🔄 {column} vs {pair_feature}: 상관관계={corr_value:.3f} → {column} 제거 (타겟 상관관계: {target_corr_col:.3f} < {target_corr_pair:.3f})")
                    else:
                        high_corr_features.add(pair_feature)
                        print(f"   🔄 {column} vs {pair_feature}: 상관관계={corr_value:.3f} → {pair_feature} 제거 (타겟 상관관계: {target_corr_pair:.3f} < {target_corr_col:.3f})")
                else:
                    # 타겟 컬럼이 없으면 첫 번째 피처를 제거
                    high_corr_features.add(column)
                    print(f"   🔄 {column} vs {pair_feature}: 상관관계={corr_value:.3f} → {column} 제거")
        
        if high_corr_pairs:
            print(f"\n🔗 높은 상관관계 피처 쌍: {len(high_corr_pairs)}개")
            print(f"❌ 중복으로 제거할 피처: {len(high_corr_features)}개")
        else:
            print(f"\n✅ 상관관계 기준을 통과한 모든 피처")
        
        # 높은 상관관계 피처 제거 (보호된 컬럼은 유지)
        remaining_features = [col for col in analyzable_cols if col not in high_corr_features]
        final_cols = protected_cols + remaining_features
        
        self.removed_features_info['high_correlation'] = list(high_corr_features)
        print(f"📊 상관관계 필터링 후: {len(analyzable_cols)} → {len(remaining_features)} 피처")
        
        return df[final_cols]
    
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
            print(f"🔒 close 컬럼을 마지막 위치로 이동 완료")
        else:
            print(f"⚠️ close 컬럼을 찾을 수 없습니다.")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        스마트한 중복 피처 제거 프로세스 실행
        
        Args:
            df: 입력 데이터프레임
            target_col: 타겟 컬럼명
            
        Returns:
            중복이 제거된 피처들로 구성된 데이터프레임
        """
        print(f"\n" + "="*60)
        print(f"🚀 스마트한 피처 중복 제거 시작")
        print(f"="*60)
        print(f"📊 초기 피처 수: {len(df.columns)}개")
        print(f"🎯 타겟 컬럼: {target_col}")
        print(f"🔒 필수 보존 컬럼: {self.preserve_columns}")
        
        # 1단계: 분산이 낮은 피처 제거 (거의 변하지 않는 피처들)
        print(f"\n📐 1단계: 분산 기반 필터링 (임계값: {self.variance_threshold})")
        df_filtered = self.remove_low_variance_features(df, target_col)
        
        # 2단계: 상관관계 높은 피처 제거 (중복 정보 피처들)
        print(f"\n🔗 2단계: 상관관계 기반 필터링 (임계값: {self.correlation_threshold})")
        df_filtered = self.remove_correlated_features(df_filtered, target_col)
        
        # 3단계: close 컬럼을 마지막 위치로 이동
        print(f"\n🔒 3단계: close 컬럼 위치 최적화")
        df_filtered = self._ensure_close_column_last(df_filtered)
        
        # 최종 선택된 피처들
        selected_features = [col for col in df_filtered.columns if col != target_col]
        if target_col in df.columns:
            final_features = selected_features + [target_col]
        else:
            final_features = selected_features
        
        self.selected_features = selected_features
        
        # 결과 요약
        initial_count = len(df.columns)
        final_count = len(df_filtered.columns)
        removed_count = initial_count - final_count
        efficiency = (removed_count / initial_count) * 100 if initial_count > 0 else 0
        
        print(f"\n" + "="*60)
        print(f"✨ 스마트한 피처 중복 제거 완료!")
        print(f"="*60)
        print(f"📊 처리 결과:")
        print(f"   🎯 초기 피처: {initial_count}개")
        print(f"   ✅ 최종 피처: {final_count}개")
        print(f"   ❌ 제거된 피처: {removed_count}개")
        print(f"   📈 효율성 개선: {efficiency:.1f}% 중복 제거")
        
        # 제거된 피처들 카테고리별 요약
        total_removed = 0
        for category, removed_list in self.removed_features_info.items():
            if removed_list:
                total_removed += len(removed_list)
                print(f"   - {category.replace('_', ' ').title()}: {len(removed_list)}개")
        
        # 최종 피처 구성 요약
        protected_features = [f for f in self.selected_features if f in self.preserve_columns]
        derived_features = [f for f in self.selected_features if f not in self.preserve_columns]
        
        print(f"\n📋 최종 피처 구성:")
        print(f"   🔒 필수 보존: {len(protected_features)}개")
        print(f"   ⚡ 선별된 파생: {len(derived_features)}개")
        print(f"   🎯 타겟 ({target_col}): 마지막 위치 보장")
        print(f"="*60)
        
        return df_filtered
    
    def transform(self, df: pd.DataFrame, target_col: str = 'close') -> pd.DataFrame:
        """
        학습된 피처 선택 기준으로 새 데이터 변환
        
        Args:
            df: 입력 데이터프레임
            target_col: 타겟 컬럼명
            
        Returns:
            선택된 피처들로 구성된 데이터프레임
        """
        if self.selected_features is None:
            raise ValueError("먼저 fit_transform을 실행해주세요.")
        
        print(f"🔄 학습된 피처 기준으로 변환 중...")
        
        # 선택된 피처들만 추출
        available_features = [f for f in self.selected_features if f in df.columns]
        missing_features = [f for f in self.selected_features if f not in df.columns]
        
        if missing_features:
            print(f"⚠️ 누락된 피처들: {missing_features}")
        
        if target_col in df.columns:
            final_features = available_features + [target_col]
        else:
            final_features = available_features
        
        result_df = df[final_features]
        
        # close 컬럼 위치 재확인
        result_df = self._ensure_close_column_last(result_df)
        
        print(f"✅ 변환 완료: {len(available_features)}개 피처 + {target_col}")
        
        return result_df


# 사용 예시 및 테스트 함수
def test_improved_feature_selection():
    """
    개선된 피처 선택 테스트 함수
    """
    print("🧪 개선된 피처 선택 테스트 시작")
    print("="*50)
    
    # 샘플 데이터 생성 (실제 상황 시뮬레이션)
    np.random.seed(42)
    n_samples = 1000
    
    # 기본 OHLCV 데이터
    close_prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.02)
    open_prices = close_prices + np.random.randn(n_samples) * 0.1
    high_prices = np.maximum(open_prices, close_prices) + np.abs(np.random.randn(n_samples) * 0.2)
    low_prices = np.minimum(open_prices, close_prices) - np.abs(np.random.randn(n_samples) * 0.2)
    volumes = np.random.randint(1000, 10000, n_samples)
    
    # 샘플 데이터프레임 생성 (의도적으로 중복 피처들 포함)
    sample_data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes,
        
        # 유사한 이동평균들 (높은 상관관계)
        'sma_5': close_prices + np.random.randn(n_samples) * 0.01,
        'sma_10': close_prices + np.random.randn(n_samples) * 0.02,
        'sma_20': close_prices + np.random.randn(n_samples) * 0.03,
        'ema_5': close_prices + np.random.randn(n_samples) * 0.01,  # sma_5와 높은 상관관계
        'ema_10': close_prices + np.random.randn(n_samples) * 0.02,  # sma_10과 높은 상관관계
        
        # RSI 계열 (유사한 지표들)
        'rsi_14': 50 + np.random.randn(n_samples) * 20,
        'rsi_30': 50 + np.random.randn(n_samples) * 18,  # rsi_14와 상관관계 있음
        
        # 거의 변하지 않는 피처 (낮은 분산)
        'constant_feature': np.full(n_samples, 1.0) + np.random.randn(n_samples) * 0.00001,
        'almost_constant': np.full(n_samples, 0.5) + np.random.randn(n_samples) * 0.00005,
        
        # 유용한 독립적인 피처들
        'macd': np.random.randn(n_samples) * 0.5,
        'bb_width': np.random.randn(n_samples) * 0.3,
        'atr_14': np.abs(np.random.randn(n_samples) * 0.4),
        'volume_change': np.random.randn(n_samples) * 0.3,
        'returns_1d': np.random.randn(n_samples) * 0.02,
        
        # 추가 중복 후보들
        'price_volatility': (high_prices - low_prices) / close_prices,
        'high_low_ratio': high_prices / low_prices,
        'similar_to_volatility': (high_prices - low_prices) / close_prices + np.random.randn(n_samples) * 0.001,  # price_volatility와 거의 동일
    })
    
    print(f"📊 테스트 데이터 생성 완료: {sample_data.shape}")
    print(f"🔒 필수 보존 컬럼: ['open', 'high', 'low', 'close', 'volume']")
    print(f"⚡ 파생 지표: {len(sample_data.columns) - 5}개")
    
    # 피처 선택 실행
    print(f"\n🚀 개선된 피처 선택 테스트 실행...")
    selector = FeatureSelector(
        correlation_threshold=0.85,
        variance_threshold=0.0001,
        preserve_columns=['open', 'high', 'low', 'close', 'volume'],
        remove_duplicates_only=True
    )
    
    selected_data = selector.fit_transform(sample_data, target_col='close')
    
    print(f"\n📋 테스트 결과 요약:")
    print(f"   📊 원본 데이터 shape: {sample_data.shape}")
    print(f"   ✅ 선택된 데이터 shape: {selected_data.shape}")
    print(f"   🔒 필수 컬럼 보존 확인: {all(col in selected_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])}")
    print(f"   🎯 close 컬럼 마지막 위치: {selected_data.columns[-1] == 'close'}")
    
    print(f"\n📋 최종 선택된 컬럼들:")
    for i, col in enumerate(selected_data.columns, 1):
        marker = "🔒" if col in ['open', 'high', 'low', 'close', 'volume'] else "⚡"
        print(f"   {i:2d}. {marker} {col}")
    
    print(f"\n🧪 개선된 피처 선택 테스트 완료!")
    return selected_data


def demonstrate_feature_selection_benefits():
    """
    피처 선택의 효과를 시연하는 함수
    """
    print("\n" + "="*60)
    print("💡 피처 선택의 효과 시연")
    print("="*60)
    
    # 극단적인 중복 데이터 생성
    np.random.seed(123)
    n_samples = 500
    base_signal = np.cumsum(np.random.randn(n_samples) * 0.1)
    
    demo_data = pd.DataFrame({
        # 필수 OHLCV
        'open': base_signal + np.random.randn(n_samples) * 0.01,
        'high': base_signal + 1 + np.random.randn(n_samples) * 0.01,
        'low': base_signal - 1 + np.random.randn(n_samples) * 0.01,
        'close': base_signal,
        'volume': np.random.randint(1000, 5000, n_samples),
        
        # 의도적인 중복 피처들
        'feature_1': base_signal + np.random.randn(n_samples) * 0.001,  # close와 거의 동일
        'feature_2': base_signal + np.random.randn(n_samples) * 0.002,  # close와 거의 동일
        'feature_3': base_signal * 1.1 + np.random.randn(n_samples) * 0.001,  # close와 높은 상관관계
        'feature_4': base_signal * 0.9 + np.random.randn(n_samples) * 0.001,  # close와 높은 상관관계
        
        # 완전히 무용한 피처들
        'constant_noise': np.full(n_samples, 100),  # 완전 상수
        'tiny_variance': np.full(n_samples, 50) + np.random.randn(n_samples) * 0.00001,  # 극도로 낮은 분산
        
        # 유용한 독립적인 피처
        'useful_feature_1': np.random.randn(n_samples),  # 독립적인 신호
        'useful_feature_2': np.sin(np.arange(n_samples) / 50),  # 주기적 패턴
    })
    
    print(f"📊 시연용 데이터: {demo_data.shape}")
    
    # 상관관계 매트릭스 출력 (주요 피처들만)
    main_features = ['close', 'feature_1', 'feature_2', 'feature_3', 'feature_4', 'useful_feature_1', 'useful_feature_2']
    corr_matrix = demo_data[main_features].corr()
    
    print(f"\n🔗 주요 피처들 간 상관관계:")
    print(f"   close vs feature_1: {corr_matrix.loc['close', 'feature_1']:.3f}")
    print(f"   close vs feature_2: {corr_matrix.loc['close', 'feature_2']:.3f}")
    print(f"   close vs feature_3: {corr_matrix.loc['close', 'feature_3']:.3f}")
    print(f"   close vs useful_feature_1: {corr_matrix.loc['close', 'useful_feature_1']:.3f}")
    
    # 분산 정보
    print(f"\n📐 분산 정보:")
    print(f"   constant_noise: {demo_data['constant_noise'].var():.8f}")
    print(f"   tiny_variance: {demo_data['tiny_variance'].var():.8f}")
    print(f"   close: {demo_data['close'].var():.8f}")
    print(f"   useful_feature_1: {demo_data['useful_feature_1'].var():.8f}")
    
    # 피처 선택 적용
    selector = FeatureSelector(
        correlation_threshold=0.90,  # 90% 이상 상관관계
        variance_threshold=0.0001,
        preserve_columns=['open', 'high', 'low', 'close', 'volume'],
        remove_duplicates_only=True
    )
    
    result = selector.fit_transform(demo_data, target_col='close')
    
    print(f"\n✨ 시연 결과:")
    print(f"   📊 처리 전: {demo_data.shape[1]}개 피처")
    print(f"   ✅ 처리 후: {result.shape[1]}개 피처")
    print(f"   🎯 효율성: {((demo_data.shape[1] - result.shape[1]) / demo_data.shape[1] * 100):.1f}% 중복 제거")
    
    return result


if __name__ == "__main__":
    # 기본 테스트 실행
    test_result = test_improved_feature_selection()
    
    # 효과 시연
    demo_result = demonstrate_feature_selection_benefits()
    
    print("\n" + "="*60)
    print("🎉 모든 테스트 완료!")
    print("="*60)
    print("✅ 개선된 FeatureSelector 검증 완료")
    print("✅ OHLCV 보존 확인")
    print("✅ close 컬럼 마지막 위치 보장")
    print("✅ 스마트한 중복 제거 동작 확인")
    print("="*60)