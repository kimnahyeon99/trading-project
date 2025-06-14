"""
SAC 트레이딩 시스템에서 사용되는 유틸리티 함수들
"""
import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Union, Any

def set_korean_font():
    """
    한글 폰트 설정 함수
    """
    # Windows 환경에서 기본 한글 폰트 설정
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    if os.path.exists(font_path):
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        # 맑은 고딕이 없는 경우 시스템 기본 폰트 사용
        plt.rcParams['font.family'] = 'NanumGothic'
    
    plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

def create_directory(directory_path: Union[str, Path]) -> None:
    """
    디렉토리가 존재하지 않으면 생성하는 함수
    
    Args:
        directory_path: 생성할 디렉토리 경로
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"디렉토리 생성됨: {directory_path}")

def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = True) -> None:
    """
    데이터프레임을 CSV 파일로 저장하는 함수
    
    Args:
        df: 저장할 데이터프레임
        file_path: 저장할 파일 경로
        index: 인덱스 포함 여부
    """
    create_directory(os.path.dirname(file_path))
    df.to_csv(file_path, index=index, encoding='utf-8-sig')
    print(f"파일 저장됨: {file_path}")

def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    CSV 파일에서 데이터프레임을 로드하는 함수
    
    Args:
        file_path: 로드할 파일 경로
        
    Returns:
        로드된 데이터프레임
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없음: {file_path}")
    
    return pd.read_csv(file_path, encoding='utf-8-sig')

def normalize_data(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    데이터 정규화 함수 (Min-Max 스케일링)
    
    Args:
        data: 정규화할 데이터 배열
        
    Returns:
        정규화된 데이터, 최소값, 최대값
    """
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    
    # 최대값과 최소값이 같은 경우(상수) 처리
    denom = max_vals - min_vals
    denom[denom == 0] = 1  # 0으로 나누기 방지
    
    normalized_data = (data - min_vals) / denom
    
    return normalized_data, min_vals, max_vals

def denormalize_data(normalized_data: np.ndarray, min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
    """
    정규화된 데이터를 원래 스케일로 되돌리는 함수
    
    Args:
        normalized_data: 정규화된 데이터
        min_vals: 원본 데이터의 최소값
        max_vals: 원본 데이터의 최대값
        
    Returns:
        원래 스케일로 되돌린 데이터
    """
    return normalized_data * (max_vals - min_vals) + min_vals

def calculate_sharpe_ratio(returns: Union[List[float], np.ndarray], risk_free_rate: float = 0.0) -> float:
    """
    샤프 비율 계산 함수
    
    Args:
        returns: 수익률 리스트 또는 배열
        risk_free_rate: 무위험 수익률 (연간 기준)
        
    Returns:
        샤프 비율
    """
    returns = np.array(returns)
    excess_returns = returns - risk_free_rate / 252  # 일별 무위험 수익률로 변환
    
    if len(excess_returns) == 0 or np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # 연간화된 샤프 비율

def calculate_max_drawdown(equity_curve: Union[List[float], np.ndarray]) -> float:
    """
    최대 낙폭(MDD) 계산 함수
    
    Args:
        equity_curve: 자산 가치 곡선
        
    Returns:
        최대 낙폭 (%)
    """
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdown)
    
    return max_drawdown * 100  # 퍼센트로 변환

def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float) -> None:
    """
    타겟 네트워크의 소프트 업데이트 함수
    
    Args:
        target: 타겟 네트워크
        source: 소스 네트워크
        tau: 소프트 업데이트 계수
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

def plot_learning_curve(rewards: List[float], ma_window: int = 100, save_path: Union[str, Path] = None) -> None:
    """
    학습 곡선 시각화 함수
    
    Args:
        rewards: 에피소드별 보상 리스트
        ma_window: 이동 평균 윈도우 크기
        save_path: 그래프 저장 경로 (None인 경우 저장하지 않음)
    """
    set_korean_font()
    
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, alpha=0.3, color='blue', label='원본 보상')
    
    if len(rewards) >= ma_window:
        ma_rewards = pd.Series(rewards).rolling(window=ma_window).mean().values
        plt.plot(ma_rewards, color='red', label=f'{ma_window}회 이동 평균')
    
    plt.title('학습 진행 곡선')
    plt.xlabel('에피소드')
    plt.ylabel('에피소드 보상')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"학습 곡선 저장됨: {save_path}")
    
    plt.close()

def plot_equity_curve(equity_curve: List[float], save_path: Union[str, Path] = None) -> None:
    """
    자산 가치 곡선 시각화 함수
    
    Args:
        equity_curve: 자산 가치 리스트
        save_path: 그래프 저장 경로 (None인 경우 저장하지 않음)
    """
    set_korean_font()
    
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, color='blue')
    plt.title('자산 가치 곡선')
    plt.xlabel('거래일')
    plt.ylabel('자산 가치')
    plt.grid(True, alpha=0.3)
    
    # 최대 낙폭 표시
    mdd = calculate_max_drawdown(equity_curve)
    plt.figtext(0.01, 0.01, f'최대 낙폭(MDD): {mdd:.2f}%', fontsize=10)
    
    # 최종 수익률 표시
    if len(equity_curve) > 0:
        final_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
        plt.figtext(0.01, 0.04, f'최종 수익률: {final_return:.2f}%', fontsize=10)
    
    if save_path:
        create_directory(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"자산 가치 곡선 저장됨: {save_path}")
    
    plt.close()

def get_timestamp() -> str:
    """
    현재 시간 타임스탬프 문자열 반환 함수
    
    Returns:
        YYYYMMDD_HHMMSS 형식의 타임스탬프 문자열
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S") 