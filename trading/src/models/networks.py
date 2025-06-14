"""
SAC 알고리즘의 Actor와 Critic 네트워크 구현 모듈 (LSTM + Mamba 추가 버전)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, List, Union, Optional
import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(project_root)

from src.config.ea_teb_config import (
    HIDDEN_DIM,
    DEVICE,
    LOGGER,
    WINDOW_SIZE
)

class ActorNetwork(nn.Module):
    """
    SAC 알고리즘의 Actor 네트워크 (정책 네트워크)
    연속적인 행동 공간에 대한 확률적 정책을 모델링
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        """
        ActorNetwork 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            log_std_min: 로그 표준편차의 최소값
            log_std_max: 로그 표준편차의 최대값
            device: 모델이 실행될 장치
        """
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        # 공통 특성 추출 레이어
        print(f'state_dim : {state_dim}')
        print(f'hidden_dim : {hidden_dim}')
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        
        # 평균값 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 출력 레이어
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"Actor 네트워크 초기화 완료: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 텐서
            
        Returns:
            평균과 로그 표준편차 텐서의 튜플
        """
        # 입력 차원 검사 및 조정
        if state.size(1) != self.state_dim:
            print(f"경고: 입력 상태 차원 ({state.size(1)})이 모델의 상태 차원 ({self.state_dim})과 일치하지 않습니다.")
            # 차원이 더 큰 경우 자르기
            if state.size(1) > self.state_dim:
                state = state[:, :self.state_dim]
                print(f"입력을 {self.state_dim} 차원으로 잘랐습니다.")
            # 차원이 더 작은 경우 패딩
            else:
                padding = torch.zeros(state.size(0), self.state_dim - state.size(1), device=state.device)
                state = torch.cat([state, padding], dim=1)
                print(f"입력을 {self.state_dim} 차원으로 패딩했습니다.")
        
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 평균값 계산
        mean = self.mean(x)
        
        # 로그 표준편차 계산 및 클리핑
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동을 샘플링
        
        Args:
            state: 상태 텐서
            
        Returns:
            (행동, 로그 확률, 평균) 튜플
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 재매개변수화 트릭을 사용하여 정규 분포에서 샘플링
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # 재매개변수화된 샘플
        
        # Tanh 변환을 통해 행동 범위 제한 (-1, 1)
        y_t = torch.tanh(x_t)
        
        # 정책의 로그 확률 계산
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def to(self, device: torch.device) -> 'ActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(ActorNetwork, self).to(device)


class CriticNetwork(nn.Module):
    """
    SAC 알고리즘의 Critic 네트워크 (Q-함수)
    상태-행동 쌍의 가치를 평가
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        device: torch.device = DEVICE
    ):
        """
        CriticNetwork 클래스 초기화
        
        Args:
            state_dim: 상태 공간의 차원
            action_dim: 행동 공간의 차원
            hidden_dim: 은닉층의 뉴런 수
            device: 모델이 실행될 장치
        """
        super(CriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # Q1 네트워크
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 네트워크 (Double Q-learning)
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"Critic 네트워크 초기화 완료: 상태 차원 {state_dim}, 행동 차원 {action_dim}")
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 텐서
            action: 행동 텐서
            
        Returns:
            두 Q 값의 튜플
        """
        # 입력 차원 검사 및 조정
        if state.size(1) != self.state_dim:
            print(f"경고: 입력 상태 차원 ({state.size(1)})이 모델의 상태 차원 ({self.state_dim})과 일치하지 않습니다.")
            # 차원이 더 큰 경우 자르기
            if state.size(1) > self.state_dim:
                state = state[:, :self.state_dim]
                print(f"입력을 {self.state_dim} 차원으로 잘랐습니다.")
            # 차원이 더 작은 경우 패딩
            else:
                padding = torch.zeros(state.size(0), self.state_dim - state.size(1), device=state.device)
                state = torch.cat([state, padding], dim=1)
                print(f"입력을 {self.state_dim} 차원으로 패딩했습니다.")
        
        # 상태와 행동을 연결
        sa = torch.cat([state, action], 1)
        
        # Q1 값 계산
        q1 = F.relu(self.fc1(sa))
        q1 = F.relu(self.fc2(q1))
        q1 = F.relu(self.fc3(q1))
        q1 = self.q1(q1)
        
        # Q2 값 계산
        q2 = F.relu(self.fc4(sa))
        q2 = F.relu(self.fc5(q2))
        q2 = F.relu(self.fc6(q2))
        q2 = self.q2(q2)
        
        return q1, q2
    
    def to(self, device: torch.device) -> 'CriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CriticNetwork, self).to(device)


class CNNActorNetwork(nn.Module):
    """
    개선된 CNN 기반 Actor 네트워크
    - 더 깊은 컨볼루션 레이어
    - Batch Normalization과 Dropout 추가
    - Residual Connection 적용
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        dropout_rate: float = 0.1,
        device: torch.device = DEVICE
    ):
        super(CNNActorNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device

        # 🆕 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim  # 모델이 기대하는 차원
        self.adaptive_input = True  # 동적 입력 조정 활성화
        
        LOGGER.info(f"🖼️ CNN Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # 개선된 컨볼루션 블록들
        self.conv_block1 = self._make_conv_block(self.feature_dim, 64, 3, 1, dropout_rate)
        self.conv_block2 = self._make_conv_block(64, 128, 3, 1, dropout_rate)
        self.conv_block3 = self._make_conv_block(128, 256, 3, 1, dropout_rate)
        
        # ✅ 동적 Adaptive pooling 크기 계산
        # MaxPool stride=2가 3번 적용되므로 길이가 1/8로 감소
        pooled_length = max(4, self.window_size // 8)  # 최소 4, 최대 window_size//8
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pooled_length)
        
        # ✅ 동적 컨볼루션 출력 크기 계산
        self.conv_output_size = 256 * pooled_length
        
        # 포트폴리오 상태 처리
        self.portfolio_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ✅ 개선된 특성 융합 네트워크
        fusion_input_size = self.conv_output_size + hidden_dim // 4  # 단순 concat
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 가중치 초기화
        self._initialize_weights()
        
        self.to(device)
        LOGGER.info(f"CNN Actor 네트워크 초기화 완료: 입력 형태 {input_shape}, 풀링 크기 {pooled_length}")
    
    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        """컨볼루션 블록 생성"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2, stride=2, padding=0)  # 경미한 다운샘플링
        )
    
    def _initialize_weights(self):
        """Xavier 초기화 적용"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """개선된 순전파"""
        market_data = state['market_data']
        portfolio_state = state['portfolio_state']
        
        # 🆕 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 처리
        if len(market_data.shape) == 3:  # (B, W, F)
            market_data = market_data.permute(0, 2, 1)  # (B, F, W)
        elif len(market_data.shape) == 2:  # (W, F)
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)  # (1, F, W)
        
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        # 컨볼루션 처리
        x = self.conv_block1(market_data)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        market_features = x.view(x.size(0), -1)  # 평탄화
        
        # 포트폴리오 상태 처리
        portfolio_features = self.portfolio_fc(portfolio_state)
        
        # 단순한 특성 융합 (concat)
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
        
        # 최종 특성 처리
        fused_features = self.fusion_network(combined_features)
        
        # 출력 계산
        mean = self.mean(fused_features)
        log_std = self.log_std(fused_features)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 샘플링"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정
        
        Args:
            market_data: 입력 마켓 데이터 (B, W, F) 또는 (W, F)
            
        Returns:
            조정된 마켓 데이터
        """
        # 현재 입력의 특성 차원 확인
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)  # (1, W, F)로 변환
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data

    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)

    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 CNN 동적 입력 조정: {'활성화' if enabled else '비활성화'}")

    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 CNN 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
        
        # 컨볼루션 레이어 재구성이 필요한 경우 경고
        if old_feature_dim != new_feature_dim:
            LOGGER.warning("⚠️ 특성 차원 변경으로 컨볼루션 레이어 재초기화 필요")
    
    def to(self, device: torch.device) -> 'CNNActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CNNActorNetwork, self).to(device)


class CNNCriticNetwork(nn.Module):
    """
    개선된 CNN 기반 Critic 네트워크
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        dropout_rate: float = 0.1,
        device: torch.device = DEVICE
    ):
        super(CNNCriticNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 🆕 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim  # 모델이 기대하는 차원
        self.adaptive_input = True  # 동적 입력 조정 활성화
        
        LOGGER.info(f"🖼️ CNN Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Q1, Q2 네트워크 컨볼루션 블록들 (동일한 _make_conv_block 사용)
        self.q1_conv_block1 = self._make_conv_block(self.feature_dim, 64, 3, 1, dropout_rate)
        self.q1_conv_block2 = self._make_conv_block(64, 128, 3, 1, dropout_rate)
        self.q1_conv_block3 = self._make_conv_block(128, 256, 3, 1, dropout_rate)
        
        self.q2_conv_block1 = self._make_conv_block(self.feature_dim, 64, 3, 1, dropout_rate)
        self.q2_conv_block2 = self._make_conv_block(64, 128, 3, 1, dropout_rate)
        self.q2_conv_block3 = self._make_conv_block(128, 256, 3, 1, dropout_rate)
        
        # ✅ 동적 Adaptive pooling
        pooled_length = max(4, self.window_size // 8)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(pooled_length)
        self.conv_output_size = 256 * pooled_length
        
        # Q1, Q2 포트폴리오/행동 처리 (기존과 동일)
        self.q1_portfolio_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.q1_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Q2도 동일하게...
        self.q2_portfolio_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.q2_action_fc = nn.Sequential(
            nn.Linear(action_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # ✅ 개선된 Q1, Q2 융합 네트워크
        fusion_input_size = self.conv_output_size + hidden_dim // 2  # market + portfolio + action
        self.q1_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        # 가중치 초기화
        self._initialize_weights()
        
        self.to(device)
        LOGGER.info(f"CNN Critic 네트워크 초기화 완료")

    def _make_conv_block(self, in_channels, out_channels, kernel_size, stride, dropout_rate):
        """컨볼루션 블록 생성"""
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.MaxPool1d(2, stride=2, padding=0)
        )
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """개선된 순전파"""
        market_data = state['market_data']
        portfolio_state = state['portfolio_state']
        
        # 🆕 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 처리
        if len(market_data.shape) == 3:
            market_data = market_data.permute(0, 2, 1)
        elif len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0).permute(0, 2, 1)
        
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Q1 계산
        q1_x = self.q1_conv_block1(market_data)
        q1_x = self.q1_conv_block2(q1_x)
        q1_x = self.q1_conv_block3(q1_x)
        q1_x = self.adaptive_pool(q1_x)
        q1_market_features = q1_x.view(q1_x.size(0), -1)  # ✅ 변수명 수정
        
        q1_p = self.q1_portfolio_fc(portfolio_state)
        q1_a = self.q1_action_fc(action)
        q1_portfolio_action = torch.cat([q1_p, q1_a], dim=1)  # ✅ 변수명 수정
        
        # 단순한 특성 결합
        q1_combined = torch.cat([q1_market_features, q1_portfolio_action], dim=1)
        q1 = self.q1_fusion(q1_combined)
        
        # Q2 계산
        q2_x = self.q2_conv_block1(market_data)
        q2_x = self.q2_conv_block2(q2_x)
        q2_x = self.q2_conv_block3(q2_x)
        q2_x = self.adaptive_pool(q2_x)
        q2_market_features = q2_x.view(q2_x.size(0), -1)  # ✅ 변수명 수정
        
        q2_p = self.q2_portfolio_fc(portfolio_state)
        q2_a = self.q2_action_fc(action)
        q2_portfolio_action = torch.cat([q2_p, q2_a], dim=1)  # ✅ 변수명 수정
        
        # 단순한 특성 결합
        q2_combined = torch.cat([q2_market_features, q2_portfolio_action], dim=1)
        q2 = self.q2_fusion(q2_combined)
        
        return q1, q2
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정
        
        Args:
            market_data: 입력 마켓 데이터 (B, W, F) 또는 (W, F)
            
        Returns:
            조정된 마켓 데이터
        """
        # 현재 입력의 특성 차원 확인
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)  # (1, W, F)로 변환
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data

    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)

    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 CNN 동적 입력 조정: {'활성화' if enabled else '비활성화'}")

    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 CNN 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
        
        # 컨볼루션 레이어 재구성이 필요한 경우 경고
        if old_feature_dim != new_feature_dim:
            LOGGER.warning("⚠️ 특성 차원 변경으로 컨볼루션 레이어 재초기화 필요")
    
    def to(self, device: torch.device) -> 'CNNCriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(CNNCriticNetwork, self).to(device)


class LSTMActorNetwork(nn.Module):
    """
    LSTM 기반 Actor 네트워크
    시계열 데이터의 순차적 특성을 모델링하기 위한 LSTM 레이어 사용
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        """
        LSTMActorNetwork 클래스 초기화
        
        Args:
            input_shape: 입력 데이터의 형태 (window_size, feature_dim)
            action_dim: 행동 공간의 차원
            hidden_dim: FC 은닉층의 뉴런 수
            lstm_hidden_dim: LSTM 은닉층의 차원
            num_lstm_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
            log_std_min: 로그 표준편차의 최소값
            log_std_max: 로그 표준편차의 최대값
            device: 모델이 실행될 장치
        """
        super(LSTMActorNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        # 🆕 동적 입력 조정 설정 추가
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🧠 LSTM Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # LSTM 레이어 (market_data 처리용)
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # 포트폴리오 상태 처리용 FC 레이어
        self.portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        # LSTM 출력과 포트폴리오 상태를 결합하는 레이어  
        combined_dim = lstm_hidden_dim + hidden_dim // 4
        self.fc1 = nn.Linear(combined_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 평균값 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        
        # 로그 표준편차 출력 레이어
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"LSTM Actor 네트워크 초기화 완료: 입력 형태 {input_shape}, LSTM 차원 {lstm_hidden_dim}, 레이어 수 {num_lstm_layers}")
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 딕셔너리 {'market_data': 시장 데이터, 'portfolio_state': 포트폴리오 상태}
            
        Returns:
            평균과 로그 표준편차 텐서의 튜플
        """
        # 시장 데이터 처리 (B, W, F) 형태 확인
        market_data = state['market_data']
        
        # 🆕 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 확인 및 처리
        if len(market_data.shape) == 2:  # (W, F) 형태 - 배치 차원 추가 필요
            market_data = market_data.unsqueeze(0)  # (1, W, F) 형태로 변환
        
        # 포트폴리오 상태 처리
        portfolio_state = state['portfolio_state']
        
        # 포트폴리오 상태 차원 확인 및 처리
        if len(portfolio_state.shape) == 1:  # (2,) 형태 - 배치 차원 추가 필요
            portfolio_state = portfolio_state.unsqueeze(0)  # (1, 2) 형태로 변환
        
        # LSTM을 통한 시장 데이터 처리
        # market_data: (batch_size, sequence_length, input_size)
        lstm_out, (h_n, c_n) = self.lstm(market_data)
        
        # 마지막 타임스텝의 출력 사용
        lstm_features = lstm_out[:, -1, :]  # (batch_size, lstm_hidden_dim)
        
        # 포트폴리오 상태 처리
        portfolio_features = F.relu(self.portfolio_fc(portfolio_state))
        
        # 특성 결합
        combined = torch.cat([lstm_features, portfolio_features], dim=1)
        
        # FC 레이어를 통한 처리
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # 평균값 계산
        mean = self.mean(x)
        
        # 로그 표준편차 계산 및 클리핑
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        상태에서 행동을 샘플링
        
        Args:
            state: 상태 딕셔너리
            
        Returns:
            (행동, 로그 확률, 평균) 튜플
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # 재매개변수화 트릭을 사용하여 정규 분포에서 샘플링
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        # 정책의 로그 확률 계산
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정
        
        Args:
            market_data: 입력 마켓 데이터 (B, W, F) 또는 (W, F)
            
        Returns:
            조정된 마켓 데이터
        """
        # 현재 입력의 특성 차원 확인
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)  # (1, W, F)로 변환
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data

    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)

    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 LSTM 동적 입력 조정: {'활성화' if enabled else '비활성화'}")

    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 LSTM 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
        
        # LSTM 레이어 재구성이 필요한 경우 경고
        if old_feature_dim != new_feature_dim:
            LOGGER.warning("⚠️ 특성 차원 변경으로 LSTM 레이어 재초기화 필요")
    
    def to(self, device: torch.device) -> 'LSTMActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(LSTMActorNetwork, self).to(device)


class LSTMCriticNetwork(nn.Module):
    """
    LSTM 기반 Critic 네트워크
    시계열 데이터의 순차적 특성을 모델링하기 위한 LSTM 레이어 사용
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
        device: torch.device = DEVICE
    ):
        """
        LSTMCriticNetwork 클래스 초기화
        
        Args:
            input_shape: 입력 데이터의 형태 (window_size, feature_dim)
            action_dim: 행동 공간의 차원
            hidden_dim: FC 은닉층의 뉴런 수
            lstm_hidden_dim: LSTM 은닉층의 차원
            num_lstm_layers: LSTM 레이어 수
            dropout: 드롭아웃 비율
            device: 모델이 실행될 장치
        """
        super(LSTMCriticNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.device = device
        
        # 🆕 동적 입력 조정 설정 추가 (이 부분이 누락되어 있었음)
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🧠 LSTM Critic 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Q1 네트워크의 LSTM
        self.q1_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # Q2 네트워크의 LSTM
        self.q2_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )
        
        # 포트폴리오 상태 처리용 FC 레이어
        self.q1_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        self.q2_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        # 행동 처리용 FC 레이어
        self.q1_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        self.q2_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        
        # LSTM 출력, 포트폴리오 상태, 행동을 결합하는 레이어
        self.q1_fc1 = nn.Linear(lstm_hidden_dim + hidden_dim // 2, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(lstm_hidden_dim + hidden_dim // 2, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        # 디버깅 플래그
        self._debug_first_forward = False
        
        LOGGER.info(f"LSTM Critic 네트워크 초기화 완료: 입력 형태 {input_shape}, LSTM 차원 {lstm_hidden_dim}, 레이어 수 {num_lstm_layers}")
    
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 딕셔너리 {'market_data': 시장 데이터, 'portfolio_state': 포트폴리오 상태}
            action: 행동 텐서
            
        Returns:
            두 Q 값의 튜플
        """
        # 시장 데이터 처리 (B, W, F) 형태 확인
        market_data = state['market_data']
        
        # 🆕 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 확인 및 처리
        if len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # 포트폴리오 상태 차원 확인
        portfolio_state = state['portfolio_state']
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        # 디버깅: 차원 정보 출력
        if hasattr(self, '_debug_first_forward') and not self._debug_first_forward:
            self._debug_first_forward = True
            LOGGER.debug(f"🔍 LSTM Critic 차원 정보:")
            LOGGER.debug(f"   market_data: {market_data.shape}")
            LOGGER.debug(f"   portfolio_state: {portfolio_state.shape}")
            LOGGER.debug(f"   action: {action.shape}")
            LOGGER.debug(f"   lstm_hidden_dim: {self.lstm_hidden_dim}")
            LOGGER.debug(f"   hidden_dim: {self.hidden_dim}")
        
        # Q1 계산
        q1_lstm_out, _ = self.q1_lstm(market_data)
        q1_lstm_features = q1_lstm_out[:, -1, :]
        
        q1_p = F.relu(self.q1_portfolio_fc(portfolio_state))
        q1_a = F.relu(self.q1_action_fc(action))
        q1_combined = torch.cat([q1_lstm_features, q1_p, q1_a], dim=1)
        
        q1_x = F.relu(self.q1_fc1(q1_combined))
        q1_x = self.dropout(q1_x)
        q1_x = F.relu(self.q1_fc2(q1_x))
        q1_x = self.dropout(q1_x)
        q1 = self.q1(q1_x)
        
        # Q2 계산
        q2_lstm_out, _ = self.q2_lstm(market_data)
        q2_lstm_features = q2_lstm_out[:, -1, :]
        
        q2_p = F.relu(self.q2_portfolio_fc(portfolio_state))
        q2_a = F.relu(self.q2_action_fc(action))
        q2_combined = torch.cat([q2_lstm_features, q2_p, q2_a], dim=1)
        
        q2_x = F.relu(self.q2_fc1(q2_combined))
        q2_x = self.dropout(q2_x)
        q2_x = F.relu(self.q2_fc2(q2_x))
        q2_x = self.dropout(q2_x)
        q2 = self.q2(q2_x)
        
        return q1, q2
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정
        
        Args:
            market_data: 입력 마켓 데이터 (B, W, F) 또는 (W, F)
            
        Returns:
            조정된 마켓 데이터
        """
        # 현재 입력의 특성 차원 확인
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)  # (1, W, F)로 변환
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data

    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)

    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 LSTM 동적 입력 조정: {'활성화' if enabled else '비활성화'}")

    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 LSTM 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
        
        # LSTM 레이어 재구성이 필요한 경우 경고
        if old_feature_dim != new_feature_dim:
            LOGGER.warning("⚠️ 특성 차원 변경으로 LSTM 레이어 재초기화 필요")
    
    def to(self, device: torch.device) -> 'LSTMCriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(LSTMCriticNetwork, self).to(device)


class MambaBlock(nn.Module):
    """
    Simplified Mamba Block for Trading
    State Space Model with selective mechanism
    """
    def __init__(
        self, 
        d_model: int, 
        d_state: int = 16, 
        d_conv: int = 4,
        expand_factor: int = 2,
        device: torch.device = DEVICE
    ):
        super(MambaBlock, self).__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand_factor * d_model
        self.device = device
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Convolution for local interactions
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise convolution
        )
        
        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # B, C projections
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # A parameter (learnable)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Residual connection
        residual = x
        
        # Input projection
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # Convolution (need to transpose for conv1d)
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim to original length
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # SSM
        x = F.silu(x)
        
        # Get B, C, and dt
        x_proj_out = self.x_proj(x)  # (B, L, 2*d_state)
        B, C = x_proj_out.split([self.d_state, self.d_state], dim=-1)
        
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Apply SSM
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        y = self._ssm(x, A, B, C, dt)
        
        # Skip connection and gating
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection and layer norm
        return self.norm(output + residual)
    
    def _ssm(self, x, A, B, C, dt):
        """Simplified SSM computation"""
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[1]
        
        # Discretize A and B
        dt = dt.unsqueeze(-1)  # (B, L, d_inner, 1)
        A_disc = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt)  # (B, L, d_inner, d_state)
        B_disc = dt * B.unsqueeze(2)  # (B, L, d_inner, d_state)
        
        # Initialize state
        h = torch.zeros(batch_size, d_inner, d_state, device=x.device)
        
        outputs = []
        for i in range(seq_len):
            # Update state
            h = A_disc[:, i] * h + B_disc[:, i] * x[:, i:i+1]  # Broadcasting
            
            # Compute output
            y = torch.sum(h * C[:, i:i+1].unsqueeze(1), dim=-1)  # (B, d_inner)
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        # Add skip connection
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaActorNetwork(nn.Module):
    """
    Mamba-based Actor Network for Trading
    Efficient state space model for long-range dependencies
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        mamba_layers: int = 4,
        d_state: int = 16,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        dropout_rate: float = 0.1,
        device: torch.device = DEVICE
    ):
        super(MambaActorNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        # 🆕 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🌊 Mamba Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Input embedding
        self.input_embedding = nn.Linear(self.feature_dim, hidden_dim)
        
        # Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(
                d_model=hidden_dim,
                d_state=d_state,
                device=device
            ) for _ in range(mamba_layers)
        ])
        
        # Portfolio state processing
        self.portfolio_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Feature fusion
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._initialize_weights()
        self.to(device)
        
        LOGGER.info(f"Mamba Actor 네트워크 초기화 완료: {mamba_layers}개 Mamba 블록")
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        market_data = state['market_data']
        portfolio_state = state['portfolio_state']
        
        # 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 처리
        if len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0)
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        # Market data embedding
        x = self.input_embedding(market_data)  # (B, W, hidden_dim)
        
        # Mamba blocks
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        
        # Global average pooling for sequence aggregation
        market_features = torch.mean(x, dim=1)  # (B, hidden_dim)
        
        # Portfolio features
        portfolio_features = self.portfolio_fc(portfolio_state)
        
        # Feature fusion
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        # Output
        mean = self.mean(fused_features)
        log_std = self.log_std(fused_features)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 샘플링"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return y_t, log_prob, mean
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """입력 차원 동적 조정"""
        if len(market_data.shape) == 3:
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data
    
    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)
    
    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 Mamba 동적 입력 조정: {'활성화' if enabled else '비활성화'}")
    
    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 Mamba 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
    
    def to(self, device: torch.device) -> 'MambaActorNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(MambaActorNetwork, self).to(device)


class MambaCriticNetwork(nn.Module):
    """
    Mamba-based Critic Network for Trading
    """
    def __init__(
        self,
        input_shape: Tuple[int, int],
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        mamba_layers: int = 4,
        d_state: int = 16,
        dropout_rate: float = 0.1,
        device: torch.device = DEVICE
    ):
        super(MambaCriticNetwork, self).__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.device = device
        
        # 🆕 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🧠 LSTM Critic 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Q1 네트워크의 LSTM
        self.q1_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=mamba_layers,
            batch_first=True,
            dropout=dropout_rate if mamba_layers > 1 else 0,
            bidirectional=False
        )
        
        # Q2 네트워크의 LSTM
        self.q2_lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=mamba_layers,
            batch_first=True,
            dropout=dropout_rate if mamba_layers > 1 else 0,
            bidirectional=False
        )
        
        # 포트폴리오 상태 처리용 FC 레이어
        self.q1_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        self.q2_portfolio_fc = nn.Linear(2, hidden_dim // 4)
        
        # 행동 처리용 FC 레이어
        self.q1_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        self.q2_action_fc = nn.Linear(action_dim, hidden_dim // 4)
        
        # LSTM 출력, 포트폴리오 상태, 행동을 결합하는 레이어
        self.q1_fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1 = nn.Linear(hidden_dim, 1)
        
        self.q2_fc1 = nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2 = nn.Linear(hidden_dim, 1)
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)
        
        # 모델을 지정된 장치로 이동
        self.to(device)
        
        LOGGER.info(f"LSTM Critic 네트워크 초기화 완료: 입력 형태 {input_shape}, LSTM 차원 {hidden_dim}, 레이어 수 {mamba_layers}")
    
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파 함수
        
        Args:
            state: 상태 딕셔너리 {'market_data': 시장 데이터, 'portfolio_state': 포트폴리오 상태}
            action: 행동 텐서
            
        Returns:
            두 Q 값의 튜플
        """
        # 시장 데이터 처리 (B, W, F) 형태 확인
        market_data = state['market_data']
        
        # 🆕 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 확인 및 처리
        if len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # 포트폴리오 상태 차원 확인
        portfolio_state = state['portfolio_state']
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        # 디버깅: 차원 정보 출력
        if hasattr(self, '_debug_first_forward') and not self._debug_first_forward:
            self._debug_first_forward = True
            LOGGER.debug(f"🔍 LSTM Critic 차원 정보:")
            LOGGER.debug(f"   market_data: {market_data.shape}")
            LOGGER.debug(f"   portfolio_state: {portfolio_state.shape}")
            LOGGER.debug(f"   action: {action.shape}")
            LOGGER.debug(f"   lstm_hidden_dim: {self.lstm_hidden_dim}")
            LOGGER.debug(f"   hidden_dim: {self.hidden_dim}")
        
        # Q1 계산
        q1_lstm_out, _ = self.q1_lstm(market_data)
        q1_lstm_features = q1_lstm_out[:, -1, :]
        
        q1_p = F.relu(self.q1_portfolio_fc(portfolio_state))
        q1_a = F.relu(self.q1_action_fc(action))
        q1_combined = torch.cat([q1_lstm_features, q1_p, q1_a], dim=1)
        
        q1_x = F.relu(self.q1_fc1(q1_combined))
        q1_x = self.dropout(q1_x)
        q1_x = F.relu(self.q1_fc2(q1_x))
        q1_x = self.dropout(q1_x)
        q1 = self.q1(q1_x)
        
        # Q2 계산
        q2_lstm_out, _ = self.q2_lstm(market_data)
        q2_lstm_features = q2_lstm_out[:, -1, :]
        
        q2_p = F.relu(self.q2_portfolio_fc(portfolio_state))
        q2_a = F.relu(self.q2_action_fc(action))
        q2_combined = torch.cat([q2_lstm_features, q2_p, q2_a], dim=1)
        
        q2_x = F.relu(self.q2_fc1(q2_combined))
        q2_x = self.dropout(q2_x)
        q2_x = F.relu(self.q2_fc2(q2_x))
        q2_x = self.dropout(q2_x)
        q2 = self.q2(q2_x)
        
        return q1, q2
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정
        
        Args:
            market_data: 입력 마켓 데이터 (B, W, F) 또는 (W, F)
            
        Returns:
            조정된 마켓 데이터
        """
        # 현재 입력의 특성 차원 확인
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)  # (1, W, F)로 변환
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정 필요
        if current_feature_dim > self.expected_feature_dim:
            # 특성이 더 많은 경우: 앞에서부터 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
            
        else:
            # 특성이 더 적은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim} (제로 패딩)")
        
        return adjusted_data

    def get_expected_input_shape(self) -> Tuple[int, int]:
        """모델이 기대하는 입력 형태 반환"""
        return (self.window_size, self.expected_feature_dim)

    def set_adaptive_input(self, enabled: bool):
        """동적 입력 조정 기능 활성화/비활성화"""
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 LSTM 동적 입력 조정: {'활성화' if enabled else '비활성화'}")

    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: int = None):
        """기대하는 입력 차원 업데이트 (모델 로드 시 사용)"""
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 LSTM 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
        
        # LSTM 레이어 재구성이 필요한 경우 경고
        if old_feature_dim != new_feature_dim:
            LOGGER.warning("⚠️ 특성 차원 변경으로 LSTM 레이어 재초기화 필요")
    
    def to(self, device: torch.device) -> 'MambaCriticNetwork':
        """
        모델을 지정된 장치로 이동
        
        Args:
            device: 이동할 장치
            
        Returns:
            장치로 이동된 모델
        """
        self.device = device
        return super(MambaCriticNetwork, self).to(device)


if __name__ == "__main__":
    # 모듈 테스트 코드
    # 일반 네트워크 테스트
    state_dim = 10
    action_dim = 1
    batch_size = 4
    
    actor = ActorNetwork(state_dim, action_dim)
    critic = CriticNetwork(state_dim, action_dim)
    
    # 랜덤 데이터로 테스트
    state = torch.randn(batch_size, state_dim).to(DEVICE)
    action = torch.randn(batch_size, action_dim).to(DEVICE)
    
    # Actor 테스트
    mean, log_std = actor(state)
    action_sample, log_prob, _ = actor.sample(state)
    
    print(f"Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # Critic 테스트
    q1, q2 = critic(state, action)
    
    print(f"Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")
    
    # CNN 네트워크 테스트
    window_size = WINDOW_SIZE
    feature_dim = 5
    
    cnn_actor = CNNActorNetwork((window_size, feature_dim), action_dim)
    cnn_critic = CNNCriticNetwork((window_size, feature_dim), action_dim)
    
    # 랜덤 데이터로 테스트
    market_data = torch.randn(batch_size, window_size, feature_dim).to(DEVICE)
    portfolio_state = torch.randn(batch_size, 2).to(DEVICE)
    state_dict = {
        'market_data': market_data,
        'portfolio_state': portfolio_state
    }
    
    # CNN Actor 테스트
    mean, log_std = cnn_actor(state_dict)
    action_sample, log_prob, _ = cnn_actor.sample(state_dict)
    
    print(f"CNN Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"CNN Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # CNN Critic 테스트
    q1, q2 = cnn_critic(state_dict, action_sample)
    
    print(f"CNN Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")
    
    # LSTM 네트워크 테스트
    print("\n=== LSTM 네트워크 테스트 ===")
    
    lstm_actor = LSTMActorNetwork((window_size, feature_dim), action_dim)
    lstm_critic = LSTMCriticNetwork((window_size, feature_dim), action_dim)
    
    # LSTM Actor 테스트
    mean, log_std = lstm_actor(state_dict)
    action_sample, log_prob, _ = lstm_actor.sample(state_dict)
    
    print(f"LSTM Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"LSTM Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # LSTM Critic 테스트
    q1, q2 = lstm_critic(state_dict, action_sample)
    
    print(f"LSTM Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")
    
    print("\n모든 네트워크 테스트 완료!")