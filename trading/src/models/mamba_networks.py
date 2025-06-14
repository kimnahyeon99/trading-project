"""
Mamba 기반 SAC 네트워크 구현 모듈
State Space Model을 활용한 트레이딩에 최적화된 네트워크
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


class MambaBlock(nn.Module):
    """
    Simplified Mamba Block for Trading
    State Space Model with selective mechanism optimized for financial time series
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
        
        # Convolution for local interactions (financial patterns)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner  # Depthwise convolution for efficiency
        )
        
        # SSM parameters for market dynamics
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)  # B, C projections
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        # A parameter (learnable state transition matrix)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D parameter (skip connection for immediate market signals)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Layer norm for training stability
        self.norm = nn.LayerNorm(d_model)
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Mamba block
        
        Args:
            x: (batch_size, seq_len, d_model) - Financial time series
        Returns:
            output: (batch_size, seq_len, d_model) - Processed features
        """
        batch_size, seq_len, d_model = x.shape
        
        # Residual connection for gradient flow
        residual = x
        
        # Input projection to inner dimension
        x_and_res = self.in_proj(x)  # (B, L, 2*d_inner)
        x, res = x_and_res.split([self.d_inner, self.d_inner], dim=-1)
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # 1D Convolution for local pattern recognition
        x = x.transpose(1, 2)  # (B, d_inner, L)
        x = self.conv1d(x)[:, :, :seq_len]  # Trim to original length
        x = x.transpose(1, 2)  # (B, L, d_inner)
        
        # Apply activation again
        x = F.silu(x)
        
        # Get selective scan parameters
        x_proj_out = self.x_proj(x)  # (B, L, 2*d_state)
        B, C = x_proj_out.split([self.d_state, self.d_state], dim=-1)
        
        # Delta (time step) computation
        dt = F.softplus(self.dt_proj(x))  # (B, L, d_inner)
        
        # Apply State Space Model
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        y = self._selective_scan(x, A, B, C, dt)
        
        # Gating mechanism
        y = y * F.silu(res)
        
        # Output projection
        output = self.out_proj(y)
        
        # Residual connection and normalization
        return self.norm(output + residual)
    
    def _selective_scan(self, x, A, B, C, dt):
        """
        Simplified selective scan for financial time series (No in-place operations)
        """
        batch_size, seq_len, d_inner = x.shape
        
        # Very simplified approach: just apply learned transformations
        # This maintains the Mamba concept while being computationally stable
        
        # Apply temporal mixing with simple recurrence (avoiding in-place operations)
        y = x.clone()
        
        # Create a list to store results without in-place modification
        y_list = [y[:, 0]]  # Start with first timestep
        
        # Simple temporal processing without in-place operations
        for i in range(1, seq_len):
            # Simple exponential moving average-like update
            alpha = torch.sigmoid(dt[:, i])  # Use dt as mixing coefficient
            # ✅ No in-place operation: create new tensor
            new_timestep = alpha * y[:, i] + (1 - alpha) * y_list[i-1]
            y_list.append(new_timestep)
        
        # Stack results back into tensor
        y = torch.stack(y_list, dim=1)  # (B, L, d_inner)
        
        # Add skip connection for immediate price signals
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MambaActorNetwork(nn.Module):
    """
    Mamba-based Actor Network for Trading
    Captures both short-term patterns and long-term market trends efficiently
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
        
        # 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🌊 Mamba Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Input embedding for financial features
        self.input_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Mamba blocks for temporal pattern learning
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
        
        # Feature fusion network
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
        
        # Policy output layers
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        self._initialize_weights()
        self.to(device)
        
        LOGGER.info(f"Mamba Actor 네트워크 초기화 완료: {mamba_layers}개 Mamba 블록")
    
    def _initialize_weights(self):
        """Xavier 초기화 적용"""
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
        
        # Mamba blocks for temporal learning
        for mamba_block in self.mamba_blocks:
            x = mamba_block(x)
        
        # Temporal aggregation (attention-like pooling)
        # Using the last timestep (most recent market state)
        market_features = x[:, -1, :]  # (B, hidden_dim)
        
        # Portfolio features
        portfolio_features = self.portfolio_fc(portfolio_state)
        
        # Feature fusion
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
        fused_features = self.fusion_network(combined_features)
        
        # Policy output
        mean = self.mean(fused_features)
        log_std = self.log_std(fused_features)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """행동 샘플링 with reparameterization trick"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        y_t = torch.tanh(x_t)  # Squash to [-1, 1]
        
        # Log probability with Jacobian correction
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
        
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        if current_feature_dim > self.expected_feature_dim:
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
        else:
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim}")
        
        return adjusted_data
    
    def get_expected_input_shape(self) -> Tuple[int, int]:
        return (self.window_size, self.expected_feature_dim)
    
    def set_adaptive_input(self, enabled: bool):
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 Mamba 동적 입력 조정: {'활성화' if enabled else '비활성화'}")
    
    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: Optional[int] = None):
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 Mamba 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
    
    def to(self, device: torch.device) -> 'MambaActorNetwork':
        self.device = device
        return super(MambaActorNetwork, self).to(device)


class MambaCriticNetwork(nn.Module):
    """
    Mamba-based Critic Network for Trading
    Dual Q-networks with state space modeling for value estimation
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
        
        # 동적 차원 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🌊 Mamba Critic 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Q1 network components
        self.q1_input_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.q1_mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, d_state=d_state, device=device) 
            for _ in range(mamba_layers)
        ])
        
        # Q2 network components
        self.q2_input_embedding = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.q2_mamba_blocks = nn.ModuleList([
            MambaBlock(d_model=hidden_dim, d_state=d_state, device=device) 
            for _ in range(mamba_layers)
        ])
        
        # Portfolio and action processing for Q1
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
        
        # Portfolio and action processing for Q2
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
        
        # Fusion networks
        fusion_input_size = hidden_dim + hidden_dim // 2  # market + portfolio + action
        
        self.q1_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2_fusion = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        
        self._initialize_weights()
        self.to(device)
        
        LOGGER.info(f"Mamba Critic 네트워크 초기화 완료: {mamba_layers}개 Mamba 블록")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Double Q-function forward pass"""
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
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        
        # Q1 computation
        q1_x = self.q1_input_embedding(market_data)
        for mamba_block in self.q1_mamba_blocks:
            q1_x = mamba_block(q1_x)
        q1_market_features = q1_x[:, -1, :]  # Use last timestep
        
        q1_p = self.q1_portfolio_fc(portfolio_state)
        q1_a = self.q1_action_fc(action)
        q1_combined = torch.cat([q1_market_features, q1_p, q1_a], dim=1)
        q1 = self.q1_fusion(q1_combined)
        
        # Q2 computation
        q2_x = self.q2_input_embedding(market_data)
        for mamba_block in self.q2_mamba_blocks:
            q2_x = mamba_block(q2_x)
        q2_market_features = q2_x[:, -1, :]  # Use last timestep
        
        q2_p = self.q2_portfolio_fc(portfolio_state)
        q2_a = self.q2_action_fc(action)
        q2_combined = torch.cat([q2_market_features, q2_p, q2_a], dim=1)
        q2 = self.q2_fusion(q2_combined)
        
        return q1, q2
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """입력 차원 동적 조정 (Actor와 동일)"""
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
        
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        if current_feature_dim > self.expected_feature_dim:
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"🔧 특성 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
        else:
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, 
                                device=market_data.device, dtype=market_data.dtype)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"🔧 특성 차원 확장: {current_feature_dim} → {self.expected_feature_dim}")
        
        return adjusted_data
    
    def get_expected_input_shape(self) -> Tuple[int, int]:
        return (self.window_size, self.expected_feature_dim)
    
    def set_adaptive_input(self, enabled: bool):
        self.adaptive_input = enabled
        LOGGER.info(f"🔧 Mamba 동적 입력 조정: {'활성화' if enabled else '비활성화'}")
    
    def update_expected_dimensions(self, new_feature_dim: int, new_window_size: Optional[int] = None):
        old_feature_dim = self.expected_feature_dim
        self.expected_feature_dim = new_feature_dim
        
        if new_window_size:
            self.window_size = new_window_size
        
        LOGGER.info(f"🔧 Mamba 기대 차원 업데이트: 특성 {old_feature_dim} → {new_feature_dim}")
    
    def to(self, device: torch.device) -> 'MambaCriticNetwork':
        self.device = device
        return super(MambaCriticNetwork, self).to(device)


if __name__ == "__main__":
    # Mamba 네트워크 테스트 코드
    print("🌊 Mamba 네트워크 테스트 시작")
    
    window_size = 30
    feature_dim = 10
    batch_size = 4
    action_dim = 1
    
    # 테스트 데이터 생성
    market_data = torch.randn(batch_size, window_size, feature_dim)
    portfolio_state = torch.randn(batch_size, 2)
    action = torch.randn(batch_size, action_dim)
    
    state_dict = {
        'market_data': market_data,
        'portfolio_state': portfolio_state
    }
    
    # Mamba Actor 테스트
    print("\n🎭 Mamba Actor 테스트")
    mamba_actor = MambaActorNetwork((window_size, feature_dim), action_dim)
    
    mean, log_std = mamba_actor(state_dict)
    action_sample, log_prob, _ = mamba_actor.sample(state_dict)
    
    print(f"✅ Actor 출력 - 평균: {mean.shape}, 로그 표준편차: {log_std.shape}")
    print(f"✅ Actor 샘플 - 행동: {action_sample.shape}, 로그 확률: {log_prob.shape}")
    
    # Mamba Critic 테스트
    print("\n🎯 Mamba Critic 테스트")
    mamba_critic = MambaCriticNetwork((window_size, feature_dim), action_dim)
    
    q1, q2 = mamba_critic(state_dict, action_sample)
    
    print(f"✅ Critic 출력 - Q1: {q1.shape}, Q2: {q2.shape}")
    
    print("\n🎉 Mamba 네트워크 테스트 완료!")
    print("🌊 메모리 효율적인 장기 의존성 학습 준비 완료") 