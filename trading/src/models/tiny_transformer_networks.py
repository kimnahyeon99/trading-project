"""
TinyTransformer 기반 SAC Actor와 Critic 네트워크 구현
시계열 데이터에 최적화된 경량 Transformer 아키텍처
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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


class PositionalEncoding(nn.Module):
    """시계열 데이터를 위한 위치 인코딩"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TinyTransformerBlock(nn.Module):
    """경량화된 Transformer 블록"""
    
    def __init__(
        self, 
        d_model: int, 
        nhead: int = 4, 
        dim_feedforward: int = None, 
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        
        if dim_feedforward is None:
            dim_feedforward = d_model * 2  # 일반적인 4배 대신 2배로 경량화
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'gelu':
            self.activation = F.gelu
        elif activation == 'relu':
            self.activation = F.relu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, d_model]
            src_mask: attention mask
        """
        # Self-attention with residual connection
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward with residual connection
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TinyTransformerEncoder(nn.Module):
    """경량화된 Transformer 인코더"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = None,
        dropout: float = 0.1,
        max_seq_len: int = 200
    ):
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TinyTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model)
        
        # 가중치 초기화
        self._initialize_weights()
    
    def _initialize_weights(self):
        """가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            src: [batch_size, seq_len, input_dim]
            mask: attention mask
        Returns:
            encoded: [batch_size, seq_len, d_model]
        """
        # Input projection
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Positional encoding (transpose for positional encoding, then back)
        src = src.transpose(0, 1)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply transformer blocks
        for transformer_block in self.transformer_blocks:
            src = transformer_block(src, mask)
        
        # Final layer norm
        src = self.norm(src)
        
        return src


class TinyTransformerActorNetwork(nn.Module):
    """
    TinyTransformer 기반 Actor 네트워크
    시계열 데이터의 temporal dependency를 효율적으로 모델링
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        log_std_min: float = -2.0,
        log_std_max: float = 2.0,
        device: torch.device = DEVICE
    ):
        super().__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.device = device
        
        # 동적 입력 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🤖 TinyTransformer Actor 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # TinyTransformer 인코더 (market_data 처리용)
        self.transformer_encoder = TinyTransformerEncoder(
            input_dim=self.feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout_rate,
            max_seq_len=self.window_size + 10  # 여유분 추가
        )
        
        # Global average pooling 또는 attention pooling
        self.attention_pooling = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout_rate, batch_first=True
        )
        self.pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 포트폴리오 상태 처리
        self.portfolio_fc = nn.Sequential(
            nn.Linear(2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 특성 융합 네트워크
        fusion_input_size = d_model + hidden_dim // 4
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_input_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 출력 레이어
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # 가중치 초기화
        self._initialize_weights()
        
        self.to(device)
        LOGGER.info(f"TinyTransformer Actor 네트워크 초기화 완료: "
                   f"d_model={d_model}, heads={nhead}, layers={num_layers}")
    
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
        if len(market_data.shape) == 2:  # (W, F)
            market_data = market_data.unsqueeze(0)  # (1, W, F)
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        batch_size = market_data.size(0)
        
        # Transformer encoding
        encoded = self.transformer_encoder(market_data)  # [B, W, d_model]
        
        # Attention pooling to get global representation
        query = self.pooling_query.expand(batch_size, -1, -1)  # [B, 1, d_model]
        pooled_output, _ = self.attention_pooling(query, encoded, encoded)  # [B, 1, d_model]
        market_features = pooled_output.squeeze(1)  # [B, d_model]
        
        # 포트폴리오 상태 처리
        portfolio_features = self.portfolio_fc(portfolio_state)
        
        # 특성 융합
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
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
        """입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정"""
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정
        if current_feature_dim > self.expected_feature_dim:
            # 특성 차원이 더 큰 경우: 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"📊 TinyTransformer Actor 입력 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
        else:
            # 특성 차원이 더 작은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, device=market_data.device)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"📊 TinyTransformer Actor 입력 차원 확장: {current_feature_dim} → {self.expected_feature_dim}")
        
        return adjusted_data


class TinyTransformerCriticNetwork(nn.Module):
    """
    TinyTransformer 기반 Critic 네트워크
    Double Q-learning을 위한 두 개의 Q 네트워크
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, int],  # (window_size, feature_dim)
        action_dim: int = 1,
        hidden_dim: int = HIDDEN_DIM,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout_rate: float = 0.1,
        device: torch.device = DEVICE
    ):
        super().__init__()
        
        self.window_size, self.feature_dim = input_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.d_model = d_model
        self.device = device
        
        # 동적 입력 조정 설정
        self.expected_feature_dim = self.feature_dim
        self.adaptive_input = True
        
        LOGGER.info(f"🤖 TinyTransformer Critic 초기화: 기대 특성 차원 {self.expected_feature_dim}")
        
        # Q1 네트워크의 Transformer 인코더
        self.q1_transformer_encoder = TinyTransformerEncoder(
            input_dim=self.feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout_rate,
            max_seq_len=self.window_size + 10
        )
        
        # Q2 네트워크의 Transformer 인코더
        self.q2_transformer_encoder = TinyTransformerEncoder(
            input_dim=self.feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout_rate,
            max_seq_len=self.window_size + 10
        )
        
        # Attention pooling for Q1 and Q2
        self.q1_attention_pooling = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout_rate, batch_first=True
        )
        self.q2_attention_pooling = nn.MultiheadAttention(
            d_model, num_heads=1, dropout=dropout_rate, batch_first=True
        )
        self.q1_pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        self.q2_pooling_query = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Q1 포트폴리오 및 행동 처리
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
        
        # Q2 포트폴리오 및 행동 처리
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
        
        # Q1, Q2 융합 네트워크
        fusion_input_size = d_model + hidden_dim // 2  # market + portfolio + action
        
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
        
        # 가중치 초기화
        self._initialize_weights()
        
        self.to(device)
        LOGGER.info(f"TinyTransformer Critic 네트워크 초기화 완료: "
                   f"d_model={d_model}, heads={nhead}, layers={num_layers}")
    
    def _initialize_weights(self):
        """Xavier 초기화 적용"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """순전파"""
        market_data = state['market_data']
        portfolio_state = state['portfolio_state']
        
        # 동적 입력 차원 조정
        if self.adaptive_input:
            market_data = self._adjust_input_dimensions(market_data)
        
        # 차원 처리
        if len(market_data.shape) == 2:
            market_data = market_data.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        if len(portfolio_state.shape) == 1:
            portfolio_state = portfolio_state.unsqueeze(0)
        
        batch_size = market_data.size(0)
        
        # Q1 계산
        q1_encoded = self.q1_transformer_encoder(market_data)
        q1_query = self.q1_pooling_query.expand(batch_size, -1, -1)
        q1_pooled_output, _ = self.q1_attention_pooling(q1_query, q1_encoded, q1_encoded)
        q1_market_features = q1_pooled_output.squeeze(1)
        
        q1_portfolio_features = self.q1_portfolio_fc(portfolio_state)
        q1_action_features = self.q1_action_fc(action)
        q1_combined = torch.cat([q1_market_features, q1_portfolio_features, q1_action_features], dim=1)
        q1 = self.q1_fusion(q1_combined)
        
        # Q2 계산
        q2_encoded = self.q2_transformer_encoder(market_data)
        q2_query = self.q2_pooling_query.expand(batch_size, -1, -1)
        q2_pooled_output, _ = self.q2_attention_pooling(q2_query, q2_encoded, q2_encoded)
        q2_market_features = q2_pooled_output.squeeze(1)
        
        q2_portfolio_features = self.q2_portfolio_fc(portfolio_state)
        q2_action_features = self.q2_action_fc(action)
        q2_combined = torch.cat([q2_market_features, q2_portfolio_features, q2_action_features], dim=1)
        q2 = self.q2_fusion(q2_combined)
        
        return q1, q2
    
    def _adjust_input_dimensions(self, market_data: torch.Tensor) -> torch.Tensor:
        """입력 데이터의 차원을 모델이 기대하는 차원으로 동적 조정"""
        if len(market_data.shape) == 3:  # (B, W, F)
            current_feature_dim = market_data.shape[2]
            batch_size, window_size = market_data.shape[0], market_data.shape[1]
        elif len(market_data.shape) == 2:  # (W, F)
            current_feature_dim = market_data.shape[1]
            batch_size, window_size = 1, market_data.shape[0]
            market_data = market_data.unsqueeze(0)
        else:
            LOGGER.warning(f"⚠️ 예상치 못한 입력 형태: {market_data.shape}")
            return market_data
        
        # 차원이 일치하면 그대로 반환
        if current_feature_dim == self.expected_feature_dim:
            return market_data
        
        # 차원 조정
        if current_feature_dim > self.expected_feature_dim:
            # 특성 차원이 더 큰 경우: 자르기
            adjusted_data = market_data[:, :, :self.expected_feature_dim]
            LOGGER.debug(f"📊 TinyTransformer Critic 입력 차원 축소: {current_feature_dim} → {self.expected_feature_dim}")
        else:
            # 특성 차원이 더 작은 경우: 제로 패딩
            padding_size = self.expected_feature_dim - current_feature_dim
            padding = torch.zeros(batch_size, window_size, padding_size, device=market_data.device)
            adjusted_data = torch.cat([market_data, padding], dim=2)
            LOGGER.debug(f"📊 TinyTransformer Critic 입력 차원 확장: {current_feature_dim} → {self.expected_feature_dim}")
        
        return adjusted_data


if __name__ == "__main__":
    # 테스트 코드
    print("=== TinyTransformer 네트워크 테스트 ===")
    
    # 테스트 파라미터
    batch_size = 4
    window_size = WINDOW_SIZE
    feature_dim = 5
    action_dim = 1
    d_model = 64  # 테스트용 작은 모델
    
    # 네트워크 생성
    actor = TinyTransformerActorNetwork(
        input_shape=(window_size, feature_dim),
        action_dim=action_dim,
        d_model=d_model,
        nhead=2,
        num_layers=2
    )
    
    critic = TinyTransformerCriticNetwork(
        input_shape=(window_size, feature_dim),
        action_dim=action_dim,
        d_model=d_model,
        nhead=2,
        num_layers=2
    )
    
    # 테스트 데이터 생성
    market_data = torch.randn(batch_size, window_size, feature_dim)
    portfolio_state = torch.randn(batch_size, 2)
    action = torch.randn(batch_size, action_dim)
    
    state_dict = {
        'market_data': market_data,
        'portfolio_state': portfolio_state
    }
    
    # Actor 테스트
    print(f"입력 형태: market_data={market_data.shape}, portfolio_state={portfolio_state.shape}")
    
    mean, log_std = actor(state_dict)
    sampled_action, log_prob, _ = actor.sample(state_dict)
    
    print(f"Actor 출력: mean={mean.shape}, log_std={log_std.shape}")
    print(f"Actor 샘플: action={sampled_action.shape}, log_prob={log_prob.shape}")
    
    # Critic 테스트
    q1, q2 = critic(state_dict, action)
    print(f"Critic 출력: Q1={q1.shape}, Q2={q2.shape}")
    
    print("✅ TinyTransformer 네트워크 테스트 완료!") 