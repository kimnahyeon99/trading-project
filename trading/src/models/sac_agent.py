"""
SAC (Soft Actor-Critic) 알고리즘 구현 모듈 (Config 통합 버전)
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Union, Optional, Any
from collections import deque
import random
import pickle
import time
from pathlib import Path
import sys
import os

from src.config.ea_teb_config import (
    DEVICE,
    HIDDEN_DIM,
    GAMMA,
    TAU,
    LEARNING_RATE_ACTOR,
    LEARNING_RATE_CRITIC,
    LEARNING_RATE_ALPHA,
    ALPHA_INIT,
    REPLAY_BUFFER_SIZE,
    TARGET_UPDATE_INTERVAL,
    BATCH_SIZE,
    MODELS_DIR,
    LOGGER,
    MAX_STEPS_PER_EPISODE,
    WINDOW_SIZE,
    WEIGHT_DECAY,
    TARGET_ENTROPY_FACTOR
)

from src.models.networks import (
    ActorNetwork, 
    CriticNetwork, 
    CNNActorNetwork, 
    CNNCriticNetwork,
    LSTMActorNetwork,
    LSTMCriticNetwork
)
from src.models.mamba_networks import (
    MambaActorNetwork,
    MambaCriticNetwork
)
from src.models.tiny_transformer_networks import (
    TinyTransformerActorNetwork,
    TinyTransformerCriticNetwork
)

from src.utils.utils import soft_update, create_directory

    
class ReplayBuffer:
    """
    리플레이 버퍼 (정리된 버전 - 핵심 기능만 유지)
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        """
        리플레이 버퍼 초기화
        
        Args:
            capacity: 버퍼 최대 용량
        """
        self.capacity = capacity
        self.buffer: List[Tuple] = []
        self.position = 0
    
    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        경험을 버퍼에 저장
        
        Args:
            state: 현재 상태
            action: 수행한 액션
            reward: 받은 보상
            next_state: 다음 상태  
            done: 에피소드 종료 여부
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        배치 샘플링
        
        Args:
            batch_size: 샘플링할 배치 크기
            
        Returns:
            샘플링된 경험들의 튜플
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
            
        experiences = random.sample(self.buffer, batch_size)
        
        # 각 요소별로 분리
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """
        버퍼 크기 반환
        
        Returns:
            현재 버퍼에 저장된 경험의 수
        """
        return len(self.buffer)





class SACAgent:
    """
    SAC 알고리즘 에이전트 (정리된 버전)
    """
        
    def __init__(
        self,
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        hidden_dim: int = HIDDEN_DIM,
        actor_lr: float = LEARNING_RATE_ACTOR,
        critic_lr: float = LEARNING_RATE_CRITIC,
        alpha_lr: float = LEARNING_RATE_ALPHA,
        gamma: float = GAMMA,
        tau: float = TAU,
        alpha_init: float = ALPHA_INIT,
        target_update_interval: int = TARGET_UPDATE_INTERVAL,
        use_automatic_entropy_tuning: bool = True,
        device: torch.device = DEVICE,
        buffer_capacity: int = REPLAY_BUFFER_SIZE,
        input_shape: Optional[Tuple[int, int]] = None,
        use_cnn: bool = False,
        use_lstm: bool = False,
        use_mamba: bool = False,
        use_tinytransformer: bool = False,
        model_type: Optional[str] = None,
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.1,
        training_symbols: Optional[List[str]] = None,
        lstm_dropout: float = 0.2,
        symbol: Optional[str] = None,
        gradient_clip_norm: float = 1.0,
        use_gradient_clipping: bool = True,
        adaptive_alpha: bool = True
    ):
        """SACAgent 클래스 초기화 (Config 통합 버전)"""
        
        # 기본 속성 설정
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau = tau
        self.target_update_interval = target_update_interval
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.device = device
        self.use_cnn = use_cnn
        self.use_lstm = use_lstm
        self.use_mamba = use_mamba
        self.use_tinytransformer = use_tinytransformer
        self.input_shape = input_shape
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_dropout = lstm_dropout
        self.symbol = symbol
        
        # model_type 처리 (auto-detect if not provided)
        if model_type is None:
            if use_tinytransformer:
                model_type = 'tinytransformer'
            elif use_mamba:
                model_type = 'mamba'
            elif use_lstm:
                model_type = 'lstm'
            elif use_cnn:
                model_type = 'cnn'
            else:
                model_type = 'mlp'
        self.model_type = model_type
        
        # 통합된 파라미터 설정
        self.dropout_rate = dropout_rate
        self.training_symbols = training_symbols or []
        
        # ✅ 전달받은 하이퍼파라미터 사용 (하이퍼파라미터 최적화 호환)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.alpha_init = alpha_init
        
        LOGGER.info(f"📊 {model_type.upper() if model_type else 'SAC'} 에이전트 파라미터:")
        LOGGER.info(f"   └─ Actor LR: {self.actor_lr:.6f}")
        LOGGER.info(f"   └─ Critic LR: {self.critic_lr:.6f}")
        LOGGER.info(f"   └─ Dropout: {self.dropout_rate}")
        
        # 안정성 개선 설정 (통합)
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clip_norm = gradient_clip_norm
        self.adaptive_alpha = adaptive_alpha
        
        # ✅ SAC 표준: 자동 엔트로피 튜닝 (고정 target_entropy)
        if self.adaptive_alpha:
            self.recent_entropies: deque = deque(maxlen=100)
        
        # ✅ SAC 표준: 고정 엔트로피 타겟 설정 (설정값 반영)
        action_dim_safe = action_dim or 1
        self.target_entropy = action_dim_safe * TARGET_ENTROPY_FACTOR  # config에서 설정한 값 사용
        LOGGER.info(f"📊 엔트로피 타겟: {self.target_entropy} (Factor: {TARGET_ENTROPY_FACTOR})")

        # 학습 단계 카운터
        self.train_step_counter = 0
        self.update_counter = 0

        # 모델 타입 검증
        model_count = sum([use_cnn, use_lstm, use_mamba, use_tinytransformer])
        if model_count > 1:
            raise ValueError("CNN, LSTM, Mamba, TinyTransformer 모델 중 하나만 선택할 수 있습니다.")

        # 모델 타입 로깅
        if use_tinytransformer:
            LOGGER.info("🤖 TinyTransformer 기반 SAC 에이전트 초기화 중...")
        elif use_mamba:
            LOGGER.info("🌊 Mamba 기반 SAC 에이전트 초기화 중...")
        elif use_lstm:
            LOGGER.info("LSTM 기반 SAC 에이전트 초기화 중...")
        elif use_cnn:
            LOGGER.info("개선된 CNN 기반 SAC 에이전트 초기화 중...")
        else:
            LOGGER.info("MLP 기반 SAC 에이전트 초기화 중...")

        # TradingEnvironment를 위한 상태 차원 자동 계산
        if not use_cnn and not use_lstm and not use_mamba and not use_tinytransformer and state_dim is None:
            if input_shape is None:
                raise ValueError("input_shape를 명시적으로 제공해야 합니다.")
            self.state_dim = input_shape[0] * input_shape[1] + 2
            state_dim = self.state_dim
            LOGGER.info(f"📏 상태 차원 자동 계산: {self.state_dim}")

        # 네트워크 초기화
        self._initialize_networks()

        # 타겟 네트워크 초기화
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # ✅ 옵티마이저 설정 (SAC 표준: 고정 학습률)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, weight_decay=WEIGHT_DECAY)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, weight_decay=WEIGHT_DECAY)
        
        # ✅ Alpha 자동 튜닝 (SAC 핵심 기능) - 이론적으로 올바른 초기화
        if self.use_automatic_entropy_tuning:
            # ✅ 안정적인 초기화 (더 보수적인 alpha 학습률)
            conservative_alpha_lr = alpha_lr * 0.3  # Alpha 학습률을 더 보수적으로
            
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.log_alpha.data.fill_(np.log(alpha_init))  # alpha_init의 로그값으로 초기화
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=conservative_alpha_lr)
            self.alpha = self.log_alpha.exp()
            
            LOGGER.info(f"🎯 SAC 자동 엔트로피 튜닝 활성화:")
            LOGGER.info(f"   └─ 초기 Alpha: {self.alpha.item():.4f}")
            LOGGER.info(f"   └─ 초기 log_Alpha: {self.log_alpha.item():.4f}")
            LOGGER.info(f"   └─ Alpha 학습률: {conservative_alpha_lr:.6f} (보수적)")
            LOGGER.info(f"   └─ Target Entropy: {self.target_entropy}")
        else:
            self.alpha = torch.tensor(self.alpha_init, device=device)
            LOGGER.info(f"🎯 고정 Alpha 사용: {self.alpha.item():.4f}")
        
        # ✅ 리플레이 버퍼
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # ✅ 배치 크기 저장 (하이퍼파라미터 최적화에서 필요)
        self.batch_size = BATCH_SIZE
        
        # 학습 통계
        self.actor_losses: List[float] = []
        self.critic_losses: List[float] = []
        self.alpha_losses: List[float] = []
        self.entropy_values: List[float] = []
        
        # 최종 로그
        model_type_display = "LSTM" if use_lstm else ("CNN" if use_cnn else "MLP")
        LOGGER.info(f"🎉 SAC 에이전트 초기화 완료!")
        LOGGER.info(f"   └─ 모델 타입: {model_type_display}")
        LOGGER.info(f"   └─ 행동 차원: {action_dim}")
        LOGGER.info(f"   └─ 상태 차원: {self.state_dim if not (use_cnn or use_lstm) else input_shape}")
        LOGGER.info(f"   └─ 장치: {device}")
        if use_cnn:
            LOGGER.info(f"   └─ CNN 설정 소스: {'심볼별 최적화' if symbol else '기본 CNN 설정'}")
        
    def _initialize_networks(self):
        """개선된 네트워크 초기화"""
        if self.use_lstm:
            # LSTM 네트워크
            if self.input_shape is None:
                raise ValueError("LSTM 모델을 사용할 때는 input_shape가 필요합니다.")
            
            self.actor = LSTMActorNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                num_lstm_layers=self.num_lstm_layers,
                dropout=self.lstm_dropout,
                device=self.device
            )
            
            self.critic = LSTMCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                num_lstm_layers=self.num_lstm_layers,
                dropout=self.lstm_dropout,
                device=self.device
            )
            
            self.critic_target = LSTMCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                lstm_hidden_dim=self.lstm_hidden_dim,
                num_lstm_layers=self.num_lstm_layers,
                dropout=self.lstm_dropout,
                device=self.device
            )
            
        elif self.use_cnn:
            # 🖼️ 개선된 CNN 네트워크 사용 (config 값 적용)
            if self.input_shape is None:
                raise ValueError("CNN 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ 개선된 CNN 네트워크 생성 중...")
            
            self.actor = CNNActorNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,  # ✅ 수정: cnn_dropout_rate → dropout_rate
                device=self.device
            )
            
            self.critic = CNNCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,  # ✅ 수정: cnn_dropout_rate → dropout_rate
                device=self.device
            )
            
            self.critic_target = CNNCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.dropout_rate,  # ✅ 수정: cnn_dropout_rate → dropout_rate
                device=self.device
            )
            
            LOGGER.info("✅ 개선된 CNN 네트워크 생성 완료")
            
        elif self.use_mamba:
            # 🌊 Mamba 네트워크 사용
            if self.input_shape is None:
                raise ValueError("Mamba 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ Mamba 네트워크 생성 중...")
            
            self.actor = MambaActorNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                mamba_layers=4,  # Default Mamba layers
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            self.critic = MambaCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                mamba_layers=4,  # Default Mamba layers
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            self.critic_target = MambaCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                mamba_layers=4,  # Default Mamba layers
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            LOGGER.info("✅ Mamba 네트워크 생성 완료")
            
        elif self.use_tinytransformer:
            # 🤖 TinyTransformer 네트워크 사용
            if self.input_shape is None:
                raise ValueError("TinyTransformer 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ TinyTransformer 네트워크 생성 중...")
            
            self.actor = TinyTransformerActorNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                d_model=128,  # Default transformer dimension
                nhead=4,      # Default number of attention heads
                num_layers=2, # Default number of transformer layers
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            self.critic = TinyTransformerCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            self.critic_target = TinyTransformerCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                d_model=128,
                nhead=4,
                num_layers=2,
                dropout_rate=self.dropout_rate,
                device=self.device
            )
            
            LOGGER.info("✅ TinyTransformer 네트워크 생성 완료")
                
        else:
            # MLP 네트워크
            if self.state_dim is None:
                raise ValueError("일반 모델을 사용할 때는 state_dim이 필요합니다.")
            
            self.actor = ActorNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                device=self.device
            )
            
            self.critic = CriticNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                device=self.device
            )
            
            self.critic_target = CriticNetwork(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                device=self.device
            )
    def select_action(self, state: Union[Dict[str, np.ndarray], Dict[str, torch.Tensor], np.ndarray], evaluate: bool = False) -> float:
        """
        상태에 따른 행동 선택 (실시간 트레이딩 호환)
        """
        try:
            # 상태를 네트워크 입력 형태로 변환
            if isinstance(state, dict):
                state_tensor = self._process_state_for_network(state)
            elif isinstance(state, np.ndarray):
                if self.use_cnn or self.use_lstm or self.use_mamba or self.use_tinytransformer:
                    if self.input_shape and len(state) == (self.input_shape[0] * self.input_shape[1] + 2):
                        market_data_flat = state[:-2]
                        portfolio_state = state[-2:]
                        market_data = market_data_flat.reshape(self.input_shape)
                        state = {
                            'market_data': market_data,
                            'portfolio_state': portfolio_state
                        }
                        state_tensor = self._process_state_for_network(state)
                    else:
                        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            elif isinstance(state, torch.Tensor):
                state_tensor = state.to(self.device)
                if state_tensor.dim() == 1:
                    state_tensor = state_tensor.unsqueeze(0)
            else:
                raise ValueError(f"지원하지 않는 상태 타입: {type(state)}")
            
            # 행동 선택 - BatchNorm 오류 방지를 위해 eval 모드 설정
            self.actor.eval()
            with torch.no_grad():
                if evaluate:
                    # ✅ Evaluation 모드: 약간의 변동성을 가진 deterministic 행동
                    action, log_prob, mean = self.actor.sample(state_tensor)
                    # 평균값 기반이지만 약간의 변동성 유지 (실제 샘플링 값 사용)
                    action_value = float(action.detach().cpu().numpy().flatten()[0])
                    
                    # 추가 안정성: 극단값 클리핑
                    action_value = np.clip(action_value, -0.99, 0.99)
                else:
                    # ✅ Training 모드: 탐험적 샘플링
                    action, log_prob, mean = self.actor.sample(state_tensor)
                    action_value = float(action.detach().cpu().numpy().flatten()[0])
            self.actor.train()
            
            # 학습 초기에 추가 탐색 노이즈 주입 (훈련 모드에만)
            if not evaluate and self.train_step_counter < 5000:
                noise_std = max(0.1, 0.5 * (1 - self.train_step_counter / 5000))
                noise = np.random.normal(0, noise_std)
                action_value = np.clip(action_value + noise, -1.0, 1.0)
                
            return action_value
            
        except Exception as e:
            LOGGER.error(f"행동 선택 중 오류: {e}")
            return 0.0

    def _process_state_for_network(self, state: Dict[str, Union[np.ndarray, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        상태를 네트워크 입력 형태로 변환
        
        Args:
            state: TradingEnvironment 상태 딕셔너리
            
        Returns:
            네트워크 입력용 텐서 또는 텐서 딕셔너리
        """
        try:
            if self.use_cnn or self.use_lstm or self.use_mamba or self.use_tinytransformer:
                # CNN / LSTM / Mamba / TinyTransformer: 상태를 딕셔너리 형태로 유지
                market_data = state["market_data"]
                portfolio_state = state["portfolio_state"]
                
                # numpy 배열을 텐서로 변환
                if isinstance(market_data, np.ndarray):
                    market_tensor = torch.FloatTensor(market_data).to(self.device)
                else:
                    market_tensor = market_data.to(self.device)
                    
                if isinstance(portfolio_state, np.ndarray):
                    portfolio_tensor = torch.FloatTensor(portfolio_state).to(self.device)
                else:
                    portfolio_tensor = portfolio_state.to(self.device)
                
                # 배치 차원 확인 및 추가
                if market_tensor.dim() == 2:  # (window_size, feature_dim)
                    market_tensor = market_tensor.unsqueeze(0)  # (1, window_size, feature_dim)
                if portfolio_tensor.dim() == 1:  # (2,)
                    portfolio_tensor = portfolio_tensor.unsqueeze(0)  # (1, 2)
                
                return {
                    "market_data": market_tensor,
                    "portfolio_state": portfolio_tensor
                }
            else:
                # MLP: 상태를 flatten
                market_data = state['market_data']
                portfolio_state = state['portfolio_state']
                
                # numpy 배열을 텐서로 변환
                if isinstance(market_data, np.ndarray):
                    market_flat = market_data.flatten()
                else:
                    market_flat = market_data.flatten().cpu().numpy()
                    
                if isinstance(portfolio_state, np.ndarray):
                    portfolio_flat = portfolio_state
                else:
                    portfolio_flat = portfolio_state.cpu().numpy()
                
                # 결합
                combined_state = np.concatenate([market_flat, portfolio_flat])
                
                return torch.FloatTensor(combined_state).unsqueeze(0).to(self.device)
            
        except Exception as e:
            LOGGER.error(f"상태 변환 중 오류: {e}")
            # 기본 상태 반환
            if self.use_cnn or self.use_lstm or self.use_mamba or self.use_tinytransformer:
                if self.input_shape is not None:
                    return {
                        "market_data": torch.zeros((1, self.input_shape[0], self.input_shape[1]), device=self.device),
                        "portfolio_state": torch.zeros((1, 2), device=self.device)
                    }
                else:
                    return {
                        "market_data": torch.zeros((1, 10, 5), device=self.device),  # fallback dimensions
                        "portfolio_state": torch.zeros((1, 2), device=self.device)
                    }
            else:
                return torch.zeros((1, self.state_dim or 1), device=self.device)

    def _process_batch_states(self, states: List[Dict[str, np.ndarray]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        배치 상태들을 네트워크 입력으로 변환
        """
        # 🔄 GPU 메모리 체크를 맨 앞으로 이동
        if self.device.type == 'cuda':
            total_memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            memory_threshold = total_memory_gb * 0.85
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            
            if memory_allocated > memory_threshold:
                LOGGER.warning(f"GPU 메모리 사용량 높음: {memory_allocated:.1f}GB/{total_memory_gb:.1f}GB ({memory_allocated/total_memory_gb*100:.1f}%)")
        
        if self.use_cnn or self.use_lstm or self.use_mamba or self.use_tinytransformer:
            market_batch = []
            portfolio_batch = []
            for state in states:
                market_batch.append(state["market_data"])
                portfolio_batch.append(state["portfolio_state"])
            
            market_tensor = torch.FloatTensor(np.stack(market_batch)).to(self.device)
            portfolio_tensor = torch.FloatTensor(np.stack(portfolio_batch)).to(self.device)

            return {
                "market_data": market_tensor,
                "portfolio_state": portfolio_tensor
            }
        else:
            batch_states = []
            for state in states:
                processed_state = self._process_state_for_network(state)
                batch_states.append(processed_state)
            return torch.cat(batch_states, dim=0).to(self.device)
        
    def add_experience(self, state: Dict[str, np.ndarray], action: float, reward: float, 
                      next_state: Dict[str, np.ndarray], done: bool) -> None:
        """
        경험을 리플레이 버퍼에 추가
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 종료 여부
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update_parameters(self, batch_size: int = BATCH_SIZE) -> Dict[str, float]:
        """통합된 네트워크 파라미터 업데이트 (안정성 개선)"""
        if len(self.replay_buffer) < batch_size:
            return {
                'actor_loss': 0.0,
                'critic_loss': 0.0,
                'alpha_loss': 0.0,
                'entropy': 0.0,
                'alpha': self.alpha.item() if isinstance(self.alpha, torch.Tensor) else self.alpha
            }
        
        # 랜덤 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 배치 텐서 변환
        batched_states = self._process_batch_states(states)
        batched_next_states = self._process_batch_states(next_states)
        
        batched_actions = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        batched_rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        batched_dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # ✅ 통합된 업데이트 (모든 모델에 동일한 안정성 적용)
        critic_loss = self._update_critic(
            batched_states, batched_actions, batched_rewards, batched_next_states, batched_dones
        )
        actor_loss = self._update_actor(batched_states)
        alpha_loss = self._update_alpha(batched_states)
        
        # ✅ 개선된 타겟 네트워크 업데이트
        self.train_step_counter += 1
        self.update_counter += 1  # ✅ update_counter 증가 추가
        if self.train_step_counter % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        # 엔트로피 계산 (올바른 공식 + 디버깅)
        entropy_value = 0.0
        if self.update_counter % 10 == 0:
            with torch.no_grad():
                _, log_probs, _ = self.actor.sample(batched_states)
                # ✅ 정확한 엔트로피 계산: H = -E[log π(a|s)]
                entropy_value = -log_probs.mean().item()
                
                # 🔍 디버깅: 매우 작은 음수 엔트로피 감지
                if entropy_value < -0.1:  # 의미있는 음수 엔트로피만 경고
                    LOGGER.debug(f"📊 엔트로피 디버깅 - log_probs 평균: {log_probs.mean().item():.6f}, "
                               f"엔트로피: {entropy_value:.6f}")

        # 학습 통계
        stats = {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha_loss': alpha_loss if self.use_automatic_entropy_tuning else 0.0,
            'entropy': entropy_value,
            'alpha': self.alpha.item()
        }
        
        self.actor_losses.append(stats['actor_loss'])
        self.critic_losses.append(stats['critic_loss'])
        self.alpha_losses.append(stats['alpha_loss'])
        self.entropy_values.append(stats['entropy'])
        
        return stats
    
    def _update_critic(self, states, actions, rewards, next_states, dones):
        """통합된 Critic 네트워크 업데이트 (안정성 개선)"""
        # 현재 정책에서 다음 행동 샘플링
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            
            # 타겟 Q 값 계산
            next_q1_target, next_q2_target = self.critic_target(next_states, next_actions)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_q_target = next_q_target - self.alpha * next_log_probs
            expected_q = rewards + (1.0 - dones) * self.gamma * next_q_target
        
        current_q1, current_q2 = self.critic(states, actions)
        
        q1_loss = F.mse_loss(current_q1, expected_q)
        q2_loss = F.mse_loss(current_q2, expected_q)
        
        critic_loss = q1_loss + q2_loss
        
        # ✅ 안정성 개선: 손실 값 체크
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            LOGGER.warning("Critic 손실에 NaN/Inf 감지됨. 업데이트 건너뜀.")
            return 0.0
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # ✅ 통합된 그래디언트 클리핑 (모든 모델에 적용)
        if self.use_gradient_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), 
                max_norm=self.gradient_clip_norm
            )
            
            # 극단적인 그래디언트 감지
            if grad_norm > self.gradient_clip_norm * 5:
                LOGGER.warning(f"Critic 큰 그래디언트 감지: {grad_norm:.6f}")
        
        self.critic_optimizer.step()
        
        return critic_loss.item()

    def _update_actor(self, states):
        """통합된 Actor 네트워크 업데이트 (안정성 개선)"""
        new_actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, new_actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # ✅ 안정성 개선: 손실 값 체크
        if torch.isnan(actor_loss) or torch.isinf(actor_loss):
            LOGGER.warning("Actor 손실에 NaN/Inf 감지됨. 업데이트 건너뜀.")
            return 0.0
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # ✅ 통합된 그래디언트 클리핑
        if self.use_gradient_clipping:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), 
                max_norm=self.gradient_clip_norm
            )
            
            if grad_norm > self.gradient_clip_norm * 5:
                LOGGER.warning(f"Actor 큰 그래디언트 감지: {grad_norm:.6f}")
        
        self.actor_optimizer.step()
        
        return actor_loss.item()

    def _update_alpha(self, states):
        """표준 SAC Alpha 업데이트 (이론적으로 올바른 방법)"""
        if not self.use_automatic_entropy_tuning:
            return 0.0
        
        # 현재 정책의 행동과 log_prob 샘플링
        with torch.no_grad():
            _, log_probs, _ = self.actor.sample(states)
            current_entropy = -log_probs.mean().item()
        
        # ✅ SAC 표준: 엔트로피 통계 수집
        if self.adaptive_alpha:
            self.recent_entropies.append(current_entropy)
        
        # ✅ SAC 표준 Alpha Loss 공식 (detach() 올바른 위치)
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # ✅ 안정성 체크 (NaN/Inf만 체크)
        if torch.isnan(alpha_loss) or torch.isinf(alpha_loss):
            LOGGER.warning("Alpha 손실에 NaN/Inf 감지됨. 업데이트 건너뜀.")
            return 0.0
        
        # ✅ 적응적 학습률 (극단적 loss 시에만)
        current_alpha_lr = self.alpha_lr
        if abs(alpha_loss.item()) > 5:  # 임계값 완화
            current_alpha_lr = self.alpha_lr * 0.5
            LOGGER.debug(f"Alpha Loss 큼: {alpha_loss.item():.4f}, 학습률 임시 감소")
        
        # Alpha 업데이트
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        
        # ✅ 보수적인 그래디언트 클리핑 (alpha만)
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
        
        # ✅ 적응적 학습률 적용
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = current_alpha_lr
        
        self.alpha_optimizer.step()
        
        # ✅ 학습률 복원
        for param_group in self.alpha_optimizer.param_groups:
            param_group['lr'] = self.alpha_lr
        
        # Alpha 값 업데이트 (제한 없음 - SAC 이론 준수)
        self.alpha = self.log_alpha.exp()
        
        # ✅ 경고만 출력 (값 제한하지 않음)
        if self.alpha.item() > 10:
            LOGGER.warning(f"Alpha 값이 큼: {self.alpha.item():.4f} - 학습이 불안정할 수 있음")
        elif self.alpha.item() < 0.001:
            LOGGER.warning(f"Alpha 값이 작음: {self.alpha.item():.6f} - 탐험이 부족할 수 있음")
        
        # ✅ 디버깅 정보 (주기적) alpha값 디버깅 100스텝 마다 찍고 싶으면 % 100
        if hasattr(self, 'update_counter') and self.update_counter % 1000 == 0:
            LOGGER.info(f"🔍 Alpha 상태 - Loss: {alpha_loss.item():.4f}, Alpha: {self.alpha.item():.4f}, "
                       f"Entropy: {current_entropy:.4f}, Target: {self.target_entropy}")
        
        return alpha_loss.item()
    
    def save_model(self, save_dir: Optional[Union[str, Path]] = None, prefix: str = "",
                model_type: Optional[str] = None, symbol: Optional[str] = None, symbols: Optional[List[str]] = None) -> str:
            """
            모델 저장 (실시간 트레이딩 호환성 개선)

            Args:
                save_dir: 저장 디렉토리
                prefix: 파일명 접두사
                model_type: 모델 타입 ('mlp', 'cnn', 'lstm')
                symbol: 단일 심볼 (MLP용)
                symbols: 다중 심볼 리스트

            Returns:
                str: 저장된 모델의 경로
            """
            if save_dir is None:
                save_dir = MODELS_DIR

            create_directory(save_dir)

            # 타임스탬프 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 모델 타입 자동 감지 (지정되지 않은 경우)
            if model_type is None:
                model_type = getattr(self, 'model_type', None)
                if model_type is None:
                    if self.use_lstm:
                        model_type = 'lstm'
                    elif self.use_cnn:
                        model_type = 'cnn'
                    else:
                        model_type = 'mlp'

            # 파일명 패턴 생성
            if model_type.lower() == 'mlp':
                # MLP: 심볼별로 구분
                if symbol:
                    model_name = f"final_sac_model_{symbol}_{timestamp}"
                elif symbols and len(symbols) == 1:
                    model_name = f"final_sac_model_{symbols[0]}_{timestamp}"
                else:
                    # 다중 심볼이거나 심볼 정보 없음
                    model_name = f"final_sac_model_multi_{timestamp}"
            else:
                # CNN, LSTM: 심볼 구분 없음
                model_name = f"final_{model_type.lower()}_sac_model_{timestamp}"

            # 접두사 적용
            if prefix:
                model_name = f"{prefix}{model_name}"

            # 저장 경로 생성
            model_path = Path(save_dir) / model_name
            create_directory(model_path)

            # 네트워크 가중치 저장
            torch.save(self.actor.state_dict(), model_path / "actor.pth")
            torch.save(self.critic.state_dict(), model_path / "critic.pth")
            torch.save(self.critic_target.state_dict(), model_path / "critic_target.pth")

            # 옵티마이저 상태 저장
            torch.save(self.actor_optimizer.state_dict(), model_path / "actor_optimizer.pth")
            torch.save(self.critic_optimizer.state_dict(), model_path / "critic_optimizer.pth")

            # Alpha 관련 상태 저장
            if self.use_automatic_entropy_tuning:
                torch.save(self.log_alpha, model_path / "log_alpha.pth")
                torch.save(self.alpha_optimizer.state_dict(), model_path / "alpha_optimizer.pth")

            # 학습 통계 저장
            training_stats = {
                'actor_losses': self.actor_losses,
                'critic_losses': self.critic_losses,
                'alpha_losses': self.alpha_losses,
                'entropy_values': self.entropy_values,
                'train_step_counter': self.train_step_counter
            }
            torch.save(training_stats, model_path / "training_stats.pth")

            # ✅ 설정 저장 (CNN config 정보 포함) - 안전한 속성 접근으로 수정
            config = {
                # 기본 모델 정보
                'model_type': model_type.lower(),
                'symbol': symbol or getattr(self, 'symbol', None),
                'symbols': symbols,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'hidden_dim': self.hidden_dim,
                
                # 모델 아키텍처
                'use_cnn': self.use_cnn,
                'use_lstm': self.use_lstm,
                'input_shape': self.input_shape,
                
                # LSTM 전용 파라미터
                'lstm_hidden_dim': getattr(self, 'lstm_hidden_dim', None),
                'num_lstm_layers': getattr(self, 'num_lstm_layers', None),
                'lstm_dropout': getattr(self, 'lstm_dropout', None),
                
                # ✅ CNN/공통 파라미터 (안전한 접근으로 수정)
                'dropout_rate': getattr(self, 'dropout_rate', None),
                'learning_rate_actor': getattr(self, 'actor_lr', None),
                'learning_rate_critic': getattr(self, 'critic_lr', None),
                'learning_rate_alpha': getattr(self, 'alpha_lr', None),
                'alpha_init': getattr(self, 'alpha_init', None),
                'gradient_clip_norm': getattr(self, 'gradient_clip_norm', None),
                
                # 실시간 트레이딩 호환성 정보
                'realtime_compatible': True,
                'supports_dict_state': self.use_cnn or self.use_lstm,
                'supports_flat_state': not (self.use_cnn or self.use_lstm),
                'state_format': 'dict' if (self.use_cnn or self.use_lstm) else 'flat',
                
                # 메타 정보
                'timestamp': timestamp,
                'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                'device': str(self.device),
                'framework': 'pytorch',
                'sac_version': '2.0'
            }
            torch.save(config, model_path / "config.pth")

            model_type_display = model_type.upper()
            LOGGER.info(f"✅ {model_type_display} 모델 저장 완료: {model_path}")
            LOGGER.info(f"   └─ 모델 타입: {model_type_display}")
            LOGGER.info(f"   └─ 심볼: {symbol or symbols or 'Multi'}")
            LOGGER.info(f"   └─ 실시간 호환: {'YES' if config['realtime_compatible'] else 'NO'}")
            LOGGER.info(f"   └─ 상태 형식: {config['state_format'].upper()}")
            LOGGER.info(f"   └─ 타임스탬프: {timestamp}")
            
            # ✅ 모델별 설정 정보 출력 (안전한 접근으로 수정)
            if self.use_cnn:
                dropout_rate = getattr(self, 'dropout_rate', 'N/A')
                actor_lr = getattr(self, 'actor_lr', 'N/A')
                LOGGER.info(f"   └─ CNN 설정: Dropout={dropout_rate}, Actor LR={actor_lr}")
            elif self.use_lstm:
                lstm_hidden = getattr(self, 'lstm_hidden_dim', 'N/A')
                lstm_layers = getattr(self, 'num_lstm_layers', 'N/A')
                LOGGER.info(f"   └─ LSTM 설정: Hidden={lstm_hidden}, Layers={lstm_layers}")
            else:
                actor_lr = getattr(self, 'actor_lr', 'N/A')
                LOGGER.info(f"   └─ MLP 설정: Actor LR={actor_lr}")

            return str(model_path)
        
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        모델 로드
        
        Args:
            model_path: 모델 디렉토리 경로
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            LOGGER.error(f"❌ 모델 경로가 존재하지 않습니다: {model_path}")
            return
        
        # 모델 타입 감지 및 로그
        model_name = model_path.name.lower()
        # 현재 에이전트 설정을 기준으로 모델 타입 판단
        if self.use_lstm:
            LOGGER.info(f"🔄 LSTM SAC 모델 로드 중: {model_path}")
        elif self.use_cnn:
            LOGGER.info(f"🔄 CNN SAC 모델 로드 중: {model_path}")
        elif self.use_mamba:
            LOGGER.info(f"🔄 Mamba SAC 모델 로드 중: {model_path}")
        else:
            LOGGER.info(f"🔄 MLP SAC 모델 로드 중: {model_path}")
        
        try:
            # 먼저 원래 방식으로 로드 시도
            self.actor.load_state_dict(torch.load(model_path / "actor.pth", map_location=self.device))
            self.critic.load_state_dict(torch.load(model_path / "critic.pth", map_location=self.device))
            self.critic_target.load_state_dict(torch.load(model_path / "critic_target.pth", map_location=self.device))
            
            # 옵티마이저 상태 로드
            self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device))
            
            # 옵티마이저 상태도 GPU로 이동
            if self.device.type == 'cuda':
                for state in self.actor_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                            
                for state in self.critic_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            
            # Alpha 관련 로드
            if self.use_automatic_entropy_tuning:
                self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device)
                self.log_alpha.requires_grad = True
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device))
                
                # alpha_optimizer 상태도 GPU로 이동
                if self.device.type == 'cuda':
                    for state in self.alpha_optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
            else:
                self.alpha = torch.tensor(self.alpha_init, device=self.device)
            
            # 학습 통계 로드
            training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device)
            self.actor_losses = training_stats.get('actor_losses', [])
            self.critic_losses = training_stats.get('critic_losses', [])
            self.alpha_losses = training_stats.get('alpha_losses', [])
            self.entropy_values = training_stats.get('entropy_values', [])
            self.train_step_counter = training_stats.get('train_step_counter', 0)
            
            # 성공 로그
            model_type = "LSTM" if self.use_lstm else ("CNN" if self.use_cnn else "MLP")
            LOGGER.info(f"✅ {model_type} 모델 로드 완료: {model_path}")
            LOGGER.info(f"   └─ 학습 스텝: {self.train_step_counter:,}")
            LOGGER.info(f"   └─ 버퍼 크기: {len(self.replay_buffer):,}")
            
        except RuntimeError as e:
            # 크기 불일치 오류 발생 시 크기 조정 로드 메서드 사용
            LOGGER.warning(f"⚠️ 표준 모델 로드 실패: {e}")
            LOGGER.info("🔄 크기 조정 방식으로 재시도 중...")
            self.load_model_with_resize(model_path)
    
    def get_latest_model_path(self, save_dir: Union[str, Path] = None, prefix: str = '') -> Optional[Path]:
        """
        최신 모델 경로 반환
        
        Args:
            save_dir: 모델 디렉토리 (None인 경우 기본값 사용)
            prefix: 파일명 접두사
            
        Returns:
            최신 모델 경로 (없으면 None)
        """
        if save_dir is None:
            save_dir = MODELS_DIR
        
        save_dir = Path(save_dir)
        if not save_dir.exists():
            return None
        
        # 모델 타입에 따른 패턴 확인
        patterns = [f"{prefix}lstm_sac_model_", f"{prefix}cnn_sac_model_", f"{prefix}mlp_sac_model_", f"{prefix}sac_model_"]
        
        model_dirs = []
        for pattern in patterns:
            model_dirs.extend([d for d in save_dir.iterdir() if d.is_dir() and d.name.startswith(pattern)])
        
        if not model_dirs:
            return None
        
        # 타임스탬프 기준으로 정렬
        model_dirs.sort(key=lambda d: d.name, reverse=True)
        return model_dirs[0]
    
    def load_model_with_resize(self, model_path):
        """
        크기가 다른 모델을 부분적으로 로드
        
        Args:
            model_path: 모델 디렉토리 경로
        """
        model_path = Path(model_path)
        
        LOGGER.info("🔧 크기 조정 방식으로 모델 로드 시도 중...")
        
        # 저장된 상태 사전 로드
        saved_actor_dict = torch.load(model_path / "actor.pth", map_location=self.device)
        saved_critic_dict = torch.load(model_path / "critic.pth", map_location=self.device)
        
        # 현재 모델의 상태 사전
        current_actor_dict = self.actor.state_dict()
        current_critic_dict = self.critic.state_dict()
        
        # 🆕 CNN/LSTM 모델 전용 호환성 처리
        if self.use_cnn or self.use_lstm:
            LOGGER.info("🖼️ CNN/LSTM 모델 호환성 처리 중...")
            
            # CNN 컨볼루션 레이어 호환성 검사
            if self.use_cnn:
                # 첫 번째 컨볼루션 레이어의 입력 채널 수 확인
                saved_conv1_weight = saved_actor_dict.get('conv_block1.0.weight')  # 첫 번째 Conv1d
                current_conv1_weight = current_actor_dict.get('conv_block1.0.weight')
                
                if saved_conv1_weight is not None and current_conv1_weight is not None:
                    saved_in_channels = saved_conv1_weight.shape[1]  # [out_ch, in_ch, kernel]
                    current_in_channels = current_conv1_weight.shape[1]
                    
                    LOGGER.info(f"🔍 CNN 입력 채널 비교:")
                    LOGGER.info(f"   └─ 저장된 모델: {saved_in_channels}개 채널")
                    LOGGER.info(f"   └─ 현재 모델: {current_in_channels}개 채널")
                    
                    if saved_in_channels != current_in_channels:
                        LOGGER.warning(f"⚠️ CNN 입력 채널 불일치! 컨볼루션 레이어 제외하고 로드")
                        
                        # 컨볼루션 관련 레이어 제외 목록
                        conv_layer_prefixes = [
                            'conv_block1', 'conv_block2', 'conv_block3',
                            'q1_conv_block1', 'q1_conv_block2', 'q1_conv_block3',
                            'q2_conv_block1', 'q2_conv_block2', 'q2_conv_block3'
                        ]
                        
                        # Actor에서 컨볼루션 레이어 제외
                        filtered_actor_dict = {}
                        for k, v in saved_actor_dict.items():
                            if not any(k.startswith(prefix) for prefix in conv_layer_prefixes):
                                if k in current_actor_dict and v.shape == current_actor_dict[k].shape:
                                    filtered_actor_dict[k] = v
                        
                        # Critic에서 컨볼루션 레이어 제외  
                        filtered_critic_dict = {}
                        for k, v in saved_critic_dict.items():
                            if not any(k.startswith(prefix) for prefix in conv_layer_prefixes):
                                if k in current_critic_dict and v.shape == current_critic_dict[k].shape:
                                    filtered_critic_dict[k] = v
                        
                        LOGGER.info(f"📊 CNN 호환 로딩 결과:")
                        LOGGER.info(f"   └─ Actor: {len(filtered_actor_dict)}/{len(current_actor_dict)} 레이어")
                        LOGGER.info(f"   └─ Critic: {len(filtered_critic_dict)}/{len(current_critic_dict)} 레이어")
                        LOGGER.info(f"   └─ 컨볼루션 레이어는 새로 초기화됩니다")
                        
                        # 필터링된 가중치 적용
                        current_actor_dict.update(filtered_actor_dict)
                        current_critic_dict.update(filtered_critic_dict)
                        
                    else:
                        # 채널 수가 같으면 기존 방식으로 처리
                        LOGGER.info("✅ CNN 입력 채널 일치, 일반 호환성 검사 진행")
                        actor_dict = {k: v for k, v in saved_actor_dict.items() 
                                    if k in current_actor_dict and v.shape == current_actor_dict[k].shape}
                        critic_dict = {k: v for k, v in saved_critic_dict.items() 
                                    if k in current_critic_dict and v.shape == current_critic_dict[k].shape}
                        
                        current_actor_dict.update(actor_dict)
                        current_critic_dict.update(critic_dict)
                # LSTM 모델 호환성 처리
                elif self.use_lstm:
                    LOGGER.info("🧠 LSTM 모델 호환성 처리")
                    # LSTM의 경우 hidden_size가 다를 수 있음
                    lstm_layer_prefixes = ['lstm', 'q1_lstm', 'q2_lstm']
                    
                    filtered_actor_dict = {}
                    for k, v in saved_actor_dict.items():
                        if not any(k.startswith(prefix) for prefix in lstm_layer_prefixes):
                            if k in current_actor_dict and v.shape == current_actor_dict[k].shape:
                                filtered_actor_dict[k] = v
                    
                    filtered_critic_dict = {}
                    for k, v in saved_critic_dict.items():
                        if not any(k.startswith(prefix) for prefix in lstm_layer_prefixes):
                            if k in current_critic_dict and v.shape == current_critic_dict[k].shape:
                                filtered_critic_dict[k] = v
                    
                    current_actor_dict.update(filtered_actor_dict)
                    current_critic_dict.update(filtered_critic_dict)
                    
                    LOGGER.info(f"📊 LSTM 호환 로딩 결과:")
                    LOGGER.info(f"   └─ LSTM 레이어는 새로 초기화됩니다")
        
        # 크기가 일치하는 파라미터만 로드
        else:
            actor_dict = {k: v for k, v in saved_actor_dict.items() if k in current_actor_dict and v.shape == current_actor_dict[k].shape}
            critic_dict = {k: v for k, v in saved_critic_dict.items() if k in current_critic_dict and v.shape == current_critic_dict[k].shape}
            
            # 호환 통계 로그
            actor_loaded = len(actor_dict)
            actor_total = len(current_actor_dict)
            critic_loaded = len(critic_dict)
            critic_total = len(current_critic_dict)
            
            LOGGER.info(f"📊 MLP 모델 호환성 분석:")
            LOGGER.info(f"   └─ Actor: {actor_loaded}/{actor_total} 레이어 호환 ({actor_loaded/actor_total*100:.1f}%)")
            LOGGER.info(f"   └─ Critic: {critic_loaded}/{critic_total} 레이어 호환 ({critic_loaded/critic_total*100:.1f}%)")
        
            # 상태 사전 업데이트
            current_actor_dict.update(actor_dict)
            current_critic_dict.update(critic_dict)
        
        # 모델에 적용
        self.actor.load_state_dict(current_actor_dict)
        self.critic.load_state_dict(current_critic_dict)
        
        # critic_target 업데이트
        try:
            saved_critic_target_dict = torch.load(model_path / "critic_target.pth", map_location=self.device)
            current_critic_target_dict = self.critic_target.state_dict()
            critic_target_dict = {k: v for k, v in saved_critic_target_dict.items() if k in current_critic_target_dict and v.shape == current_critic_target_dict[k].shape}
            current_critic_target_dict.update(critic_target_dict)
            self.critic_target.load_state_dict(current_critic_target_dict)
        except:
            # critic_target 파일이 없으면 critic을 복사
            LOGGER.warning("⚠️ critic_target 파일이 없어 critic에서 복사합니다.")
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(param.data)
        
        # 옵티마이저와 기타 상태 로드 (가능한 경우)
        try:
            self.actor_optimizer.load_state_dict(torch.load(model_path / "actor_optimizer.pth", map_location=self.device))
            self.critic_optimizer.load_state_dict(torch.load(model_path / "critic_optimizer.pth", map_location=self.device))
            
            # 옵티마이저 상태도 GPU로 이동
            if self.device.type == 'cuda':
                for state in self.actor_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                            
                for state in self.critic_optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            
            if self.use_automatic_entropy_tuning:
                self.log_alpha = torch.load(model_path / "log_alpha.pth", map_location=self.device)
                self.log_alpha.requires_grad = True
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(torch.load(model_path / "alpha_optimizer.pth", map_location=self.device))
                
                # alpha_optimizer 상태도 GPU로 이동
                if self.device.type == 'cuda':
                    for state in self.alpha_optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.to(self.device)
        except:
            LOGGER.warning(f"⚠️ 옵티마이저 상태 로드 실패. 기본값 유지.")
        
        # 학습 통계 로드 (가능한 경우)
        try:
            training_stats = torch.load(model_path / "training_stats.pth", map_location=self.device)
            self.actor_losses = training_stats.get('actor_losses', [])
            self.critic_losses = training_stats.get('critic_losses', [])
            self.alpha_losses = training_stats.get('alpha_losses', [])
            self.entropy_values = training_stats.get('entropy_values', [])
            self.train_step_counter = training_stats.get('train_step_counter', 0)
        except:
            LOGGER.warning(f"⚠️ 학습 통계 로드 실패. 기본값 유지.")
        
        model_type = "LSTM" if self.use_lstm else ("CNN" if self.use_cnn else "MLP")
        LOGGER.info(f"✅ {model_type} 모델 크기 조정 로드 완료: {model_path}")
        LOGGER.info("💡 일부 파라미터는 새로 초기화됩니다.")


def train_sac_agent(env, agent, num_episodes: int = 1000, 
                   max_steps_per_episode=MAX_STEPS_PER_EPISODE, update_frequency: int = 1,
                   log_frequency: int = 100):
    """
    SAC 에이전트 학습 함수

    Args:
        env: TradingEnvironment 인스턴스
        agent: SAC 에이전트
        num_episodes: 학습할 에피소드 수
        max_steps_per_episode: 에피소드당 최대 스텝 수
        update_frequency: 업데이트 빈도
        log_frequency: 로그 출력 빈도
    
    Returns:
        학습된 SAC 에이전트
    """
    model_type = "LSTM" if agent.use_lstm else ("CNN" if agent.use_cnn else "MLP")
    LOGGER.info(f"🚀 {model_type} SAC 에이전트 학습 시작: {num_episodes} 에피소드")
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps_per_episode):
            # 행동 선택
            action = agent.select_action(state, evaluate=False)
            
            # 환경에서 스텝 실행
            next_state, reward, done, info = env.step(action)
            
            # 경험 저장
            agent.add_experience(state, action, reward, next_state, done)
            
            # 네트워크 업데이트
            if step % update_frequency == 0:
                stats = agent.update_parameters()
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # 로그 출력
        if episode % log_frequency == 0:
            avg_reward = np.mean(episode_rewards[-log_frequency:])
            avg_length = np.mean(episode_lengths[-log_frequency:])
            
            LOGGER.info(f"Episode {episode}")
            LOGGER.info(f"  평균 보상: {avg_reward:.2f}")
            LOGGER.info(f"  평균 길이: {avg_length:.1f}")
            LOGGER.info(f"  포트폴리오 가치: ${info['portfolio_value']:.2f}")
            LOGGER.info(f"  총 수익률: {info['total_return'] * 100:.2f}%")
            
            if len(agent.actor_losses) > 0:
                LOGGER.info(f"  Actor Loss: {agent.actor_losses[-1]:.6f}")
                LOGGER.info(f"  Critic Loss: {agent.critic_losses[-1]:.6f}")
                LOGGER.info(f"  Alpha: {agent.alpha.item():.6f}")
    
    LOGGER.info(f"🎉 {model_type} SAC 에이전트 학습 완료!")
    return agent