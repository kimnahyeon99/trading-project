"""
SAC 알고리즘 구현 모듈 (기본 버전) - model_type 파라미터 추가 수정
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
    MAX_STEPS_PER_EPISODE
)

from src.models.networks import (
    ActorNetwork, 
    CriticNetwork, 
    CNNActorNetwork, 
    CNNCriticNetwork,
    LSTMActorNetwork,
    LSTMCriticNetwork
)

from src.utils.utils import soft_update, create_directory

    
class ReplayBuffer:
    """
    경험 리플레이 버퍼
    RL 에이전트가 경험한 샘플을 저장하고 무작위로 샘플링
    """
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE):
        """
        ReplayBuffer 클래스 초기화
        
        Args:
            capacity: 버퍼의 최대 용량
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state: Any, action: Any, reward: float, next_state: Any, done: bool) -> None:
        """
        버퍼에 샘플 추가
        
        Args:
            state: 현재 상태
            action: 수행한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 종료 여부
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> Tuple:
        """
        버퍼에서 무작위로 샘플 추출
        
        Args:
            batch_size: 추출할 샘플 수
            
        Returns:
            (상태, 행동, 보상, 다음 상태, 종료 여부) 튜플
        """
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(list, zip(*batch))
        
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        """
        버퍼의 현재 크기 반환
        
        Returns:
            버퍼의 현재 크기
        """
        return len(self.buffer)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        버퍼를 파일로 저장
        
        Args:
            path: 저장할 파일 경로
        """
        with open(path, 'wb') as f:
            pickle.dump(self.buffer, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        파일에서 버퍼 로드
        
        Args:
            path: 로드할 파일 경로
        """
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
        self.position = len(self.buffer) % self.capacity


class SequentialReplayBuffer(ReplayBuffer):
    """시계열성을 유지하는 리플레이 버퍼"""
    
    def __init__(self, capacity: int = REPLAY_BUFFER_SIZE, sequence_length: int = 32):
        super().__init__(capacity)
        self.sequence_length = sequence_length
        LOGGER.info(f"🔄 순차적 리플레이 버퍼 초기화: 시퀀스 길이 {sequence_length}")
    
    def sample_sequential(self, batch_size: int) -> Tuple:
        """연속된 시퀀스들을 배치로 샘플링"""
        if len(self.buffer) < self.sequence_length:
            LOGGER.warning("버퍼 크기 부족, 기존 방식으로 샘플링")
            return self.sample(batch_size)
        
        # 연속된 시퀀스 시작점들을 랜덤 선택
        max_start_idx = len(self.buffer) - self.sequence_length
        start_indices = np.random.choice(max_start_idx, batch_size, replace=True)
        
        # 시간순으로 정렬 (중요!)
        start_indices = np.sort(start_indices)
        
        sequences = []
        for start_idx in start_indices:
            # 연속된 sequence_length만큼의 경험들 추출
            sequence = []
            for i in range(self.sequence_length):
                if start_idx + i < len(self.buffer):
                    sequence.append(self.buffer[start_idx + i])
            
            if len(sequence) == self.sequence_length:
                sequences.append(sequence)
        
        if not sequences:
            return self.sample(batch_size)
        
        return self._process_sequences(sequences)
    
    def _process_sequences(self, sequences):
        """시퀀스들을 배치로 변환"""
        # 각 시퀀스의 마지막 transition 사용
        batch = [seq[-1] for seq in sequences]
        state, action, reward, next_state, done = map(list, zip(*batch))
        return state, action, reward, next_state, done


class SACAgent:
    """
    SAC 알고리즘 에이전트 (개선된 CNN 지원)
    """
        
    def __init__(
        self,
        state_dim: int = None,
        action_dim: int = 1,
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
        input_shape: Tuple[int, int] = None,
        use_cnn: bool = False,
        use_lstm: bool = False,
        model_type: str = None,  # 🔧 추가된 파라미터
        lstm_hidden_dim: int = 128,
        num_lstm_layers: int = 2,
        lstm_dropout: float = 0.2,
        buffer_type: str = 'sequential',
        sequence_length: int = 32,
        # CNN 전용 파라미터
        cnn_dropout_rate: float = 0.1,
        cnn_learning_rate_factor: float = 0.5,  # CNN은 더 낮은 학습률 사용
        cnn_alpha_init: float = 0.1,  # CNN은 더 낮은 초기 alpha 사용
        cnn_batch_norm: bool = True,
        # ✅ 새로운 안정성 파라미터들 추가
        gradient_clip_norm: float = 1.0,
        use_gradient_clipping: bool = True,
        use_lr_scheduling: bool = True,
        lr_scheduler_factor: float = 0.8,
        lr_scheduler_patience: int = 100,
        target_update_method: str = 'soft',  # 'soft' or 'hard'
        adaptive_alpha: bool = True
    ):
        """SACAgent 클래스 초기화 (개선된 CNN 지원)"""
        
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
        self.input_shape = input_shape
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.lstm_dropout = lstm_dropout
        
        # 🔧 model_type 처리 (auto-detect if not provided)
        if model_type is None:
            if use_lstm:
                model_type = 'lstm'
            elif use_cnn:
                model_type = 'cnn'
            else:
                model_type = 'mlp'
        self.model_type = model_type
        
        # CNN 전용 설정
        self.cnn_dropout_rate = cnn_dropout_rate
        self.cnn_batch_norm = cnn_batch_norm
        
        # ✅ 안정성 개선 설정
        self.gradient_clip_norm = gradient_clip_norm
        self.use_gradient_clipping = use_gradient_clipping
        self.use_lr_scheduling = use_lr_scheduling
        self.target_update_method = target_update_method
        self.adaptive_alpha = adaptive_alpha
        
        # ✅ 학습률 스케줄링 지표 추적
        self.recent_critic_losses = deque(maxlen=50)
        self.lr_plateau_counter = 0
        self.lr_scheduler_patience = lr_scheduler_patience
        self.lr_scheduler_factor = lr_scheduler_factor
        
        # ✅ 적응적 Alpha 조정을 위한 추적
        if self.adaptive_alpha:
            self.recent_entropies = deque(maxlen=100)
            self.target_entropy_range = (-action_dim * 1.5, -action_dim * 0.5)  # 동적 범위
        
        # 🆕 모델별 엔트로피 설정 적용
        if model_type:
            from src.config.ea_teb_config import get_entropy_config_for_model
            entropy_config = get_entropy_config_for_model(model_type, action_dim)
            
            if entropy_config['use_adaptive']:
                self.use_automatic_entropy_tuning = True
                self.target_entropy_range = entropy_config['entropy_range']
                self.target_entropy = entropy_config['initial_target']
            else:
                # MLP의 경우 고정값 사용
                self.target_entropy = entropy_config['fixed_target']
                
            LOGGER.info(f"📊 {model_type.upper()} 엔트로피 설정: {entropy_config['description']}")
        else:
            # 기존 방식 (하위 호환)
            self.target_entropy = -action_dim
        
        # 📚 모델별 하이퍼파라미터 조정
        if use_cnn:
            self.actor_lr = actor_lr * cnn_learning_rate_factor
            self.critic_lr = critic_lr * cnn_learning_rate_factor  
            self.alpha_lr = alpha_lr * cnn_learning_rate_factor
            self.alpha_init = max(cnn_alpha_init, 0.2)  # 최소 0.2로 설정
            
            LOGGER.info("🖼️ CNN 전용 하이퍼파라미터 적용:")
            LOGGER.info(f"   └─ Actor LR: {self.actor_lr:.6f} (원본의 {cnn_learning_rate_factor:.1f}배)")
            LOGGER.info(f"   └─ Critic LR: {self.critic_lr:.6f}")
            LOGGER.info(f"   └─ Alpha LR: {self.alpha_lr:.6f}")
            LOGGER.info(f"   └─ 초기 Alpha: {self.alpha_init}")
            LOGGER.info(f"   └─ Dropout: {self.cnn_dropout_rate}")
        else:
            self.actor_lr = actor_lr
            self.critic_lr = critic_lr
            self.alpha_lr = alpha_lr
            self.alpha_init = alpha_init
        
        # 학습 단계 카운터
        self.train_step_counter = 0
        self.update_counter = 0
        
        # 모델 타입 검증
        if use_cnn and use_lstm:
            raise ValueError("CNN과 LSTM 모델을 동시에 사용할 수 없습니다.")
        
        # 모델 타입 로깅
        if use_lstm:
            LOGGER.info("🧠 LSTM 기반 SAC 에이전트 초기화 중...")
        elif use_cnn:
            LOGGER.info("🖼️ 개선된 CNN 기반 SAC 에이전트 초기화 중...")
            LOGGER.info("🔀 개선사항: BatchNorm, Dropout, 적응형 풀링, 가중치 초기화")
        else:
            LOGGER.info("🔢 MLP 기반 SAC 에이전트 초기화 중...")
        
        # TradingEnvironment를 위한 상태 차원 자동 계산
        if not use_cnn and not use_lstm and state_dim is None:
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
        
        # 옵티마이저 (모델별 다른 학습률 적용)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # ✅ 학습률 스케줄러 추가
        if self.use_lr_scheduling:
            self.actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.actor_optimizer, mode='min', factor=self.lr_scheduler_factor, 
                patience=self.lr_scheduler_patience, min_lr=1e-6
            )
            self.critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.critic_optimizer, mode='min', factor=self.lr_scheduler_factor,
                patience=self.lr_scheduler_patience, min_lr=1e-6
            )
            LOGGER.info("✅ 학습률 스케줄링 활성화됨")
        
        # 📚 CNN일 때 추가 정규화 기법
        if use_cnn:
            # L2 정규화 추가
            for param_group in self.actor_optimizer.param_groups:
                param_group['weight_decay'] = 1e-4
            for param_group in self.critic_optimizer.param_groups:
                param_group['weight_decay'] = 1e-4
            
            LOGGER.info("✅ CNN 모델에 L2 정규화 (weight_decay=1e-4) 적용")
        
        # 자동 엔트로피 조정
        if self.use_automatic_entropy_tuning:
            self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        else:
            self.alpha = torch.tensor(self.alpha_init, device=device)
        
        # CUDA 최적화
        self._setup_cuda_streams()
        self._optimize_gpu_settings()
        self._move_all_optimizers_to_device()
        
        # 리플레이 버퍼
        if buffer_type == 'sequential':
            self.replay_buffer = SequentialReplayBuffer(
                capacity=buffer_capacity,
                sequence_length=sequence_length
            )
            self.use_sequential_sampling = True
        else:
            self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
            self.use_sequential_sampling = False
        
        # 학습 통계
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.entropy_values = []
        
        # 최종 로그
        model_type_display = "LSTM" if use_lstm else ("CNN" if use_cnn else "MLP")
        LOGGER.info(f"🎉 SAC 에이전트 초기화 완료!")
        LOGGER.info(f"   └─ 모델 타입: {model_type_display}")
        LOGGER.info(f"   └─ 행동 차원: {action_dim}")
        LOGGER.info(f"   └─ 상태 차원: {self.state_dim if not (use_cnn or use_lstm) else input_shape}")
        LOGGER.info(f"   └─ 장치: {device}")
        
    def _move_optimizer_to_device(self, optimizer, device):
        """
        옵티마이저 상태를 지정된 디바이스로 이동
        
        Args:
            optimizer: PyTorch 옵티마이저
            device: 이동할 디바이스
        """
        if device.type == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)

    def _move_all_optimizers_to_device(self):
        """모든 옵티마이저를 현재 디바이스로 이동"""
        self._move_optimizer_to_device(self.actor_optimizer, self.device)
        self._move_optimizer_to_device(self.critic_optimizer, self.device)
        
        if hasattr(self, 'alpha_optimizer'):
            self._move_optimizer_to_device(self.alpha_optimizer, self.device)

    def clear_gpu_cache(self):
        """GPU 캐시 정리"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()

    def get_gpu_memory_info(self):
        """GPU 메모리 사용량 정보 반환"""
        if self.device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3  # GB
            cached = torch.cuda.memory_reserved(self.device) / 1024**3     # GB
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_memory_gb': torch.cuda.get_device_properties(self.device).total_memory / 1024**3
            }
        return {'allocated_gb': 0, 'cached_gb': 0, 'total_memory_gb': 0}

    def _setup_cuda_streams(self):
        """CUDA 스트림 설정"""
        if self.device.type == 'cuda':
            self.data_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
            LOGGER.info(f"✅ CUDA 스트림 생성 완료: {self.device}")
        else:
            self.data_stream = None
            self.compute_stream = None
            
    def _optimize_gpu_settings(self):
        """GPU 최적화 설정"""
        if self.device.type == 'cuda':
            # 메모리 할당 최적화
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            
            # 혼합 정밀도 사용 준비 (옵션)
            if hasattr(torch.cuda, 'amp'):
                self.use_amp = True
                self.scaler = torch.cuda.amp.GradScaler()
            else:
                self.use_amp = False
                
            LOGGER.info(f"✅ GPU 최적화 설정 완료: {self.device}")
    
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
            # 🖼️ 개선된 CNN 네트워크 사용
            if self.input_shape is None:
                raise ValueError("CNN 모델을 사용할 때는 input_shape가 필요합니다.")
            
            LOGGER.info("🏗️ 개선된 CNN 네트워크 생성 중...")
            
            self.actor = CNNActorNetwork(
            input_shape=self.input_shape,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            dropout_rate=self.cnn_dropout_rate,
            device=self.device
            )
            
            self.critic = CNNCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.cnn_dropout_rate,
                device=self.device
            )
            
            self.critic_target = CNNCriticNetwork(
                input_shape=self.input_shape,
                action_dim=self.action_dim,
                hidden_dim=self.hidden_dim,
                dropout_rate=self.cnn_dropout_rate,
                device=self.device
            )
            
            LOGGER.info("✅ 개선된 CNN 네트워크 생성 완료")
                
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
                if self.use_cnn or self.use_lstm:
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
                    _, _, action = self.actor.sample(state_tensor)
                else:
                    action, _, _ = self.actor.sample(state_tensor)
            self.actor.train()
            
            action_value = float(action.detach().cpu().numpy().flatten()[0])
            
            # 학습 초기에 추가 탐색 노이즈 주입
            if not evaluate and self.train_step_counter < 5000:  # 추가
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
            if self.use_cnn or self.use_lstm:
                # CNN / LSTM: 상태를 딕셔너리 형태로 유지
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
                
                return torch.FloatTensor(combined_state).unsqueeze(0).pin_memory().to(self.device, non_blocking=True)
            
        except Exception as e:
            LOGGER.error(f"상태 변환 중 오류: {e}")
            # 기본 상태 반환
            if self.use_cnn or self.use_lstm:
                return {
                    "market_data": torch.zeros((1, self.input_shape[0], self.input_shape[1]), device=self.device),
                    "portfolio_state": torch.zeros((1, 2), device=self.device)
                }
            else:
                return torch.zeros((1, self.state_dim), device=self.device)

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
        
        if self.use_cnn or self.use_lstm:
            market_batch = []
            portfolio_batch = []
            for state in states:
                market_batch.append(state["market_data"])
                portfolio_batch.append(state["portfolio_state"])
            
            market_tensor = torch.FloatTensor(np.stack(market_batch)).pin_memory().to(self.device, non_blocking=True)
            portfolio_tensor = torch.FloatTensor(np.stack(portfolio_batch)).pin_memory().to(self.device, non_blocking=True)

            return {
                "market_data": market_tensor,
                "portfolio_state": portfolio_tensor
            }
        else:
            batch_states = []
            for state in states:
                processed_state = self._process_state_for_network(state)
                batch_states.append(processed_state)
            return torch.cat(batch_states, dim=0).pin_memory().to(self.device, non_blocking=True)
        
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
        
        # 샘플링
        try:
            if self.use_sequential_sampling and hasattr(self.replay_buffer, 'sample_sequential'):
                states, actions, rewards, next_states, dones = self.replay_buffer.sample_sequential(batch_size)
            else:
                states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        except Exception as e:
            LOGGER.warning(f"샘플링 실패, 기본 방식 사용: {e}")
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # 배치 텐서 변환
        batched_states = self._process_batch_states(states)
        batched_next_states = self._process_batch_states(next_states)
        
        batched_actions = torch.FloatTensor(actions).unsqueeze(1).pin_memory().to(self.device, non_blocking=True)
        batched_rewards = torch.FloatTensor(rewards).unsqueeze(1).pin_memory().to(self.device, non_blocking=True)
        batched_dones = torch.FloatTensor(dones).unsqueeze(1).pin_memory().to(self.device, non_blocking=True)
        
        # ✅ 통합된 업데이트 (모든 모델에 동일한 안정성 적용)
        critic_loss = self._update_critic(
            batched_states, batched_actions, batched_rewards, batched_next_states, batched_dones
        )
        actor_loss = self._update_actor(batched_states)
        alpha_loss = self._update_alpha(batched_states)
        
        # ✅ 개선된 타겟 네트워크 업데이트
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_interval == 0:
            if self.target_update_method == 'soft':
                soft_update(self.critic_target, self.critic, self.tau)
            else:  # hard update
                self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 메모리 정리
        if self.device.type == 'cuda' and self.update_counter % 100 == 0:
            torch.cuda.empty_cache()
        
        # 엔트로피 계산
        entropy_value = 0.0
        if self.update_counter % 10 == 0:
            with torch.no_grad():
                _, log_probs, _ = self.actor.sample(batched_states)
                entropy_value = -log_probs.mean().item()

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
                LOGGER.warning(f"큰 그래디언트 감지: {grad_norm:.6f}")
        
        self.critic_optimizer.step()
        
        # ✅ 학습률 스케줄링 업데이트
        critic_loss_value = critic_loss.item()
        if self.use_lr_scheduling:
            self.recent_critic_losses.append(critic_loss_value)
            if len(self.recent_critic_losses) >= 50:
                avg_loss = sum(self.recent_critic_losses) / len(self.recent_critic_losses)
                self.critic_scheduler.step(avg_loss)
        
        return critic_loss_value

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
        
        # ✅ 학습률 스케줄링
        if self.use_lr_scheduling:
            self.actor_scheduler.step(actor_loss.item())
        
        return actor_loss.item()

    def _update_alpha(self, states):
        """개선된 Alpha 업데이트 (적응적 조정)"""
        if not self.use_automatic_entropy_tuning:
            return 0.0
        
        with torch.no_grad():
            _, log_probs, _ = self.actor.sample(states)
            current_entropy = -log_probs.mean().item()
        
        # ✅ 적응적 target_entropy 조정
        if self.adaptive_alpha:
            self.recent_entropies.append(current_entropy)
            
            if len(self.recent_entropies) >= 50:
                avg_entropy = sum(self.recent_entropies) / len(self.recent_entropies)
                
                # 엔트로피가 너무 낮으면 target을 낮춤, 너무 높으면 높임
                if avg_entropy < self.target_entropy_range[0]:
                    self.target_entropy = max(self.target_entropy * 1.02, self.target_entropy_range[0])
                elif avg_entropy > self.target_entropy_range[1]:
                    self.target_entropy = min(self.target_entropy * 0.98, self.target_entropy_range[1])
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        # ✅ 안정성 체크
        if torch.isnan(alpha_loss) or torch.isinf(alpha_loss):
            LOGGER.warning("Alpha 손실에 NaN/Inf 감지됨. 업데이트 건너뜀.")
            return 0.0
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        
        # ✅ Alpha도 그래디언트 클리핑
        if self.use_gradient_clipping:
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=self.gradient_clip_norm)
        
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # ✅ Alpha 값 범위 제한
        with torch.no_grad():
            self.log_alpha.clamp_(min=-10, max=2)  # alpha 범위를 [exp(-10), exp(2)]로 제한
            self.alpha = self.log_alpha.exp()
        
        return alpha_loss.item()
    
    def save_model(self, save_dir: Union[str, Path] = None, prefix: str = "",
               model_type: str = None, symbol: str = None, symbols: List[str] = None) -> str:
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

        # 설정 저장 (실시간 트레이딩 호환성 정보 포함)
        config = {
            # 기본 모델 정보
            'model_type': model_type.lower(),
            'symbol': symbol,
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
        if "lstm" in model_name:
            LOGGER.info(f"🔄 LSTM SAC 모델 로드 중: {model_path}")
        elif "cnn" in model_name:
            LOGGER.info(f"🔄 CNN SAC 모델 로드 중: {model_path}")
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
        
        # 크기가 일치하는 파라미터만 로드
        actor_dict = {k: v for k, v in saved_actor_dict.items() if k in current_actor_dict and v.shape == current_actor_dict[k].shape}
        critic_dict = {k: v for k, v in saved_critic_dict.items() if k in current_critic_dict and v.shape == current_critic_dict[k].shape}
        
        # 호환 통계 로그
        actor_loaded = len(actor_dict)
        actor_total = len(current_actor_dict)
        critic_loaded = len(critic_dict)
        critic_total = len(current_critic_dict)
        
        LOGGER.info(f"📊 모델 호환성 분석:")
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