"""
TinyTransformer 전용 최적화 설정
1분봉 주식 데이터에 특화된 하이퍼파라미터
"""

# TinyTransformer 최적 설정
TINYTRANSFORMER_CONFIG = {
    # 🎯 배치 및 메모리 설정
    'batch_size': 256,                    # 메모리 효율성을 위해 축소
    'replay_buffer_size': 250000,         # 적당한 크기로 조정
    'max_steps_per_episode': 1500,        # 더 긴 에피소드로 패턴 학습
    
    # 🤖 Transformer 아키텍처
    'd_model': 128,                       # 기본 차원
    'nhead': 4,                          # 어텐션 헤드 수
    'num_layers': 2,                     # 레이어 수 (경량화)
    'dropout_rate': 0.15,                # 적당한 정규화
    
    # 📚 학습률 설정 (보수적)
    'actor_lr': 8e-5,                   # 기본값보다 낮게
    'critic_lr': 1e-4,                  # 기본값 유지
    'alpha_lr': 8e-5,                   # 보수적 엔트로피 학습
    
    # 🎛️ SAC 파라미터
    'gamma': 0.95,                      # 약간 높은 할인율
    'tau': 0.005,                       # 부드러운 타겟 업데이트
    'alpha_init': 0.15,                 # 낮은 초기 엔트로피
    'target_entropy_factor': -0.3,      # 보수적 엔트로피 타겟
    
    # 🔧 안정성 설정
    'gradient_clip_norm': 0.5,          # 강한 그래디언트 클리핑
    'weight_decay': 1e-5,               # 가벼운 L2 정규화
    'target_update_interval': 2,        # 조금 더 느린 타겟 업데이트
    
    # 📊 학습 스케줄
    'num_episodes': 150,                 # 적당한 에피소드 수
    'evaluate_interval': 15,             # 평가 간격
    'save_model_interval': 30,           # 모델 저장 간격
    'warmup_steps': 5000,               # 워밍업 스텝
    
    # 🎯 시퀀스 설정
    'window_size': 30,                   # 기본 윈도우 크기
    'max_seq_len': 40,                   # 여유분 포함
    
    # 🚀 최적화 설정
    'update_frequency': 1,               # 매 스텝마다 업데이트
    'batch_norm': True,                  # 배치 정규화 사용
    'layer_norm': True,                  # 레이어 정규화 사용
    'use_residual': True,                # 잔차 연결 사용
}

# 데이터 규모별 설정 조정
def get_tinytransformer_config_for_data_size(train_size: int):
    """데이터 크기에 따른 설정 조정"""
    config = TINYTRANSFORMER_CONFIG.copy()
    
    if train_size > 90000:  # 큰 데이터셋 (95,247)
        config.update({
            'batch_size': 256,
            'max_steps_per_episode': 1500,
            'replay_buffer_size': 250000,
            'num_episodes': 120,
            'warmup_steps': 8000,
        })
    elif train_size > 50000:  # 중간 데이터셋
        config.update({
            'batch_size': 128,
            'max_steps_per_episode': 1000,
            'replay_buffer_size': 150000,
            'num_episodes': 100,
            'warmup_steps': 5000,
        })
    else:  # 작은 데이터셋
        config.update({
            'batch_size': 64,
            'max_steps_per_episode': 800,
            'replay_buffer_size': 100000,
            'num_episodes': 80,
            'warmup_steps': 3000,
        })
    
    return config

# 메모리 사용량에 따른 동적 조정
def adjust_config_for_memory(config: dict, available_memory_gb: float):
    """사용 가능한 메모리에 따른 설정 조정"""
    if available_memory_gb < 8:  # 8GB 미만
        config['batch_size'] = min(config['batch_size'], 128)
        config['d_model'] = min(config['d_model'], 96)
        config['replay_buffer_size'] = min(config['replay_buffer_size'], 150000)
    elif available_memory_gb < 16:  # 16GB 미만
        config['batch_size'] = min(config['batch_size'], 256)
        config['d_model'] = min(config['d_model'], 128)
    # 16GB 이상은 기본 설정 유지
    
    return config

# 학습 단계별 설정
TRAINING_PHASES = {
    'warmup': {
        'actor_lr': 5e-5,
        'critic_lr': 8e-5,
        'alpha_lr': 5e-5,
        'update_frequency': 2,
        'exploration_noise': 0.3,
    },
    'main': {
        'actor_lr': 8e-5,
        'critic_lr': 1e-4,
        'alpha_lr': 8e-5,
        'update_frequency': 1,
        'exploration_noise': 0.2,
    },
    'fine_tune': {
        'actor_lr': 3e-5,
        'critic_lr': 5e-5,
        'alpha_lr': 3e-5,
        'update_frequency': 1,
        'exploration_noise': 0.1,
    }
}

# 사용 예시
def get_optimized_config():
    """95,247 train 데이터에 최적화된 설정 반환"""
    return get_tinytransformer_config_for_data_size(95247)

if __name__ == "__main__":
    config = get_optimized_config()
    print("🤖 TinyTransformer 최적 설정:")
    for key, value in config.items():
        print(f"   {key}: {value}") 