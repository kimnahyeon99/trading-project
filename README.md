# SAC 강화학습 기반 주식 트레이딩 시스템 프로젝트 계획

## 1. 프로젝트 개요
- **프로젝트명**: SAC 강화학습 기반 주식 트레이딩 시스템
- **목적**: Soft Actor-Critic(SAC) 알고리즘을 활용한 주식 트레이딩 시스템 구축
- **개발환경**: Python 3.10, PyTorch GPU(CUDA), RTX 1660 이상 호환

## 2. 프로젝트 구조
```
trading/
├── data/                      # 수집된 주식 데이터 저장
├── logs/                      # 로그 파일 저장
├── models/                    # 학습된 모델 저장
├── notebooks/                 # 분석 및 시각화 노트북
├── results/                   # 백테스트 결과 저장
├── tests/                     # 단위 테스트 및 통합 테스트
│   ├── unit/                  # 단위 테스트
│   └── integration/           # 통합 테스트
└── src/                       # 소스 코드
    ├── config/                # 설정 파일
    ├── data_collection/       # 데이터 수집 모듈
    ├── preprocessing/         # 데이터 전처리 모듈
    ├── environment/           # 트레이딩 환경 모듈
    ├── models/                # SAC 모델 구현
    ├── training/              # 모델 학습 모듈
    ├── evaluation/            # 모델 평가 모듈
    ├── backtesting/           # 백테스트 모듈
    ├── trading/               # 실시간 트레이딩 모듈
    └── utils/                 # 유틸리티 함수
```

## 3. 개발 단계

### 3.1 개발 환경 설정 
- Conda 가상환경 생성 (sac_trading_py310, Python 3.10)
- 필수 라이브러리 설치 (PyTorch GPU, NumPy 1.23.5 등)
- CUDA 설정 및 GPU 작동 확인
- 프로젝트 기본 구조 생성

### 3.2 데이터 수집 모듈 개발 
- Alpha Vantage API 연동
- 미국 빅테크 기업 주식 데이터 수집 (5-10년)
- 데이터 저장 및 관리 기능 구현
- 단위 테스트 구현 및 검증

### 3.3 데이터 전처리 모듈 개발 
- 수집 데이터 정제 및 정규화
- 기술적 지표 계산 및 특성 추출
- 학습/검증/테스트 데이터 분할
- 단위 테스트 구현 및 검증

### 3.4 트레이딩 환경 모듈 개발
- 커스텀 트레이딩 환경 설계 (Gym 사용 X)
- 상태 공간, 행동 공간, 보상 함수 정의
- 환경 시뮬레이션 기능 구현
- 단위 테스트 구현 및 검증

### 3.5 SAC 모델 개발 
- Actor 네트워크 구현
- Critic 네트워크 구현
- 온도 파라미터 자동 조정 메커니즘 구현
- 경험 리플레이 버퍼 구현
- 단위 테스트 구현 및 검증

### 3.6 학습 모듈 개발
- SAC 학습 알고리즘 구현
- 모델 체크포인트 저장 및 로드 기능
- 학습 진행 모니터링 및 로깅
- 단위 테스트 구현 및 검증

### 3.7 백테스트 모듈 개발
- 백테스트 프레임워크 구현
- 성능 지표 계산 (수익률, 샤프 비율, MDD 등)
- 백테스트 결과 시각화
- 단위 테스트 구현 및 검증

### 3.8 실시간 트레이딩 모듈 개발
- 실시간 데이터 수집 기능 구현
- 트레이딩 시그널 생성 기능 구현
- 모의 트레이딩 기능 구현
- 단위 테스트 구현 및 검증

### 3.9 통합 테스트 및 시스템 최적화
- 모듈별 통합 테스트 수행
- 전체 시스템 통합 테스트 수행
- 성능 최적화 및 버그 수정
- 문서화 및 최종 검토

## 4. 테스트 계획

### 4.1 단위 테스트
- 각 모듈별 기능 검증
- 입출력 유효성 검사
- 예외 처리 검증
- 코드 커버리지 분석

### 4.2 통합 테스트
- 모듈 간 인터페이스 검증
- 데이터 흐름 검증
- 시스템 안정성 검증

### 4.3 성능 테스트
- 학습 속도 및 효율성 검증
- GPU 활용률 검증
- 메모리 사용량 모니터링

## 5. 클래스 다이어그램

```
+------------------------+       +------------------------+
|   DataCollector        |       |   DataPreprocessor    |
+------------------------+       +------------------------+
|                        |       | - features: list      |
| - symbols: list        |       | - window_size: int    |
+------------------------+       +------------------------+
| + collect_data()       |------>| + preprocess_data()   |
| + save_data()          |       | + create_features()   |
| + load_data()          |       | + normalize_data()    |
+------------------------+       +------------------------+
                                           |
                                           v
+------------------------+       +------------------------+
|   TradingEnvironment   |<------|   DataLoader          |
+------------------------+       +------------------------+
| - data: DataFrame      |       | - data_path: str      |
| - window_size: int     |       | - batch_size: int     |
| - current_step: int    |       +------------------------+
+------------------------+       | + load_batch()        |
| + reset()              |       | + split_data()        |
| + step()               |       +------------------------+
| + get_state()          |
| + calculate_reward()   |
+------------------------+
           |
           v
+------------------------+       +------------------------+ 
|   SACAgent             |<------|   ReplayBuffer        |
+------------------------+       +------------------------+
| - actor: ActorNetwork  |       | - capacity: int       |
| - critic: CriticNetwork|       | - buffer: list        |
| - target_entropy: float|       +------------------------+
| - log_alpha: Parameter |       | + add()               |
+------------------------+       | + sample()            |
| + select_action()      |       | + __len__()           |
| + update_parameters()  |       +------------------------+
| + save_model()         |
| + load_model()         |
+------------------------+
           |
           v
+------------------------+       +------------------------+
|   Trainer              |------>|   Evaluator           |
+------------------------+       +------------------------+
| - agent: SACAgent      |       | - agent: SACAgent     |
| - env: TradingEnv      |       | - env: TradingEnv     |
| - buffer: ReplayBuffer |       +------------------------+
+------------------------+       | + evaluate()          |
| + train()              |       | + calculate_metrics() |
| + validate()           |       +------------------------+
+------------------------+
           |
           v
+------------------------+       +------------------------+
|   Backtester           |------>|   Visualizer          |
+------------------------+       +------------------------+
| - agent: SACAgent      |       | - results: dict       |
| - data: DataFrame      |       +------------------------+
+------------------------+       | + plot_returns()      |
| + run_backtest()       |       | + plot_trades()       |
| + calculate_metrics()  |       | + plot_drawdown()     |
+------------------------+       +------------------------+
           |
           v
+------------------------+
|   LiveTrader           |
+------------------------+
| - agent: SACAgent      |
| - api_connector: object|
+------------------------+
| + connect()            |
| + get_market_data()    |
| + execute_trade()      |
| + monitor_positions()  |
+------------------------+
```

## 6. 흐름도

```
[데이터 수집] -> [데이터 전처리] -> [특성 추출]
       |
       v
[학습/검증/테스트 데이터 분할]
       |
       v
[트레이딩 환경 초기화] <----+
       |                   |
       v                   |
[상태 관측] -> [행동 선택] -+
       |
       v
[모델 학습] -> [모델 평가] -> [모델 저장]
       |
       v
[백테스트 수행] -> [성능 지표 계산] -> [결과 시각화]
       |
       v
[실시간 트레이딩 시스템 구축]
```

## 7. 구조

![image](https://github.com/user-attachments/assets/d384736c-261f-4c40-a898-83d93cf2a22f)

## 8. 일정

- 총 개발 기간: 약 2달
- 각 단계별 단위 테스트 수행 및 검증
- 주간 진행 상황 점검 및 이슈 해결

## 9. 리스크 관리

- 데이터 품질 문제: 대체 데이터 소스 준비
- GPU 호환성 문제: CPU 폴백 옵션 구현
- API 한도 초과: 데이터 캐싱 및 요청 제한 관리
- 모델 성능 부족: 하이퍼파라미터 튜닝 및 앙상블 방법 고려 
---
# 훈련 화면


https://github.com/user-attachments/assets/13952cf9-204a-4052-bbb0-63f934e3181a


---
# 결과 화면

[▶️ result_video.mp4 직접 보기](https://raw.githubusercontent.com/sw-dreamer/trading-project/master/video/result_video.mp4)


https://github.com/user-attachments/assets/4bfb6a06-4185-4a76-a7f1-9845d3369fba


# trading-project
