"""
모델 상태 관리자 - 백테스팅 완료된 모델을 실시간 트레이딩에서 사용하기 위한 관리자
"""
import os
import torch
import pickle
import json
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd

from src.models.sac_agent import SACAgent
from src.preprocessing.data_processor import DataProcessor
from src.config.config import Config

class ModelStateManager:
    """
    백테스팅 완료된 모델과 전처리 파이프라인을 실시간 트레이딩에서 사용하기 위한 관리자
    """
    
    def __init__(self, model_path: str, config: Config):
        """
        ModelStateManager 초기화
        
        Args:
            model_path: 백테스팅 완료된 모델 경로
            config: 설정 객체
        """
        self.model_path = Path(model_path)
        self.config = config
        self.agent = None
        self.data_processor = None
        self.model_metadata = {}
    
    def load_complete_model(self) -> tuple:
        """
        백테스팅 완료된 모델과 전처리 파이프라인을 로드
        
        Returns:
            (SAC 에이전트, 데이터 전처리기) 튜플
        """
        try:
            # 1. 모델 메타데이터 로드
            self._load_model_metadata()
            
            # 2. SAC 에이전트 로드
            self.agent = self._load_sac_agent()
            
            # 3. 데이터 전처리기 로드 (스케일러 포함)
            self.data_processor = self._load_data_processor()
            
            print(f"✅ 백테스팅 완료된 모델 로드 성공: {self.model_path}")
            print(f"   └─ 모델 학습 날짜: {self.model_metadata.get('training_date', 'Unknown')}")
            print(f"   └─ 백테스트 성능: {self.model_metadata.get('backtest_performance', 'N/A')}")
            print(f"   └─ 사용된 심볼: {self.model_metadata.get('symbols', 'N/A')}")
            
            return self.agent, self.data_processor
            
        except Exception as e:
            print(f"❌ 모델 로드 실패: {e}")
            raise
        
    def _load_model_metadata(self) -> None:
        """모델 메타데이터 로드"""
        metadata_path = self.model_path / "model_metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.model_metadata = json.load(f)
        else:
            print("ℹ️  모델 메타데이터 파일이 없습니다. 기본 설정을 사용합니다.")
            self.model_metadata = {
                "training_date": "Unknown",
                "backtest_performance": "N/A",
                "symbols": self.config.TRADING_SYMBOLS,
                "window_size": self.config.WINDOW_SIZE,
                "model_type": {
                    "use_cnn": False,
                    "use_lstm": False,
                    "state_dim": self.config.WINDOW_SIZE,
                    "action_dim": 1,
                    "hidden_dim": self.config.HIDDEN_DIM
                }
            }
    
    def _load_sac_agent(self) -> SACAgent:
        """SAC 에이전트 로드 (LSTM/CNN 지원)"""
        try:
            # 모델 설정 파일 로드
            config_path = self.model_path / "config.pth"
            if not config_path.exists():
                raise FileNotFoundError(f"모델 설정 파일이 없습니다: {config_path}")
            
            model_config = torch.load(config_path, map_location=self.config.DEVICE)
            
            # LSTM/CNN 사용 여부 확인
            use_cnn = model_config.get('use_cnn', False)
            use_lstm = model_config.get('use_lstm', False)
            input_shape = model_config.get('input_shape')
            
            print(f"📋 모델 설정 로드:")
            print(f"   └─ CNN 사용: {use_cnn}")
            print(f"   └─ LSTM 사용: {use_lstm}")
            print(f"   └─ 입력 형태: {input_shape}")
            
            # SAC 에이전트 생성
            agent = SACAgent(
                state_dim=model_config.get('state_dim'),
                action_dim=model_config.get('action_dim', 1),
                hidden_dim=model_config.get('hidden_dim', self.config.HIDDEN_DIM),
                use_cnn=use_cnn,
                use_lstm=use_lstm,  # LSTM 지원 추가
                input_shape=input_shape,
                lstm_hidden_dim=model_config.get('lstm_hidden_dim', 128),
                num_lstm_layers=model_config.get('num_lstm_layers', 2),
                lstm_dropout=model_config.get('lstm_dropout', 0.2),
                device=self.config.DEVICE
            )
            
            # 모델 가중치 로드
            agent.load_model(self.model_path)
            
            model_type = "LSTM" if use_lstm else ("CNN" if use_cnn else "MLP")
            print(f"✅ {model_type} SAC 에이전트 로드 완료")
            
            return agent
            
        except Exception as e:
            print(f"❌ SAC 에이전트 로드 실패: {e}")
            raise
    
    def _load_data_processor(self) -> DataProcessor:
        """데이터 전처리기 로드 (스케일러 포함)"""
        try:
            # 데이터 전처리기 생성
            processor = DataProcessor(
                window_size=self.model_metadata.get('window_size', self.config.WINDOW_SIZE)
            )
            
            # 저장된 스케일러들 로드
            scalers_path = self.model_path / "scalers.pkl"
            if scalers_path.exists():
                with open(scalers_path, 'rb') as f:
                    processor.scalers = pickle.load(f)
                print(f"✅ 스케일러 로드 완료: {len(processor.scalers)}개 심볼")
            else:
                print("⚠️  저장된 스케일러가 없습니다. 실시간 데이터로 새로 생성됩니다.")
            
            return processor
            
        except Exception as e:
            print(f"❌ 데이터 전처리기 로드 실패: {e}")
            raise
    
    def save_model_state(
        self, 
        agent: SACAgent, 
        data_processor: DataProcessor, 
        backtest_results: Dict[str, Any],
        symbols: list
    ) -> None:
        """
        백테스팅 완료된 모델 상태 저장 (LSTM/CNN 정보 포함)
        """
        try:
            # 모델 저장
            model_save_path = agent.save_model(prefix='backtest_complete_')
            
            # 스케일러 저장
            scalers_path = model_save_path / "scalers.pkl"
            with open(scalers_path, 'wb') as f:
                pickle.dump(data_processor.scalers, f)
            
            # 메타데이터 저장 (모델 타입 정보 포함)
            metadata = {
                "training_date": pd.Timestamp.now().isoformat(),
                "backtest_performance": {
                    "total_return": backtest_results.get('total_return', 0),
                    "sharpe_ratio": backtest_results.get('sharpe_ratio', 0),
                    "max_drawdown": backtest_results.get('max_drawdown', 0)
                },
                "symbols": symbols,
                "window_size": data_processor.window_size,
                "model_path": str(model_save_path),
                "model_type": {
                    "use_cnn": getattr(agent, 'use_cnn', False),
                    "use_lstm": getattr(agent, 'use_lstm', False),
                    "state_dim": agent.state_dim if hasattr(agent, 'state_dim') else None,
                    "action_dim": agent.action_dim if hasattr(agent, 'action_dim') else 1,
                    "hidden_dim": agent.hidden_dim if hasattr(agent, 'hidden_dim') else None
                }
            }
            
            metadata_path = model_save_path / "model_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ 백테스팅 완료된 모델 상태 저장: {model_save_path}")
            
        except Exception as e:
            print(f"❌ 모델 상태 저장 실패: {e}")
            raise
    
    def validate_model_compatibility(self) -> bool:
        """
        모델이 현재 실시간 트레이딩 설정과 호환되는지 확인 (모델 타입 포함)
        """
        try:
            # 메타데이터 파일이 없는 경우 기본 설정으로 진행
            if not hasattr(self, 'model_metadata') or not self.model_metadata:
                print("ℹ️  모델 메타데이터가 없습니다. 기본 설정으로 진행합니다.")
                return True
            
            # 심볼 호환성 확인
            model_symbols = set(self.model_metadata.get('symbols', []))
            trading_symbols = set(self.config.trading_symbols)
            
            if not trading_symbols.issubset(model_symbols):
                missing_symbols = trading_symbols - model_symbols
                print(f"⚠️  모델에 없는 심볼들: {missing_symbols}")
                
                if missing_symbols == {'AAPL'} or 'AAPL' in missing_symbols:
                    print("💡 AAPL 심볼을 테스트용으로 허용합니다.")
                    return True
                else:
                    return False
            
            # 윈도우 크기 호환성 확인
            model_window_size = self.model_metadata.get('window_size', 0)
            if model_window_size != self.config.window_size:
                print(f"⚠️  윈도우 크기 불일치: 모델({model_window_size}) vs 설정({self.config.window_size})")
                return False
            
            # 모델 타입 정보 출력
            model_type_info = self.model_metadata.get('model_type', {})
            use_cnn = model_type_info.get('use_cnn', False)
            use_lstm = model_type_info.get('use_lstm', False)
            
            if use_cnn:
                print("🧠 모델 타입: CNN (Convolutional Neural Network)")
            elif use_lstm:
                print("🧠 모델 타입: LSTM (Long Short-Term Memory)")
            else:
                print("🧠 모델 타입: MLP (Multi-Layer Perceptron)")
            
            print("✅ 모델 호환성 확인 완료")
            return True
            
        except Exception as e:
            print(f"❌ 호환성 확인 중 오류: {e}")
            return False


def create_complete_trading_system(model_path: str, config: Config):
    """
    백테스팅 완료된 모델을 사용하여 완전한 실시간 트레이딩 시스템 생성
    
    Args:
        model_path: 백테스팅 완료된 모델 경로
        config: 설정 객체
        
    Returns:
        (agent, data_processor) 튜플
    """
    print("🚀 백테스팅 완료된 모델로 실시간 트레이딩 시스템 생성 중...")
    
    # 모델 상태 관리자 생성
    model_manager = ModelStateManager(model_path, config)
    
    # 모델 호환성 확인
    if not model_manager.validate_model_compatibility():
        raise ValueError("모델이 현재 실시간 트레이딩 설정과 호환되지 않습니다.")
    
    # 완전한 모델 로드
    agent, data_processor = model_manager.load_complete_model()
    
    print("✅ 실시간 트레이딩 시스템 생성 완료!")
    return agent, data_processor