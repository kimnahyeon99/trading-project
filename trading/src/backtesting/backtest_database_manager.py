#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
백테스트 결과 데이터베이스 관리 모듈
MySQL 데이터베이스에 백테스트 결과를 저장하고 관리합니다.
"""

import mysql.connector
import json
import os
from datetime import datetime, date
from typing import Dict, Any, Optional
import traceback

from src.config.ea_teb_config import (
    DEVICE,
    TARGET_SYMBOLS,
    LOGGER,
    WINDOW_SIZE,
    INITIAL_BALANCE,
    TRANSACTION_FEE_PERCENT
)

class BacktestDatabaseManager:
    """
    백테스트 결과를 MySQL 데이터베이스에 저장하고 관리하는 클래스
    """

    def __init__(self, host: str, database: str, user: str, password: str, port: int = 3306):
        """
        MySQL 데이터베이스 연결 초기화

        Args:
            host: MySQL 서버 호스트
            database: 데이터베이스 이름
            user: 사용자명
            password: 비밀번호
            port: 포트 번호 (기본값: 3306)
        """
        self.connection_config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'port': port,
            'charset': 'utf8mb4',
            'autocommit': False
        }
        self.connection = None

    def connect(self) -> bool:
        """
        데이터베이스 연결

        Returns:
            bool: 연결 성공 여부
        """
        try:
            self.connection = mysql.connector.connect(**self.connection_config)
            print("✅ MySQL 데이터베이스에 성공적으로 연결되었습니다.")
            return True
        except mysql.connector.Error as error:
            print(f"❌ 데이터베이스 연결 실패: {error}")
            return False

    def disconnect(self) -> None:
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            print("📋 MySQL 연결이 종료되었습니다.")

    def create_tables_if_not_exists(self) -> bool:
        """
        기존 테이블 구조에 맞게 확인 (테이블 생성하지 않음)

        Returns:
            bool: 테이블 확인 성공 여부
        """
        if not self.connection:
            print("❌ 데이터베이스가 연결되지 않았습니다.")
            return False

        try:
            cursor = self.connection.cursor()

            print("🔄 기존 테이블 확인 중...")
            
            # 테이블 존재 여부 확인
            cursor.execute("SHOW TABLES LIKE 'backtest_results'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                print("📋 기존 backtest_results 테이블 발견됨.")
                cursor.execute("DESCRIBE backtest_results")
                columns = [row[0] for row in cursor.fetchall()]
                print(f"📋 backtest_results 테이블 컬럼: {columns}")
                
                # 필수 컬럼 확인
                required_columns = [
                    'id', 'symbol', 'model_id', 'backtest_date', 'start_date', 'end_date',
                    'initial_balance', 'final_balance', 'total_return', 'win_rate', 
                    'total_trades', 'annualized_return', 'sharpe_ratio', 'max_drawdown', 
                    'profit_factor', 'parameters'
                ]
                
                missing_columns = [col for col in required_columns if col not in columns]
                if missing_columns:
                    print(f"⚠️  누락된 컬럼: {missing_columns}")
                    return False
                else:
                    print("✅ 모든 필수 컬럼이 존재합니다.")
            else:
                print("❌ backtest_results 테이블이 존재하지 않습니다.")
                print("💡 다음 SQL로 테이블을 생성하세요:")
                print("""
CREATE TABLE backtest_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    symbol VARCHAR(50) NOT NULL,
    model_id VARCHAR(50) NOT NULL,
    backtest_date DATE NOT NULL,
    start_date DATETIME,
    end_date DATETIME,
    initial_balance DECIMAL(15,2) NOT NULL,
    final_balance DECIMAL(15,2) NOT NULL,
    total_return DECIMAL(15,2) NOT NULL,
    win_rate DECIMAL(10,6),
    total_trades INT DEFAULT 0,
    annualized_return DECIMAL(10,6),
    sharpe_ratio DECIMAL(10,6),
    max_drawdown DECIMAL(10,6),
    profit_factor DECIMAL(10,6),
    parameters JSON
);
                """)
                return False
            
            cursor.close()
            return True

        except mysql.connector.Error as error:
            print(f"❌ 테이블 확인 중 오류: {error}")
            return False

    def convert_backtest_results_to_db_format(self, results: Dict[str, Any], metrics: Dict[str, Any],
                                               symbol: str, model_path: str, data_type: str = 'test',
                                               backtest_start_time=None, backtest_end_time=None) -> Dict[str, Any]:
        """
        백테스트 결과를 기존 데이터베이스 저장 형식으로 변환 (기존 테이블 구조에 맞춤)

        Args:
            results: 백테스트 결과 딕셔너리
            metrics: 성능 지표 딕셔너리
            symbol: 주식 심볼
            model_path: 모델 경로
            data_type: 데이터 타입 ('train', 'valid', 'test')
            backtest_start_time: 백테스트 시작 시간
            backtest_end_time: 백테스트 종료 시간

        Returns:
            Dict: 기존 데이터베이스 저장용 형식
        """
        try:
            # 날짜/시간 정보 (실제 백테스팅 시간 사용)
            if backtest_start_time and backtest_end_time:
                start_date = backtest_start_time  # 실제 백테스팅 시작 시간
                end_date = backtest_end_time      # 실제 백테스팅 종료 시간
                backtest_date = backtest_start_time.date()  # 백테스팅 시작 날짜
            else:
                # 백업: 현재 시간 사용 (기존 방식)
                now = datetime.now()
                start_date = now
                end_date = now
                backtest_date = now.date()
            
            # 기본 값들
            initial_balance = results.get('initial_portfolio_value', INITIAL_BALANCE)
            final_balance = results.get('final_portfolio_value', initial_balance)
            
            # 메트릭스에서 값들 추출 (기본값 설정)
            total_return = metrics.get('total_return', 0.0)
            annualized_return = metrics.get('annual_return', 0.0)
            sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
            max_drawdown = metrics.get('max_drawdown', 0.0)
            win_rate = metrics.get('win_rate', 0.0)
            total_trades = metrics.get('total_trades', 0)
            
            # Profit Factor 계산
            profit_factor = self._calculate_profit_factor(results)
            
            # 모델 타입 감지 (모델 경로에서 추출)
            model_type = self._detect_model_type(model_path)
            
            # Model ID 생성 (새로운 형식: 날짜_모델타입_수익률)
            date_str = backtest_date.strftime("%Y%m%d")
            return_str = f"{total_return:.2f}"
            model_id = f"{date_str}_{model_type}_{return_str}"
            
            # Parameters JSON 생성 (모델 경로 및 시간 정보 포함)
            parameters = {
                'model_path': model_path,
                'data_type': data_type,
                'model_type': model_type,  # 모델 타입 추가
                'window_size': 30,  # 기본값
                'initial_balance': initial_balance,
                'algorithm': 'sac',
                'total_steps': metrics.get('total_steps', 0),
                'backtest_start_time': start_date.isoformat(),
                'backtest_end_time': end_date.isoformat(),
                'backtest_duration_seconds': (end_date - start_date).total_seconds() if backtest_start_time and backtest_end_time else 0
            }
            
            # 기존 테이블 구조에 맞는 데이터 생성
            db_data = {
                'symbol': symbol,
                'model_id': model_id,
                'backtest_date': backtest_date,
                'start_date': start_date,  # 실제 백테스팅 시작 시간
                'end_date': end_date,      # 실제 백테스팅 종료 시간
                'initial_balance': float(initial_balance),
                'final_balance': float(final_balance),
                'total_return': float(total_return),
                'win_rate': float(win_rate / 100.0) if win_rate > 1.0 else float(win_rate),  # 비율로 변환
                'total_trades': int(total_trades),
                'annualized_return': float(annualized_return / 100.0) if annualized_return > 1.0 else float(annualized_return),  # 비율로 변환
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(abs(max_drawdown) / 100.0) if abs(max_drawdown) > 1.0 else float(abs(max_drawdown)),  # 비율로 변환
                'profit_factor': float(profit_factor),
                'parameters': parameters
            }
            
            return db_data
            
        except Exception as e:
            print(f"❌ 데이터 변환 중 오류: {str(e)}")
            # 기본 데이터 반환
            now = datetime.now()
            return {
                'symbol': symbol,
                'model_id': f"{now.strftime('%Y%m%d')}_unknown_0.00",
                'backtest_date': now.date(),
                'start_date': now,
                'end_date': now,
                'initial_balance': INITIAL_BALANCE,
                'final_balance': INITIAL_BALANCE,
                'total_return': 0.0,
                'win_rate': 0.0,
                'total_trades': 0,
                'annualized_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 0.0,
                'parameters': {'model_path': model_path, 'data_type': data_type}
            }

    def _detect_model_type(self, model_path: str) -> str:
        """
        모델 경로에서 모델 타입 감지
        
        Args:
            model_path: 모델 파일 경로
            
        Returns:
            str: 모델 타입 ('mlp', 'cnn', 'lstm', 'transformer')
        """
        try:
            model_name = os.path.basename(model_path.rstrip('/\\')).lower()
            
            # 모델 이름에서 타입 감지
            if 'cnn' in model_name:
                return 'cnn'
            elif 'lstm' in model_name:
                return 'lstm'
            elif 'transformer' in model_name:
                return 'transformer'
            elif 'mlp' in model_name or 'sac_model' in model_name:
                return 'mlp'
            else:
                # 기본값으로 mlp 반환
                return 'mlp'
                
        except Exception as e:
            print(f"⚠️  모델 타입 감지 실패: {e}")
            return 'unknown'

    def _calculate_profit_factor(self, results: Dict[str, Any]) -> float:
        """
        Profit Factor 계산 (총 이익 / 총 손실)

        Args:
            results: 백테스트 결과

        Returns:
            float: Profit Factor 값
        """
        try:
            rewards = results.get('rewards', [])
            if not rewards:
                return 0.0

            total_profit = sum(r for r in rewards if r > 0)
            total_loss = abs(sum(r for r in rewards if r < 0))

            if total_loss == 0:
                return float('inf') if total_profit > 0 else 0.0

            return total_profit / total_loss

        except Exception:
            return 0.0

    def insert_backtest_result(self, backtest_data: Dict[str, Any],
                               trade_details: Optional[Dict[str, Any]] = None) -> Optional[int]:
        """
        백테스트 결과를 기존 데이터베이스 테이블에 저장

        Args:
            backtest_data: 백테스트 결과 데이터
            trade_details: 상세 거래 내역 (현재 미사용)

        Returns:
            Optional[int]: 저장된 레코드의 ID (실패시 None)
        """
        if not self.connection:
            print("❌ 데이터베이스가 연결되지 않았습니다.")
            return None

        try:
            cursor = self.connection.cursor()

            # 기존 테이블 구조에 맞는 INSERT 쿼리
            insert_query = """
            INSERT INTO backtest_results (
                symbol, model_id, backtest_date, start_date, end_date,
                initial_balance, final_balance, total_return, win_rate, total_trades,
                annualized_return, sharpe_ratio, max_drawdown, profit_factor, parameters
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
            )
            """

            # 날짜/시간 처리
            backtest_date = backtest_data['backtest_date']  # date 객체
            start_date = backtest_data['start_date']  # datetime 객체
            end_date = backtest_data['end_date']  # datetime 객체

            # 데이터 준비 (기존 테이블 구조에 맞게)
            values = (
                backtest_data['symbol'],
                backtest_data['model_id'],
                backtest_date,
                start_date,
                end_date,
                float(backtest_data['initial_balance']),
                float(backtest_data['final_balance']),
                float(backtest_data['total_return']),
                float(backtest_data.get('win_rate', 0.0)),
                int(backtest_data.get('total_trades', 0)),
                float(backtest_data.get('annualized_return', 0.0)),
                float(backtest_data.get('sharpe_ratio', 0.0)),
                float(backtest_data.get('max_drawdown', 0.0)),
                float(backtest_data.get('profit_factor', 0.0)),
                json.dumps(backtest_data.get('parameters', {}))
            )

            cursor.execute(insert_query, values)
            backtest_id = cursor.lastrowid

            self.connection.commit()

            print(f"✅ 백테스트 결과가 성공적으로 저장되었습니다.")
            print(f"   └─ Backtest ID: {backtest_id}")
            print(f"   └─ Model ID: {backtest_data['model_id']}")
            print(f"   └─ Symbol: {backtest_data['symbol']}")
            print(f"   └─ Backtest Date: {backtest_date}")
            print(f"   └─ Total Return: {backtest_data['total_return']:.4f}%")
            print(f"   └─ Win Rate: {backtest_data.get('win_rate', 0.0)*100:.2f}%")
            print(f"   └─ Total Trades: {backtest_data.get('total_trades', 0):,}")

            cursor.close()
            return backtest_id

        except mysql.connector.Error as error:
            print(f"❌ 데이터 저장 중 MySQL 오류: {error}")
            if self.connection:
                self.connection.rollback()
            return None
        except Exception as error:
            print(f"❌ 데이터 저장 중 예상치 못한 오류: {error}")
            print(traceback.format_exc())
            if self.connection:
                self.connection.rollback()
            return None

    def check_existing_record(self, model_id: str, symbol: str) -> bool:
        """
        이미 존재하는 레코드인지 확인 (기존 테이블 구조에 맞춤)

        Args:
            model_id: 모델 ID
            symbol: 주식 심볼

        Returns:
            bool: 레코드 존재 여부
        """
        if not self.connection:
            return False

        try:
            cursor = self.connection.cursor()
            query = """
            SELECT COUNT(*) FROM backtest_results 
            WHERE model_id = %s AND symbol = %s
            """
            cursor.execute(query, (model_id, symbol))
            count = cursor.fetchone()[0]
            cursor.close()
            return count > 0
        except mysql.connector.Error as error:
            print(f"❌ 레코드 확인 중 오류: {error}")
            return False

    def get_performance_summary(self, symbol: str = None, limit: int = 10) -> Optional[list]:
        """
        성능 요약 정보 조회 (기존 테이블 구조에 맞춤)

        Args:
            symbol: 특정 심볼 필터 (선택사항)
            limit: 조회할 최대 레코드 수

        Returns:
            Optional[list]: 성능 요약 리스트
        """
        if not self.connection:
            return None

        try:
            cursor = self.connection.cursor(dictionary=True)

            if symbol:
                query = """
                SELECT symbol, model_id, backtest_date, total_return, 
                       sharpe_ratio, max_drawdown, total_trades, win_rate
                FROM backtest_results 
                WHERE symbol = %s
                ORDER BY backtest_date DESC, total_return DESC
                LIMIT %s
                """
                cursor.execute(query, (symbol, limit))
            else:
                query = """
                SELECT symbol, model_id, backtest_date, total_return, 
                       sharpe_ratio, max_drawdown, total_trades, win_rate
                FROM backtest_results 
                ORDER BY backtest_date DESC, total_return DESC
                LIMIT %s
                """
                cursor.execute(query, (limit,))

            results = cursor.fetchall()
            cursor.close()
            return results

        except mysql.connector.Error as error:
            print(f"❌ 성능 요약 조회 중 오류: {error}")
            return None

    def display_summary(self, backtest_data: Dict[str, Any]) -> None:
        """
        백테스트 결과 요약 정보 출력 (기존 테이블 구조에 맞춤)

        Args:
            backtest_data: 백테스트 결과 데이터
        """
        print("\n" + "=" * 50)
        print("📊 백테스트 결과 요약")
        print("=" * 50)
        print(f"📈 심볼: {backtest_data.get('symbol', 'N/A')}")
        print(f"🤖 모델 ID: {backtest_data.get('model_id', 'N/A')}")
        print(f"📅 백테스트 날짜: {backtest_data.get('backtest_date', 'N/A')}")
        
        # 시작/종료 시간 정보 추가
        start_time = backtest_data.get('start_date')
        end_time = backtest_data.get('end_date')
        if start_time and end_time:
            print(f"⏰ 시작 시각: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⏰ 종료 시각: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            # 소요 시간 계산
            duration = end_time - start_time
            duration_str = str(duration).split('.')[0]  # 마이크로초 제거
            print(f"⏱️  소요 시간: {duration_str}")
        
        print(f"💰 초기 자산: ${backtest_data.get('initial_balance', 0):,.2f}")
        print(f"💵 최종 자산: ${backtest_data.get('final_balance', 0):,.2f}")
        print(f"📈 총 수익률: {backtest_data.get('total_return', 0):.2f}%")
        print(f"📊 연간 수익률: {backtest_data.get('annualized_return', 0)*100:.2f}%")
        print(f"⚖️  샤프 비율: {backtest_data.get('sharpe_ratio', 0):.4f}")
        print(f"📉 최대 낙폭: {backtest_data.get('max_drawdown', 0)*100:.2f}%")
        print(f"🎯 승률: {backtest_data.get('win_rate', 0)*100:.2f}%")
        print(f"🔄 총 거래 수: {backtest_data.get('total_trades', 0):,}")
        print(f"⚡ Profit Factor: {backtest_data.get('profit_factor', 0):.4f}")
        
        # Parameters 정보 출력
        parameters = backtest_data.get('parameters', {})
        if isinstance(parameters, str):
            try:
                parameters = json.loads(parameters)
            except:
                parameters = {}
        
        print(f"📁 모델 경로: {parameters.get('model_path', 'N/A')}")
        print(f"📊 데이터 타입: {parameters.get('data_type', 'N/A')}")
        print(f"🏷️  모델 타입: {parameters.get('model_type', 'N/A')}")
        
        # 백테스팅 소요 시간 (Parameters에서)
        duration_seconds = parameters.get('backtest_duration_seconds', 0)
        if duration_seconds > 0:
            hours = int(duration_seconds // 3600)
            minutes = int((duration_seconds % 3600) // 60)
            seconds = int(duration_seconds % 60)
            if hours > 0:
                duration_display = f"{hours}시간 {minutes}분 {seconds}초"
            elif minutes > 0:
                duration_display = f"{minutes}분 {seconds}초"
            else:
                duration_display = f"{seconds}초"
            print(f"⏱️  실행 시간: {duration_display}")
        
        print("=" * 50 + "\n")

    def __enter__(self):
        """Context manager 진입"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        self.disconnect()