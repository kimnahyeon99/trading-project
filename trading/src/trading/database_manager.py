#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mysql.connector
from mysql.connector import Error
import json
from datetime import datetime
from decimal import Decimal
import logging
from typing import Dict, List, Optional, Any
import time
import os
import torch
import pandas as pd


class DatabaseManager:
    """MySQL 데이터베이스 연결 및 트레이딩 데이터 저장 관리 클래스 - 기존 테이블 구조 호환 버전"""
    
    def __init__(self, host='192.168.40.199', database='trading', 
                 user='root', password='mysecretpassword', logger=None):
        """
        데이터베이스 매니저 초기화
        
        Args:
            host: MySQL 서버 호스트
            database: 데이터베이스 명
            user: 사용자명
            password: 비밀번호
            logger: 로거 인스턴스
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        self.logger = logger or logging.getLogger(__name__)
        self.connection_config = {
            'host': host,
            'database': database,
            'user': user,
            'password': password,
            'autocommit': True,
            'charset': 'utf8mb4',
            'connection_timeout': 30,
            'sql_mode': 'TRADITIONAL'  # 엄격한 모드로 오류 감지 향상
        }
        
    def connect(self) -> bool:
        """데이터베이스 연결"""
        try:
            if self.connection and self.connection.is_connected():
                return True
                
            self.connection = mysql.connector.connect(**self.connection_config)
            
            if self.connection.is_connected():
                self.logger.info(f"✅ MySQL 데이터베이스 연결 성공: {self.host}/{self.database}")
                return True
            
        except Error as e:
            self.logger.error(f"❌ MySQL 연결 실패: {e}")
            return False
        
        return False
    
    def disconnect(self):
        """데이터베이스 연결 종료"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            self.logger.info("🔌 MySQL 연결 종료")
    
    def is_connected(self) -> bool:
        """연결 상태 확인 및 재연결"""
        try:
            if self.connection and self.connection.is_connected():
                # 연결 상태 테스트
                try:
                    self.connection.ping(attempts=3, delay=1)
                    return True
                except:
                    # ping 실패 시 연결이 끊어진 것으로 간주
                    pass
        except:
            pass
        
        # 연결이 끊어진 경우 재연결 시도
        self.logger.warning("⚠️ MySQL 연결이 끊어졌습니다. 재연결을 시도합니다...")
        return self.connect()
    
    def execute_query(self, query: str, params: tuple = None, retry_count: int = 3) -> bool:
        """쿼리 실행 (INSERT, UPDATE, DELETE) - 재시도 로직 포함"""
        for attempt in range(retry_count):
            try:
                # 연결 상태 확인 및 재연결
                if not self.is_connected():
                    self.logger.error("❌ 데이터베이스 연결을 설정할 수 없습니다.")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return False
                
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                cursor.close()
                return True
                
            except Error as e:
                self.logger.error(f"❌ 쿼리 실행 실패 (시도 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    self.logger.info("🔄 1초 후 재시도...")
                    time.sleep(1)
                    # 연결 재설정
                    try:
                        if self.connection:
                            self.connection.close()
                    except:
                        pass
                    self.connection = None
                else:
                    self.logger.error(f"Query: {query}")
                    if params:
                        self.logger.error(f"Params: {params}")
                    
        return False
    
    def fetch_query(self, query: str, params: tuple = None, retry_count: int = 3) -> List[Dict]:
        """쿼리 실행 (SELECT) 및 결과 반환 - 재시도 로직 포함"""
        for attempt in range(retry_count):
            try:
                # 연결 상태 확인 및 재연결
                if not self.is_connected():
                    self.logger.error("❌ 데이터베이스 연결을 설정할 수 없습니다.")
                    if attempt < retry_count - 1:
                        time.sleep(1)
                        continue
                    return []
                
                cursor = self.connection.cursor(dictionary=True)
                cursor.execute(query, params)
                results = cursor.fetchall()
                cursor.close()
                return results
                
            except Error as e:
                self.logger.error(f"❌ 조회 쿼리 실행 실패 (시도 {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    self.logger.info("🔄 1초 후 재시도...")
                    time.sleep(1)
                    # 연결 재설정
                    try:
                        if self.connection:
                            self.connection.close()
                    except:
                        pass
                    self.connection = None
                else:
                    self.logger.error(f"Query: {query}")
                    if params:
                        self.logger.error(f"Params: {params}")
                    
        return []
    
    def save_model_info_detailed(self, model_id: str, file_path: str, 
                           symbols: List[str], 
                           description: str = None, 
                           is_active: bool = False,
                           model_metadata: Dict = None,
                           config_info: Dict = None) -> bool:
        """
        모델 파일 자체(BLOB)까지 포함하여 상세 모델 정보를 저장 - 디버깅 강화 버전
        """
        print(f"🔍 DEBUG: save_model_info_detailed 시작")
        print(f"   └─ model_id: {model_id}")
        print(f"   └─ file_path: {file_path}")
        print(f"   └─ symbols: {symbols}")
        print(f"   └─ description: {description}")
        print(f"   └─ is_active: {is_active}")
        print(f"   └─ model_metadata type: {type(model_metadata)}")
        print(f"   └─ config_info type: {type(config_info)}")
        
        try:
            
            # None 체크 강화
            if model_metadata is None:
                model_metadata = {}
            if config_info is None:
                config_info = {}
            if symbols is None:
                symbols = []
            
            # dict 타입이 아닌 경우도 체크
            if not isinstance(model_metadata, dict):
                model_metadata = {}
            if not isinstance(config_info, dict):
                config_info = {}
            if not isinstance(symbols, list):
                symbols = []
            
        
            # None 체크 및 기본값 설정
            if model_metadata is None:
                model_metadata = {}
                print("   └─ model_metadata를 빈 dict로 초기화")
                
            if config_info is None:
                config_info = {}
                print("   └─ config_info를 빈 dict로 초기화")
                
            if symbols is None:
                symbols = []
                print("   └─ symbols를 빈 list로 초기화")
            
            # 파일 정보 확인
            file_exists = os.path.exists(file_path)
            print(f"   └─ 파일 존재 여부: {file_exists}")
            
            if file_exists:
                file_size = os.path.getsize(file_path)
                created_time = datetime.fromtimestamp(os.path.getctime(file_path))
                modified_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                print(f"   └─ 파일 크기: {file_size} bytes")
            else:
                file_size = 0
                created_time = datetime.now()
                modified_time = datetime.now()
                print("   └─ 파일이 없어서 기본값 사용")
            
            # 모델 타입 추론
            model_type = "MLP"
            if config_info:
                use_cnn = config_info.get('use_cnn', False)
                use_lstm = config_info.get('use_lstm', False)
                if use_cnn:
                    model_type = "CNN"
                elif use_lstm:
                    model_type = "LSTM"
            print(f"   └─ 모델 타입: {model_type}")
            
            # 상세 정보 구성
            detailed_info = {
                "model_type": model_type,
                "symbols": symbols,
                "training_date": model_metadata.get('training_date') if model_metadata else None,
                "backtest_performance": model_metadata.get('backtest_performance') if model_metadata else None,
                "window_size": model_metadata.get('window_size') if model_metadata else config_info.get('window_size', 30),
                "model_config": config_info,
                "metadata": model_metadata,
                "activation_timestamp": datetime.now().isoformat(),
                "file_size_bytes": file_size
            }
            print(f"   └─ detailed_info 생성 완료")
            
            # description 생성
            if description:
                full_description = f"{description}\n\n[상세정보]\n{json.dumps(detailed_info, ensure_ascii=False, indent=2)}"
            else:
                basic_desc = f"실시간 트레이딩 모델 - {', '.join(symbols)} ({model_type})"
                full_description = f"{basic_desc}\n\n[상세정보]\n{json.dumps(detailed_info, ensure_ascii=False, indent=2)}"
            
            print(f"   └─ description 길이: {len(full_description)} 문자")

            # 모델 파일 읽기 (BLOB)
            model_blob = None
            if file_exists:
                try:
                    if os.path.isdir(file_path):
                        # 디렉토리인 경우 모든 파일을 하나의 바이너리로 결합
                        model_files = []
                        for root, _, files in os.walk(file_path):
                            for file in files:
                                if file.endswith('.pth'):  # PyTorch 모델 파일만 처리
                                    file_full_path = os.path.join(root, file)
                                    with open(file_full_path, 'rb') as f:
                                        model_files.append(f.read())
                        if model_files:
                            # 파일들을 하나의 바이너리로 결합
                            model_blob = b''.join(model_files)
                            print(f"   └─ 모델 디렉토리 읽기 성공: {len(model_files)}개 파일, 총 {len(model_blob)} bytes")
                    else:
                        # 단일 파일인 경우
                        with open(file_path, 'rb') as f:
                            model_blob = f.read()
                        print(f"   └─ 모델 파일 읽기 성공: {len(model_blob)} bytes")
                except Exception as e:
                    print(f"   └─ 모델 파일 읽기 실패: {e}")
                    model_blob = None
            else:
                print("   └─ 파일이 없어서 model_blob = None")

            # 연결 상태 확인
            if not self.is_connected():
                print("   └─ ❌ 데이터베이스 연결 실패")
                return False
            
            print("   └─ ✅ 데이터베이스 연결 확인")

            # 쿼리 준비
            query = """
            INSERT INTO models 
            (model_id, file_path, created_time, modified_time, file_size, description, is_active, model_blob)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                description = VALUES(description),
                is_active = VALUES(is_active),
                modified_time = VALUES(modified_time),
                model_blob = VALUES(model_blob)
            """

            params = (
                model_id,
                file_path,
                created_time,
                modified_time,
                file_size,
                full_description,
                is_active,
                model_blob
            )

            print(f"   └─ 쿼리 파라미터 준비 완료")
            print(f"      ├─ model_id: {params[0]}")
            print(f"      ├─ file_path: {params[1]}")
            print(f"      ├─ created_time: {params[2]}")
            print(f"      ├─ modified_time: {params[3]}")
            print(f"      ├─ file_size: {params[4]}")
            print(f"      ├─ description 길이: {len(params[5])}")
            print(f"      ├─ is_active: {params[6]}")
            print(f"      └─ model_blob 크기: {len(params[7]) if params[7] else 0}")

            # 쿼리 실행 전 로그
            print("   └─ 🚀 쿼리 실행 시작...")
            
            # execute_query 메서드를 직접 호출하는 대신 여기서 직접 실행해보기
            try:
                cursor = self.connection.cursor()
                cursor.execute(query, params)
                affected_rows = cursor.rowcount
                cursor.close()
                
                print(f"   └─ ✅ 쿼리 실행 성공! 영향받은 행: {affected_rows}")
                
                # 저장 확인
                check_query = "SELECT model_id, file_size, is_active FROM models WHERE model_id = %s"
                check_result = self.fetch_query(check_query, (model_id,))
                
                if check_result:
                    print(f"   └─ ✅ 저장 확인 성공: {check_result[0]}")
                else:
                    print(f"   └─ ⚠️ 저장 확인 실패: 데이터를 찾을 수 없음")
                
                self.logger.info(f"🤖 모델 정보 + 파일 저장 완료: {model_id}")
                self.logger.info(f"   └─ 타입: {model_type}")
                self.logger.info(f"   └─ 심볼: {', '.join(symbols)}")
                self.logger.info(f"   └─ 파일 크기: {file_size:,} bytes")
                return True
                
            except mysql.connector.Error as db_error:
                print(f"   └─ ❌ MySQL 오류: {db_error}")
                print(f"      └─ 오류 코드: {db_error.errno}")
                print(f"      └─ SQL 상태: {db_error.sqlstate}")
                self.logger.error(f"❌ MySQL 오류: {db_error}")
                return False
            except Exception as e:
                print(f"   └─ ❌ 일반 오류: {e}")
                self.logger.error(f"❌ 쿼리 실행 중 오류: {e}")
                return False

        except Exception as e:
            print(f"🔍 DEBUG: save_model_info_detailed 전체 오류: {e}")
            print(f"   └─ 오류 타입: {type(e)}")
            import traceback
            print(f"   └─ 스택 트레이스:\n{traceback.format_exc()}")
            self.logger.error(f"❌ save_model_info_detailed 오류: {e}")
            return False
    
    def save_trading_session_info(self, session_id: str, model_ids: List[str], 
                                symbols: List[str], status: str = 'STARTED') -> bool:
        """
        트레이딩 세션 정보를 기존 테이블에 저장 (description 필드를 활용)
        
        Args:
            session_id: 세션 ID
            model_ids: 사용되는 모델 ID 목록
            symbols: 트레이딩 심볼 목록
            status: 세션 상태
            
        Returns:
            저장 성공 여부
        """
        try:
            # 세션 정보를 JSON으로 구성
            session_info = {
                "session_id": session_id,
                "model_ids": model_ids,
                "symbols": symbols,
                "status": status,
                "start_time": datetime.now().isoformat(),
                "type": "trading_session"
            }
            
            # 특별한 model_id로 세션 정보 저장 (session_ 접두사)
            session_model_id = f"session_{session_id}"
            description = f"트레이딩 세션 - {status}\n\n[세션정보]\n{json.dumps(session_info, ensure_ascii=False, indent=2)}"
            
            query = """
            INSERT INTO models 
            (model_id, file_path, description, is_active)
            VALUES (%s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            description = VALUES(description),
            is_active = VALUES(is_active)
            """
            
            params = (
                session_model_id,
                f"session/{session_id}",
                description,
                status == 'STARTED'
            )
            
            if self.execute_query(query, params):
                self.logger.info(f"🚀 트레이딩 세션 정보 저장: {session_id}")
                self.logger.info(f"   └─ 모델: {', '.join(model_ids)}")
                self.logger.info(f"   └─ 심볼: {', '.join(symbols)}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ save_trading_session_info 오류: {e}")
            return False
    
    def update_trading_session_status(self, session_id: str, status: str = 'STOPPED', 
                                    final_stats: Dict = None) -> bool:
        """트레이딩 세션 상태 업데이트"""
        try:
            session_model_id = f"session_{session_id}"
            
            # 기존 정보 조회
            query = "SELECT description FROM models WHERE model_id = %s"
            result = self.fetch_query(query, (session_model_id,))
            
            if not result:
                self.logger.warning(f"세션 정보를 찾을 수 없습니다: {session_id}")
                return False
            
            current_description = result[0]['description']
            
            # 세션 종료 정보 추가
            end_info = {
                "end_time": datetime.now().isoformat(),
                "status": status,
                "final_stats": final_stats or {}
            }
            
            updated_description = current_description + f"\n\n[종료정보]\n{json.dumps(end_info, ensure_ascii=False, indent=2)}"
            
            update_query = """
            UPDATE models 
            SET description = %s, is_active = %s
            WHERE model_id = %s
            """
            
            params = (updated_description, False, session_model_id)
            
            if self.execute_query(update_query, params):
                self.logger.info(f"🏁 트레이딩 세션 종료 정보 업데이트: {session_id} ({status})")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"❌ update_trading_session_status 오류: {e}")
            return False
    
    def save_trading_stats(self, portfolio_value: float, cash_balance: float, 
                          equity_value: float, daily_pnl: float = 0, 
                          total_pnl: float = 0, session_id: str = None) -> bool:
        """trading_stats 테이블에 통계 저장"""
        query = """
        INSERT INTO trading_stats 
        (timestamp, portfolio_value, cash_balance, equity_value, daily_pnl, total_pnl)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        params = (
            datetime.now(),
            Decimal(str(portfolio_value)),
            Decimal(str(cash_balance)),
            Decimal(str(equity_value)),
            Decimal(str(daily_pnl)),
            Decimal(str(total_pnl))
        )
        
        if self.execute_query(query, params):
            self.logger.debug(f"📊 trading_stats 저장 완료 (세션: {session_id})")
            return True
        return False
    
    def save_trade(self, symbol: str, side: str, quantity: float, 
                   price: float, fee: float = 0, pnl: Optional[float] = None,
                   model_id: Optional[str] = None, session_id: str = None) -> bool:
        """trades 테이블에 거래 저장"""
        try:
            # 값 검증 및 변환
            if not symbol or not side:
                self.logger.error(f"❌ 필수 값 누락: symbol={symbol}, side={side}")
                return False
            
            if quantity <= 0 or price <= 0:
                self.logger.error(f"❌ 잘못된 값: quantity={quantity}, price={price}")
                return False
            
            query = """
            INSERT INTO trades 
            (timestamp, symbol, side, quantity, price, fee, pnl, model_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            
            params = (
                datetime.now(),
                str(symbol).upper(),  # 심볼을 대문자로 변환
                str(side).lower(),    # 사이드를 소문자로 변환
                Decimal(str(quantity)),
                Decimal(str(price)),
                Decimal(str(fee)) if fee is not None else Decimal('0'),
                Decimal(str(pnl)) if pnl is not None else None,
                str(model_id) if model_id else None
            )
            
            if self.execute_query(query, params):
                self.logger.info(f"💰 거래 저장 완료: {symbol} {side} {quantity}@{price} (세션: {session_id})")
                return True
            else:
                self.logger.error(f"❌ 거래 저장 실패: {symbol} {side} {quantity}@{price}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ save_trade 오류: {e}")
            return False
    
    def save_position(self, symbol: str, quantity: float, avg_entry_price: float,
                     current_price: Optional[float] = None, 
                     unrealized_pnl: float = 0, session_id: str = None) -> bool:
        """positions 테이블에 포지션 저장"""
        try:
            # 값 검증
            if not symbol:
                self.logger.error("❌ 심볼이 없습니다.")
                return False
            
            # None 값들을 0으로 변환
            quantity = quantity if quantity is not None else 0
            avg_entry_price = avg_entry_price if avg_entry_price is not None else 0
            current_price = current_price if current_price is not None else 0
            unrealized_pnl = unrealized_pnl if unrealized_pnl is not None else 0
            
            query = """
            INSERT INTO positions 
            (symbol, quantity, avg_entry_price, current_price, unrealized_pnl, timestamp)
            VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            params = (
                str(symbol).upper(),  # 심볼을 대문자로 변환
                Decimal(str(quantity)),
                Decimal(str(avg_entry_price)),
                Decimal(str(current_price)),
                Decimal(str(unrealized_pnl)),
                datetime.now()
            )
            
            if self.execute_query(query, params):
                self.logger.debug(f"🏢 포지션 누적 저장: {symbol} {quantity}주 @${avg_entry_price:.2f} (세션: {session_id})")
                return True
            else:
                self.logger.error(f"❌ 포지션 저장 실패: {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ save_position 오류: {e}")
            return False
    
    def get_latest_trading_stats(self, limit: int = 10) -> List[Dict]:
        """최근 거래 통계 조회"""
        query = """
        SELECT * FROM trading_stats 
        ORDER BY timestamp DESC 
        LIMIT %s
        """
        return self.fetch_query(query, (limit,))
    
    def get_recent_trades(self, symbol: str = None, limit: int = 20) -> List[Dict]:
        """최근 거래 내역 조회"""
        if symbol:
            query = """
            SELECT * FROM trades 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            params = (symbol, limit)
        else:
            query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            params = (limit,)
        
        return self.fetch_query(query, params)
    
    def get_current_positions(self) -> List[Dict]:
        """현재 포지션 조회 (각 심볼별 최신 레코드만)"""
        query = """
        SELECT p1.* FROM positions p1
        INNER JOIN (
            SELECT symbol, MAX(timestamp) as max_timestamp
            FROM positions 
            GROUP BY symbol
        ) p2 ON p1.symbol = p2.symbol AND p1.timestamp = p2.max_timestamp
        WHERE ABS(p1.quantity) > 0.000001
        ORDER BY p1.timestamp DESC
        """
        return self.fetch_query(query)
    
    def get_position_history(self, symbol: str = None, limit: int = 50) -> List[Dict]:
        """포지션 변경 이력 조회"""
        if symbol:
            query = """
            SELECT * FROM positions 
            WHERE symbol = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            return self.fetch_query(query, (symbol, limit))
        else:
            query = """
            SELECT * FROM positions 
            ORDER BY timestamp DESC 
            LIMIT %s
            """
            return self.fetch_query(query, (limit,))
    
    def get_model_usage_stats(self) -> List[Dict]:
        """모델 사용 통계 조회 (description에서 정보 추출)"""
        query = """
        SELECT 
            m.model_id,
            m.description,
            m.is_active,
            COUNT(t.id) as total_trades,
            SUM(CASE WHEN t.side = 'buy' THEN 1 ELSE 0 END) as buy_trades,
            SUM(CASE WHEN t.side = 'sell' THEN 1 ELSE 0 END) as sell_trades,
            MAX(t.timestamp) as last_trade_time
        FROM models m
        LEFT JOIN trades t ON m.model_id = t.model_id
        WHERE m.model_id NOT LIKE 'session_%'
        GROUP BY m.model_id, m.description, m.is_active
        ORDER BY total_trades DESC, m.model_id
        """
        return self.fetch_query(query)
    
    def get_trading_sessions(self, limit: int = 20) -> List[Dict]:
        """트레이딩 세션 목록 조회"""
        query = """
        SELECT model_id, description, is_active
        FROM models 
        WHERE model_id LIKE 'session_%'
        ORDER BY model_id DESC
        LIMIT %s
        """
        return self.fetch_query(query, (limit,))
    
    def test_connection(self) -> bool:
        """데이터베이스 연결 테스트"""
        try:
            if not self.is_connected():
                return False
            
            # 간단한 쿼리로 연결 테스트
            result = self.fetch_query("SELECT 1 as test")
            if result and result[0]['test'] == 1:
                self.logger.info("✅ 데이터베이스 연결 테스트 성공")
                return True
            else:
                self.logger.error("❌ 데이터베이스 연결 테스트 실패")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 데이터베이스 연결 테스트 중 오류: {e}")
            return False
    
    def save_market_data(self, symbol: str, data: pd.DataFrame) -> bool:
        """
        시장 데이터를 데이터베이스에 저장
        
        Args:
            symbol: 심볼/티커
            data: 시장 데이터 DataFrame (timestamp, open, high, low, close, volume 포함)
            
        Returns:
            저장 성공 여부
        """
        try:
            if data.empty:
                self.logger.warning(f"❌ 저장할 시장 데이터가 없습니다: {symbol}")
                return False
            
            # 데이터 검증
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                self.logger.error(f"❌ 필수 컬럼 누락: {symbol}")
                return False
            
            # 데이터베이스 연결 확인
            if not self.is_connected():
                self.logger.error("❌ 데이터베이스 연결을 설정할 수 없습니다.")
                return False
            
            # market_data 테이블이 없으면 생성
            create_table_query = """
            CREATE TABLE IF NOT EXISTS market_data (
                id BIGINT AUTO_INCREMENT PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                timestamp DATETIME NOT NULL,
                open DECIMAL(20,8) NOT NULL,
                high DECIMAL(20,8) NOT NULL,
                low DECIMAL(20,8) NOT NULL,
                close DECIMAL(20,8) NOT NULL,
                volume DECIMAL(30,8) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE KEY unique_market_data (symbol, timestamp)
            )
            """
            
            if not self.execute_query(create_table_query):
                self.logger.error("❌ market_data 테이블 생성 실패")
                return False
            
            # 데이터 저장
            query = """
            INSERT INTO market_data 
            (symbol, timestamp, open, high, low, close, volume)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                open = VALUES(open),
                high = VALUES(high),
                low = VALUES(low),
                close = VALUES(close),
                volume = VALUES(volume)
            """
            
            success_count = 0
            for index, row in data.iterrows():
                params = (
                    str(symbol).upper(),
                    index.to_pydatetime(),
                    Decimal(str(row['open'])),
                    Decimal(str(row['high'])),
                    Decimal(str(row['low'])),
                    Decimal(str(row['close'])),
                    Decimal(str(row['volume']))
                )
                
                if self.execute_query(query, params):
                    success_count += 1
            
            if success_count > 0:
                self.logger.info(f"✅ 시장 데이터 저장 완료: {symbol} ({success_count}개)")
                return True
            else:
                self.logger.error(f"❌ 시장 데이터 저장 실패: {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ save_market_data 오류: {e}")
            return False
    
    def get_market_data(self, symbol: str, start_time: Optional[datetime] = None, 
                       end_time: Optional[datetime] = None, limit: int = 100,
                       api_connector = None) -> pd.DataFrame:
        """
        저장된 시장 데이터 조회 (APIConnector 사용)
        
        Args:
            symbol: 심볼/티커
            start_time: 시작 시간 (옵션)
            end_time: 종료 시간 (옵션)
            limit: 조회할 최대 건수
            api_connector: APIConnector 인스턴스
            
        Returns:
            시장 데이터 DataFrame
        """
        try:
            if api_connector is None:
                self.logger.error("❌ APIConnector가 필요합니다.")
                return pd.DataFrame()
            
            # APIConnector를 통해 데이터 가져오기
            df = api_connector.get_market_data(
                symbol=symbol,
                limit=limit,
                save_to_db=True,
                db_manager=self
            )
            
            if df.empty:
                self.logger.warning(f"⚠️ 시장 데이터를 가져올 수 없습니다: {symbol}")
                return pd.DataFrame()
            
            # 시간 필터링
            if start_time:
                df = df[df.index >= start_time]
            if end_time:
                df = df[df.index <= end_time]
            
            # limit 적용
            if len(df) > limit:
                df = df.tail(limit)
            
            self.logger.info(f"✅ 시장 데이터 조회 완료: {symbol} ({len(df)}개)")
            return df
            
        except Exception as e:
            self.logger.error(f"❌ get_market_data 오류: {e}")
            return pd.DataFrame()