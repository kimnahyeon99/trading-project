"""
    MYSQL 데이터베이스 기반 대시보드 데이터 관리 모듈
    database.py에서 제공하는 DatabaseManager 인스턴스를 사용. 
    이미 연결된 데이터베이스 연결을 이용하여 데이터 로직을 처리
"""
import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import time

from src.utils.logger import Logger
from src.utils.database import DatabaseManager


class DBDataManager:
    """
    데이터베이스를 사용하는 대시보드 데이터 관리 클래스
    """
    
    def __init__(
        self, 
        db_manager: DatabaseManager,
        data_dir: Optional[str] = None,
        logger: Optional[Logger] = None
    ):
        """
        DBDataManager 클래스 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 객체
            data_dir: 데이터 디렉토리 경로 (필요시 사용, 옵션)
            logger: 로깅을 위한 Logger 인스턴스 (옵션)
        """
        self.db_manager = db_manager
        self.data_dir = data_dir
        self.logger = logger
        
        if self.logger:
            self.logger.info("DBDataManager 초기화 완료")
    
    # 트레이딩 통계
    def get_trading_stats(self, model_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        """
        트레이딩 통계 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            트레이딩 통계 정보
        """
        try:
            # 최근 트레이딩 통계 조회
            stats_query = """
            SELECT * FROM trading_stats 
            ORDER BY timestamp DESC 
            LIMIT 100
            """
            
            stats_rows = self.db_manager.execute_query(stats_query)
            
            if not stats_rows:
                if self.logger:
                    self.logger.warning("트레이딩 통계 데이터가 없습니다.")
                return {}
            
            # 최근 거래 내역 조회
            trades_query = """
            SELECT * FROM trades 
            ORDER BY timestamp DESC 
            """
            
            trades = self.db_manager.execute_query(trades_query)
            
            
            # 통계 데이터 구성
            # portfolio_values = 자산 변화만 뽑은 리스트 
            # timestamps = 타임스탬프만 뽑은 리스트
            portfolio_values = [row['portfolio_value'] for row in reversed(stats_rows)]    
            timestamps = [row['timestamp'].isoformat() for row in reversed(stats_rows)]
            
            # 일일 수익률 계산
            returns = []
            for i in range(1, len(portfolio_values)):
                daily_return = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
                returns.append(daily_return)
            returns.insert(0, 0)  # 첫 날의 수익률은 0
            
            # 낙폭 계산
            drawdowns = []
            peak = portfolio_values[0] if portfolio_values else 0
            
            for value in portfolio_values:
                if value > peak:
                    peak = value
                    drawdowns.append(0)
                else:
                    drawdown = (peak - value) / peak
                    drawdowns.append(drawdown)
                    
            
            
            # 현재 포지션 조회
            positions_query = """
            SELECT * FROM positions
            """
            
            positions = self.db_manager.execute_query(positions_query)
            
            # 포지션 데이터 가공
            positions_dict = {}
            for position in positions:
                try:
                    current_price = float(position['current_price'])
                except (TypeError, ValueError):
                    current_price = 0

                positions_dict[position['symbol']] = {
                    'quantity': float(position['quantity']),
                    'avg_entry_price': float(position['avg_entry_price']),
                    'current_price': current_price,
                    'unrealized_pnl': float(position['unrealized_pnl']),
                    'timestamp': position['timestamp'].isoformat()
                }

            
            # 오늘의 손익 합산
            today_str = datetime.now().strftime('%Y-%m-%d')
            today_pnl_query = """
            SELECT SUM(daily_pnl) AS today_pnl
            FROM trading_stats
            WHERE DATE(timestamp) = %s
            """
            today_pnl_result = self.db_manager.execute_query(today_pnl_query, (today_str,))
            today_pnl = today_pnl_result[0]['today_pnl'] if today_pnl_result and today_pnl_result[0]['today_pnl'] else 0
            
            
            # 기간 계산용 데이터
            query = """
            SELECT MIN(timestamp) AS start_date, MAX(timestamp) AS end_date
            FROM trading_stats
            """
            start_end_result = self.db_manager.execute_query(query)
            start_date = start_end_result[0]['start_date'].isoformat() if start_end_result and start_end_result[0]['start_date'] else None
            end_date = start_end_result[0]['end_date'].isoformat() if start_end_result and start_end_result[0]['end_date'] else None
            
            # 최신 트레이딩 통계 데이터
            latest_stats = stats_rows[0]
            
            # 결과 구성
            result = {
                'portfolio_values': portfolio_values,
                'timestamps': timestamps,
                'returns': returns,
                'drawdowns': drawdowns,
                'trades': [
                    {
                        'timestamp': trade['timestamp'].isoformat(),
                        'symbol': trade['symbol'],
                        'side': trade['side'],
                        'quantity': float(trade['quantity']),
                        'price': float(trade['price']),
                        'fee': float(trade['fee']),
                        'pnl': float(trade['pnl']) if trade['pnl'] else 0
                    } for trade in trades
                ],
                'positions': positions_dict,
                'trading_stats': {
                    # 현재 총 자산가치
                    'portfolio_value': float(latest_stats['portfolio_value']),
                    # 현재 보유 현금  
                    'cash_balance': float(latest_stats['cash_balance']),
                    # 주식의 현재 시가 총합(주식자산 평가 금액)
                    'equity_value': float(latest_stats['equity_value']),
                    # 오늘 손익
                    'daily_pnl': float(latest_stats['daily_pnl']or 0),
                    # 누적 손익
                    'total_pnl': float(latest_stats['total_pnl']),
                    'timestamp': latest_stats['timestamp'].isoformat(),
                    'start_date': start_date, 
                    'end_date': end_date        
                    
                }
            }
            
            if self.logger:
                self.logger.info(f"트레이딩 통계 조회 완료: {len(stats_rows)}개 레코드")
            
            return result
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 조회 중 오류 발생: {e}")
            return {}
    
    
    # 모델별 수익률 데이터 가져오기
    def get_performance_chart_data_from_trading_stats(self) -> Dict[str, Tuple[float, float]]:
        """
        trades 테이블 기반: 모델별 누적 손익 비교용 데이터 생성
        Returns:
            dict: {model_id: (0, total_pnl)} 형식 (initial=0은 비교용, final은 누적 손익)
        """
        try:
            query = """
            SELECT model_id, SUM(pnl) AS total_pnl
            FROM trades
            GROUP BY model_id
            """
            results = self.db_manager.execute_query(query)

            data = {}
            for row in results:
                model_id = row['model_id']
                total_pnl = float(row['total_pnl']) if row['total_pnl'] is not None else 0
                data[model_id] = (0, total_pnl)  # 초기값 0, 최종값 = 누적 손익

            return data
        except Exception as e:
            if self.logger:
                self.logger.error(f"[수익 비교 실패] trades 기반 모델 누적 손익 계산 실패: {e}")
            return {}

       


    
    # 매수 vs 매도 비율 데이터 가져오기
    def get_trade_side_distribution(self, model_id: Optional[str] = None):
        if model_id:
            query = """
                SELECT UPPER(TRIM(side)) AS side, COUNT(*) AS count
                FROM trades
                WHERE model_id = %s
                GROUP BY UPPER(TRIM(side))
            """
            return self.db_manager.execute_query(query, (model_id,))
        else:
            query = """
                SELECT UPPER(TRIM(side)) AS side, COUNT(*) AS count
                FROM trades
                GROUP BY UPPER(TRIM(side))
            """
            return self.db_manager.execute_query(query)



    
    
    def get_backtest_results(self, model_id: Optional[str] = None, refresh: bool = False) -> Dict[str, Any]:
        """
        백테스트 결과 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            백테스트 결과 정보
        """
        try:
            if model_id:
                # 특정 모델 결과만 조회
                query = """
                SELECT * FROM backtest_results
                WHERE model_id = %s
                ORDER BY backtest_date DESC
                """
                params = (model_id,)
            else:
                # 모든 모델 결과 조회
                query = """
                SELECT * FROM backtest_results
                ORDER BY model_id, backtest_date DESC
                """
                params = None
            
            results = self.db_manager.execute_query(query, params)
            
            if not results:
                if self.logger:
                    self.logger.warning(f"백테스트 결과가 없습니다: {model_id if model_id else '모든 모델'}")
                return {}
            
            # 모델별로 결과 정리
            backtest_results = {}
            
            for result in results:
                model_id_val = result['model_id']
                
                
                # 첫 번째 결과만 사용 (가장 최근 백테스트)
                if model_id_val not in backtest_results:
                    # JSON 파라미터 파싱
                    parameters = json.loads(result['parameters']) if result['parameters'] else {}
                    
                    backtest_results[model_id_val] = {
                        'model_id': model_id_val,
                        'backtest_date': result['backtest_date'].isoformat(),
                        'start_date': result['start_date'].isoformat(),
                        'end_date': result['end_date'].isoformat(),
                        'initial_balance': float(result['initial_balance']),
                        'final_balance': float(result['final_balance']),
                        'parameters': parameters,
                        'metrics': {
                            'total_return': float(result['total_return']),
                            'annualized_return': float(result['annualized_return']) if result['annualized_return'] else 0,
                            'sharpe_ratio': float(result['sharpe_ratio']) if result['sharpe_ratio'] else 0,
                            'max_drawdown': float(result['max_drawdown']) if result['max_drawdown'] else 0,
                            'win_rate': float(result['win_rate']) if result['win_rate'] else 0,
                            'profit_factor': float(result['profit_factor']) if result['profit_factor'] else 0,
                            'total_trades': result['total_trades'] or 0
                        },
                    }
            
            if self.logger:
                self.logger.info(f"백테스트 결과 조회 완료: {len(backtest_results)}개 모델")
            
            return backtest_results
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"백테스트 결과 조회 중 오류 발생: {e}")
            return {}
        
    # 거래 내역 조회
    def get_trades(self, model_id: str) -> List[Dict[str, Any]]:
        query = """
            SELECT * FROM trades
            WHERE model_id = %s
            ORDER BY timestamp ASC
        """
        return self.db_manager.execute_query(query, (model_id,))

    
    
    # 모델 정보
    def get_model_info(self, refresh: bool = False) -> Dict[str, Any]:
        """
        학습된 모델 정보 조회
        
        Args:
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            모델 정보 목록
        """
        try:
            query = """
            SELECT * FROM models
            ORDER BY created_time DESC
            """
            
            models = self.db_manager.execute_query(query)
            
            if not models:
                if self.logger:
                    self.logger.warning("등록된 모델이 없습니다.")
                return {}
            
            # 모델 정보 정리
            model_info = {}
            
            for model in models:
                model_id = model['model_id']
                
                model_info[model_id] = {
                    'model_id': model_id,
                    'file_path': model['file_path'],
                    'created_time': model['created_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'modified_time': model['modified_time'].strftime('%Y-%m-%d %H:%M:%S'),
                    'file_size': model['file_size'],
                    'file_size_mb': round(model['file_size'] / (1024 * 1024), 2),
                    'description': model['description'],
                    'is_active': bool(model['is_active'])
                }
            
            if self.logger:
                self.logger.info(f"모델 정보 조회 완료: {len(model_info)}개")
            
            return model_info
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"모델 정보 조회 중 오류 발생: {e}")
            return {}
    
    def get_market_data(self, symbol: str, interval: str = '1d', limit: int = 100, refresh: bool = False) -> pd.DataFrame:
        """
        시장 데이터 조회
        
        Args:
            symbol: 심볼/티커
            interval: 시간 간격 (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            limit: 가져올 데이터 개수
            refresh: 캐시 갱신 여부 (데이터베이스 방식에서는 무시됨)
            
        Returns:
            시장 데이터 DataFrame
        """
        try:
            query = """
            SELECT * FROM market_data
            WHERE symbol = %s
            ORDER BY timestamp DESC
            LIMIT %s
            """
            
            params = (symbol, limit)
            
            market_data = self.db_manager.execute_query(query, params)
            
            if not market_data:
                if self.logger:
                    self.logger.warning(f"{symbol} 시장 데이터가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame으로 변환
            df = pd.DataFrame(market_data)
            
            # 날짜 열 이름 변경 (timestamp -> date)
            if 'timestamp' in df.columns:
                df['date'] = df['timestamp']
            
            # 최신 순으로 정렬되어 있으므로 역순으로 변경
            df = df.iloc[::-1].reset_index(drop=True)
            
            if self.logger:
                self.logger.info(f"{symbol} 시장 데이터 조회 완료: {len(df)} 행")
            
            return df
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"{symbol} 시장 데이터 조회 중 오류 발생: {e}")
            return pd.DataFrame()
    
    def get_performance_metrics(self, model_id: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 성능 지표 조회
        
        Args:
            model_id: 모델 ID (옵션, 특정 모델 결과만 조회)
            
        Returns:
            성능 지표 정보
        """
        # 백테스트 결과에서 성능 지표 추출
        backtest_results = self.get_backtest_results(model_id)
        
        metrics = {}
        for model_id, result in backtest_results.items():
            if 'metrics' in result:
                metrics[model_id] = result['metrics']
            else:
                # 기본 지표 구조 생성
                metrics[model_id] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'win_rate': result.get('win_rate', 0),
                    'profit_factor': result.get('profit_factor', 0)
                }
        
        return metrics

    def sync_file_to_db(self,json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)

        query = """
            INSERT INTO models (model_id, description, created_time, modified_time, file_size, file_path, file_name, is_active)
            VALUES (%s, %s, NOW(), NOW(), %s, %s, %s, %s)
        """
        
        file_path = f"uploads/{data['model_id']}.pt"  
        file_size = os.path.getsize(file_path)
        file_name = data['file_name']
        is_active = int(data.get("is_active", 1))

        values = (
            data['model_id'],
            data.get('description', ''),
            file_size,
            file_path,
            file_name,
            is_active
        )

        # DB 연결 후 INSERT
        cursor = self.db.cursor()
        cursor.execute(query, values)
        self.db.commit()
        return True

    
    def _sync_trading_stats(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        트레이딩 통계 동기화
        """
        try:
            # 파일명에서 타임스탬프 추출
            filename = os.path.basename(file_path)
            timestamp_str = filename.split('_')[-1].split('.')[0]
            timestamp = datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
            
            # 트레이딩 통계 삽입
            if 'portfolio_value' in data and 'cash_balance' in data:
                query = """
                INSERT INTO trading_stats 
                (timestamp, portfolio_value, cash_balance, equity_value, daily_pnl, total_pnl)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                portfolio_value = VALUES(portfolio_value),
                cash_balance = VALUES(cash_balance),
                equity_value = VALUES(equity_value),
                daily_pnl = VALUES(daily_pnl),
                total_pnl = VALUES(total_pnl)
                """
                
                params = (
                    timestamp,
                    data['portfolio_value'],
                    data['cash_balance'],
                    data.get('equity_value', data['portfolio_value'] - data['cash_balance']),
                    data.get('daily_pnl', 0),
                    data.get('total_pnl', 0)
                )
                
                self.db_manager.execute_query(query, params, fetch=False)
            
            # 포지션 정보 삽입
            if 'positions' in data:
                for symbol, position in data['positions'].items():
                    query = """
                    INSERT INTO positions
                    (symbol, quantity, avg_entry_price, current_price, unrealized_pnl, timestamp)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE
                    quantity = VALUES(quantity),
                    avg_entry_price = VALUES(avg_entry_price),
                    current_price = VALUES(current_price),
                    unrealized_pnl = VALUES(unrealized_pnl),
                    timestamp = VALUES(timestamp)
                    """
                    
                    params = (
                        symbol,
                        position['quantity'],
                        position['avg_entry_price'],
                        position.get('current_price', 0),
                        position.get('unrealized_pnl', 0),
                        timestamp
                    )
                    
                    self.db_manager.execute_query(query, params, fetch=False)
            
            # 거래 내역 삽입
            if 'trades' in data:
                for trade in data['trades']:
                    query = """
                    INSERT INTO trades
                    (timestamp, symbol, side, quantity, price, fee, pnl, model_id)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """
                    
                    trade_timestamp = datetime.fromisoformat(trade['timestamp']) if 'timestamp' in trade else timestamp
                    
                    params = (
                        trade_timestamp,
                        trade['symbol'],
                        trade['side'],
                        trade['quantity'],
                        trade['price'],
                        trade.get('fee', 0),
                        trade.get('pnl', None),
                        trade.get('model_id', None)
                    )
                    
                    self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"트레이딩 통계 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"트레이딩 통계 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_backtest_results(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        백테스트 결과 동기화
        """
        try:
            # 파일명에서 모델 ID 추출
            filename = os.path.basename(file_path)
            model_id = filename.split('_')[1] if '_' in filename else filename.split('.')[0]
            
            # 백테스트 결과 삽입
            backtest_date = datetime.fromisoformat(data.get('backtest_date', datetime.now().isoformat()))
            start_date = datetime.fromisoformat(data.get('start_date', '')).date() if 'start_date' in data else None
            end_date = datetime.fromisoformat(data.get('end_date', '')).date() if 'end_date' in data else None
            
            # 지표 추출
            metrics = data.get('metrics', {})
            if not metrics and 'performance' in data:
                metrics = data['performance']
            
            query = """
            INSERT INTO backtest_results
            (model_id, backtest_date, start_date, end_date, initial_balance, final_balance,
             total_return, annualized_return, sharpe_ratio, max_drawdown, win_rate,
             profit_factor, total_trades, parameters)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            start_date = VALUES(start_date),
            end_date = VALUES(end_date),
            initial_balance = VALUES(initial_balance),
            final_balance = VALUES(final_balance),
            total_return = VALUES(total_return),
            annualized_return = VALUES(annualized_return),
            sharpe_ratio = VALUES(sharpe_ratio),
            max_drawdown = VALUES(max_drawdown),
            win_rate = VALUES(win_rate),
            profit_factor = VALUES(profit_factor),
            total_trades = VALUES(total_trades),
            parameters = VALUES(parameters)
            """
            
            params = (
                model_id,
                backtest_date,
                start_date,
                end_date,
                data.get('initial_balance', 0),
                data.get('final_balance', 0),
                metrics.get('total_return', 0),
                metrics.get('annualized_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0),
                metrics.get('total_trades', 0),
                json.dumps(data.get('parameters', {}))
            )
            
            self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"백테스트 결과 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"백테스트 결과 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_model_info(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        모델 정보 동기화
        """
        try:
            model_id = data.get('model_id', os.path.basename(file_path).split('.')[0])
            
            # 파일 정보 추출
            file_stats = os.stat(file_path)
            created_time = datetime.fromtimestamp(file_stats.st_ctime)
            modified_time = datetime.fromtimestamp(file_stats.st_mtime)
            file_size = file_stats.st_size
            
            query = """
            INSERT INTO models
            (model_id, file_path, created_time, modified_time, file_size, description, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            file_path = VALUES(file_path),
            modified_time = VALUES(modified_time),
            file_size = VALUES(file_size),
            description = VALUES(description)
            """
            
            params = (
                model_id,
                file_path,
                created_time,
                modified_time,
                file_size,
                data.get('description', ''),
                data.get('is_active', False)
            )
            
            self.db_manager.execute_query(query, params, fetch=False)
            
            if self.logger:
                self.logger.info(f"모델 정보 동기화 완료: {file_path}")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"모델 정보 동기화 중 오류 발생: {e}")
            return False
    
    def _sync_market_data(self, data: Union[Dict[str, Any], pd.DataFrame], file_path: str) -> bool:
        """
        시장 데이터 동기화
        """
        try:
            # 파일명에서 심볼 추출
            filename = os.path.basename(file_path)
            symbol = filename.split('_')[0]
            
            # DataFrame으로 변환
            if isinstance(data, dict):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # 데이터 삽입
            if len(df) > 0:
                # 일괄 삽입을 위한 파라미터 리스트 생성
                params_list = []
                
                for _, row in df.iterrows():
                    # 날짜 처리
                    if 'date' in row:
                        timestamp = pd.to_datetime(row['date'])
                    elif 'timestamp' in row:
                        timestamp = pd.to_datetime(row['timestamp'])
                    else:
                        continue
                    
                    params = (
                        symbol,
                        timestamp,
                        row['open'],
                        row['high'],
                        row['low'],
                        row['close'],
                        row.get('volume', 0)
                    )
                    
                    params_list.append(params)
                
                if params_list:
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
                    
                    self.db_manager.execute_many(query, params_list)
            
            if self.logger:
                self.logger.info(f"시장 데이터 동기화 완료: {file_path}, {len(df)} 행")
            
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"시장 데이터 동기화 중 오류 발생: {e}")
            return False 