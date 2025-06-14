"""
대시보드 웹 애플리케이션 모듈
"""
import os
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from flask import Flask
import tempfile
from flask import Flask, render_template, jsonify, request, Response, redirect
import plotly
from pymongo import MongoClient

from src.dashboard.data_manager_file import FileDataManager
from src.dashboard.data_manager_db import DBDataManager
from src.dashboard.visualization import Visualizer
from src.utils.logger import Logger
from src.utils.database import DatabaseManager
from werkzeug.utils import secure_filename
from src.alpaca.alpaca_client import AlpacaClient

# <라우팅 설정>
# 라우팅 : "어떤 url이 들어왔을 때 어떤 함수를 실행할지 정하는 것"

# 몽고DB 연동 코드 
try:
    mongo_client = MongoClient('mongodb://192.168.40.192/', serverSelectionTimeoutMS=5000)
    # 연결 테스트
    mongo_client.server_info()
    
    # polygon DB 접속
    polygon_db = mongo_client['polygon']                        #DB 이름
    polygon_bulletin = polygon_db['bulletin']                    #컬렉션 이름

    # yahoo DB 접속
    yahoo_db = mongo_client['yahoo']                           #DB 이름
    yahoo_news = yahoo_db['news']                               #컬렉션 이름
    
except Exception as e:
    print(f"MongoDB 연결 실패: {str(e)}")
    # 기본값 설정
    polygon_bulletin = None
    yahoo_news = None

class DashboardApp:
    """
    SAC 트레이딩 시스템 웹 대시보드 애플리케이션
    """
    
    def __init__(
        self,
        alpaca,
        data_manager,
        static_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        host: str = '0.0.0.0',
        port: int = 5000,
        debug: bool = False
    ):
        """
        DashboardApp 클래스 초기화
        
        Args:
            data_manager: 데이터 관리자 객체 (DataManager 또는 DBDataManager)
            static_dir: 정적 파일 디렉토리 경로 (옵션)
            templates_dir: 템플릿 디렉토리 경로 (옵션)
            host: 호스트 주소
            port: 포트 번호
            debug: 디버그 모드 여부
        """
        # alpaca API
        self.alpaca = alpaca
        # 데이터 관리자 및 로거
        self.data_manager = data_manager
        self.logger = data_manager.logger
        
        # 시각화 도구 초기화
        self.visualizer = Visualizer()
        
        # Flask 앱 초기화
        template_folder = templates_dir if templates_dir else os.path.join(os.path.dirname(__file__), 'templates')
        static_folder = static_dir if static_dir else os.path.join(os.path.dirname(__file__), 'static')
        
        self.app = Flask(__name__, template_folder=template_folder, static_folder=static_folder)
        
        # 설정
        self.host = host
        self.port = port
        self.debug = debug
        
        # 라우트 설정
        self._setup_routes()
        
        # 서버 상태
        self.is_running = False
        
        if self.logger:
            self.logger.info(f"대시보드 앱 초기화 완료: http://{host}:{port}")
    
    # 라우트 설정 함수
    def _setup_routes(self):
        """
        Flask 라우트 설정
        """
        # 메인 페이지
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        # 실시간 트레이딩 페이지
        @self.app.route('/live-trading')
        def live_trading():
            return render_template('live_trading.html')
        
        # 실시간 차트 페이지
        @self.app.route('/live-chart')
        def live_chart():
            return render_template('live_chart.html')
        
        # 백테스트 결과 페이지
        @self.app.route('/backtest')
        def backtest():
            return render_template('backtest.html')
        
        # 모델 관리 페이지
        @self.app.route('/models')
        def models():
            return render_template('models.html')
        
        
        # 기사 페이지
        @self.app.route('/news')
        def news():
            ticker = request.args.get('name')  
            source = request.args.get('source', 'polygon')  # 기본값 polygon

            query_filter = {}
            if ticker:
                query_filter = {"name": ticker}

            polygon_data = []
            yahoo_data = []

            if source == 'polygon':
                polygon_data = list(polygon_bulletin.find(query_filter, {
                    '_id': 0,
                    'name': 1,
                    'title': 1,
                    'summary': 1,
                    'sentiment': 1,
                    'date': 1,
                    'url': 1
                }))
            elif source == 'yahoo':
                yahoo_data = list(yahoo_news.find(query_filter, {
                    '_id': 0,
                    'name': 1,
                    'title': 1,
                    'summary': 1,
                    'sentiment': 1,
                    'date': 1,
                    'url': 1
                }))

            return render_template('news.html', 
                polygon=polygon_data, 
                yahoo=yahoo_data,
                ticker=ticker,
                source=source
            )

        # 개인 계정 조회
        @self.app.route('/api/account', methods=['GET'])
        def get_account():
            # 여기서는 임시 데이터 예제
            # 실제 구현에선 broker API에서 데이터를 가져와야 함
            account = self.alpaca.get_account()  # 객체 반환
            account_data = {
                "portfolio_value": float(account['portfolio_value']), 
                "cash": float(account['cash']),
                "positions": [
                    {"symbol": "AAPL", "quantity": 10, "current_price": 180.25},
                    {"symbol": "TSLA", "quantity": 5, "current_price": 850.50}
                ]
            }
            
            self.logger.info(f"✅ Alpaca 포트폴리오가치: ${account['portfolio_value']}")
            self.logger.info(f"✅ Alpaca 현금 잔고: ${account['cash']}")
            
            return jsonify(account_data)
    
        
        # API 라우트 설정
        
        # 트레이딩 통계 API
        @self.app.route('/api/trading-stats')
        def get_trading_stats():
            model_id = request.args.get('model_id')
            refresh = request.args.get('refresh', 'false').lower() == 'true'
            self.logger.info(f"📊 트레이딩 통계 요청됨 | model_id: {model_id}, refresh: {refresh}")
            stats = self.data_manager.get_trading_stats(model_id=model_id, refresh=refresh)
            return jsonify(stats)
        
        # 모델 정보 API
        @self.app.route('/api/models')
        def get_models():
            refresh = request.args.get('refresh', 'false').lower() == 'true'
            models = self.data_manager.get_model_info(refresh=refresh)
            return jsonify(models)
        
        # 초기자산 vs 최종자산 차트
        @self.app.route('/api/charts/initial-vs-final')
        def get_initial_vs_final_chart():
            results = self.data_manager.get_backtest_results()

            model_ids = []
            initial_balances = []
            final_balances = []

            for model_id, data in results.items():
                model_ids.append(model_id)
                initial_balances.append(data.get("initial_balance", 0))
                final_balances.append(data.get("final_balance", 0))

            fig = self.visualizer.create_initial_vs_final_balance_chart(
                model_ids=model_ids,
                initial_balances=initial_balances,
                final_balances=final_balances,
                title="초기 자산 vs 최종 자산 비교"
            )

            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        # 위험 수익률 차트
        @self.app.route("/api/charts/risk-return")
        def get_risk_return_chart():
            results = self.data_manager.get_backtest_results()
            fig = self.visualizer.create_risk_return_chart(results)
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
            
      # 기사 데이터 조회 API
        @self.app.route('/api/news')
        def get_news():
            ticker = request.args.get('name')
            try:
                query = {}
                if ticker:
                    query = {"name": ticker}

                polygon_bulletin_data = list(polygon_bulletin.find(query, {
                    '_id': 0,
                    'name': 1,
                    'title': 1,
                    'summary': 1,
                    'sentiment': 1,
                    'published_date_kst': 1,
                    'url': 1
                }))

                yahoo_news_data = list(yahoo_news.find(query, {
                    '_id': 0,
                    'name': 1,
                    'title': 1,
                    'summary': 1,
                    'sentiment': 1,
                    'date': 1,
                    'url': 1
                }))
                
                return jsonify({
                    'polygon': polygon_bulletin_data,
                    'yahoo': yahoo_news_data
                })
            except Exception as e:
                return jsonify({'error': str(e)})
                

                
                
                
        # 백테스트 결과 API
        @self.app.route('/api/backtest-results')
        def get_backtest_results():
            model_id = request.args.get('model_id')
            refresh = request.args.get('refresh', 'false').lower() == 'true'
            results = self.data_manager.get_backtest_results(model_id=model_id, refresh=refresh)
            # trades = 거래내역 데이터 조회
            trades = self.data_manager.get_trades(model_id=model_id)
            if model_id in results:
                results[model_id]['trades'] = trades
            return jsonify(results)
        
    
        
        # 성능 지표 API
        @self.app.route('/api/performance-metrics')
        def get_performance_metrics():
            model_id = request.args.get('model_id')
            metrics = self.data_manager.get_performance_metrics(model_id=model_id)
            return jsonify(metrics)
        
        # 시장 데이터 API
        @self.app.route('/api/market-data')
        def get_market_data():
            symbol = request.args.get('symbol', 'AAPL')
            interval = request.args.get('interval', '1d')
            limit = int(request.args.get('limit', 100))
            refresh = request.args.get('refresh', 'false').lower() == 'true'
            
            df = self.data_manager.get_market_data(
                symbol=symbol,
                interval=interval,
                limit=limit,
                refresh=refresh
            )
            
            return jsonify(df.to_dict(orient='records'))
        
        # 차트 API 엔드포인트
        
        # 포트폴리오 가치 차트 API
        @self.app.route('/api/charts/portfolio')
        def get_portfolio_chart():
            model_id = request.args.get('model_id')
            
            # 모델 ID가 제공된 경우 백테스트 결과에서 데이터 가져오기
            if model_id:
                results = self.data_manager.get_backtest_results(model_id=model_id)
                
                if not results or model_id not in results:
                    return jsonify({'error': f'모델 ID에 대한 결과를 찾을 수 없습니다: {model_id}'})
                
                model_results = results[model_id]
                
                if 'portfolio_values' not in model_results or 'timestamps' not in model_results:
                    return jsonify({'error': '포트폴리오 데이터가 없습니다'})
                
                fig = self.visualizer.create_portfolio_value_chart(
                    portfolio_values=model_results.get('portfolio_values', []),
                    timestamps=model_results.get('timestamps', []),
                    benchmark_values=model_results.get('benchmark_values', None),
                    title=f"{model_id} 포트폴리오 가치 변화"
                )
            else:
                # 모델 ID가 없는 경우 일반 트레이딩 통계 사용
                stats = self.data_manager.get_trading_stats()
                
                if not stats or 'portfolio_values' not in stats or 'timestamps' not in stats:
                    return jsonify({'error': '포트폴리오 데이터가 없습니다'})
                
                fig = self.visualizer.create_portfolio_value_chart(
                    portfolio_values=stats.get('portfolio_values', []),
                    timestamps=stats.get('timestamps', []),
                    title="포트폴리오 가치 변화"
                )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        # 수익률 차트 API
        @self.app.route('/api/charts/returns')
        def get_returns_chart():
            model_id = request.args.get('model_id')
            
            # 모델 ID가 제공된 경우 백테스트 결과에서 데이터 가져오기
            if model_id:
                results = self.data_manager.get_backtest_results(model_id=model_id)
                
                if not results or model_id not in results:
                    return jsonify({'error': f'모델 ID에 대한 결과를 찾을 수 없습니다: {model_id}'})
                
                model_results = results[model_id]
                
                if 'returns' not in model_results or 'timestamps' not in model_results:
                    return jsonify({'error': '수익률 데이터가 없습니다'})
                
                fig = self.visualizer.create_returns_chart(
                    returns=model_results.get('returns', []),
                    timestamps=model_results.get('timestamps', []),
                    benchmark_returns=model_results.get('benchmark_returns', None),
                    title=f"{model_id} 수익률 변화"
                )
            else:
                # 모델 ID가 없는 경우 일반 트레이딩 통계 사용
                stats = self.data_manager.get_trading_stats()
                
                if not stats or 'returns' not in stats or 'timestamps' not in stats:
                    return jsonify({'error': '수익률 데이터가 없습니다'})
                
                fig = self.visualizer.create_returns_chart(
                    returns=stats.get('returns', []),
                    timestamps=stats.get('timestamps', []),
                    title="수익률 변화"
                )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        # 거래 분포 차트 API
        @self.app.route("/api/charts/trade-distribution")
        def get_trade_distribution_chart():
            results = self.data_manager.get_backtest_results()
            fig = self.visualizer.create_trade_count_chart(results)
            return jsonify(json.loads(plotly.io.to_json(fig)))

        
        
        # 낙폭 차트 API
        @self.app.route('/api/charts/drawdown')
        def get_drawdown_chart():
            model_id = request.args.get('model_id')
            
            # 모델 ID가 제공된 경우 백테스트 결과에서 데이터 가져오기
            if model_id:
                results = self.data_manager.get_backtest_results(model_id=model_id)
                
                if not results or model_id not in results:
                    return jsonify({'error': f'모델 ID에 대한 결과를 찾을 수 없습니다: {model_id}'})
                
                model_results = results[model_id]
                
                if 'drawdowns' not in model_results or 'timestamps' not in model_results:
                    return jsonify({'error': '낙폭 데이터가 없습니다'})
                
                fig = self.visualizer.create_drawdown_chart(
                    drawdowns=model_results.get('drawdowns', []),
                    timestamps=model_results.get('timestamps', []),
                    title=f"{model_id} 낙폭(Drawdown) 변화"
                )
            else:
                # 모델 ID가 없는 경우 일반 트레이딩 통계 사용
                stats = self.data_manager.get_trading_stats()
                
                if not stats or 'drawdowns' not in stats or 'timestamps' not in stats:
                    return jsonify({'error': '낙폭 데이터가 없습니다'})
                
                fig = self.visualizer.create_drawdown_chart(
                    drawdowns=stats.get('drawdowns', []),
                    timestamps=stats.get('timestamps', []),
                    title="낙폭(Drawdown) 변화"
                )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        
        # 매수 vs 매도 비율 차트 API
        @self.app.route('/api/charts/trade-buy-sell')
        def get_trade_buy_sell_chart():
            model_id = request.args.get('model_id')
            
            # 집계된 BUY/SELL 수만 가져오기
            side_counts = self.data_manager.get_trade_side_distribution(model_id=model_id)

            fig = self.visualizer.create_trade_buy_sell_chart_from_aggregate(
                side_counts,
                title="매수 vs 매도 비율" if model_id else "전체 매수 vs 매도 비율"
            )

            return jsonify(json.loads(plotly.io.to_json(fig)))



        # 가격 차트 API
        @self.app.route('/api/charts/price')
        def get_price_chart():
            symbol = request.args.get('symbol', 'AAPL')
            buy_signals_str = request.args.get('buy_signals', '[]')
            sell_signals_str = request.args.get('sell_signals', '[]')
            
            try:
                buy_signals = json.loads(buy_signals_str)
                sell_signals = json.loads(sell_signals_str)
            except json.JSONDecodeError:
                buy_signals = []
                sell_signals = []
            
            # 시장 데이터 조회
            price_data = self.data_manager.get_market_data(
                symbol=symbol,
                interval='1d',
                limit=100
            )
            
            if price_data.empty:
                return jsonify({'error': f'No price data available for {symbol}'})
            
            # 차트 생성
            fig = self.visualizer.create_trade_chart(
                prices=price_data['close'].tolist(),
                timestamps=price_data['date'].astype(str).tolist(),
                buy_signals=buy_signals,
                sell_signals=sell_signals,
                title=f"{symbol} 가격 및 거래 신호"
            )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        # 모델 성능 비교 차트 API
        @self.app.route('/api/charts/model-comparison')
        def get_model_comparison_chart():
            metric_name = request.args.get('metric', 'total_return')
            title = request.args.get('title', f"{metric_name} 모델 비교")
            
            metrics = self.data_manager.get_performance_metrics()
            
            if not metrics:
                return jsonify({'error': 'No performance metrics available'})
            
            fig = self.visualizer.create_performance_comparison_chart(
                metrics=metrics,
                metric_name=metric_name,
                title=title
            )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        
        # 모델 성능 차트 API
        @self.app.route('/api/charts/model-performance-alt')
        def get_model_performance_alt_chart():
            try:
                data = self.data_manager.get_performance_chart_data_from_trading_stats()
                fig = self.visualizer.create_model_performance_chart(data)
                return jsonify(json.loads(plotly.io.to_json(fig)))
            except Exception as e:
                return jsonify({'error': str(e)})
        
        
        # 레이더 차트 API
        @self.app.route('/api/charts/radar')
        def get_radar_chart():
            model_ids = request.args.get('models', '').split(',')
            if not model_ids or model_ids[0] == '':
                model_ids = None
                
            metrics = self.data_manager.get_performance_metrics()
            
            if not metrics:
                return jsonify({'error': 'No performance metrics available'})
            
            metric_names = [
                'total_return',
                'sharpe_ratio',
                'max_drawdown',
                'win_rate',
                'profit_factor'
            ]
            
            fig = self.visualizer.create_model_comparison_radar_chart(
                metrics=metrics,
                metric_names=metric_names,
                model_ids=model_ids,
                title="모델 성능 비교"
            )
            
            return jsonify(json.loads(plotly.io.to_json(fig)))
        
        # 모델 파일로 직접 업로드

        @self.app.route('/models', methods=['POST'])
        def upload_model():
            model_id = request.form['model_id']
            description = request.form.get('description', '')
            file = request.files['model_file']

            if file.filename == '':
                return '파일이 선택되지 않았습니다.', 400
            
            upload_dir = 'uploads'
            os.makedirs(upload_dir, exist_ok=True)
            
            filename = secure_filename(file.filename)
            model_file_path = os.path.join('uploads', filename)

            # uploads 폴더가 없으면 자동으로 생성
            os.makedirs(os.path.dirname(model_file_path), exist_ok=True)

            file.save(model_file_path)

            # JSON 메타파일도 같이 생성
            model_data = {
                'model_id': model_id,
                'description': description,
                'is_active': True,
                'file_path': model_file_path  
            }
            temp_json_path = os.path.join(upload_dir, f'{model_id}_meta.json')
            with open(temp_json_path, 'w') as f:
                json.dump(model_data, f)

            #모델 정보 등록
            success = self.data_manager.sync_file_to_db('models', temp_json_path)
            if success:
                return redirect('/models')
            else:
                return '모델 업로드 실패', 500

            
            
    def run(self) -> None:
        if self.is_running:
            self.logger.warning("🚫 서버가 이미 실행 중입니다.")
            return
            
        self.is_running = True
        self.logger.info(f"✅ 대시보드 웹 서버 시작됨: http://{self.host}:{self.port}")
        
        try:
            self.app.run(host=self.host, port=self.port, debug=self.debug)
        except Exception as e:
            self.logger.error(f"❗ 서버 실행 중 오류 발생: {e}")
            self.is_running = False
    
    def stop(self) -> None:
        """
        대시보드 웹 서버 중지
        """
        if not self.is_running:
            self.logger.warning("서버가 실행 중이 아닙니다.")
            return
            
        # Flask 개발 서버는 프로그래밍 방식으로 중지할 방법이 없어서
        # 실제로는 프로세스를 종료해야 함
        self.is_running = False
        self.logger.info("대시보드 웹 서버 중지")


def create_app(
    data_manager,
    static_dir: Optional[str] = None,
    templates_dir: Optional[str] = None,
    host: str = '0.0.0.0',
    port: int = 5000,
    debug: bool = False
) -> Flask:
    """
    Flask 애플리케이션 생성 헬퍼 함수
    
    Args:
        data_manager: 데이터 관리자 객체 (DataManager 또는 DBDataManager)
        static_dir: 정적 파일 디렉토리 경로 (옵션)
        templates_dir: 템플릿 디렉토리 경로 (옵션)
        host: 호스트 주소
        port: 포트 번호
        debug: 디버그 모드 여부
        
    Returns:
        Flask 애플리케이션 인스턴스
    """
    dashboard = DashboardApp(
        data_manager=data_manager,
        static_dir=static_dir,
        templates_dir=templates_dir,
        host=host,
        port=port,
        debug=debug
    )
    
    return dashboard.app


if __name__ == '__main__':
    import argparse
    
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='SAC 트레이딩 시스템 대시보드 실행')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='데이터 디렉토리 경로 (기본값: ./data)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='호스트 주소 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='포트 번호 (기본값: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드 활성화')
    
    args = parser.parse_args()
    
    # 대시보드 실행
    dashboard = DashboardApp(
        data_manager=DatabaseManager(args.data_dir),
        host=args.host,
        port=args.port,
        debug=args.debug
    )
    
    dashboard.run() 