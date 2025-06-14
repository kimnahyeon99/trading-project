#!/usr/bin/env python
# -*- coding: utf-8 -*-

# <서버 실행, 설정, 데이터 관리 프로세스 전체 실행 로직>


"""
SAC 트레이딩 시스템 웹 대시보드 실행 스크립트
"""

import os
import sys
import argparse
import logging
from pathlib import Path



# 현재 디렉토리를 상위 경로에 추가하여 src 모듈을 찾을 수 있도록 함
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.dashboard.dashboard_app import DashboardApp
from src.dashboard.data_manager_file import FileDataManager
from src.dashboard.data_manager_factory import DataManagerFactory
from src.utils.database import DatabaseManager
from src.utils.logger import Logger

#from src.alpaca import alpaca_client
from src.alpaca.alpaca_client import AlpacaClient
from src.config import config  # Alpaca API 설정 불러오기

# 스크랩트를 실행 할 때 여러 옵션을 줄 수 있게 만드는 함수 
def parse_args():
    """
    명령행 인자 파싱
    """
    parser = argparse.ArgumentParser(description='SAC 트레이딩 시스템 웹 대시보드')
    
    # 일반 설정 (서버를 어떤 주소, 어떤 포트로 열지 지정)
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='호스트 주소 (기본값: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000,
                        help='포트 번호 (기본값: 5000)')
    parser.add_argument('--debug', action='store_true',
                        help='디버그 모드 활성화')
    parser.add_argument('--data-dir', type=str, default='results',
                        help='데이터 디렉토리 경로 (기본값: results)')
    
    # 데이터베이스 설정(DB 접속 설정 값들)
    parser.add_argument('--db-mode', action='store_true', default=True,
                        help='데이터베이스 모드 활성화')
    parser.add_argument('--db-host', type=str, default='192.168.40.199',
                        help='데이터베이스 호스트')
    parser.add_argument('--db-port', type=int, default=3306,
                        help='데이터베이스 포트 ')
    parser.add_argument('--db-user', type=str, default='root',
                        help='데이터베이스 사용자')
    parser.add_argument('--db-password', type=str, default='mysecretpassword',
                        help='데이터베이스 비밀번호')
    parser.add_argument('--db-name', type=str, default='trading',
                        help='데이터베이스 이름')
    
    # 동기화 설정
    parser.add_argument('--sync-to-db', action='store_true',
                        help='파일 데이터를 데이터베이스로 동기화')
    
    return parser.parse_args()

# 전체 실행 로직 
def main():
    """
    메인 함수
    """
    # 명령행 인자 파싱
    args = parse_args()
    
    # 데이터 디렉토리 설정
    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    
    # 로깅 설정
    log_dir = os.path.join(os.path.dirname(data_dir), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'dashboard.log')
    logger = Logger(log_file, log_level=logging.DEBUG if args.debug else logging.INFO)
    logger.info(f"대시보드 시작 - 데이터 디렉토리: {data_dir}")
    
    try:
        # 데이터 관리자 유형 설정(DB 관리자 or 파일 관리자)
        manager_type = 'db' if args.db_mode else 'file'
        
        # 데이터베이스 구성 설정 (데이터베이스 모드인 경우)
        # db_config: 데이터베이스 설정 정보를 저장하는 딕셔너리
        db_config = None
        if args.db_mode:
            db_config = {
                'host': args.db_host,
                'port': args.db_port,
                'user': args.db_user,
                'password': args.db_password,
                'database': args.db_name
            }
            logger.info(f"데이터베이스 설정: {args.db_host}:{args.db_port}/{args.db_name}")
        
        # 데이터 관리자 생성
        db_manager = DatabaseManager(**db_config, logger=logger)
        
        data_manager = DataManagerFactory.create_manager(
            manager_type=manager_type,
            data_dir=data_dir,
            db_manager=db_manager,
            logger=logger
        )
        
        # 파일에서 데이터베이스로 동기화 (요청된 경우)
        if args.sync_to_db and args.db_mode:
            try:
                logger.info("파일 데이터를 데이터베이스로 동기화 중...")
                
                # 파일 데이터 관리자 생성
                file_data_manager = DataManagerFactory.create_manager(
                    manager_type='file',
                    data_dir=data_dir,
                    logger=logger
                )
                
                # 동기화 수행
                sync_results = DataManagerFactory.sync_file_to_db(
                    file_data_manager=file_data_manager,
                    db_data_manager=data_manager
                )
                
                logger.info(f"동기화 결과: {sync_results}")
            except Exception as e:
                logger.error(f"동기화 중 오류 발생: {e}")
                
        # Alpaca API 연동 (config에서 키 불러오기)
        try:
            alpaca = AlpacaClient(config.API_KEY, config.API_SECRET, config.BASE_URL, data_feed=config.DATA_FEED)
            account = alpaca.get_account()
            logger.info(f"✅ Alpaca 계좌 연결 성공! 현금 잔고: ${account['cash']}")
            logger.info(f"✅ 데이터 피드 설정: {config.DATA_FEED}")
        except Exception as e:
            logger.info(f"❌ Alpaca 계좌 연결 실패: {e}")
            exit(1)
        
        # 대시보드 앱 생성 및 실행(서버 실행)
        app = DashboardApp(
            alpaca=alpaca,
            data_manager=data_manager,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
        
        
        logger.info(f"대시보드 실행 - http://{args.host}:{args.port}")
        app.run()
        
    except Exception as e:
        logger.error(f"대시보드 실행 중 오류 발생: {e}")
        return 1
    
    logger.info("대시보드 종료")
    return 0


if __name__ == '__main__':
    sys.exit(main()) 