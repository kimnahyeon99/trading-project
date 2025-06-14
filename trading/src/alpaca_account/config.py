import os

API_KEY = os.getenv("APCA_API_KEY_ID", "PK5SQ65O444Z6PRDV4UI")
API_SECRET = os.getenv("APCA_API_SECRET_KEY", "GIavNSovfJIwAnVZC14DI73pBhNBcAuY2CGhOKl5")

BASE_URL = "https://paper-api.alpaca.markets"

DATA_FEED = 'iex' # 무료플랜: iex, 유료플랜: sip

# 기타 설정
DEBUG = False
MAX_RETRIES = 3