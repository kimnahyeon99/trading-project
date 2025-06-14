import json
import os
import time
import gc
from bs4 import BeautifulSoup
import requests
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
import yfinance as yf
import json
from datetime import datetime
import os

def get_yahoo_finance_news(ticker):
    stock = yf.Ticker(ticker)
    news = stock.news  # Get news information
    
    if news:
        # Process the news articles if the title and summary keys are present
        for article in news:
            if 'title' in article:
                article['title'] = article['title'].encode('utf-8').decode('unicode_escape')
            if 'summary' in article:
                article['summary'] = article['summary'].encode('utf-8').decode('unicode_escape')
        
        # Convert the news to JSON format
        news_json = json.dumps(news, indent=4)
        return news_json
    else:
        return json.dumps({"message": "No news found for this ticker."}, indent=4)

def fetch_multiple_tickers_news(tickers):
    news_data = {}
    for ticker in tickers:
        print(f"Fetching news for {ticker}...")
        news_data[ticker] = get_yahoo_finance_news(ticker)
    
    return news_data

def format_publishday(pubDate):
    dt = datetime.strptime(pubDate, '%Y-%m-%dT%H:%M:%SZ')
    return dt.strftime('%b %d, %Y, %I:%M %p')

def process_and_save_news(tickers, output_file='ticker_news.json'):
    all_news = fetch_multiple_tickers_news(tickers)
    all_articles = []
    
    for ticker, news in all_news.items():
        try:
            news_list = json.loads(news)
            if isinstance(news_list, list) and len(news_list) > 0:
                ticker_articles = []
                for article in news_list:
                    content = article.get('content', {})
                    title = content.get('title', 'No Title')
                    summary = content.get('summary', 'No Summary')
                    pubDate = content.get('pubDate', '')
                    url = content.get('canonicalUrl', {}).get('url', 'No URL')

                    formatted_pubDate = format_publishday(pubDate) if pubDate else 'No Date'

                    article_data = {
                        "name": ticker,
                        "title": title,
                        "date": formatted_pubDate,
                        "url": url,
                        "summary": summary
                    }
                    
                    ticker_articles.append(article_data)
                    all_articles.append(article_data)
                
                # Print ticker articles for debug
                print(f"\nNews for {ticker}:")
                print(json.dumps(ticker_articles, indent=4, ensure_ascii=False))
            else:
                print(f"No articles available for {ticker}.")
        except json.JSONDecodeError:
            print(f"Error decoding JSON for {ticker}: {news}")
    
    # Save to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)
    
    print(f"\n모든 뉴스 데이터가 '{output_file}' 파일에 성공적으로 저장되었습니다.")
    
    return all_articles

# 예제 티커 리스트
tickers = ['VALE', 'NEM', 'FCX', 'T', 'NFLX', 'WBD', 'TSLA', 'AMZN', 'NKE', 'PEP', 'KVUE', 'PM', 'PBR', 'KMI', 'SHEL', 'ITUB', 'USB', 'MUFG', 'AZN', 'ABT', 'UNH', 'CSX', 'BA', 'RTX', 'NVDA', 'ERIC', 'YMM', 'NEE']  # 원하는 티커를 여기에 추가하세요

# 뉴스 가져오기, 처리 및 JSON 파일로 저장
processed_news = process_and_save_news(tickers)

# CUDA 환경변수 설정 추가 (디버깅용)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# GPU 메모리 초기화 함수
def clear_gpu_memory():
    """GPU 메모리 초기화 함수"""
    if torch.cuda.is_available():
        # 모든 PyTorch 캐시 비우기
        torch.cuda.empty_cache()
        # 가비지 컬렉션 강제 실행
        gc.collect()
        print("🧹 GPU 메모리가 초기화되었습니다.")

# 시작 시 GPU 메모리 초기화
clear_gpu_memory()

# 경로 설정
input_path = './ticker_news.json'
output_path = './summaries/Bart_summary_news.json'

# 결과 저장 디렉토리 생성
os.makedirs(os.path.dirname(output_path), exist_ok=True)

def get_full_article_text(url):
    """웹 페이지에서 기사 전체 내용 추출"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36'
        }
        res = requests.get(url, headers=headers, timeout=15)
        res.raise_for_status()  # HTTP 오류 발생 시 예외 발생
        soup = BeautifulSoup(res.text, 'html.parser')

        # 대부분의 기사는 p 태그 내에 본문 내용이 있음
        paragraphs = soup.find_all('p')
        full_text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
        
        # 텍스트 정리
        full_text = full_text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
        while '  ' in full_text:
            full_text = full_text.replace('  ', ' ')
            
        return full_text.strip()
    except Exception as e:
        print(f"[ERROR] Failed to fetch: {url} - {str(e)}")
        return ""

# BART 모델 및 토크나이저 로드
print("Loading BART model and tokenizer...")
model_name = "facebook/bart-large-cnn"  # CNN 뉴스 데이터로 학습된 요약 모델
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# GPU 사용 가능 시 GPU로 이동
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")

# 원본 JSON 파일 로드
print(f"Loading article URLs from {input_path}...")
with open(input_path, 'r', encoding='utf-8') as f:
    articles = json.load(f)

# URL 목록 추출
urls = [article['url'] for article in articles if 'url' in article]
print(f"Found {len(urls)} URLs to process")

# 결과를 저장할 리스트
summarized_articles = []

# GPU 메모리 모니터링을 위한 변수
processed_count = 0
memory_clear_interval = 3  # 3개 기사마다 메모리 초기화 (더 자주 초기화)

for idx, article in enumerate(articles, 1):
    if 'url' not in article:
        print(f"[{idx}/{len(articles)}] Skipping (no URL found)")
        continue
        
    url = article['url']
    print(f"[{idx}/{len(articles)}] Processing: {url}")
    
    # 기사 내용 크롤링
    print(f"  - Fetching content...")
    content = get_full_article_text(url)
    
    # 크롤링 실패 또는 매우 짧은 내용 건너뛰기
    if not content or len(content) < 100:
        print(f"  - Skipping (too short or failed to fetch): {url}")
        summarized_articles.append({
            "url": url,
            "title": article.get('title', ''),
            "date": article.get('date', ''),
            "original_content": content,
            "summary": "Content too short to summarize or failed to fetch",
            "status": "skipped"
        })
        continue
    
    print(f"  - Content fetched: {len(content)} characters")
    print(f"  - Summarizing...")
    
    try:
        # 입력 텍스트 길이 제한 - 오류 방지를 위해 더 보수적인 값 사용
        max_input_length = 1024  # 더 낮은 값으로 설정
        
        # 토큰화 및 요약 생성
        inputs = tokenizer(content, max_length=max_input_length, return_tensors="pt", truncation=True)
        inputs = inputs.to(device)
        
        # 더 안전한 파라미터로 요약 생성
        summary_ids = model.generate(
            inputs["input_ids"],
            num_beams=2,  # 4에서 2로 감소 (메모리 사용량 감소)
            min_length=80,
            max_length=200,
            early_stopping=True,
            length_penalty=1.0,  # 더 보수적인 값으로 변경
            no_repeat_ngram_size=3,  # 반복 방지
        )
        
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # 요약 길이 확인 및 조정 (필요한 경우)
        if len(summary) > 200:
            # 문장 단위로 잘라서 조정
            sentences = summary.split('.')
            adjusted_summary = ""
            for sentence in sentences:
                if len(adjusted_summary) + len(sentence) <= 197:  # 마침표와 공백 고려
                    adjusted_summary += sentence + "."
                else:
                    break
            summary = adjusted_summary.strip()
        
        print(f"  - Summary created: {len(summary)} characters")
        
        # 원본 기사의 모든 필드 보존하고 요약 결과 추가
        result = article.copy()
        result.update({
            "original_content": content,
            "summary": summary,
            "summary_length": len(summary),
            "status": "success"
        })
        
        summarized_articles.append(result)
        
        # 매 기사 처리 후 GPU 메모리 초기화 (오류 방지)
        clear_gpu_memory()
        
    except Exception as e:
        print(f"  - [ERROR] Failed to summarize: {url} - {str(e)}")
        
        # 원본 기사의 모든 필드 보존하고 에러 정보 추가
        result = article.copy()
        result.update({
            "original_content": content,
            "summary": "Error in summarization process",
            "error": str(e),
            "status": "error"
        })
        
        summarized_articles.append(result)
        
        # 오류 발생 시 GPU 메모리 즉시 초기화
        clear_gpu_memory()
    
    # 서버에 부담을 주지 않기 위해 요청 간 일정 시간 대기
    time.sleep(2)

# 결과 저장
print(f"Saving summaries to {output_path}...")
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(summarized_articles, f, ensure_ascii=False, indent=4)

# 요약 결과 통계
successful_summaries = [article for article in summarized_articles if article.get('status') == 'success']
summary_lengths = [len(article['summary']) for article in successful_summaries]

if summary_lengths:
    avg_length = sum(summary_lengths) / len(summary_lengths)
    print(f"\n✅ Finished. {len(successful_summaries)}/{len(articles)} articles successfully summarized.")
    print(f"Average summary length: {avg_length:.1f} characters")
    print(f"Min summary length: {min(summary_lengths)} characters")
    print(f"Max summary length: {max(summary_lengths)} characters")
else:
    print("\n❌ No successful summaries were generated.")

print(f"Summaries saved to: {output_path}")

import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 1. JSON 파일 로드
with open("./summaries/Bart_summary_news.json", "r", encoding="utf-8") as f:
    data_list = json.load(f)  # data_list는 리스트임

# 2. FinBERT 모델 및 토크나이저 로드
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3. 감성 분석 파이프라인 생성
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# 4. 각 뉴스 요약에 대해 감성 분석 수행 및 결과 추가
for item in data_list:
    summary_text = item.get("summary", "")
    if summary_text:
        result = sentiment_pipeline(summary_text)[0]
        item["sentiment"] = result["label"]
        item["sentiment_score"] = round(result["score"], 4)
    else:
        item["sentiment"] = "UNKNOWN"
        item["sentiment_score"] = 0.0

final_output = [
    {
        "name": item.get("name", ""),
        "title": item.get("title", ""),
        "date": item.get("date", ""),
        "url": item.get("url", ""),
        "summary": item.get("summary", ""),
        "sentiment": item.get("sentiment", ""),
        "sentiment_score": item.get("sentiment_score", 0.0)
    }
    for item in summarized_articles
]

# 5. 결과 저장
with open("Bart_summary_news_with_sentiment.json", "w", encoding="utf-8") as f:
    json.dump(final_output, f, indent=4, ensure_ascii=False)

print("✅ 전체 뉴스에 대해 감성 분석 완료!")

from pymongo import MongoClient, UpdateOne
import json

# MongoDB 설정
MONGO_URI = "mongodb://192.168.40.192:27017"  # 필요 시 포트, 인증 정보 등 수정
DB_NAME = "yahoo"
COLLECTION_NAME = "articles"

# MongoDB 연결
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

# JSON 파일 로드
with open("Bart_summary_news_with_sentiment.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

# 중복 제거 및 MongoDB에 저장 (url 기준 upsert)
operations = []

for article in articles:
    url = article.get("url")
    if url:
        # URL 기준으로 기존 문서 있으면 업데이트, 없으면 삽입
        operations.append(
            UpdateOne(
                {"url": url},       # filter 조건
                {"$set": article},  # 업데이트 내용
                upsert=True         # 없으면 삽입
            )
        )

if operations:
    result = collection.bulk_write(operations)
    print(f"✅ MongoDB 저장 완료: 삽입 {result.upserted_count}건, 수정 {result.modified_count}건")
else:
    print("⚠️ 저장할 데이터가 없습니다.")

# 연결 종료
client.close()
