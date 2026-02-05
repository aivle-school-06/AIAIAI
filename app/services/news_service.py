"""
뉴스 서비스
==========

금융 뉴스 검색, 필터링, 감성분석, 요약 기능.
네이버 뉴스 API + GPT 필터링 + KR-FinBert 감성분석.
본문 요약은 비동기 병렬 처리로 성능 최적화.
"""
import json
import html
import asyncio
import logging
import urllib.request
import urllib.parse
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI, AsyncOpenAI
from newspaper import Article

from app.config import Settings
from app.core.sentiment_analyzer import SentimentAnalyzer

logger = logging.getLogger(__name__)


class NewsService:
    """
    뉴스 분석 서비스

    파이프라인:
    1. 네이버 뉴스 API로 검색 (동기)
    2. 제목 유사도 기반 중복 제거 (동기)
    3. GPT로 동일 이벤트 기사 필터링 (동기)
    4. 본문 추출 및 요약 (비동기 병렬)
    5. 감성 분석 (동기)
    """

    def __init__(self, settings: Settings, sentiment_analyzer: SentimentAnalyzer):
        """
        NewsService 초기화

        Args:
            settings: 앱 설정 (API 키 포함)
            sentiment_analyzer: 감성 분석기 인스턴스
        """
        self.settings = settings
        self.sentiment_analyzer = sentiment_analyzer

        # Naver API 설정
        self.naver_client_id = settings.NAVER_CLIENT_ID
        self.naver_client_secret = settings.NAVER_CLIENT_SECRET

        # OpenAI 클라이언트 설정 (동기 + 비동기)
        if settings.OPENAI_API_KEY:
            self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
            self.openai_async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.openai_client = None
            self.openai_async_client = None
            logger.warning("OPENAI_API_KEY not set - GPT filtering/summarization disabled")

        # ThreadPoolExecutor for blocking I/O (article extraction)
        self._executor = ThreadPoolExecutor(max_workers=10)

    async def analyze_news_async(
        self,
        company_name: str,
        max_results: int = 50,
        similarity_threshold: float = 0.3,
        enable_gpt_filter: bool = True,
        enable_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        기업 뉴스 분석 (비동기 버전 - 요약 병렬 처리)

        Args:
            company_name: 기업명 (검색 키워드)
            max_results: 최대 검색 결과 수
            similarity_threshold: 제목 유사도 임계값
            enable_gpt_filter: GPT 필터링 활성화
            enable_summary: 요약 생성 활성화

        Returns:
            Dict: 분석 결과
        """
        logger.info(f"Starting async news analysis for: {company_name}")

        # 1. 뉴스 검색 (동기)
        raw_items = self._search_news(company_name, max_results)
        logger.info(f"Found {len(raw_items)} raw news items")

        if not raw_items:
            return {
                "company_name": company_name,
                "total_count": 0,
                "news": [],
                "average_score": None,
                "analyzed_at": datetime.now().isoformat()
            }

        # 2. 유사도 기반 중복 제거 (동기)
        filtered_items = self._filter_by_similarity(raw_items, similarity_threshold)
        logger.info(f"After similarity filter: {len(filtered_items)} items")

        # 3. GPT 필터링 (동기)
        if enable_gpt_filter and self.openai_client:
            filtered_items = self._filter_with_gpt(filtered_items, company_name)
            logger.info(f"After GPT filter: {len(filtered_items)} items")

        # 4. 각 뉴스 분석 (비동기 병렬 처리)
        logger.info(f"Starting parallel analysis for {len(filtered_items)} items")
        tasks = [
            self._analyze_single_news_async(item, enable_summary)
            for item in filtered_items
        ]
        news_results = await asyncio.gather(*tasks)
        logger.info(f"Parallel analysis completed")

        # 5. 평균 점수 계산
        positive_scores = [r["score"] for r in news_results]
        avg_score = sum(positive_scores) / len(positive_scores) if positive_scores else None

        return {
            "company_name": company_name,
            "total_count": len(news_results),
            "news": news_results,
            "average_score": round(avg_score, 4) if avg_score else None,
            "analyzed_at": datetime.now().isoformat()
        }

    def analyze_news(
        self,
        company_name: str,
        max_results: int = 50,
        similarity_threshold: float = 0.3,
        enable_gpt_filter: bool = True,
        enable_summary: bool = True,
    ) -> Dict[str, Any]:
        """
        기업 뉴스 분석 (동기 버전 - 하위 호환용)
        """
        logger.info(f"Starting news analysis for: {company_name}")

        # 1. 뉴스 검색
        raw_items = self._search_news(company_name, max_results)
        logger.info(f"Found {len(raw_items)} raw news items")

        if not raw_items:
            return {
                "company_name": company_name,
                "total_count": 0,
                "news": [],
                "average_score": None,
                "analyzed_at": datetime.now().isoformat()
            }

        # 2. 유사도 기반 중복 제거
        filtered_items = self._filter_by_similarity(raw_items, similarity_threshold)
        logger.info(f"After similarity filter: {len(filtered_items)} items")

        # 3. GPT 필터링 (선택)
        if enable_gpt_filter and self.openai_client:
            filtered_items = self._filter_with_gpt(filtered_items, company_name)
            logger.info(f"After GPT filter: {len(filtered_items)} items")

        # 4. 각 뉴스 분석 (순차)
        news_results = []
        positive_scores = []

        for item in filtered_items:
            result = self._analyze_single_news(item, enable_summary)
            news_results.append(result)
            positive_scores.append(result["score"])

        # 5. 평균 점수 계산
        avg_score = sum(positive_scores) / len(positive_scores) if positive_scores else None

        return {
            "company_name": company_name,
            "total_count": len(news_results),
            "news": news_results,
            "average_score": round(avg_score, 4) if avg_score else None,
            "analyzed_at": datetime.now().isoformat()
        }

    def _search_news(self, query: str, display: int = 50) -> List[Dict]:
        """네이버 뉴스 API 검색"""
        if not self.naver_client_id or not self.naver_client_secret:
            logger.error("Naver API credentials not configured")
            return []

        enc_text = urllib.parse.quote(query)
        url = f"https://openapi.naver.com/v1/search/news?query={enc_text}&display={display}&sort=sim"

        req = urllib.request.Request(url)
        req.add_header("X-Naver-Client-Id", self.naver_client_id)
        req.add_header("X-Naver-Client-Secret", self.naver_client_secret)

        try:
            with urllib.request.urlopen(req, timeout=10) as res:
                data = json.loads(res.read().decode('utf-8'))
                return data.get('items', [])
        except Exception as e:
            logger.error(f"News search failed: {e}")
            return []

    def _clean_text(self, text: str) -> str:
        """HTML 태그 제거 및 정규화"""
        if not text:
            return ""
        return html.unescape(text).replace('<b>', '').replace('</b>', '').strip()

    def _get_similarity(self, a: str, b: str) -> float:
        """두 문자열 유사도 (0~1)"""
        return SequenceMatcher(None, a, b).ratio()

    def _filter_by_similarity(
        self,
        news_items: List[Dict],
        threshold: float = 0.5
    ) -> List[Dict]:
        """제목 유사도 기반 중복 제거"""
        unique_items = []

        for item in news_items:
            current_title = self._clean_text(item.get('title', ''))
            is_duplicate = False

            for seen_item in unique_items:
                seen_title = self._clean_text(seen_item.get('title', ''))
                if self._get_similarity(current_title, seen_title) > threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_items.append(item)

        return unique_items

    def _filter_with_gpt(
        self,
        news_items: List[Dict],
        keyword: str
    ) -> List[Dict]:
        """GPT를 사용한 동일 이벤트 기사 필터링"""
        if not news_items or not self.openai_client:
            return news_items

        news_context = "\n".join([
            f"ID: {i} | 제목: {self._clean_text(item.get('title', ''))} | 설명: {self._clean_text(item.get('description', ''))}"
            for i, item in enumerate(news_items)
        ])

        prompt = f"""
"{keyword}" 뉴스에서 중복을 제거하세요.

[중복 예시]
- "SK하이닉스 성과급 2964%" = "SK하이닉스 성과급 잭팟" = "연봉 1억에 보너스 1.5억" → 같은 뉴스, 1개만 선택
- "삼성전자 시총 1000조" = "시가총액 1천조 돌파" → 같은 뉴스, 1개만 선택
- "1월 판매량 30만대" = "1월 글로벌 판매 1% 감소" → 같은 뉴스, 1개만 선택

[규칙]
1. 같은 수치/실적/사건을 다루면 무조건 중복
2. 제목이 달라도 설명이 같은 내용이면 중복
3. 중복 그룹에서 가장 상세한 기사 1개만 선택

[뉴스 목록]
{news_context}

반드시 JSON으로 응답: {{"selected_ids": [0, 2, 5]}}
"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "뉴스 중복 제거 전문가입니다. 같은 사건을 다룬 기사는 반드시 1개만 남깁니다."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                timeout=30
            )

            result = json.loads(response.choices[0].message.content)
            ids = result.get("selected_ids", [])
            return [news_items[int(i)] for i in ids if int(i) < len(news_items)]

        except Exception as e:
            logger.error(f"GPT filtering failed: {e}")
            return news_items

    def _extract_article_text(self, url: str) -> str:
        """기사 본문 추출"""
        try:
            article = Article(url, language='ko')
            article.download()
            article.parse()
            return article.text
        except Exception as e:
            logger.warning(f"Article extraction failed for {url}: {e}")
            return ""

    async def _extract_article_text_async(self, url: str) -> str:
        """기사 본문 추출 (비동기 - ThreadPool 사용)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self._extract_article_text, url)

    def _summarize_text(self, text: str, max_tokens: int = 300) -> str:
        """GPT를 사용한 텍스트 요약 (동기)"""
        if not text or not self.openai_client:
            return ""

        prompt = (
            "다음 뉴스 기사의 핵심 내용을 2~3문장으로 요약해 주세요. "
            "주요 사실, 수치, 전망 등을 포함해 주세요.\n\n"
            f"기사 본문:\n{text}\n\n"
            "요약:"
        )

        if len(prompt) > 30000:
            prompt = prompt[:30000] + "\n\n[중간 내용 생략]..."

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 객관적인 금융 데이터 분석가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
                timeout=30
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return ""

    async def _summarize_text_async(self, text: str, max_tokens: int = 300) -> str:
        """GPT를 사용한 텍스트 요약 (비동기)"""
        if not text or not self.openai_async_client:
            return ""

        prompt = (
            "다음 뉴스 기사의 핵심 내용을 2~3문장으로 요약해 주세요. "
            "주요 사실, 수치, 전망 등을 포함해 주세요.\n\n"
            f"기사 본문:\n{text}\n\n"
            "요약:"
        )

        if len(prompt) > 30000:
            prompt = prompt[:30000] + "\n\n[중간 내용 생략]..."

        try:
            response = await self.openai_async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "당신은 객관적인 금융 데이터 분석가입니다."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0,
                timeout=30
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Async summarization failed: {e}")
            return ""

    def _parse_pub_date(self, pub_date: str) -> str:
        """날짜 문자열 파싱"""
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(pub_date)
            return dt.isoformat()
        except Exception:
            return pub_date

    def _analyze_single_news(
        self,
        item: Dict,
        enable_summary: bool = True
    ) -> Dict[str, Any]:
        """단일 뉴스 분석 (동기)"""
        title = self._clean_text(item.get('title', ''))
        description = self._clean_text(item.get('description', ''))
        link = item.get('link', '')
        original_link = item.get('originallink', link)
        pub_date = item.get('pubDate', '')

        # 감성 분석
        combined_text = f"{title} {description}"
        sentiment_result = self.sentiment_analyzer.predict_single(combined_text)
        positive_score = sentiment_result['probs'][2]

        # 요약 생성
        summary = ""
        if enable_summary:
            article_text = self._extract_article_text(original_link)
            if not article_text or len(article_text) < 100:
                article_text = self._extract_article_text(link)
            if article_text:
                summary = self._summarize_text(article_text)

        return {
            "title": title,
            "summary": summary,
            "score": round(positive_score, 4),
            "date": self._parse_pub_date(pub_date),
            "link": link,
            "sentiment": sentiment_result['label']
        }

    async def _analyze_single_news_async(
        self,
        item: Dict,
        enable_summary: bool = True
    ) -> Dict[str, Any]:
        """단일 뉴스 분석 (비동기)"""
        title = self._clean_text(item.get('title', ''))
        description = self._clean_text(item.get('description', ''))
        link = item.get('link', '')
        original_link = item.get('originallink', link)
        pub_date = item.get('pubDate', '')

        # 감성 분석 (CPU-bound, 빠름)
        combined_text = f"{title} {description}"
        sentiment_result = self.sentiment_analyzer.predict_single(combined_text)
        positive_score = sentiment_result['probs'][2]

        # 요약 생성 (I/O-bound, 병렬화 대상)
        summary = ""
        if enable_summary:
            # 본문 추출 (비동기)
            article_text = await self._extract_article_text_async(original_link)
            if not article_text or len(article_text) < 100:
                article_text = await self._extract_article_text_async(link)

            # GPT 요약 (비동기)
            if article_text:
                summary = await self._summarize_text_async(article_text)

        return {
            "title": title,
            "summary": summary,
            "score": round(positive_score, 4),
            "date": self._parse_pub_date(pub_date),
            "link": link,
            "sentiment": sentiment_result['label']
        }
