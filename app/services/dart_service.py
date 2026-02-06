"""
DART 사업보고서 서비스
=====================

DART 정기보고서에서 '사업의 내용' 7개 항목을 추출하고 요약합니다.
OpenDartReader + AsyncOpenAI를 사용한 비동기 병렬 처리.
"""
import re
import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import requests
from bs4 import BeautifulSoup
import OpenDartReader
from openai import AsyncOpenAI

from app.config import Settings

logger = logging.getLogger(__name__)


class DartService:
    """
    DART 사업보고서 분석 서비스

    파이프라인:
    1. OpenDartReader로 최신 정기보고서 조회
    2. '사업의 내용' 하위 7개 항목 URL 추출
    3. 각 항목 본문 텍스트 추출 (병렬)
    4. GPT로 요약 생성 (병렬)
    5. 감성 분석

    Returns:
        NewsItem 형식과 동일한 응답
    """

    # 사업의 내용 7개 항목
    BUSINESS_SECTIONS = [
        '1. 사업의 개요',
        '2. 주요 제품 및 서비스',
        '3. 원재료 및 생산설비',
        '4. 매출 및 수주상황',
        '5. 위험관리 및 파생거래',
        '6. 주요계약 및 연구개발활동',
        '7. 기타 참고사항'
    ]

    def __init__(self, settings: Settings):
        self.settings = settings

        # OpenDartReader 초기화
        if settings.DART_API_KEY:
            self.dart = OpenDartReader(settings.DART_API_KEY)
        else:
            self.dart = None
            logger.warning("DART_API_KEY not set - report analysis disabled")

        # AsyncOpenAI 클라이언트
        if settings.OPENAI_API_KEY:
            self.openai_async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        else:
            self.openai_async_client = None
            logger.warning("OPENAI_API_KEY not set - summarization disabled")

        # ThreadPoolExecutor for blocking I/O
        self._executor = ThreadPoolExecutor(max_workers=7)

    async def analyze_report_async(self, corp_code: str) -> Dict[str, Any]:
        """
        사업보고서 분석 (비동기 버전)

        Args:
            corp_code: 기업코드 (종목코드, 예: 005930)

        Returns:
            NewsAnalysisResponse 형식 (단일 항목, 7개 섹션 통합)
        """
        logger.info(f"Starting report analysis for: {corp_code}")

        if not self.dart:
            logger.error("DART API not configured")
            return self._empty_response(corp_code)

        # 1. 최신 정기보고서 조회
        report = self._get_latest_report(corp_code)
        if report is None:
            logger.warning(f"No report found for {corp_code}")
            return self._empty_response(corp_code)

        corp_name = report['corp_name']
        rcept_no = report['rcept_no']
        report_nm = report['report_nm']
        rcept_dt = report['rcept_dt']
        logger.info(f"Found report: {corp_name} - {report_nm} ({rcept_dt})")

        # 2. 사업의 내용 하위 문서 추출 (병렬)
        sections = await self._get_business_sections_async(rcept_no)
        logger.info(f"Extracted {len(sections)} business sections")

        if not sections:
            return self._empty_response(corp_code, corp_name)

        # 3. 각 섹션 요약 (병렬) - 감성분석 없음
        tasks = [
            self._summarize_section_only_async(section, corp_name)
            for section in sections
        ]
        section_summaries = await asyncio.gather(*tasks)

        # 4. 7개 섹션을 <br /><br />로 합쳐서 단일 summary 생성
        unified_summary = "<br /><br />".join([
            f"{i}. {s['title']}<br />{s['summary']}"
            for i, s in enumerate(section_summaries, 1)
        ])

        # 5. 제목 생성: "기업 년도 분기 사업내용"
        title = self._make_report_title(corp_name, report_nm)

        # 6. 단일 항목 응답
        news_item = {
            "title": title,
            "summary": unified_summary,
            "score": None,
            "date": None,
            "link": None,
            "sentiment": None
        }

        return {
            "company_name": corp_name,
            "total_count": 1,
            "news": [news_item],
            "average_score": None,
            "analyzed_at": datetime.now().isoformat()
        }

    def _make_report_title(self, corp_name: str, report_nm: str) -> str:
        """보고서 제목 생성: 기업 년도 분기 사업내용"""
        import re
        # report_nm 예시: "분기보고서 (2025.09)", "사업보고서 (2024.12)"
        match = re.search(r'\((\d{4})\.(\d{2})\)', report_nm)
        if match:
            year = match.group(1)
            month = int(match.group(2))
            # 분기 판단
            if month <= 3:
                quarter = "1분기"
            elif month <= 6:
                quarter = "2분기"
            elif month <= 9:
                quarter = "3분기"
            else:
                quarter = "4분기"
            return f"{corp_name} {year}년 {quarter} 사업내용"
        return f"{corp_name} 사업내용"

    async def _summarize_section_only_async(
        self,
        section: Dict,
        corp_name: str
    ) -> Dict[str, str]:
        """단일 섹션 요약만 수행 (감성분석 없음)"""
        title = section.get("title", "")
        content = section.get("content", "")

        # GPT 요약
        result = await self._summarize_section_async(corp_name, title, content)
        return {
            "title": result.get("title", title),
            "summary": result.get("summary", "")
        }

    def _get_latest_report(self, corp_code: str) -> Optional[Dict]:
        """최신 정기보고서 조회 (동기)"""
        try:
            df = self.dart.list(corp_code, kind='A')
            if df is None or len(df) == 0:
                return None
            return df.iloc[0][['rcept_no', 'corp_name', 'report_nm', 'rcept_dt']].to_dict()
        except Exception as e:
            logger.error(f"Failed to get report list: {e}")
            return None

    async def _get_business_sections_async(self, rcept_no: str) -> List[Dict]:
        """사업의 내용 7개 항목 추출 (비동기 병렬)"""
        try:
            sub_docs = self.dart.sub_docs(rcept_no)
        except Exception as e:
            logger.error(f"Failed to get sub_docs: {e}")
            return []

        # 매칭되는 항목 필터링
        matched_docs = []
        for _, row in sub_docs.iterrows():
            title = row['title'].strip()
            for bt in self.BUSINESS_SECTIONS:
                if bt in title or title in bt:
                    matched_docs.append({'title': title, 'url': row['url']})
                    break

        if not matched_docs:
            return []

        # ThreadPoolExecutor로 URL 병렬 요청
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(self._executor, self._fetch_section_content, doc['url'])
            for doc in matched_docs
        ]
        contents = await asyncio.gather(*futures)

        # 결과 정리
        results = []
        for doc, content in zip(matched_docs, contents):
            logger.info(f"Extracted: {doc['title']}")
            results.append({
                'title': doc['title'],
                'content': content,
                'url': doc['url']
            })

        return results

    def _fetch_section_content(self, url: str) -> str:
        """하위 문서 URL에서 텍스트 추출 (테이블 제거)"""
        try:
            response = requests.get(url, timeout=10)
            html = response.content.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html, 'lxml')

            # 테이블 제거
            for table in soup.find_all('table'):
                table.decompose()

            text = soup.get_text(separator='\n', strip=True)
            return self._clean_text(text)
        except Exception as e:
            logger.warning(f"Failed to fetch content from {url}: {e}")
            return ""

    def _clean_text(self, text: str) -> str:
        """텍스트 정제"""
        if not text:
            return ""
        text = text.replace('ㆍ', ', ').replace('㈜', '(주)').replace('\xa0', ' ')
        text = text.replace('☞', '').replace('·', ', ').replace('\t', ' ')
        text = re.sub(r'[^가-힣a-zA-Z0-9\s.,!?\-()%:\'\"\n]', ' ', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        return text.strip()

    async def _summarize_section_async(
        self,
        corp_name: str,
        section_title: str,
        content: str
    ) -> Dict[str, str]:
        """GPT로 섹션 요약 (비동기)"""
        if not content or not self.openai_async_client:
            return {"title": section_title, "summary": ""}

        if len(content) < 100:
            return {"title": section_title, "summary": content}

        prompt = f"""당신은 기업 리스크 분석 전문가입니다.

[서비스 목적]
AI 분석 기반 협력사 리스크 사전 진단 및 기업 관리 플랫폼에서 사용됩니다.
기업 담당자가 협력사 정보를 빠르게 파악할 수 있도록 카드뉴스 형식으로 제공됩니다.

[기업명] {corp_name}
[섹션] {section_title}

[원문 내용]
{content[:8000]}

[요청사항]
1. 위 내용을 협력사 리스크 관점에서 핵심만 요약하세요.
2. 요약은 3~5문장으로 작성하세요.
3. 객관적 사실 위주로 작성하고, 추측이나 평가는 제외하세요.
4. 카드뉴스에 표시될 짧은 제목(15자 이내)도 작성하세요.

[출력 형식 - JSON만 출력]
{{"title": "카드 제목", "summary": "요약 내용"}}
"""

        try:
            response = await self.openai_async_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "JSON 형식으로만 응답하세요."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0,
                timeout=30
            )

            result = json.loads(response.choices[0].message.content)
            return {
                "title": result.get("title", section_title),
                "summary": result.get("summary", "")
            }

        except Exception as e:
            logger.error(f"Section summarization failed: {e}")
            return {"title": section_title, "summary": ""}

    def _empty_response(self, corp_code: str, corp_name: str = None) -> Dict[str, Any]:
        """빈 응답 생성"""
        return {
            "company_name": corp_name or corp_code,
            "total_count": 0,
            "news": [],
            "average_score": None,
            "analyzed_at": datetime.now().isoformat()
        }
