"""
비동기 PDF 생성 서비스
====================

완전한 비동기 설계로 LLM 호출을 병렬 처리합니다.

아키텍처:
- FastAPI (async) → LLM 호출 (AsyncOpenAI, 병렬) → PDF 렌더링 (스레드풀)

성능:
- 기존: 17회 순차 LLM 호출 → 약 2분
- 개선: 17회 병렬 LLM 호출 → 약 30-40초
"""
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# 상수 (기존 config에서 가져옴)
# =============================================================================
TARGET_METRICS = {
    '수익성': ['ROA', 'ROE', '매출액영업이익률'],
    '안정성': ['부채비율', '차입금의존도'],
    '차입금': ['순차입금비율', '이자보상배율'],
    '유동성': ['유동비율', '당좌비율'],
    '현금흐름': ['CFO_자산비율', 'FCF_자산비율', 'CAPEX_자산비율', '영업현금흐름비율'],
}

METRIC_DESCRIPTION = {
    'ROA': '자산 대비 순이익률',
    'ROE': '자기자본 대비 순이익률',
    '매출액영업이익률': '매출 대비 영업이익 비율',
    '부채비율': '자기자본 대비 부채 비율',
    '차입금의존도': '총자산 대비 차입금 비율',
    '순차입금비율': '순차입금 대비 자기자본 비율',
    '이자보상배율': '영업이익 대비 이자비용 배율',
    '유동비율': '유동부채 대비 유동자산 비율',
    '당좌비율': '재고 제외 유동자산 비율',
    'CFO_자산비율': '영업현금흐름 자산 비율',
    'FCF_자산비율': '잉여현금흐름 자산 비율',
    'CAPEX_자산비율': '자본적 지출 자산 비율',
    '영업현금흐름비율': '영업현금흐름 영업이익 비율',
}

METRIC_DIRECTION = {
    'ROA': 'higher', 'ROE': 'higher', '매출액영업이익률': 'higher',
    '부채비율': 'lower', '차입금의존도': 'lower', '순차입금비율': 'lower',
    '이자보상배율': 'higher', '유동비율': 'higher', '당좌비율': 'higher',
    'CFO_자산비율': 'higher', 'FCF_자산비율': 'higher',
    'CAPEX_자산비율': 'context', '영업현금흐름비율': 'higher',
}


class AsyncLLMService:
    """비동기 LLM 서비스"""

    def __init__(self, max_concurrent: int = 10):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 설정되어 있지 않습니다.")

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2000
    ) -> str:
        """단일 LLM API 호출"""
        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                logger.error(f"LLM API 호출 실패: {e}")
                return f"오류: {str(e)}"


class AsyncPDFService:
    """비동기 PDF 생성 서비스"""

    def __init__(self):
        self.llm = AsyncLLMService()
        self._import_backend_modules()

    def _import_backend_modules(self):
        """app/report 모듈 import"""
        from app.report.report_generator import generate_report
        from app.report.data_loader import get_data_loader
        from app.report.pdf_generator import PDFReportGenerator, PDFReport
        from app.report.config import TARGET_METRICS as TM

        self._generate_report = generate_report
        self._get_data_loader = get_data_loader
        self._PDFReportGenerator = PDFReportGenerator
        self._PDFReport = PDFReport

    async def generate(self, company_code: str, output_dir: str) -> str:
        """
        비동기 PDF 생성

        Args:
            company_code: 기업 코드 (예: '[005930]')
            output_dir: 출력 디렉토리

        Returns:
            str: 생성된 PDF 파일 경로
        """
        logger.info(f"[Async PDF] Starting for {company_code}")
        start_time = asyncio.get_event_loop().time()

        # 1. 데이터 준비 (스레드풀에서 실행)
        logger.info("[Async PDF] Step 1: Preparing data...")
        report_data, company_data, industry_comparison = await self._prepare_data(company_code)

        if 'error' in report_data:
            raise ValueError(report_data['error'])

        # 2. 모든 LLM 분석 병렬 실행
        logger.info("[Async PDF] Step 2: Running LLM analysis in parallel...")
        llm_results = await self._run_all_llm_analysis(
            report_data, company_data, industry_comparison
        )
        logger.info(f"[Async PDF] LLM analysis completed: {len(llm_results)} results")

        # 3. PDF 렌더링 (스레드풀에서 실행)
        logger.info("[Async PDF] Step 3: Rendering PDF...")
        pdf_path = await self._render_pdf(
            company_code, output_dir, report_data,
            company_data, industry_comparison, llm_results
        )

        elapsed = asyncio.get_event_loop().time() - start_time
        logger.info(f"[Async PDF] Completed in {elapsed:.1f}s: {pdf_path}")

        return pdf_path

    async def _prepare_data(self, company_code: str) -> tuple:
        """데이터 준비 (병렬)"""
        loop = asyncio.get_event_loop()

        # report_data 생성
        report_task = loop.run_in_executor(
            None, self._generate_report, company_code
        )

        # company_data & industry_comparison
        data_loader = self._get_data_loader()

        company_task = loop.run_in_executor(
            None, data_loader.get_company_data, company_code
        )

        industry_task = loop.run_in_executor(
            None, data_loader.get_industry_metric_comparison, company_code
        )

        report_data, company_data, industry_comparison = await asyncio.gather(
            report_task, company_task, industry_task
        )

        return report_data, company_data, industry_comparison

    async def _run_all_llm_analysis(
        self,
        report_data: Dict,
        company_data: Dict,
        industry_comparison: Dict
    ) -> Dict[str, Any]:
        """모든 LLM 분석을 병렬로 실행"""
        tasks = {}

        # 데이터 추출
        opinion_data = report_data.get('opinion_data', {})
        sections = report_data.get('sections', {})
        meta = report_data.get('meta', {})

        ai_prediction = sections.get('ai_prediction', {})
        all_predictions = ai_prediction.get('all_predictions', {})
        industry_section = sections.get('industry_comparison', {})
        trend_section = sections.get('trend_analysis', {})

        company_name = meta.get('기업명', '')
        industry = meta.get('업종명', '')

        # SHAP 데이터 추출
        shap_data = self._extract_shap_data(report_data)
        opinion_data['shap_analysis'] = shap_data

        # 1. 종합 의견
        tasks['opinion'] = self._generate_opinion(opinion_data)

        # 2. 카테고리별 분석 (5개)
        for category, metrics in TARGET_METRICS.items():
            cat_predictions = {m: all_predictions.get(m) for m in metrics if m in all_predictions}
            if cat_predictions:
                tasks[f'category_{category}'] = self._generate_category_analysis(
                    category, metrics, opinion_data, cat_predictions
                )

        # 3. 시계열 분석 (5개)
        historical = company_data.get('historical', {}) if company_data else {}
        trend_data = company_data.get('trend', {}) if company_data else {}
        relative_data = company_data.get('relative', {}) if company_data else {}

        for category, metrics in TARGET_METRICS.items():
            metrics_data = {m: historical.get('metrics', {}).get(m, []) for m in metrics}
            if any(metrics_data.values()):
                tasks[f'timeseries_{category}'] = self._generate_timeseries(
                    company_name, industry, category,
                    metrics_data, trend_data, relative_data
                )

        # 4. 업종 비교 종합
        if industry_section and industry_comparison:
            category_positions = industry_section.get('category_position', {})
            strengths = industry_section.get('strengths', [])
            weaknesses = industry_section.get('weaknesses', [])
            metrics_info = industry_comparison.get('metrics', {})

            comparison_data = {
                m: {'company': info.get('company'), 'industry_mean': info.get('industry_mean'), 'percentile': info.get('percentile')}
                for m, info in metrics_info.items() if info
            }

            tasks['industry_comparison'] = self._generate_industry_comparison(
                company_name, industry, category_positions,
                comparison_data, strengths, weaknesses
            )

            # 5. 카테고리별 업종 비교 (5개)
            for category, metrics in TARGET_METRICS.items():
                cat_metrics = {m: comparison_data.get(m) for m in metrics if comparison_data.get(m)}
                position = category_positions.get(category)
                if cat_metrics and position:
                    tasks[f'category_industry_{category}'] = self._generate_category_industry(
                        company_name, industry, category, cat_metrics, position
                    )

        # 모든 태스크 병렬 실행
        keys = list(tasks.keys())
        coros = list(tasks.values())

        logger.info(f"[Async PDF] Running {len(tasks)} LLM tasks...")
        results = await asyncio.gather(*coros, return_exceptions=True)

        output = {}
        for key, result in zip(keys, results):
            if isinstance(result, Exception):
                logger.warning(f"LLM task failed ({key}): {result}")
                output[key] = None
            else:
                output[key] = result

        return output

    def _extract_shap_data(self, report_data: Dict) -> Dict:
        """SHAP 분석 데이터 추출"""
        ai_pred = report_data.get('sections', {}).get('ai_prediction', {})
        all_predictions = ai_pred.get('all_predictions', {})

        shap_data = {}
        for metric, pred in all_predictions.items():
            if pred and 'shap_analysis' in pred:
                shap_data[metric] = pred['shap_analysis']

        return shap_data

    # =========================================================================
    # LLM 프롬프트 생성 및 호출
    # =========================================================================

    async def _generate_opinion(self, opinion_data: Dict) -> Dict:
        """종합 의견 생성"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
주어진 재무 데이터와 AI 예측 결과를 분석하여 종합 의견을 작성해주세요.

응답 형식:
[전문가용]
(금융 전문가를 위한 상세 분석. 5~7문장)

[비전문가용]
(일반인을 위한 쉬운 설명)

[핵심 포인트]
• (핵심 사항 1)
• (핵심 사항 2)
• (핵심 사항 3)

[주의 사항]
(리스크가 있다면 설명)
"""

        user_prompt = self._build_opinion_prompt(opinion_data)
        content = await self.llm.call(system_prompt, user_prompt, max_tokens=3000)
        return self._parse_opinion_response(content)

    def _build_opinion_prompt(self, data: Dict) -> str:
        """종합 의견 프롬프트 생성"""
        company = data.get('company', {})
        grades = data.get('grades', {})
        current = data.get('current_metrics', {})
        prediction = data.get('prediction', {})

        prompt = f"""기업 재무 상태 분석:

## 기업 정보
- 기업명: {company.get('name', 'N/A')}
- 업종: {company.get('industry', 'N/A')}
- 기준일: {company.get('period', 'N/A')}

## 종합 등급: {grades.get('overall', 'N/A')} (점수: {grades.get('score', 'N/A')})
- 강점: {', '.join(grades.get('strengths', [])) or '없음'}
- 약점: {', '.join(grades.get('weaknesses', [])) or '없음'}

## 주요 지표
"""
        for metric in ['ROA', 'ROE', '매출액영업이익률', '부채비율', '유동비율']:
            value = current.get(metric)
            if value is not None:
                prompt += f"- {metric}: {value:.2f}%\n"

        prompt += f"""
## AI 예측
- 전망: {prediction.get('outlook', 'N/A')}
- 개선 예측: {', '.join(prediction.get('improving', [])) or '없음'}
- 악화 예측: {', '.join(prediction.get('declining', [])) or '없음'}
"""
        return prompt

    def _parse_opinion_response(self, content: str) -> Dict:
        """종합 의견 응답 파싱"""
        result = {
            'expert': '', 'simple': '', 'key_points': [],
            'risk_summary': None, 'raw_response': content
        }

        sections = content.split('[')
        for section in sections:
            if section.startswith('전문가용]'):
                result['expert'] = section.replace('전문가용]', '').strip()
            elif section.startswith('비전문가용]'):
                result['simple'] = section.replace('비전문가용]', '').strip()
            elif section.startswith('핵심 포인트]'):
                text = section.replace('핵심 포인트]', '').strip()
                result['key_points'] = [
                    line.lstrip('•-').strip()
                    for line in text.split('\n')
                    if line.strip().startswith(('•', '-'))
                ]
            elif section.startswith('주의 사항]'):
                result['risk_summary'] = section.replace('주의 사항]', '').strip()

        return result

    async def _generate_category_analysis(
        self, category: str, metrics: List[str],
        opinion_data: Dict, predictions: Dict
    ) -> Dict:
        """카테고리별 분석"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
응답 형식:
[카테고리 종합]
(3~4문장 분석)

[전망]
(positive/negative/neutral)

[핵심 메시지]
• (인사이트 1)
• (인사이트 2)
"""

        current = opinion_data.get('current_metrics', {})
        relative = opinion_data.get('relative_metrics', {})

        prompt = f"'{category}' 카테고리 분석:\n\n"
        for metric in metrics:
            value = current.get(metric)
            rel = relative.get(metric)
            pred = predictions.get(metric, {})

            if value is not None:
                prompt += f"### {metric}\n"
                prompt += f"- 현재값: {value:.2f}%"
                if rel is not None:
                    prompt += f" (업종대비 {rel:+.1f}%p)"
                prompt += "\n"
                if pred and 'predicted' in pred:
                    prompt += f"- 예측: {pred['predicted']:.2f}% (변화: {pred.get('change', 0):+.2f}%p)\n"

        content = await self.llm.call(system_prompt, prompt, max_tokens=1500)
        return self._parse_category_response(content, category, metrics, predictions)

    def _parse_category_response(
        self, content: str, category: str,
        metrics: List[str], predictions: Dict
    ) -> Dict:
        """카테고리 분석 응답 파싱"""
        result = {
            'summary': '', 'outlook': 'neutral',
            'key_messages': [], 'metrics': {}, 'raw_response': content
        }

        sections = content.split('[')
        for section in sections:
            if section.startswith('카테고리 종합]'):
                result['summary'] = section.replace('카테고리 종합]', '').strip().split('\n\n')[0]
            elif section.startswith('전망]'):
                text = section.replace('전망]', '').strip().lower()
                if 'positive' in text:
                    result['outlook'] = 'positive'
                elif 'negative' in text:
                    result['outlook'] = 'negative'
            elif section.startswith('핵심 메시지]'):
                text = section.replace('핵심 메시지]', '').strip()
                result['key_messages'] = [
                    line.lstrip('•-').strip()
                    for line in text.split('\n')
                    if line.strip().startswith(('•', '-'))
                ][:3]

        # metrics 기본값
        for metric in metrics:
            pred = predictions.get(metric, {})
            shap = pred.get('shap_analysis', {}) if pred else {}
            result['metrics'][metric] = {
                'insight': '',
                'factors': {
                    'positive': shap.get('positive_factors', [])[:3],
                    'negative': shap.get('negative_factors', [])[:3],
                }
            }

        return result

    async def _generate_timeseries(
        self, company_name: str, industry: str, category: str,
        metrics_data: Dict, trend_data: Dict, relative_data: Dict
    ) -> str:
        """시계열 분석"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다. 시계열 데이터를 분석하여 간결한 인사이트를 제공합니다."

        prompt = f"'{category}' 카테고리 시계열 분석:\n"
        prompt += f"- 기업명: {company_name}\n- 업종: {industry}\n\n"

        for metric in TARGET_METRICS.get(category, []):
            values = metrics_data.get(metric, [])
            trend = trend_data.get(metric, {})
            rel = relative_data.get(metric)

            prompt += f"### {metric}\n"
            if values:
                prompt += f"- 추이: {' → '.join([f'{v:.1f}%' if v else 'N/A' for v in values[-4:]])}\n"
            if trend.get('yoy') is not None:
                prompt += f"- YoY: {trend['yoy']:+.1f}%\n"
            if rel is not None:
                prompt += f"- 업종대비: {rel:+.1f}%p\n"

        prompt += "\n3~4문장으로 분석해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=500)

    async def _generate_industry_comparison(
        self, company_name: str, industry: str,
        category_positions: Dict, comparison_data: Dict,
        strengths: List, weaknesses: List
    ) -> str:
        """업종 비교 종합"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다."

        prompt = f"업종 내 위치 분석:\n- 기업명: {company_name}\n- 업종: {industry}\n\n"
        prompt += "## 카테고리별 순위\n"
        for cat, pos in category_positions.items():
            if pos:
                prompt += f"- {cat}: {'상위' if pos <= 50 else '하위'} {pos:.0f}%\n"

        if strengths:
            prompt += f"\n## 강점: {', '.join([s['metric'] for s in strengths[:3]])}\n"
        if weaknesses:
            prompt += f"## 약점: {', '.join([w['metric'] for w in weaknesses[:3]])}\n"

        prompt += "\n4~5문장으로 분석해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=600)

    async def _generate_category_industry(
        self, company_name: str, industry: str, category: str,
        metrics_comparison: Dict, position: float
    ) -> str:
        """카테고리별 업종 비교"""
        system_prompt = "당신은 한국 상장기업 재무 분석 전문가입니다."

        rank = f"상위 {position:.0f}%" if position <= 50 else f"하위 {100-position:.0f}%"
        prompt = f"'{category}' 업종 비교:\n- 기업: {company_name}\n- 순위: {rank}\n\n"

        for metric, info in metrics_comparison.items():
            if info and info.get('company') is not None:
                prompt += f"- {metric}: {info['company']:.1f}% (평균 {info.get('industry_mean', 0):.1f}%)\n"

        prompt += "\n2~3문장으로 분석해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=300)

    # =========================================================================
    # PDF 렌더링
    # =========================================================================

    async def _render_pdf(
        self,
        company_code: str,
        output_dir: str,
        report_data: Dict,
        company_data: Dict,
        industry_comparison: Dict,
        llm_results: Dict
    ) -> str:
        """PDF 렌더링 (스레드풀에서 실행)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._render_pdf_sync,
            company_code, output_dir, report_data,
            company_data, industry_comparison, llm_results
        )

    def _render_pdf_sync(
        self,
        company_code: str,
        output_dir: str,
        report_data: Dict,
        company_data: Dict,
        industry_comparison: Dict,
        llm_results: Dict
    ) -> str:
        """PDF 렌더링 (동기, 스레드풀에서 호출됨)"""
        from app.report import llm_opinion
        from app.report import pdf_generator as pdf_gen_module

        # pdf_generator 모듈에서 직접 사용하는 함수들을 저장
        original_funcs = {
            'generate_opinion': pdf_gen_module.generate_opinion,
            'generate_category_analysis': pdf_gen_module.generate_category_analysis,
            'generate_timeseries_analysis': pdf_gen_module.generate_timeseries_analysis,
            'generate_industry_comparison_analysis': pdf_gen_module.generate_industry_comparison_analysis,
            'generate_category_industry_analysis': pdf_gen_module.generate_category_industry_analysis,
        }

        try:
            # 캐시 버전 함수 정의
            def cached_opinion(*args, **kwargs):
                result = llm_results.get('opinion')
                if result:
                    logger.debug("Using cached opinion")
                    return result
                logger.debug("Cache miss: opinion")
                return original_funcs['generate_opinion'](*args, **kwargs)

            def cached_category_analysis(*args, **kwargs):
                result = self._merge_category_results(llm_results)
                if result:
                    logger.debug("Using cached category_analysis")
                    return result
                logger.debug("Cache miss: category_analysis")
                return original_funcs['generate_category_analysis'](*args, **kwargs)

            def cached_timeseries(company_name, industry, category, *args, **kwargs):
                cached = llm_results.get(f'timeseries_{category}')
                if cached:
                    logger.debug(f"Using cached timeseries_{category}")
                    return cached
                logger.debug(f"Cache miss: timeseries_{category}")
                return original_funcs['generate_timeseries_analysis'](company_name, industry, category, *args, **kwargs)

            def cached_industry_comp(*args, **kwargs):
                cached = llm_results.get('industry_comparison')
                if cached:
                    logger.debug("Using cached industry_comparison")
                    return cached
                logger.debug("Cache miss: industry_comparison")
                return original_funcs['generate_industry_comparison_analysis'](*args, **kwargs)

            def cached_category_industry(company_name, industry, category, *args, **kwargs):
                cached = llm_results.get(f'category_industry_{category}')
                if cached:
                    logger.debug(f"Using cached category_industry_{category}")
                    return cached
                logger.debug(f"Cache miss: category_industry_{category}")
                return original_funcs['generate_category_industry_analysis'](company_name, industry, category, *args, **kwargs)

            # pdf_generator 모듈의 함수들을 캐시 버전으로 교체
            pdf_gen_module.generate_opinion = cached_opinion
            pdf_gen_module.generate_category_analysis = cached_category_analysis
            pdf_gen_module.generate_timeseries_analysis = cached_timeseries
            pdf_gen_module.generate_industry_comparison_analysis = cached_industry_comp
            pdf_gen_module.generate_category_industry_analysis = cached_category_industry

            # PDF 생성
            pdf_path = pdf_gen_module.generate_pdf_report(company_code, output_dir)

            return pdf_path

        finally:
            # 원본 함수 복원
            pdf_gen_module.generate_opinion = original_funcs['generate_opinion']
            pdf_gen_module.generate_category_analysis = original_funcs['generate_category_analysis']
            pdf_gen_module.generate_timeseries_analysis = original_funcs['generate_timeseries_analysis']
            pdf_gen_module.generate_industry_comparison_analysis = original_funcs['generate_industry_comparison_analysis']
            pdf_gen_module.generate_category_industry_analysis = original_funcs['generate_category_industry_analysis']

    def _merge_category_results(self, llm_results: Dict) -> Optional[Dict]:
        """카테고리별 분석 결과 병합"""
        merged = {}
        for category in TARGET_METRICS.keys():
            key = f'category_{category}'
            if key in llm_results and llm_results[key]:
                merged[category] = llm_results[key]

        return merged if merged else None


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================
_pdf_service: Optional[AsyncPDFService] = None


def get_pdf_service() -> AsyncPDFService:
    """AsyncPDFService 싱글톤 인스턴스"""
    global _pdf_service
    if _pdf_service is None:
        _pdf_service = AsyncPDFService()
    return _pdf_service


async def generate_pdf_async(company_code: str, output_dir: str) -> str:
    """비동기 PDF 생성 편의 함수"""
    service = get_pdf_service()
    return await service.generate(company_code, output_dir)
