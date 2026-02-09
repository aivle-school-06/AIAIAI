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
# 상수 (config에서 import)
# =============================================================================
from app.report.config import TARGET_METRICS, METRIC_DESCRIPTION, METRIC_DIRECTION


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

        # 1. 종합 의견 (기존 형식 - 폴백용)
        tasks['opinion'] = self._generate_opinion(opinion_data)

        # 1-1. 통합 종합의견 (새 내러티브 구조 - 메인)
        tasks['unified_opinion'] = self._generate_unified_opinion(opinion_data)

        # 2. 카테고리별 분석 (5개)
        for category, metrics in TARGET_METRICS.items():
            cat_predictions = {m: all_predictions.get(m) for m in metrics if m in all_predictions}
            if cat_predictions:
                tasks[f'category_{category}'] = self._generate_category_analysis(
                    category, metrics, opinion_data, cat_predictions
                )

        # 3. 시계열 분석 (5개) - SHAP 데이터 추가
        historical = company_data.get('historical', {}) if company_data else {}
        trend_data = company_data.get('trend', {}) if company_data else {}
        relative_data = company_data.get('relative', {}) if company_data else {}

        for category, metrics in TARGET_METRICS.items():
            metrics_data = {m: historical.get('metrics', {}).get(m, []) for m in metrics}
            # 해당 카테고리 지표들의 SHAP 데이터 추출
            category_shap = {m: shap_data.get(m) for m in metrics if shap_data.get(m)}
            if any(metrics_data.values()):
                tasks[f'timeseries_{category}'] = self._generate_timeseries(
                    company_name, industry, category,
                    metrics_data, trend_data, relative_data, category_shap
                )

        # 4. 업종 비교 종합 - SHAP 데이터 추가
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
                comparison_data, strengths, weaknesses, shap_data
            )

            # 5. 카테고리별 업종 비교 (5개) - SHAP 데이터 추가
            for category, metrics in TARGET_METRICS.items():
                cat_metrics = {m: comparison_data.get(m) for m in metrics if comparison_data.get(m)}
                # 해당 카테고리 지표들의 SHAP 데이터 추출
                category_shap = {m: shap_data.get(m) for m in metrics if shap_data.get(m)}
                position = category_positions.get(category)
                if cat_metrics and position:
                    tasks[f'category_industry_{category}'] = self._generate_category_industry(
                        company_name, industry, category, cat_metrics, position, category_shap
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
특히 SHAP 분석 요인을 활용하여 'AI가 왜 그런 예측을 했는지' 구체적으로 설명해주세요.
예: "AI가 ROA 상승을 예측한 주요 이유는 최근 4분기 ROA 추세(+0.70)가 긍정적으로..."

응답 형식:
[전문가용]
(금융 전문가를 위한 상세 분석. SHAP 요인을 활용하여 AI 예측 근거 설명. 5~7문장)

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

        # SHAP 분석 요약 추가
        shap_analysis = data.get('shap_analysis', {})
        if shap_analysis:
            # 주요 지표의 SHAP 요인 요약
            main_metrics = ['ROA', 'ROE', '매출액영업이익률']
            prompt += "\n## AI 예측 핵심 요인 (SHAP 분석)\n"
            for metric in main_metrics:
                if metric in shap_analysis:
                    shap = shap_analysis[metric]
                    pos = shap.get('positive_factors', [])[:5]
                    neg = shap.get('negative_factors', [])[:5]
                    if pos or neg:
                        prompt += f"\n### {metric}\n"
                        if pos:
                            factors = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in pos])
                            prompt += f"- 상승 요인: {factors}\n"
                        if neg:
                            factors = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in neg])
                            prompt += f"- 하락 요인: {factors}\n"

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

    async def _generate_unified_opinion(self, opinion_data: Dict) -> Dict:
        """통합 종합의견 생성 (내러티브 구조)"""
        system_prompt = """당신은 한국 상장기업 재무 분석 리포트 작성 전문가입니다.

주어진 데이터를 바탕으로 하나의 통합된 분석 리포트를 작성합니다.

**작성 원칙:**
- 전문가와 일반인 모두 이해할 수 있는 명확한 톤
- 숫자는 반드시 맥락과 함께 제시 (예: "ROA 5.2%로 업종 상위 20%")
- 단순 나열이 아닌 인과관계와 스토리로 연결
- 구체적이고 actionable한 인사이트 제공

**응답 형식 (반드시 준수):**

[한줄요약]
기업의 현재 재무 상태와 핵심 포인트를 한 문장으로 요약합니다.

[현황분석]
현재 재무 상태를 분석합니다. 3~4문장으로 작성합니다.

[추세분석]
최근 추세를 분석합니다. 2~3문장으로 작성합니다.

[AI전망]
AI 예측 기반 전망을 설명합니다. SHAP 요인을 활용하여 왜 그런 예측인지 설명합니다. 3~4문장.

[주시포인트]
• 포인트1
• 포인트2
• 포인트3

[결론]
종합 결론을 1~2문장으로 작성합니다.
"""

        user_prompt = self._build_unified_prompt(opinion_data)
        content = await self.llm.call(system_prompt, user_prompt, max_tokens=2500)
        return self._parse_unified_response(content, opinion_data)

    def _build_unified_prompt(self, data: Dict) -> str:
        """통합 종합의견 프롬프트 생성"""
        company = data.get('company', {})
        grades = data.get('grades', {})
        current = data.get('current_metrics', {})
        prediction = data.get('prediction', {})
        industry_position = data.get('industry_position', {})
        time_analysis = data.get('time_analysis', {})

        prompt = f"""기업 재무 상태 분석:

## 기업 정보
- 기업명: {company.get('name', 'N/A')}
- 업종: {company.get('industry', 'N/A')}
- 기준일: {company.get('period', 'N/A')}

## 종합 등급: {grades.get('overall', 'N/A')} (점수: {grades.get('score', 'N/A')})
- 강점: {', '.join(grades.get('strengths', [])) or '없음'}
- 약점: {', '.join(grades.get('weaknesses', [])) or '없음'}

## 업종 내 위치
- 종합 백분위: 상위 {industry_position.get('overall_percentile', 50):.0f}%
"""
        # 카테고리별 백분위
        cat_percentiles = industry_position.get('category_percentiles', {})
        for cat, pct in cat_percentiles.items():
            if pct is not None:
                prompt += f"- {cat}: 상위 {pct:.0f}%\n"

        prompt += "\n## 주요 지표\n"
        for metric in ['ROA', 'ROE', '매출액영업이익률', '부채비율', '유동비율']:
            value = current.get(metric)
            if value is not None:
                prompt += f"- {metric}: {value:.2f}%\n"

        prompt += f"""
## 시계열 분석 요약
{time_analysis.get('summary', '데이터 없음')}

## AI 예측
- 전망: {prediction.get('outlook', 'N/A')}
- 개선 예측: {', '.join(prediction.get('improving', [])) or '없음'}
- 악화 예측: {', '.join(prediction.get('declining', [])) or '없음'}
"""

        # SHAP 분석 요약 추가
        shap_analysis = data.get('shap_analysis', {})
        if shap_analysis:
            main_metrics = ['ROA', 'ROE', '매출액영업이익률']
            prompt += "\n## AI 예측 핵심 요인 (SHAP 분석)\n"
            for metric in main_metrics:
                if metric in shap_analysis:
                    shap = shap_analysis[metric]
                    pos = shap.get('positive_factors', [])[:3]
                    neg = shap.get('negative_factors', [])[:3]
                    if pos or neg:
                        prompt += f"\n### {metric}\n"
                        if pos:
                            factors = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in pos])
                            prompt += f"- 상승 요인: {factors}\n"
                        if neg:
                            factors = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in neg])
                            prompt += f"- 하락 요인: {factors}\n"

        return prompt

    def _parse_unified_response(self, content: str, opinion_data: Dict) -> Dict:
        """통합 종합의견 응답 파싱"""
        result = {
            'headline': '',
            'analysis': '',
            'trend': '',
            'forecast': '',
            'watch_points': [],
            'conclusion': '',
            'raw_response': content
        }

        sections = content.split('[')
        for section in sections:
            if section.startswith('한줄요약]'):
                result['headline'] = section.replace('한줄요약]', '').strip().split('\n\n')[0]
            elif section.startswith('현황분석]'):
                result['analysis'] = section.replace('현황분석]', '').strip().split('\n\n')[0]
            elif section.startswith('추세분석]'):
                result['trend'] = section.replace('추세분석]', '').strip().split('\n\n')[0]
            elif section.startswith('AI전망]'):
                result['forecast'] = section.replace('AI전망]', '').strip().split('\n\n')[0]
            elif section.startswith('주시포인트]'):
                text = section.replace('주시포인트]', '').strip()
                result['watch_points'] = [
                    line.lstrip('•-').strip()
                    for line in text.split('\n')
                    if line.strip().startswith(('•', '-'))
                ][:5]
            elif section.startswith('결론]'):
                result['conclusion'] = section.replace('결론]', '').strip().split('\n\n')[0]

        # Fallback
        if not result['headline']:
            grades = opinion_data.get('grades', {})
            company = opinion_data.get('company', {})
            result['headline'] = f"{company.get('name', '기업')}은 종합 {grades.get('overall', 'N/A')} 등급입니다."

        return result

    async def _generate_category_analysis(
        self, category: str, metrics: List[str],
        opinion_data: Dict, predictions: Dict
    ) -> Dict:
        """카테고리별 분석"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
AI 예측 분석 시, SHAP 요인을 바탕으로 '왜' 그런 예측이 나왔는지 구체적으로 설명해주세요.
예: "ROA 상승 예측의 주요 요인은 최근 4분기 ROA 추세(+0.70)가 긍정적으로 반영되었으며..."

응답 형식:
[카테고리 종합]
(3~4문장 분석 - SHAP 요인을 활용하여 AI 예측 근거 설명)

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

                    # SHAP 분석 추가 (top/bottom 5개)
                    shap = pred.get('shap_analysis', {})
                    if shap:
                        pos_factors = shap.get('positive_factors', [])[:5]
                        neg_factors = shap.get('negative_factors', [])[:5]

                        if pos_factors:
                            prompt += "- AI 예측 상승 요인:\n"
                            for f in pos_factors:
                                desc = f.get('description', f.get('feature', ''))
                                val = f.get('shap_value', 0)
                                prompt += f"  • {desc} ({val:+.2f})\n"

                        if neg_factors:
                            prompt += "- AI 예측 하락 요인:\n"
                            for f in neg_factors:
                                desc = f.get('description', f.get('feature', ''))
                                val = f.get('shap_value', 0)
                                prompt += f"  • {desc} ({val:+.2f})\n"

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

        # metrics 기본값 (SHAP top/bottom 5개)
        for metric in metrics:
            pred = predictions.get(metric, {})
            shap = pred.get('shap_analysis', {}) if pred else {}
            result['metrics'][metric] = {
                'insight': '',
                'factors': {
                    'positive': shap.get('positive_factors', [])[:5],
                    'negative': shap.get('negative_factors', [])[:5],
                }
            }

        return result

    async def _generate_timeseries(
        self, company_name: str, industry: str, category: str,
        metrics_data: Dict, trend_data: Dict, relative_data: Dict,
        shap_data: Dict = None
    ) -> str:
        """시계열 분석 (SHAP 요인 포함)"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
시계열 데이터와 AI가 분석한 주요 영향 요인을 종합하여 인사이트를 제공합니다.
데이터 추세와 AI 분석 요인을 연결하여 설명해주세요."""

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

            # SHAP 요인 추가 (설명과 값 포함, top/bottom 5개)
            if shap_data and metric in shap_data:
                shap_info = shap_data[metric]
                pos_factors = shap_info.get('positive_factors', [])[:5]
                neg_factors = shap_info.get('negative_factors', [])[:5]
                if pos_factors:
                    factors_str = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in pos_factors])
                    prompt += f"- AI 상승요인: {factors_str}\n"
                if neg_factors:
                    factors_str = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in neg_factors])
                    prompt += f"- AI 하락요인: {factors_str}\n"

        prompt += "\n위 데이터와 AI 분석 요인(SHAP 값)을 종합하여 '왜' 그런 예측이 나왔는지 3~4문장으로 설명해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=600)

    async def _generate_industry_comparison(
        self, company_name: str, industry: str,
        category_positions: Dict, comparison_data: Dict,
        strengths: List, weaknesses: List, shap_data: Dict = None
    ) -> str:
        """업종 비교 종합 (SHAP 요인 포함)"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
업종 내 위치와 AI가 분석한 주요 영향 요인을 종합하여 인사이트를 제공합니다."""

        prompt = f"업종 내 위치 분석:\n- 기업명: {company_name}\n- 업종: {industry}\n\n"
        prompt += "## 카테고리별 순위\n"
        for cat, pos in category_positions.items():
            if pos:
                prompt += f"- {cat}: {'상위' if pos <= 50 else '하위'} {pos:.0f}%\n"

        if strengths:
            prompt += f"\n## 강점: {', '.join([s['metric'] for s in strengths[:3]])}\n"
        if weaknesses:
            prompt += f"## 약점: {', '.join([w['metric'] for w in weaknesses[:3]])}\n"

        # SHAP 요인 추가 - 전체 요약 (설명과 값 포함)
        if shap_data:
            all_pos_factors = []
            all_neg_factors = []
            for metric, shap_info in shap_data.items():
                pos = shap_info.get('positive_factors', [])[:1]
                neg = shap_info.get('negative_factors', [])[:1]
                all_pos_factors.extend([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in pos])
                all_neg_factors.extend([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in neg])

            if all_pos_factors:
                prompt += f"\n## AI 분석 - 주요 상승 요인\n{', '.join(all_pos_factors[:5])}\n"
            if all_neg_factors:
                prompt += f"\n## AI 분석 - 주요 하락 요인\n{', '.join(all_neg_factors[:5])}\n"

        prompt += "\n위 데이터와 AI 분석 요인(SHAP 값)을 종합하여 '왜' 그런 예측이 나왔는지 4~5문장으로 설명해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=700)

    async def _generate_category_industry(
        self, company_name: str, industry: str, category: str,
        metrics_comparison: Dict, position: float, shap_data: Dict = None
    ) -> str:
        """카테고리별 업종 비교 (SHAP 요인 포함)"""
        system_prompt = """당신은 한국 상장기업 재무 분석 전문가입니다.
업종 비교 데이터와 AI가 분석한 영향 요인을 연결하여 설명해주세요."""

        rank = f"상위 {position:.0f}%" if position <= 50 else f"하위 {100-position:.0f}%"
        prompt = f"'{category}' 업종 비교:\n- 기업: {company_name}\n- 순위: {rank}\n\n"

        for metric, info in metrics_comparison.items():
            if info and info.get('company') is not None:
                prompt += f"### {metric}\n"
                prompt += f"- 기업값: {info['company']:.1f}% (업종평균 {info.get('industry_mean', 0):.1f}%)\n"
                prompt += f"- 백분위: {info.get('percentile', 50):.0f}%\n"

                # 해당 지표의 SHAP 요인 추가 (설명과 값 포함, top/bottom 5개)
                if shap_data and metric in shap_data:
                    shap_info = shap_data[metric]
                    pos_factors = shap_info.get('positive_factors', [])[:5]
                    neg_factors = shap_info.get('negative_factors', [])[:5]
                    if pos_factors:
                        factors_str = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in pos_factors])
                        prompt += f"- AI 상승요인: {factors_str}\n"
                    if neg_factors:
                        factors_str = ', '.join([f"{f.get('description', f['feature'])}({f.get('shap_value', 0):+.2f})" for f in neg_factors])
                        prompt += f"- AI 하락요인: {factors_str}\n"

        prompt += "\n위 데이터와 AI 분석 요인(SHAP 값)을 종합하여 '왜' 그런 예측이 나왔는지 2~3문장으로 설명해주세요."

        return await self.llm.call(system_prompt, prompt, max_tokens=400)

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
            'generate_unified_opinion': pdf_gen_module.generate_unified_opinion,
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

            def cached_unified_opinion(*args, **kwargs):
                result = llm_results.get('unified_opinion')
                if result:
                    logger.debug("Using cached unified_opinion")
                    return result
                logger.debug("Cache miss: unified_opinion")
                return original_funcs['generate_unified_opinion'](*args, **kwargs)

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
            pdf_gen_module.generate_unified_opinion = cached_unified_opinion
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
            pdf_gen_module.generate_unified_opinion = original_funcs['generate_unified_opinion']
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
