"""
등급 산정 모듈
==============

이 모듈은 기업의 재무 지표를 업종 내 백분위로 계산하고,
이를 기반으로 등급을 산정하는 기능을 제공합니다.

등급 산정 방식:
1. 각 지표별로 업종 내 백분위 계산 (상위 몇 %인지)
2. 백분위를 기준으로 등급 부여 (A+ ~ F)
3. 카테고리별 평균 점수로 카테고리 등급 산출
4. 전체 평균 점수로 종합 등급 산출

등급 기준 (백분위 기반):
- A+: 상위 10% 이내 (매우 우수)
- A : 상위 25% 이내 (우수)
- B+: 상위 40% 이내 (양호)
- B : 상위 60% 이내 (보통)
- C : 상위 80% 이내 (주의)
- F : 하위 20% (위험)

사용 예시:
    calculator = get_grade_calculator()
    grades = calculator.calculate_all_grades('[005930]')
    print(grades['overall']['grade'])  # 예: 'B+'
"""
from typing import Dict, Optional, List
from .config import (
    GRADE_THRESHOLDS,   # 등급별 백분위 기준 (A+: 10, A: 25, ...)
    GRADE_SCORES,       # 등급별 점수 (A+: 6, A: 5, ...)
    ALL_TARGETS,        # 전체 타겟 지표 13개
    TARGET_METRICS,     # 카테고리별 타겟 지표
    METRIC_DIRECTION    # 지표별 방향성 (higher/lower)
)
from .data_loader import get_data_loader


class GradeCalculator:
    """
    등급 산정 클래스

    업종 내 백분위를 기준으로 개별 지표, 카테고리, 종합 등급을 계산합니다.

    등급 체계:
    - 개별 지표 등급: 13개 지표 각각에 대해 등급 부여
    - 카테고리 등급: 5개 카테고리별 평균 등급
    - 종합 등급: 전체 지표 평균 등급

    이 클래스는 싱글톤으로 사용됩니다.
    get_grade_calculator() 함수를 통해 인스턴스를 가져옵니다.
    """

    def __init__(self):
        """
        초기화

        DataLoader 인스턴스를 가져옵니다.
        (DataLoader를 통해 백분위 계산 수행)
        """
        self.data_loader = get_data_loader()

    def calculate_all_grades(self, company_code: str) -> Dict:
        """
        모든 등급 계산 (메인 함수)

        특정 기업에 대해 개별 지표, 카테고리, 종합 등급을 모두 계산합니다.

        처리 흐름:
        1. 13개 개별 지표에 대해 백분위 계산 → 등급 변환
        2. 5개 카테고리별로 소속 지표들의 평균 점수 → 카테고리 등급
        3. 전체 지표 평균 점수 → 종합 등급

        Args:
            company_code: 기업 코드 (예: '[005930]')

        Returns:
            Dict: 등급 정보 딕셔너리
                {
                    'overall': {                    # 종합 등급
                        'grade': 'B+',
                        'score': 3.85,
                        'valid_count': 12,          # 계산 가능한 지표 수
                        'total_count': 13           # 전체 지표 수
                    },
                    'categories': {                 # 카테고리별 등급
                        '수익성': {
                            'grade': 'A',
                            'score': 4.67,
                            'metrics': {...},       # 소속 지표 상세
                            'valid_count': 3,
                            'total_count': 3
                        },
                        ...
                    },
                    'metrics': {                    # 개별 지표 등급
                        'ROA': {
                            'grade': 'A',
                            'percentile': 18.5,     # 상위 18.5%
                            'score': 5
                        },
                        ...
                    }
                }
        """
        metric_grades = {}      # 개별 지표 등급 저장
        category_grades = {}    # 카테고리별 등급 저장

        # =================================================================
        # 1단계: 개별 지표 등급 계산
        # =================================================================
        # 13개 지표 각각에 대해 업종 내 백분위를 계산하고 등급으로 변환
        for metric in ALL_TARGETS:
            # 백분위 계산 (data_loader 활용)
            # 예: ROA가 상위 18.5%라면 percentile = 18.5
            percentile = self.data_loader.calculate_percentile(company_code, metric)

            if percentile is not None:
                # 백분위를 등급으로 변환
                # 예: 18.5% → A (상위 25% 이내)
                grade = self._percentile_to_grade(percentile)

                metric_grades[metric] = {
                    'grade': grade,
                    'percentile': percentile,
                    'score': GRADE_SCORES[grade],  # 등급을 점수로 변환 (평균 계산용)
                }
            else:
                # 백분위 계산 불가 (데이터 없음)
                metric_grades[metric] = {
                    'grade': 'N/A',
                    'percentile': None,
                    'score': None,
                }

        # =================================================================
        # 2단계: 카테고리별 등급 계산
        # =================================================================
        # 5개 카테고리 (수익성, 안정성, 차입금, 유동성, 현금흐름) 각각에 대해
        # 소속 지표들의 평균 점수를 계산하여 등급 산출
        for category, metrics in TARGET_METRICS.items():
            cat_scores = []      # 해당 카테고리 지표들의 점수 리스트
            cat_metrics = {}     # 해당 카테고리 지표들의 상세 정보

            # 카테고리에 속한 지표들 순회
            for metric in metrics:
                # 점수가 있는 지표만 평균 계산에 포함
                if metric_grades[metric]['score'] is not None:
                    cat_scores.append(metric_grades[metric]['score'])
                cat_metrics[metric] = metric_grades[metric]

            # 카테고리 등급 계산
            if cat_scores:
                # 소속 지표들의 평균 점수
                avg_score = sum(cat_scores) / len(cat_scores)
                # 점수를 등급으로 변환
                cat_grade = self._score_to_grade(avg_score)
                valid_count = len(cat_scores)
            else:
                # 모든 지표가 N/A인 경우
                avg_score = None
                cat_grade = 'N/A'
                valid_count = 0

            category_grades[category] = {
                'grade': cat_grade,
                'score': avg_score,
                'metrics': cat_metrics,
                'valid_count': valid_count,         # 계산된 지표 수
                'total_count': len(metrics),        # 전체 지표 수
            }

        # =================================================================
        # 3단계: 종합 등급 계산
        # =================================================================
        # 13개 지표 전체의 평균 점수로 종합 등급 산출
        all_scores = [g['score'] for g in metric_grades.values() if g['score'] is not None]

        if all_scores:
            overall_score = sum(all_scores) / len(all_scores)
            overall_grade = self._score_to_grade(overall_score)
            valid_count = len(all_scores)
        else:
            overall_score = None
            overall_grade = 'N/A'
            valid_count = 0

        return {
            'overall': {
                'grade': overall_grade,
                'score': overall_score,
                'valid_count': valid_count,
                'total_count': len(ALL_TARGETS),
            },
            'categories': category_grades,
            'metrics': metric_grades,
        }

    def _percentile_to_grade(self, percentile: float) -> str:
        """
        백분위를 등급으로 변환 (내부 메서드)

        GRADE_THRESHOLDS 설정에 따라 등급 결정:
        - 상위 10% 이내 → A+
        - 상위 25% 이내 → A
        - 상위 40% 이내 → B+
        - 상위 60% 이내 → B
        - 상위 80% 이내 → C
        - 그 외 → F

        Args:
            percentile: 상위 백분위 (예: 18.5 = 상위 18.5%)

        Returns:
            str: 등급 ('A+', 'A', 'B+', 'B', 'C', 'F')

        예시:
            _percentile_to_grade(5.0)  → 'A+' (상위 5%)
            _percentile_to_grade(18.5) → 'A'  (상위 18.5%)
            _percentile_to_grade(35.0) → 'B+' (상위 35%)
            _percentile_to_grade(85.0) → 'F'  (상위 85% = 하위 15%)
        """
        # GRADE_THRESHOLDS = {'A+': 10, 'A': 25, 'B+': 40, 'B': 60, 'C': 80, 'F': 100}
        for grade, threshold in GRADE_THRESHOLDS.items():
            if percentile <= threshold:
                return grade
        return 'F'

    def _score_to_grade(self, score: float) -> str:
        """
        점수를 등급으로 변환 (내부 메서드)

        카테고리/종합 등급 계산 시 사용
        여러 지표의 점수를 평균낸 후 등급으로 변환

        점수 기준:
        - 5.5점 이상: A+ (거의 모든 지표가 A+ 수준)
        - 4.5점 이상: A
        - 3.5점 이상: B+
        - 2.5점 이상: B
        - 1.5점 이상: C
        - 1.5점 미만: F

        Args:
            score: 평균 점수 (1~6 범위)

        Returns:
            str: 등급 ('A+', 'A', 'B+', 'B', 'C', 'F')

        예시:
            _score_to_grade(4.67) → 'A'
            _score_to_grade(3.85) → 'B+'
            _score_to_grade(2.33) → 'C'
        """
        if score >= 5.5:
            return 'A+'
        elif score >= 4.5:
            return 'A'
        elif score >= 3.5:
            return 'B+'
        elif score >= 2.5:
            return 'B'
        elif score >= 1.5:
            return 'C'
        else:
            return 'F'

    def get_grade_summary(self, grades: Dict) -> Dict:
        """
        등급 요약 정보 생성

        calculate_all_grades()의 결과를 기반으로
        종합 등급, 강점, 약점 등을 요약합니다.

        Args:
            grades: calculate_all_grades()의 반환값

        Returns:
            Dict: 등급 요약 정보
                {
                    'overall_grade': 'B+',
                    'overall_score': 3.85,
                    'strengths': [                  # 상위 25% 이내 지표들 (강점)
                        {'metric': 'ROA', 'grade': 'A', 'percentile': 18.5},
                        ...
                    ],
                    'weaknesses': [                 # 하위 25% 이내 지표들 (약점)
                        {'metric': '부채비율', 'grade': 'F', 'percentile': 92.3},
                        ...
                    ],
                    'grade_distribution': {         # 등급 분포
                        'A+': 1, 'A': 3, 'B+': 4, 'B': 2, 'C': 2, 'F': 1, 'N/A': 0
                    }
                }
        """
        overall = grades['overall']

        # =================================================================
        # 강점/약점 분석
        # =================================================================
        # 강점: 상위 25% 이내 (백분위 ≤ 25) 지표
        # 약점: 하위 25% 이내 (백분위 ≥ 75) 지표
        strengths = []
        weaknesses = []

        for metric, grade_info in grades['metrics'].items():
            # 백분위가 없는 지표는 스킵
            if grade_info['percentile'] is None:
                continue

            if grade_info['percentile'] <= 25:  # 상위 25% → 강점
                strengths.append({
                    'metric': metric,
                    'grade': grade_info['grade'],
                    'percentile': grade_info['percentile'],
                })
            elif grade_info['percentile'] >= 75:  # 하위 25% → 약점
                weaknesses.append({
                    'metric': metric,
                    'grade': grade_info['grade'],
                    'percentile': grade_info['percentile'],
                })

        # 정렬: 강점은 백분위 낮은 순 (더 우수한 순), 약점은 높은 순 (더 취약한 순)
        strengths.sort(key=lambda x: x['percentile'])
        weaknesses.sort(key=lambda x: x['percentile'], reverse=True)

        return {
            'overall_grade': overall['grade'],
            'overall_score': overall['score'],
            'strengths': strengths[:5],    # 상위 5개 강점만 반환
            'weaknesses': weaknesses[:5],  # 상위 5개 약점만 반환
            'grade_distribution': self._get_grade_distribution(grades['metrics']),
        }

    def _get_grade_distribution(self, metric_grades: Dict) -> Dict:
        """
        등급 분포 계산 (내부 메서드)

        13개 지표가 각 등급에 몇 개씩 분포하는지 계산

        Args:
            metric_grades: 개별 지표 등급 딕셔너리

        Returns:
            Dict: 등급별 지표 수
                {'A+': 1, 'A': 3, 'B+': 4, 'B': 2, 'C': 2, 'F': 1, 'N/A': 0}

        이 정보는 시각화에 활용 가능 (등급 분포 차트 등)
        """
        # 모든 등급을 0으로 초기화
        distribution = {'A+': 0, 'A': 0, 'B+': 0, 'B': 0, 'C': 0, 'F': 0, 'N/A': 0}

        # 각 지표의 등급을 카운트
        for grade_info in metric_grades.values():
            grade = grade_info['grade']
            if grade in distribution:
                distribution[grade] += 1

        return distribution


# =============================================================================
# 싱글톤 패턴
# =============================================================================
# 전역 인스턴스 (처음에는 None)
_calculator: Optional[GradeCalculator] = None


def get_grade_calculator() -> GradeCalculator:
    """
    GradeCalculator 싱글톤 인스턴스 반환

    처음 호출 시 인스턴스 생성, 이후에는 기존 인스턴스 재사용

    Returns:
        GradeCalculator: 등급 계산기 인스턴스

    사용 예시:
        calculator = get_grade_calculator()
        grades = calculator.calculate_all_grades('[005930]')
    """
    global _calculator

    if _calculator is None:
        _calculator = GradeCalculator()

    return _calculator
