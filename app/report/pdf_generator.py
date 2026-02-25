"""
PDF 보고서 생성 모듈 (Premium Design + Visual Analytics)
========================================================

시각적 분석 + XAI + LLM 통합 보고서
"""
import os
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Tuple
from fpdf import FPDF

from .report_generator import generate_report
from .llm_opinion import generate_opinion, generate_unified_opinion, generate_category_analysis, get_llm_generator, generate_timeseries_analysis, generate_industry_comparison_analysis, generate_category_industry_analysis
from .data_loader import get_data_loader
from .config import TARGET_METRICS, ALL_TARGETS, METRIC_DIRECTION, get_feature_description, METRIC_DESCRIPTION

# 로고 이미지 경로 (ai-server/img/)
LOGO_PATH = Path(__file__).parent.parent.parent / 'img'
OWL_LOGO = LOGO_PATH / 'owllogo.png'
LOGO_NO_BG = LOGO_PATH / 'logonobg.png'


# 디자인 컬러 팔레트
class Colors:
    # 섹션 헤더 색상 (제목 박스, 가로선)
    SECTION = (0, 0, 0)              # 검정색
    SECTION_LINE = (0, 0, 0)         # 검정색

    PRIMARY = (41, 128, 185)
    PRIMARY_DARK = (31, 97, 141)
    PRIMARY_LIGHT = (174, 214, 241)
    SUCCESS = (39, 174, 96)
    DANGER = (192, 57, 43)
    WARNING = (243, 156, 18)
    DARK = (44, 62, 80)
    TEXT = (52, 73, 94)
    MUTED = (127, 140, 141)
    LIGHT = (236, 240, 241)
    WHITE = (255, 255, 255)

    # Grade Colors
    GRADE_A_PLUS = (26, 188, 156)
    GRADE_A = (46, 204, 113)
    GRADE_B_PLUS = (52, 152, 219)
    GRADE_B = (155, 89, 182)
    GRADE_C = (243, 156, 18)
    GRADE_F = (231, 76, 60)

    # Category Colors (카테고리별 고유 색상 - 긍정/부정/중립 색상과 구분)
    # SUCCESS(초록), DANGER(빨강), WARNING(주황), MUTED(회색)과 겹치지 않도록 설정
    CAT_PROFITABILITY = (41, 128, 185)     # 수익성 - 코발트 블루
    CAT_STABILITY = (22, 160, 133)          # 안정성 - 틸/청록
    CAT_DEBT = (142, 68, 173)               # 차입금 - 자주/마젠타
    CAT_LIQUIDITY = (52, 73, 94)            # 유동성 - 네이비/남색
    CAT_CASHFLOW = (211, 84, 0)             # 현금흐름 - 호박색/갈색

    # Category Light Colors (연한 버전)
    CAT_PROFITABILITY_LIGHT = (214, 234, 248)  # 연한 파랑
    CAT_STABILITY_LIGHT = (209, 242, 235)      # 연한 청록
    CAT_DEBT_LIGHT = (235, 222, 240)           # 연한 자주
    CAT_LIQUIDITY_LIGHT = (214, 219, 223)      # 연한 네이비
    CAT_CASHFLOW_LIGHT = (250, 229, 211)       # 연한 호박색


# 카테고리 색상 매핑
CATEGORY_COLORS = {
    '수익성': (Colors.CAT_PROFITABILITY, Colors.CAT_PROFITABILITY_LIGHT),
    '안정성': (Colors.CAT_STABILITY, Colors.CAT_STABILITY_LIGHT),
    '차입금': (Colors.CAT_DEBT, Colors.CAT_DEBT_LIGHT),
    '유동성': (Colors.CAT_LIQUIDITY, Colors.CAT_LIQUIDITY_LIGHT),
    '현금흐름': (Colors.CAT_CASHFLOW, Colors.CAT_CASHFLOW_LIGHT),
}

# 지표 → 카테고리 매핑
METRIC_TO_CATEGORY = {
    'ROA': '수익성', 'ROE': '수익성', '매출액영업이익률': '수익성', '매출액순이익률': '수익성',
    '부채비율': '안정성', '자기자본비율': '안정성', '자본잠식률': '안정성',
    '유동비율': '유동성', '당좌비율': '유동성', '유동부채비율': '유동성', '이자보상배율': '유동성',
    '단기차입금비율': '차입금',
    'CFO_자산비율': '현금흐름', 'CFO_매출액비율': '현금흐름', 'CFO증감률': '현금흐름',
    '총자산회전율': '현금흐름', '재고자산회전율': '현금흐름', '매출채권회전율': '현금흐름',
}

# 섹션 설명
SECTION_DESCRIPTIONS = {
    '1': '기업의 재무 건전성과 AI 예측 결과를 한눈에 확인합니다.',
    '2': '5개 카테고리별 재무지표의 현황과 시계열 추이를 분석합니다.',
    '3': '동일 업종 내 기업의 상대적 위치와 강점/약점을 파악합니다.',
    '4': 'AI 모델이 예측한 다음 분기 재무지표와 주요 영향 요인을 설명합니다.',
    '5': '전문가와 비전문가를 위한 종합적인 재무 분석 의견을 제공합니다.',
}


class PDFReport(FPDF):
    """시각적 분석이 포함된 Premium PDF 보고서"""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

        # app/assets/fonts/ 경로
        font_path = Path(__file__).parent.parent / 'assets' / 'fonts' / 'AppleSDGothicNeo-Regular.ttf'
        self.add_font('AppleSD', '', str(font_path), uni=True)
        self.set_font('AppleSD', '', 10)

        self.company_name = ""
        self.report_date = ""

    def header(self):
        if self.page_no() > 1:
            self.set_font('AppleSD', '', 8)
            self.set_text_color(*Colors.MUTED)
            self.cell(0, 8, f'{self.company_name} | 재무분석보고서', align='L')
            self.cell(0, 8, self.report_date, align='R', ln=True)
            self.set_draw_color(*Colors.LIGHT)
            self.line(10, 15, 200, 15)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        # 로고 이미지 (좌측)
        if LOGO_NO_BG.exists():
            self.image(str(LOGO_NO_BG), x=10, y=self.get_y(), h=10)
        else:
            self.set_font('AppleSD', '', 8)
            self.set_text_color(*Colors.MUTED)
            self.cell(95, 10, 'BigBig', align='L')
        # 페이지 번호 (우측)
        self.set_font('AppleSD', '', 8)
        self.set_text_color(*Colors.MUTED)
        self.cell(0, 10, f'{self.page_no()}', align='R')

    # =========================================================================
    # 기본 컴포넌트
    # =========================================================================

    def section_header(self, number: str, title: str):
        self.ln(3)
        self.set_fill_color(*Colors.SECTION)
        self.set_text_color(*Colors.WHITE)
        self.set_font('AppleSD', '', 11)
        self.cell(8, 8, number, fill=True, align='C')
        self.set_text_color(*Colors.DARK)
        self.set_font('AppleSD', '', 14)
        self.cell(0, 8, f'  {title}', ln=True)
        self.set_draw_color(*Colors.SECTION_LINE)
        self.set_line_width(0.5)
        self.line(10, self.get_y() + 2, 200, self.get_y() + 2)
        self.set_line_width(0.2)
        self.ln(5)
        # 섹션 설명 추가
        desc = SECTION_DESCRIPTIONS.get(number, '')
        if desc:
            self.set_font('AppleSD', '', 9)
            self.set_text_color(*Colors.DARK)
            self.cell(0, 5, desc, ln=True)
            self.ln(3)

    def subsection_title(self, title: str):
        self.set_font('AppleSD', '', 11)
        self.set_text_color(*Colors.TEXT)
        self.cell(0, 8, title, ln=True)
        self.ln(2)

    def body_text(self, text: str, color: Tuple = None):
        self.set_font('AppleSD', '', 10)
        self.set_text_color(*(color or Colors.TEXT))
        self.multi_cell(0, 6, text)
        self.ln(1)

    def label_value(self, label: str, value: str, label_width: float = 40):
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.MUTED)
        self.cell(label_width, 6, label)
        self.set_text_color(*Colors.DARK)
        self.set_font('AppleSD', '', 10)
        self.cell(0, 6, value, ln=True)

    # =========================================================================
    # 시각적 컴포넌트
    # =========================================================================

    def gauge_chart(self, value: float, max_value: float = 100,
                    x: float = None, y: float = None,
                    radius: float = 20, label: str = ""):
        """반원 게이지 차트"""
        if x is None:
            x = self.get_x() + radius
        if y is None:
            y = self.get_y() + radius

        # 배경 반원
        self.set_fill_color(*Colors.LIGHT)
        self._draw_arc(x, y, radius, 180, 360, Colors.LIGHT)

        # 값에 따른 각도 계산
        percentage = min(value / max_value, 1.0) if max_value > 0 else 0
        end_angle = 180 + (180 * percentage)

        # 색상 결정
        if percentage >= 0.7:
            color = Colors.SUCCESS
        elif percentage >= 0.4:
            color = Colors.PRIMARY
        elif percentage >= 0.25:
            color = Colors.WARNING
        else:
            color = Colors.DANGER

        # 값 반원
        self._draw_arc(x, y, radius, 180, end_angle, color)

        # 중앙 값 표시
        self.set_font('AppleSD', '', 10)
        self.set_text_color(*Colors.DARK)
        self.set_xy(x - radius, y - 5)
        self.cell(radius * 2, 10, f'{value:.1f}%', align='C')

        # 라벨
        if label:
            self.set_font('AppleSD', '', 8)
            self.set_text_color(*Colors.MUTED)
            self.set_xy(x - radius, y + 5)
            self.cell(radius * 2, 6, label, align='C')

    def _draw_arc(self, x: float, y: float, radius: float,
                  start_angle: float, end_angle: float, color: Tuple):
        """호 그리기 (다각형으로 근사)"""
        self.set_fill_color(*color)
        points = [(x, y)]

        for angle in range(int(start_angle), int(end_angle) + 1, 5):
            rad = math.radians(angle)
            px = x + radius * math.cos(rad)
            py = y - radius * math.sin(rad)
            points.append((px, py))

        # 다각형 그리기
        if len(points) > 2:
            self.set_fill_color(*color)
            # fpdf polygon 사용
            point_str = ' '.join([f'{p[0]:.2f} {p[1]:.2f}' for p in points])

    def metric_row(self, label: str, value_text: str, grade: str, score: float = 50):
        """지표 행 - 라벨, 바, 값, 등급을 한 줄에 표시 (테이블 형식)"""
        row_height = 10

        # 등급별 색상
        grade_colors = {
            'A+': Colors.GRADE_A_PLUS, 'A': Colors.GRADE_A,
            'B+': Colors.GRADE_B_PLUS, 'B': Colors.GRADE_B,
            'C': Colors.GRADE_C, 'F': Colors.GRADE_F
        }
        bar_color = grade_colors.get(grade, Colors.MUTED)

        # 1. 라벨 (50px)
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.TEXT)
        self.cell(45, row_height, label, border=0)

        # 2. 바 차트 영역 (90px) - 배경 + 값 바를 셀 내에 그림
        bar_x = self.get_x()
        bar_y = self.get_y()
        bar_width = 80
        bar_inner_height = 6

        # 배경 바
        self.set_fill_color(*Colors.LIGHT)
        self.rect(bar_x, bar_y + 2, bar_width, bar_inner_height, 'F')

        # 값 바
        fill_width = (score / 100) * bar_width
        fill_width = max(0, min(fill_width, bar_width))
        self.set_fill_color(*bar_color)
        self.rect(bar_x, bar_y + 2, fill_width, bar_inner_height, 'F')

        # 바 영역 이동
        self.cell(85, row_height, '', border=0)

        # 3. 값 (35px)
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.DARK)
        self.cell(35, row_height, value_text, border=0)

        # 4. 등급 뱃지 (20px)
        self.set_fill_color(*bar_color)
        self.set_text_color(*Colors.WHITE)
        self.set_font('AppleSD', '', 8)
        self.cell(18, row_height, grade, border=0, fill=True, align='C')

        self.ln(row_height + 2)

    def comparison_row(self, label: str, current: float, predicted: float):
        """현재 vs 예측 비교 행 - 테이블 형식"""
        row_height = 8
        bar_width = 70

        max_val = max(abs(current), abs(predicted), 1) * 1.2

        # 1. 라벨
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.TEXT)
        self.cell(45, row_height * 2, label, border=0)

        # 2. 바 영역
        bar_x = self.get_x()
        bar_y = self.get_y()

        # 현재값 바 (위)
        self.set_fill_color(*Colors.MUTED)
        cur_width = (abs(current) / max_val) * bar_width
        self.rect(bar_x, bar_y + 1, cur_width, row_height - 2, 'F')

        # 예측값 바 (아래)
        pred_color = Colors.SUCCESS if predicted >= current else Colors.DANGER
        self.set_fill_color(*pred_color)
        pred_width = (abs(predicted) / max_val) * bar_width
        self.rect(bar_x, bar_y + row_height, pred_width, row_height - 2, 'F')

        # 바 영역 이동
        self.cell(75, row_height * 2, '', border=0)

        # 3. 값 영역 (2줄)
        value_x = self.get_x()
        self.set_xy(value_x, bar_y)
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.MUTED)
        self.cell(30, row_height, f'{current:.1f}%')

        self.set_xy(value_x, bar_y + row_height)
        self.set_text_color(*pred_color)
        self.cell(30, row_height, f'{predicted:.1f}%')

        # 다음 줄로 이동
        self.set_xy(10, bar_y + row_height * 2 + 2)

    def metric_card(self, label: str, value: str, grade: str,
                    percentile: float = None, width: float = 60):
        """지표 카드 - 카테고리별 고유 색상 사용"""
        x = self.get_x()
        y = self.get_y()

        # 카테고리 색상 결정
        category = METRIC_TO_CATEGORY.get(label, '수익성')
        cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))

        # 카드 배경 - 카테고리 연한 색상
        self.set_fill_color(*cat_light)
        self.rect(x, y, width, 35, 'F')

        # 카테고리 색상 좌측 바
        self.set_fill_color(*cat_color)
        self.rect(x, y, 3, 35, 'F')

        # 라벨
        self.set_xy(x + 5, y + 3)
        self.set_font('AppleSD', '', 8)
        self.set_text_color(*cat_color)
        self.cell(width - 10, 5, label)

        # 값
        self.set_xy(x + 5, y + 10)
        self.set_font('AppleSD', '', 14)
        self.set_text_color(*Colors.DARK)
        self.cell(width - 10, 8, value)

        # 등급 텍스트 (박스 제거)
        self.set_xy(x + 5, y + 22)
        self.set_text_color(*Colors.DARK)
        self.set_font('AppleSD', '', 9)
        self.cell(20, 8, f'등급: {grade}')

        # 백분위
        if percentile is not None:
            self.set_xy(x + 27, y + 22)
            self.set_text_color(*Colors.MUTED)
            self.set_font('AppleSD', '', 8)
            pct_text = f"상위{percentile:.0f}%" if percentile <= 50 else f"하위{100-percentile:.0f}%"
            self.cell(width - 30, 8, pct_text)

        self.set_xy(x + width + 3, y)

    def grade_badge(self, grade: str, size: str = 'medium'):
        colors = {
            'A+': Colors.GRADE_A_PLUS, 'A': Colors.GRADE_A,
            'B+': Colors.GRADE_B_PLUS, 'B': Colors.GRADE_B,
            'C': Colors.GRADE_C, 'F': Colors.GRADE_F, 'N/A': Colors.MUTED
        }
        color = colors.get(grade, Colors.MUTED)

        if size == 'large':
            self.set_fill_color(*color)
            self.set_text_color(*Colors.WHITE)
            self.set_font('AppleSD', '', 24)
            self.cell(50, 30, grade, fill=True, align='C')
        elif size == 'medium':
            self.set_fill_color(*color)
            self.set_text_color(*Colors.WHITE)
            self.set_font('AppleSD', '', 14)
            self.cell(30, 16, grade, fill=True, align='C')
        else:
            self.set_fill_color(*color)
            self.set_text_color(*Colors.WHITE)
            self.set_font('AppleSD', '', 10)
            self.cell(20, 10, grade, fill=True, align='C')

    def styled_table(self, headers: List[str], data: List[List],
                     col_widths: List[float] = None, highlight_col: int = None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        self.set_fill_color(*Colors.PRIMARY)
        self.set_text_color(*Colors.WHITE)
        self.set_font('AppleSD', '', 9)

        for i, header in enumerate(headers):
            self.cell(col_widths[i], 9, header, border=0, fill=True, align='C')
        self.ln()

        self.set_text_color(*Colors.TEXT)
        for row_idx, row in enumerate(data):
            if row_idx % 2 == 0:
                self.set_fill_color(*Colors.WHITE)
            else:
                self.set_fill_color(250, 250, 252)

            for i, cell in enumerate(row):
                cell_str = str(cell)
                if '↑' in cell_str or '개선' in cell_str:
                    self.set_text_color(*Colors.SUCCESS)
                elif '↓' in cell_str or '악화' in cell_str:
                    self.set_text_color(*Colors.DANGER)
                elif i == highlight_col:
                    self.set_text_color(*Colors.PRIMARY)
                else:
                    self.set_text_color(*Colors.TEXT)

                self.cell(col_widths[i], 8, cell_str, border=0, fill=True, align='C')
            self.ln()
        self.ln(3)

    def insight_box(self, title: str, content: str, box_type: str = 'info'):
        """인사이트 박스"""
        x = self.get_x()
        y = self.get_y()

        # 박스 타입별 색상
        type_colors = {
            'info': Colors.PRIMARY,
            'success': Colors.SUCCESS,
            'warning': Colors.WARNING,
            'danger': Colors.DANGER
        }
        color = type_colors.get(box_type, Colors.PRIMARY)

        # 배경
        self.set_fill_color(color[0], color[1], color[2])
        self.set_draw_color(*color)

        # 좌측 바
        self.rect(x, y, 3, 25, 'F')

        # 배경 (연한 색) - 255 초과 방지
        self.set_fill_color(min(color[0] + 40, 255), min(color[1] + 40, 255), min(color[2] + 40, 255))

        # 제목
        self.set_xy(x + 5, y + 2)
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*color)
        self.cell(0, 5, title, ln=True)

        # 내용
        self.set_x(x + 5)
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.TEXT)
        self.multi_cell(180, 5, content[:150])

        self.ln(3)

    def mini_line_chart(self, values: List[float], labels: List[str],
                        width: float = 85, height: float = 45,
                        title: str = "", color: Tuple = None):
        """미니 라인 차트 - 시계열 추세 시각화 (X축, Y축 값 표시)"""
        if not values or all(v is None for v in values):
            return

        x = self.get_x()
        y = self.get_y()
        color = color or Colors.PRIMARY

        # 유효한 값만 필터링
        valid_data = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(valid_data) < 2:
            return

        # 값 범위 계산
        valid_values = [v for _, v in valid_data]
        min_val = min(valid_values)
        max_val = max(valid_values)
        val_range = max_val - min_val if max_val != min_val else 1

        # 차트 영역 (Y축 라벨 공간 확보)
        y_label_width = 18
        chart_x = x + y_label_width
        chart_y = y + 10
        chart_w = width - y_label_width - 3
        chart_h = height - 22

        # 타이틀
        if title:
            self.set_xy(x, y)
            self.set_font('AppleSD', '', 8)
            self.set_text_color(*Colors.DARK)
            self.cell(width, 6, title, align='C')

        # 배경
        self.set_fill_color(250, 251, 252)
        self.rect(chart_x, chart_y, chart_w, chart_h, 'F')

        # Y축 그리드 라인 + 값 표시
        self.set_font('AppleSD', '', 6)
        self.set_text_color(*Colors.MUTED)
        self.set_draw_color(230, 230, 230)
        self.set_line_width(0.1)

        # Y축: 최대, 중간, 최소 3개 값 표시
        for i, ratio in enumerate([0, 0.5, 1]):
            line_y = chart_y + chart_h * ratio
            y_val = max_val - val_range * ratio

            # 그리드 라인
            self.line(chart_x, line_y, chart_x + chart_w, line_y)

            # Y축 값 라벨
            self.set_xy(x, line_y - 2)
            self.cell(y_label_width - 2, 4, f'{y_val:.1f}', align='R')

        # 점과 선 그리기
        self.set_draw_color(*color)
        self.set_line_width(0.5)

        points = []
        for idx, val in valid_data:
            px = chart_x + (idx / (len(values) - 1)) * chart_w
            py = chart_y + chart_h - ((val - min_val) / val_range) * chart_h
            points.append((px, py, val))

        # 선 그리기
        for i in range(len(points) - 1):
            self.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # 점 + 값 그리기
        self.set_fill_color(*color)
        self.set_font('AppleSD', '', 5)
        for i, (px, py, val) in enumerate(points):
            self.ellipse(px - 1.5, py - 1.5, 3, 3, 'F')

            # 마지막 점에만 값 표시 (현재값)
            if i == len(points) - 1:
                self.set_text_color(*color)
                self.set_xy(px - 8, py - 6)
                self.cell(16, 4, f'{val:.1f}%', align='C')

        self.set_line_width(0.2)

        # X축 라벨 (모든 분기 표시)
        self.set_font('AppleSD', '', 5)
        self.set_text_color(*Colors.MUTED)
        if labels:
            for i, (idx, _) in enumerate(valid_data):
                label = labels[idx] if idx < len(labels) else ''
                px = chart_x + (idx / (len(values) - 1)) * chart_w
                self.set_xy(px - 8, chart_y + chart_h + 1)
                self.cell(16, 4, label[-5:], align='C')  # "2024 Q3" -> "4 Q3"

        self.set_xy(x + width + 3, y)

    def compare_bar(self, company_val: float, industry_mean: float,
                    industry_median: float, label: str = "",
                    width: float = 170, height: float = 12):
        """기업 vs 업종 비교 바 - 3개 값 비교"""
        y = self.get_y()

        # 라벨
        self.set_font('AppleSD', '', 9)
        self.set_text_color(*Colors.TEXT)
        self.cell(45, height, label)

        bar_x = self.get_x()
        bar_w = 80

        # 최대값 계산
        max_val = max(abs(company_val or 0), abs(industry_mean or 0), abs(industry_median or 0), 1) * 1.3

        # 배경
        self.set_fill_color(*Colors.LIGHT)
        self.rect(bar_x, y + 2, bar_w, height - 4, 'F')

        # 업종 중앙값 (연한 색)
        if industry_median is not None:
            med_w = (abs(industry_median) / max_val) * bar_w
            self.set_fill_color(200, 200, 200)
            self.rect(bar_x, y + 2, med_w, height - 4, 'F')

        # 업종 평균 (점선 마커)
        if industry_mean is not None:
            mean_x = bar_x + (abs(industry_mean) / max_val) * bar_w
            self.set_draw_color(*Colors.MUTED)
            self.set_line_width(0.3)
            self.line(mean_x, y + 1, mean_x, y + height - 1)

        # 기업 값 (색상 바)
        if company_val is not None:
            comp_w = (abs(company_val) / max_val) * bar_w
            if company_val >= (industry_mean or 0):
                self.set_fill_color(*Colors.SUCCESS)
            else:
                self.set_fill_color(*Colors.DANGER)
            self.rect(bar_x, y + 4, comp_w, height - 8, 'F')

        self.set_line_width(0.2)

        # 값 표시
        self.cell(85, height, '')  # 바 영역 스킵
        self.set_font('AppleSD', '', 8)

        # 기업값
        self.set_text_color(*Colors.DARK)
        comp_str = f'{company_val:.1f}%' if company_val is not None else 'N/A'
        self.cell(25, height, comp_str)

        # 업종평균
        self.set_text_color(*Colors.MUTED)
        mean_str = f'({industry_mean:.1f}%)' if industry_mean is not None else ''
        self.cell(25, height, mean_str)

        self.ln(height + 1)


class PDFReportGenerator:
    """PDF 보고서 생성 클래스"""

    def __init__(self):
        # ai-server/reports/ 경로
        self.output_dir = Path(__file__).parent.parent.parent / 'reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, company_code: str, output_dir: Optional[str] = None) -> str:
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 보고서 데이터 생성
        report = generate_report(company_code)
        if 'error' in report:
            raise ValueError(report['error'])

        # 2. 추가 데이터 로드 (시계열, 업종 비교, 피쳐 엔지니어링 데이터)
        data_loader = get_data_loader()
        company_data = data_loader.get_company_data(company_code)
        historical = company_data.get('historical', {}) if company_data else {}
        trend_data = company_data.get('trend', {}) if company_data else {}
        relative_data = company_data.get('relative', {}) if company_data else {}
        current_data = company_data.get('current', {}) if company_data else {}
        industry_comparison = data_loader.get_industry_metric_comparison(company_code)

        # 3. SHAP 분석 데이터 추출
        shap_data = self._extract_shap_data(report)
        opinion_data = report['opinion_data']
        opinion_data['shap_analysis'] = shap_data

        # 4. LLM 통합 종합의견 생성 (내러티브 구조)
        opinion = generate_unified_opinion(opinion_data)

        # 5. 카테고리별 XAI + LLM 분석 (13개 지표 전체)
        ai_pred = report['sections']['ai_prediction']
        all_predictions = ai_pred.get('all_predictions', {})
        category_analysis = generate_category_analysis(opinion_data, all_predictions)

        # 6. PDF 생성
        pdf = PDFReport()

        meta = report['meta']
        pdf.company_name = meta['기업명']
        pdf.report_date = meta['기준일']

        summary = report['sections']['summary']
        company_info = report['sections']['company_info']
        financial = report['sections']['financial_analysis']
        industry = report['sections']['industry_comparison']
        trend = report['sections']['trend_analysis']
        ai_pred = report['sections']['ai_prediction']

        # ===== 표지 =====
        pdf.add_page()
        self._add_cover_page(pdf, meta, summary, ai_pred, financial)

        # ===== 목차 =====
        pdf.add_page()
        self._add_table_of_contents(pdf)

        # ===== 1. Executive Summary =====
        pdf.add_page()
        self._add_executive_summary(pdf, summary, ai_pred, financial, industry_comparison, trend_data)

        # ===== 2. 재무 상태 분석 (시계열 시각화 + 피쳐 엔지니어링 + LLM 분석) =====
        pdf.add_page()
        self._add_financial_with_trends(pdf, financial, historical, trend_data, relative_data, current_data, meta)

        # ===== 3. 업종 비교 (지표별 비교) =====
        pdf.add_page()
        self._add_industry_with_metrics(pdf, industry, industry_comparison, meta['기업명'])

        # ===== 4. AI 예측 요약 =====
        pdf.add_page()
        self._add_ai_prediction_summary(pdf, ai_pred)

        # ===== 5. 카테고리별 XAI + LLM 분석 (13개 지표) =====
        self._add_category_xai_analysis(pdf, category_analysis, ai_pred)

        # ===== 6. 종합 의견 =====
        pdf.add_page()
        self._add_comprehensive_opinion(pdf, opinion)

        # ===== 별첨: 분석 방법론 =====
        pdf.add_page()
        self._add_methodology_appendix(pdf)

        # PDF 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{meta['기업명']}_재무분석보고서_{timestamp}.pdf"
        output_path = self.output_dir / filename

        pdf.output(str(output_path))
        return str(output_path)

    def _extract_shap_data(self, report: Dict) -> Dict:
        shap_data = {}
        ai_pred = report['sections']['ai_prediction']
        all_predictions = ai_pred.get('all_predictions', {})

        for metric, pred in all_predictions.items():
            if isinstance(pred, dict) and 'shap_analysis' in pred:
                shap_data[metric] = pred['shap_analysis']

        return shap_data

    # =========================================================================
    # 표지
    # =========================================================================

    def _add_cover_page(self, pdf: PDFReport, meta: Dict, summary: Dict, ai_pred: Dict,
                        financial: Dict = None):
        pdf.ln(15)

        pdf.set_font('AppleSD', '', 28)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(0, 15, meta['기업명'], align='C', ln=True)

        pdf.set_font('AppleSD', '', 11)
        pdf.set_text_color(*Colors.MUTED)
        pdf.cell(0, 8, f"{meta['기업코드']} | {meta['업종명']} | {meta['시장구분']}", align='C', ln=True)

        # 구분선
        pdf.ln(8)
        pdf.set_draw_color(*Colors.SECTION_LINE)
        pdf.set_line_width(0.8)
        pdf.line(60, pdf.get_y(), 150, pdf.get_y())
        pdf.set_line_width(0.2)
        pdf.ln(10)

        financial = financial or {}

        # 레이더 차트 + 종합 등급
        grade = summary.get('overall_grade') or 'N/A'
        score = summary.get('overall_score')
        score = score if score is not None else 3.0  # 기본값 3.0

        grade_colors = {
            'A+': Colors.GRADE_A_PLUS, 'A': Colors.GRADE_A,
            'B+': Colors.GRADE_B_PLUS, 'B': Colors.GRADE_B,
            'C': Colors.GRADE_C, 'F': Colors.GRADE_F
        }
        g_color = grade_colors.get(grade, Colors.MUTED)

        # 레이더 차트 그리기
        categories = ['수익성', '안정성', '차입금', '유동성', '현금흐름']
        cat_scores = []
        for cat in categories:
            cat_data = financial.get(cat, {})
            cat_score = cat_data.get('score')
            cat_scores.append(cat_score if cat_score is not None else 3)  # 기본값 3 (None 처리)

        # 차트 중심 및 크기
        center_x = 105  # A4 중앙
        center_y = pdf.get_y() + 55
        max_radius = 45

        # 결측 여부 확인 (점수가 None이거나 0이면 결측)
        is_missing = []
        for cat in categories:
            cat_data = financial.get(cat, {})
            cat_score = cat_data.get('score')  # 카테고리별 점수 (overall score와 구분)
            # 점수가 None이거나, metrics가 비어있으면 결측
            metrics_data = cat_data.get('metrics', {})
            has_data = any(
                not m.get('is_missing', True)
                for m in metrics_data.values()
            ) if metrics_data else False
            is_missing.append(not has_data and cat_score is None)

        # 5각형 배경 그리드 (6단계)
        for level in range(1, 7):
            radius = max_radius * (level / 6)
            points = []
            for i in range(5):
                angle = math.radians(-90 + i * 72)  # -90도에서 시작 (위쪽)
                x = center_x + radius * math.cos(angle)
                y = center_y + radius * math.sin(angle)
                points.append((x, y))

            # 그리드 선
            pdf.set_draw_color(220, 220, 220)
            pdf.set_line_width(0.3)
            for i in range(5):
                next_i = (i + 1) % 5
                pdf.line(points[i][0], points[i][1], points[next_i][0], points[next_i][1])

        # 축 선 (중심에서 각 꼭지점으로)
        pdf.set_draw_color(200, 200, 200)
        for i in range(5):
            angle = math.radians(-90 + i * 72)
            x = center_x + max_radius * math.cos(angle)
            y = center_y + max_radius * math.sin(angle)
            pdf.line(center_x, center_y, x, y)

        # 카테고리 라벨 (그래프에 가깝게, 결측이면 회색)
        label_radius = max_radius + 6
        for i, cat in enumerate(categories):
            angle = math.radians(-90 + i * 72)
            x = center_x + label_radius * math.cos(angle)
            y = center_y + label_radius * math.sin(angle)

            # 결측이면 회색, 아니면 카테고리 색상
            if is_missing[i]:
                label_color = Colors.MUTED
            else:
                label_color, _ = CATEGORY_COLORS.get(cat, (Colors.DARK, Colors.LIGHT))

            pdf.set_font('AppleSD', '', 11)
            pdf.set_text_color(*label_color)

            # 위치 조정
            if i == 0:  # 위
                pdf.set_xy(x - 15, y - 10)
                pdf.cell(30, 6, cat, align='C')
            elif i == 1:  # 오른쪽 위
                pdf.set_xy(x + 2, y - 3)
                pdf.cell(25, 6, cat, align='L')
            elif i == 2:  # 오른쪽 아래
                pdf.set_xy(x + 2, y - 3)
                pdf.cell(25, 6, cat, align='L')
            elif i == 3:  # 왼쪽 아래
                pdf.set_xy(x - 27, y - 3)
                pdf.cell(25, 6, cat, align='R')
            elif i == 4:  # 왼쪽 위
                pdf.set_xy(x - 27, y - 3)
                pdf.cell(25, 6, cat, align='R')

        # 데이터 다각형 좌표 계산
        data_points = []
        for i, s in enumerate(cat_scores):
            radius = max_radius * (s / 6)  # 6점 만점 기준
            angle = math.radians(-90 + i * 72)
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            data_points.append((x, y))

        # 다각형 채우기 (반투명)
        pdf.set_fill_color(41, 128, 185)   # PRIMARY 색상
        pdf.set_line_width(0.5)

        with pdf.local_context(fill_opacity=0.25):
            pdf.set_fill_color(41, 128, 185)
            pdf.polygon(data_points, style='F')  # 채우기만

        # 각 변을 개별적으로 그리기 (결측이면 회색)
        for i in range(5):
            next_i = (i + 1) % 5
            # 현재 또는 다음 꼭지점이 결측이면 회색 선
            if is_missing[i] or is_missing[next_i]:
                pdf.set_draw_color(*Colors.MUTED)
            else:
                pdf.set_draw_color(41, 128, 185)
            pdf.set_line_width(0.5)
            pdf.line(data_points[i][0], data_points[i][1],
                    data_points[next_i][0], data_points[next_i][1])

        # 각 꼭지점에 작은 점 (결측이면 회색)
        with pdf.local_context(fill_opacity=0.5):
            for i, (x, y) in enumerate(data_points):
                if is_missing[i]:
                    pdf.set_fill_color(*Colors.MUTED)
                else:
                    pdf.set_fill_color(41, 128, 185)
                pdf.ellipse(x - 0.8, y - 0.8, 1.6, 1.6, 'F')

        # 종합 등급 박스 (차트 아래)
        pdf.set_y(center_y + max_radius + 25)

        box_x = 75
        box_y = pdf.get_y()
        pdf.set_fill_color(*g_color)
        pdf.rect(box_x, box_y, 60, 35, 'F')

        # 등급 텍스트
        pdf.set_xy(box_x, box_y + 5)
        pdf.set_font('AppleSD', '', 24)
        pdf.set_text_color(*Colors.WHITE)
        pdf.cell(60, 12, grade, align='C')

        # 점수 텍스트
        pdf.set_xy(box_x, box_y + 20)
        pdf.set_font('AppleSD', '', 11)
        pdf.set_text_color(*Colors.WHITE)
        pdf.cell(60, 8, f'{score:.2f} / 6.00', align='C')

        # 기준일
        pdf.set_y(box_y + 45)
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.MUTED)
        pdf.cell(0, 5, f"분석 기준: {meta['기준일']}", align='C', ln=True)

    # =========================================================================
    # 목차
    # =========================================================================

    def _add_table_of_contents(self, pdf: PDFReport):
        """목차 페이지"""
        pdf.ln(5)

        # 목차 제목 (검정 박스 없이)
        pdf.set_font('AppleSD', '', 16)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(0, 10, '목차', ln=True)

        pdf.set_draw_color(*Colors.DARK)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
        pdf.set_line_width(0.2)
        pdf.ln(10)

        toc_items = [
            ('1', 'Executive Summary', '카테고리별 업종상대, 시계열 추세, AI 예측 요약'),
            ('2', '재무 상태 분석', '13개 지표 시계열 차트 및 피쳐 분석'),
            ('3', '업종 비교', '동일 업종 내 상대적 위치 분석'),
            ('4', 'AI 예측 요약', 'XGBoost 기반 다음 분기 예측'),
            ('5', '카테고리별 XAI 분석', 'SHAP 기반 예측 요인 분석'),
            ('6', '종합 의견', '통합 내러티브 종합 분석'),
            ('별첨', '분석 방법론', '등급 산정 기준 및 AI 모델 성능'),
        ]

        for num, title, desc in toc_items:
            row_y = pdf.get_y()

            # 번호 박스 (검정/회색)
            pdf.set_fill_color(*Colors.DARK)
            pdf.rect(15, row_y, 12, 12, 'F')
            pdf.set_xy(15, row_y + 2)
            pdf.set_font('AppleSD', '', 10)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(12, 8, num, align='C')

            # 제목
            pdf.set_xy(32, row_y + 2)
            pdf.set_font('AppleSD', '', 12)
            pdf.set_text_color(*Colors.DARK)
            pdf.cell(60, 8, title)

            # 설명
            pdf.set_xy(95, row_y + 2)
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.MUTED)
            pdf.cell(100, 8, desc)

            pdf.ln(18)

    # =========================================================================
    # Executive Summary
    # =========================================================================

    def _add_executive_summary(self, pdf: PDFReport, summary: Dict, ai_pred: Dict,
                                financial: Dict = None, industry_comparison: Dict = None,
                                trend_data: Dict = None):
        """Executive Summary - 앞으로 나올 내용의 미리보기"""
        pdf.section_header('1', 'Executive Summary')

        financial = financial or {}
        industry_comparison = industry_comparison or {}
        trend_data = trend_data or {}

        grade_colors = {
            'A+': Colors.GRADE_A_PLUS, 'A': Colors.GRADE_A,
            'B+': Colors.GRADE_B_PLUS, 'B': Colors.GRADE_B,
            'C': Colors.GRADE_C, 'F': Colors.GRADE_F
        }

        categories = ['수익성', '안정성', '차입금', '유동성', '현금흐름']

        # 카테고리별 지표 매핑
        cat_metrics = {
            '수익성': ['ROA', 'ROE', '매출액영업이익률'],
            '안정성': ['부채비율', '자기자본비율', '자본잠식률'],
            '차입금': ['단기차입금비율'],
            '유동성': ['유동비율', '당좌비율', '유동부채비율'],
            '현금흐름': ['CFO_자산비율', 'CFO_매출액비율', 'CFO증감률'],
        }

        # 지표별 방향 (up: 높을수록 좋음)
        metric_direction = {
            'ROA': 'up', 'ROE': 'up', '매출액영업이익률': 'up',
            '부채비율': 'down', '자기자본비율': 'up', '자본잠식률': 'down',
            '단기차입금비율': 'down',
            '유동비율': 'up', '당좌비율': 'up', '유동부채비율': 'down',
            'CFO_자산비율': 'up', 'CFO_매출액비율': 'up', 'CFO증감률': 'up',
        }

        section_y = pdf.get_y() + 3

        # =========================================================
        # 테이블 헤더
        # =========================================================
        header_y = section_y
        col_widths = [40, 50, 50, 50]  # 카테고리, 업종상대, 시계열추세, 예측
        col_x = [10, 50, 100, 150]

        # 헤더 배경
        pdf.set_fill_color(*Colors.DARK)
        pdf.rect(10, header_y, 190, 10, 'F')

        # 헤더 텍스트
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.WHITE)

        headers = ['카테고리', '업종상대 위치', '시계열 추세', 'AI 예측']
        for i, header in enumerate(headers):
            pdf.set_xy(col_x[i], header_y + 2)
            pdf.cell(col_widths[i], 6, header, align='C')

        # =========================================================
        # 5개 카테고리 행
        # =========================================================
        row_h = 28
        all_predictions = ai_pred.get('all_predictions', {})

        for cat_idx, cat in enumerate(categories):
            row_y = header_y + 12 + cat_idx * row_h
            cat_data = financial.get(cat, {})
            cat_grade = cat_data.get('grade', 'N/A')
            cat_color, cat_light = CATEGORY_COLORS.get(cat, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))
            g_color = grade_colors.get(cat_grade, Colors.MUTED)

            # 행 배경 (줄무늬)
            if cat_idx % 2 == 0:
                pdf.set_fill_color(248, 249, 250)
            else:
                pdf.set_fill_color(255, 255, 255)
            pdf.rect(10, row_y, 190, row_h, 'F')

            # ----- 1열: 카테고리명 + 등급 -----
            pdf.set_xy(col_x[0] + 2, row_y + 3)
            pdf.set_font('AppleSD', '', 10)
            pdf.set_text_color(*cat_color)
            pdf.cell(col_widths[0] - 4, 6, cat)

            # 등급 뱃지
            pdf.set_fill_color(*g_color)
            badge_x = col_x[0] + 2
            pdf.rect(badge_x, row_y + 12, 22, 12, 'F')
            pdf.set_xy(badge_x, row_y + 14)
            pdf.set_font('AppleSD', '', 10)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(22, 8, cat_grade, align='C')

            # ----- 2열: 업종상대 위치 -----
            metrics_in_cat = cat_metrics.get(cat, [])
            cat_percentiles = []
            for m in metrics_in_cat:
                m_data = cat_data.get('metrics', {}).get(m, {})
                pct = m_data.get('percentile')
                if pct is not None:
                    cat_percentiles.append(pct)

            avg_percentile = sum(cat_percentiles) / len(cat_percentiles) if cat_percentiles else 50

            # 위치 바 (오른쪽이 좋음)
            bar_x = col_x[1] + 5
            bar_w = col_widths[1] - 10
            bar_y = row_y + 8

            pdf.set_fill_color(220, 220, 220)
            pdf.rect(bar_x, bar_y, bar_w, 6, 'F')

            # 위치 마커 (상위 10% → 오른쪽, 상위 90% → 왼쪽)
            pos_x = bar_x + ((100 - avg_percentile) / 100) * bar_w
            if avg_percentile <= 30:
                marker_color = Colors.SUCCESS
            elif avg_percentile <= 70:
                marker_color = Colors.WARNING
            else:
                marker_color = Colors.DANGER

            pdf.set_fill_color(*marker_color)
            pdf.rect(pos_x - 3, bar_y - 2, 6, 10, 'F')

            # 위치 텍스트
            pdf.set_xy(col_x[1], row_y + 16)
            pdf.set_font('AppleSD', '', 8)
            pdf.set_text_color(*Colors.MUTED)
            position_text = f'상위 {avg_percentile:.0f}%'
            pdf.cell(col_widths[1], 5, position_text, align='C')

            # ----- 3열: 시계열 추세 -----
            # 카테고리 내 지표들의 이동평균 기반 추세 계산
            improving_count = 0
            declining_count = 0

            for m in metrics_in_cat:
                m_trend = trend_data.get(m, {})
                mom = m_trend.get('mom')  # 전분기 대비

                if mom is not None:
                    m_dir = metric_direction.get(m, 'up')
                    # 방향에 따라 좋은/나쁜 판단
                    if m_dir == 'up':
                        if mom > 0:
                            improving_count += 1
                        elif mom < 0:
                            declining_count += 1
                    else:  # down (낮을수록 좋음)
                        if mom < 0:
                            improving_count += 1
                        elif mom > 0:
                            declining_count += 1

            # 추세 표시
            pdf.set_xy(col_x[2], row_y + 5)
            pdf.set_font('AppleSD', '', 9)

            if improving_count > declining_count:
                pdf.set_text_color(*Colors.SUCCESS)
                pdf.cell(col_widths[2], 6, '▲ 개선 중', align='C')
            elif declining_count > improving_count:
                pdf.set_text_color(*Colors.DANGER)
                pdf.cell(col_widths[2], 6, '▼ 악화 중', align='C')
            else:
                pdf.set_text_color(*Colors.MUTED)
                pdf.cell(col_widths[2], 6, '― 유지', align='C')

            pdf.set_xy(col_x[2], row_y + 14)
            pdf.set_font('AppleSD', '', 8)
            pdf.set_text_color(*Colors.TEXT)
            pdf.cell(col_widths[2], 5, f'(↑{improving_count} ↓{declining_count})', align='C')

            # ----- 4열: AI 예측 요약 -----
            # direction 값은 이미 좋음/나쁨을 반영함 (improving=개선, declining=악화)
            pred_improving = 0
            pred_declining = 0

            for m in metrics_in_cat:
                pred = all_predictions.get(m, {})
                pred_dir = pred.get('direction', '')
                if pred_dir == 'improving':
                    pred_improving += 1
                elif pred_dir == 'declining':
                    pred_declining += 1
                # 'stable', 'unknown'은 무시

            pred_x = col_x[3]
            pdf.set_xy(pred_x, row_y + 5)
            pdf.set_font('AppleSD', '', 9)

            if pred_improving > pred_declining:
                pdf.set_text_color(*Colors.SUCCESS)
                pdf.cell(col_widths[3], 6, '▲ 개선 예상', align='C')
            elif pred_declining > pred_improving:
                pdf.set_text_color(*Colors.DANGER)
                pdf.cell(col_widths[3], 6, '▼ 악화 예상', align='C')
            else:
                pdf.set_text_color(*Colors.MUTED)
                pdf.cell(col_widths[3], 6, '― 유지 예상', align='C')

            pdf.set_xy(pred_x, row_y + 14)
            pdf.set_font('AppleSD', '', 8)
            pdf.set_text_color(*Colors.TEXT)
            pdf.cell(col_widths[3], 5, f'(↑{pred_improving} ↓{pred_declining})', align='C')

        # =========================================================
        # 하단 요약
        # =========================================================
        bottom_y = header_y + 12 + 5 * row_h + 5
        pdf.set_y(bottom_y)

        # 전체 요약 박스
        pdf.set_fill_color(*Colors.LIGHT)
        pdf.rect(10, bottom_y, 190, 25, 'F')

        # 종합 등급
        grade = summary['overall_grade']
        score = summary['overall_score']
        g_color = grade_colors.get(grade, Colors.MUTED)

        pdf.set_fill_color(*g_color)
        pdf.rect(15, bottom_y + 4, 30, 17, 'F')
        pdf.set_xy(15, bottom_y + 6)
        pdf.set_font('AppleSD', '', 14)
        pdf.set_text_color(*Colors.WHITE)
        pdf.cell(30, 10, grade, align='C')

        # 종합 점수
        pdf.set_xy(50, bottom_y + 8)
        pdf.set_font('AppleSD', '', 11)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(30, 8, f'{score:.2f} / 6.00')

        # AI 전망
        pred_summary = ai_pred.get('summary', {})
        outlook = pred_summary.get('overall_outlook', 'neutral')
        outlook_info = {
            'positive': ('AI 전망: 긍정적', Colors.SUCCESS),
            'negative': ('AI 전망: 주의 필요', Colors.DANGER),
            'neutral': ('AI 전망: 중립', Colors.MUTED)
        }
        outlook_text, outlook_color = outlook_info.get(outlook, ('AI 전망: 중립', Colors.MUTED))

        pdf.set_xy(100, bottom_y + 8)
        pdf.set_font('AppleSD', '', 11)
        pdf.set_text_color(*outlook_color)
        pdf.cell(50, 8, outlook_text)

        # 강점/약점 개수
        strengths = summary.get('strengths', [])
        weaknesses = summary.get('weaknesses', [])

        pdf.set_xy(155, bottom_y + 6)
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.SUCCESS)
        pdf.cell(20, 6, f'강점 {len(strengths)}개')

        pdf.set_xy(155, bottom_y + 13)
        pdf.set_text_color(*Colors.DANGER)
        pdf.cell(20, 6, f'약점 {len(weaknesses)}개')

    # =========================================================================
    # 시각적 재무 분석 + 시계열 추세
    # =========================================================================

    def _add_financial_with_trends(self, pdf: PDFReport, financial: Dict, historical: Dict,
                                     trend_data: Dict = None, relative_data: Dict = None,
                                     current_data: Dict = None, meta: Dict = None):
        """재무 분석 + 13개 지표 시계열 차트 + 피쳐 엔지니어링 데이터 + LLM 시계열 분석"""
        pdf.section_header('2', '재무 상태 분석')

        trend_data = trend_data or {}
        relative_data = relative_data or {}
        current_data = current_data or {}
        meta = meta or {}

        company_name = meta.get('기업명', '')
        industry = meta.get('업종명', '')

        grade_colors = {
            'A+': Colors.GRADE_A_PLUS, 'A': Colors.GRADE_A,
            'B+': Colors.GRADE_B_PLUS, 'B': Colors.GRADE_B,
            'C': Colors.GRADE_C, 'F': Colors.GRADE_F
        }

        # 4분기만 사용 (현재 + 이전 3분기)
        all_periods = historical.get('periods', [])
        all_metrics_data = historical.get('metrics', {})

        periods = all_periods[-4:] if len(all_periods) > 4 else all_periods
        metrics_data = {}
        for metric, values in all_metrics_data.items():
            metrics_data[metric] = values[-4:] if len(values) > 4 else values

        for category, data in financial.items():
            if pdf.get_y() > 150:
                pdf.add_page()

            # 카테고리 고유 색상 사용
            cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))
            grade_color = grade_colors.get(data['grade'], Colors.MUTED)
            header_y = pdf.get_y()

            # 헤더 배경 바 (카테고리 고유 색상)
            pdf.set_fill_color(*cat_color)
            pdf.rect(10, header_y, 190, 10, 'F')

            # 카테고리명
            pdf.set_xy(12, header_y + 2)
            pdf.set_font('AppleSD', '', 11)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(140, 6, category)

            # 등급 텍스트 (박스 없이)
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(35, 6, f'등급: {data["grade"]}', align='C')
            pdf.ln(12)

            # 재무비율 사전적 의미 (해당 카테고리의 모든 지표 - TARGET_METRICS 기준)
            metrics_list = TARGET_METRICS.get(category, [])
            pdf.set_font('AppleSD', '', 7)
            pdf.set_text_color(*cat_color)
            for metric in metrics_list:
                desc = METRIC_DESCRIPTION.get(metric, '')
                if desc:
                    pdf.set_x(12)
                    pdf.cell(0, 4, f'• {metric}: {desc}', ln=True)
            pdf.ln(3)

            # LLM 시계열 분석 (카테고리별)
            category_metrics = {m: metrics_data.get(m, []) for m in metrics_list}
            category_trends = {m: trend_data.get(m, {}) for m in metrics_list}
            category_relatives = {m: relative_data.get(m) for m in metrics_list}

            try:
                timeseries_analysis = generate_timeseries_analysis(
                    company_name, industry, category,
                    category_metrics, category_trends, category_relatives
                )
                if timeseries_analysis:
                    # 분석 박스 - multi_cell로 먼저 높이 계산
                    pdf.set_font('AppleSD', '', 8)
                    # 실제 텍스트 높이 계산 (182mm 폭, 줄간격 4mm)
                    line_width = 182
                    char_per_line = line_width / 2.2  # 폰트 크기 8 기준 약 82자/줄
                    actual_lines = 1
                    for line in timeseries_analysis.split('\n'):
                        actual_lines += max(1, len(line) / char_per_line)
                    box_height = int(actual_lines * 4) + 10  # 헤더 + 여백

                    box_y = pdf.get_y()
                    pdf.set_fill_color(*cat_light)
                    pdf.rect(10, box_y, 190, box_height, 'F')
                    pdf.set_fill_color(*cat_color)
                    pdf.rect(10, box_y, 3, box_height, 'F')

                    pdf.set_xy(15, box_y + 2)
                    pdf.set_font('AppleSD', '', 7)
                    pdf.set_text_color(*cat_color)
                    pdf.cell(0, 4, f'AI 시계열 분석', ln=True)

                    pdf.set_x(15)
                    pdf.set_font('AppleSD', '', 8)
                    pdf.set_text_color(*Colors.TEXT)
                    pdf.multi_cell(182, 4, timeseries_analysis)
                    pdf.set_y(box_y + box_height + 2)
            except Exception:
                pass

            pdf.ln(2)

            # 지표별 카드 (한 행에 2개) - 모든 지표 표시
            card_width = 93
            card_height = 58

            for idx, metric in enumerate(metrics_list):
                col = idx % 2

                if col == 0:
                    row_start_y = pdf.get_y()
                    if row_start_y > 200:
                        pdf.add_page()
                        row_start_y = pdf.get_y()

                x_pos = 10 + col * (card_width + 4)

                # 데이터 존재 여부 확인
                metric_values = metrics_data.get(metric, [])
                is_missing = not metric_values or all(v is None for v in metric_values)

                # 통합 카드 그리기 - 카테고리 색상 사용
                self._draw_metric_card(
                    pdf, metric, metric_values, periods,
                    trend_data.get(metric, {}), relative_data.get(metric),
                    current_data, cat_color,
                    x_pos, row_start_y, card_width, card_height,
                    is_missing=is_missing  # 미공시 여부 전달
                )

                # 행 끝이면 다음 줄로
                if col == 1 or idx == len(metrics_list) - 1:
                    pdf.set_y(row_start_y + card_height + 5)

            pdf.ln(5)

    def _draw_metric_card(self, pdf: PDFReport, metric: str, values: List[float],
                          periods: List[str], trend: Dict, industry_rel: float,
                          current_data: Dict, color: Tuple,
                          x: float, y: float, width: float, height: float,
                          is_missing: bool = False):
        """통합 지표 카드 (차트 + 데이터) - 세련된 디자인"""

        # 카드 배경 (그림자 효과)
        pdf.set_fill_color(230, 230, 230)
        pdf.rect(x + 1, y + 1, width, height, 'F')

        # 메인 카드 배경 (미공시는 연한 회색)
        if is_missing:
            pdf.set_fill_color(248, 248, 248)
        else:
            pdf.set_fill_color(255, 255, 255)
        pdf.rect(x, y, width, height, 'F')

        # 상단 컬러 바 (미공시는 회색)
        if is_missing:
            pdf.set_fill_color(*Colors.MUTED)
        else:
            pdf.set_fill_color(*color)
        pdf.rect(x, y, width, 3, 'F')

        # 미공시 데이터 처리
        if is_missing:
            pdf.set_xy(x + 3, y + 5)
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.MUTED)
            pdf.cell(width - 6, 5, metric)

            pdf.set_xy(x + 3, y + 20)
            pdf.set_font('AppleSD', '', 10)
            pdf.cell(width - 6, 5, '데이터 미공시')

            pdf.set_xy(x + 3, y + 35)
            pdf.set_font('AppleSD', '', 7)
            pdf.multi_cell(width - 6, 4, '해당 지표의 데이터가 공시되지 않아 분석 결과를 제공할 수 없습니다.')
            return

        # 지표명 헤더
        pdf.set_xy(x + 3, y + 5)
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(width - 6, 5, metric)

        # 현재값 추출
        current_val = None
        for cat_data in current_data.values():
            if isinstance(cat_data, dict) and metric in cat_data:
                metric_info = cat_data[metric]
                if isinstance(metric_info, dict):
                    current_val = metric_info.get('value')
                break

        # 현재값 크게 표시
        pdf.set_xy(x + 3, y + 11)
        pdf.set_font('AppleSD', '', 14)
        pdf.set_text_color(*color)
        cur_str = f'{current_val:.1f}%' if current_val is not None else '-'
        pdf.cell(30, 7, cur_str)

        # 미니 차트 (우측 상단)
        chart_x = x + 38
        chart_y = y + 6
        chart_w = 52
        chart_h = 18
        self._draw_mini_sparkline(pdf, values, chart_x, chart_y, chart_w, chart_h, color)

        # 구분선
        pdf.set_draw_color(240, 240, 240)
        pdf.line(x + 3, y + 26, x + width - 3, y + 26)

        # 하단 피쳐 데이터 (2열 그리드)
        ma4 = trend.get('ma4')
        std4 = trend.get('std4')
        yoy = trend.get('yoy')
        mom = trend.get('mom')

        data_y = y + 29
        col1_x = x + 4
        col2_x = x + 48

        pdf.set_font('AppleSD', '', 6)

        # 1행: MA4, STD4
        self._draw_data_item(pdf, col1_x, data_y, 'MA4', ma4, '%', Colors.DARK)
        self._draw_data_item(pdf, col2_x, data_y, 'STD4', std4, '', Colors.DARK, decimals=2)

        # 2행: YoY, MOM
        data_y += 10
        yoy_color = Colors.SUCCESS if yoy and yoy >= 0 else Colors.DANGER if yoy else Colors.MUTED
        mom_color = Colors.SUCCESS if mom and mom >= 0 else Colors.DANGER if mom else Colors.MUTED
        self._draw_data_item(pdf, col1_x, data_y, 'YoY', yoy, '%', yoy_color, show_sign=True, with_arrow=True)
        self._draw_data_item(pdf, col2_x, data_y, 'MOM', mom, '%', mom_color, show_sign=True, with_arrow=True)

        # 3행: 업종상대
        data_y += 10
        rel_color = Colors.SUCCESS if industry_rel and industry_rel >= 0 else Colors.DANGER if industry_rel else Colors.MUTED
        self._draw_data_item(pdf, col1_x, data_y, '업종대비', industry_rel, '', rel_color, show_sign=True, with_arrow=True)

    def _draw_mini_sparkline(self, pdf: PDFReport, values: List[float],
                              x: float, y: float, w: float, h: float, color: Tuple):
        """미니 스파크라인 차트"""
        if not values or all(v is None for v in values):
            return

        valid_data = [(i, v) for i, v in enumerate(values) if v is not None]
        if len(valid_data) < 2:
            return

        valid_values = [v for _, v in valid_data]
        min_val = min(valid_values)
        max_val = max(valid_values)
        val_range = max_val - min_val if max_val != min_val else 1

        # 배경
        pdf.set_fill_color(250, 252, 255)
        pdf.rect(x, y, w, h, 'F')

        # 그라데이션 영역 (채우기)
        points = []
        for idx, val in valid_data:
            px = x + (idx / (len(values) - 1)) * w
            py = y + h - ((val - min_val) / val_range) * (h - 2)
            points.append((px, py))

        # 선 그리기
        pdf.set_draw_color(*color)
        pdf.set_line_width(0.6)
        for i in range(len(points) - 1):
            pdf.line(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # 마지막 점 강조
        if points:
            last_x, last_y = points[-1]
            pdf.set_fill_color(*color)
            pdf.ellipse(last_x - 1.5, last_y - 1.5, 3, 3, 'F')

        pdf.set_line_width(0.2)

    def _draw_data_item(self, pdf: PDFReport, x: float, y: float, label: str,
                        value: float, unit: str, color: Tuple,
                        show_sign: bool = False, with_arrow: bool = False, decimals: int = 1):
        """데이터 항목 표시 (라벨 + 값)"""
        # 라벨
        pdf.set_xy(x, y)
        pdf.set_font('AppleSD', '', 6)
        pdf.set_text_color(*Colors.MUTED)
        pdf.cell(18, 4, label)

        # 값
        pdf.set_xy(x + 18, y)
        pdf.set_font('AppleSD', '', 7)
        pdf.set_text_color(*color)

        if value is not None:
            # 화살표
            arrow = ''
            if with_arrow and value != 0:
                arrow = '▲' if value > 0 else '▼'

            # 부호
            if show_sign and value > 0:
                val_str = f'{arrow}{value:+.{decimals}f}{unit}'
            elif show_sign and value < 0:
                val_str = f'{arrow}{value:.{decimals}f}{unit}'
            else:
                val_str = f'{value:.{decimals}f}{unit}'

            pdf.cell(22, 4, val_str)
        else:
            pdf.set_text_color(*Colors.MUTED)
            pdf.cell(22, 4, '-')

    # =========================================================================
    # 업종 비교 + 지표별 상세
    # =========================================================================

    def _add_industry_with_metrics(self, pdf: PDFReport, industry: Dict, industry_comparison: Dict,
                                     company_name: str = ''):
        """업종 비교 + 지표별 기업 vs 업종 비교 + LLM 분석 (개선된 디자인)"""
        pdf.section_header('3', '업종 비교 분석')

        industry_name = industry.get('industry', '')
        category_positions = industry.get('category_position', {})
        strengths = industry.get('strengths', [])
        weaknesses = industry.get('weaknesses', [])
        metrics_info = industry_comparison.get('metrics', {}) if industry_comparison else {}

        # 업종 정보 헤더
        pdf.set_fill_color(240, 240, 240)
        header_y = pdf.get_y()
        pdf.rect(10, header_y, 190, 12, 'F')
        pdf.set_xy(15, header_y + 3)
        pdf.set_font('AppleSD', '', 10)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(0, 6, f"비교 업종: {industry_name}")
        pdf.ln(15)

        # LLM 업종 비교 분석
        try:
            # metrics_comparison 준비
            comparison_data = {}
            key_metrics = ['ROA', 'ROE', '부채비율', '유동비율', 'CFO_자산비율']
            for metric in key_metrics:
                info = metrics_info.get(metric)
                if info:
                    comparison_data[metric] = {
                        'company': info.get('company'),
                        'industry_mean': info.get('industry_mean'),
                        'percentile': info.get('percentile')
                    }

            industry_analysis = generate_industry_comparison_analysis(
                company_name, industry_name, category_positions,
                comparison_data, strengths, weaknesses
            )

            if industry_analysis:
                # AI 분석 박스 - 텍스트 길이에 맞게 높이 계산
                pdf.set_font('AppleSD', '', 8)
                char_per_line = 82  # 182mm 폭, 폰트 8 기준
                actual_lines = 1
                for line in industry_analysis.split('\n'):
                    actual_lines += max(1, len(line) / char_per_line)
                box_height = int(actual_lines * 4) + 10

                box_y = pdf.get_y()
                pdf.set_fill_color(245, 245, 245)
                pdf.rect(10, box_y, 190, box_height, 'F')
                pdf.set_fill_color(*Colors.DARK)
                pdf.rect(10, box_y, 3, box_height, 'F')

                pdf.set_xy(15, box_y + 2)
                pdf.set_font('AppleSD', '', 7)
                pdf.set_text_color(*Colors.DARK)
                pdf.cell(0, 4, 'AI 업종 비교 분석', ln=True)

                pdf.set_x(15)
                pdf.set_font('AppleSD', '', 8)
                pdf.set_text_color(*Colors.TEXT)
                pdf.multi_cell(182, 4, industry_analysis)
                pdf.set_y(box_y + box_height + 2)
        except Exception:
            pass

        # 카테고리별 순위 카드 (가로 배치, 카테고리 고유 색상)
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(0, 6, '카테고리별 업종 내 순위', ln=True)
        pdf.ln(2)

        card_y = pdf.get_y()
        card_width = 36
        card_x = 10

        # 모든 카테고리 표시 (TARGET_METRICS 기준)
        for cat in TARGET_METRICS.keys():
            pos = category_positions.get(cat)
            is_missing = pos is None

            # 카테고리 고유 색상 사용
            cat_color, cat_light = CATEGORY_COLORS.get(cat, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))

            # 카드 배경 (미공시는 연한 회색)
            if is_missing:
                pdf.set_fill_color(245, 245, 245)
            else:
                pdf.set_fill_color(*cat_light)
            pdf.rect(card_x, card_y, card_width, 22, 'F')

            if is_missing:
                pdf.set_fill_color(*Colors.MUTED)
            else:
                pdf.set_fill_color(*cat_color)
            pdf.rect(card_x, card_y, card_width, 3, 'F')

            # 카테고리명
            pdf.set_xy(card_x + 2, card_y + 5)
            pdf.set_font('AppleSD', '', 7)
            pdf.set_text_color(*cat_color if not is_missing else Colors.MUTED)
            pdf.cell(card_width - 4, 4, cat, align='C')

            # 순위
            pdf.set_xy(card_x + 2, card_y + 11)
            pdf.set_font('AppleSD', '', 10)
            pdf.set_text_color(*cat_color if not is_missing else Colors.MUTED)
            rank_text = f"상위 {pos:.0f}%" if pos is not None else "미공시"
            pdf.cell(card_width - 4, 6, rank_text, align='C')

            card_x += card_width + 2

        pdf.set_y(card_y + 28)

        # 카테고리별 전체 지표 비교 (13개 지표 전체)
        if metrics_info:
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.DARK)
            pdf.cell(0, 6, '카테고리별 전체 지표 비교', ln=True)
            pdf.ln(2)

            for category, metrics_list in TARGET_METRICS.items():
                # 페이지 넘김 확인
                if pdf.get_y() > 180:
                    pdf.add_page()

                cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))
                cat_position = category_positions.get(category)

                # 카테고리 헤더
                cat_y = pdf.get_y()
                pdf.set_fill_color(*cat_color)
                pdf.rect(10, cat_y, 190, 7, 'F')
                pdf.set_xy(12, cat_y + 1.5)
                pdf.set_font('AppleSD', '', 9)
                pdf.set_text_color(*Colors.WHITE)
                pdf.cell(150, 4, category)
                # 순위 표시
                pdf.set_font('AppleSD', '', 8)
                rank_text = f"상위 {cat_position:.0f}%" if cat_position is not None else "미공시"
                pdf.cell(30, 4, rank_text, align='R')
                pdf.ln(8)

                # 카테고리별 AI 분석
                try:
                    cat_metrics = {m: metrics_info.get(m) for m in metrics_list if metrics_info.get(m)}
                    cat_analysis = generate_category_industry_analysis(
                        company_name, industry_name, category, cat_metrics, cat_position if cat_position is not None else 50
                    )
                    if cat_analysis:
                        pdf.set_font('AppleSD', '', 8)
                        char_per_line = 82
                        actual_lines = 1
                        for line in cat_analysis.split('\n'):
                            actual_lines += max(1, len(line) / char_per_line)
                        box_height = int(actual_lines * 4) + 4

                        box_y = pdf.get_y()
                        pdf.set_fill_color(*cat_light)
                        pdf.rect(10, box_y, 190, box_height, 'F')
                        pdf.set_fill_color(*cat_color)
                        pdf.rect(10, box_y, 3, box_height, 'F')
                        pdf.set_xy(15, box_y + 2)
                        pdf.set_text_color(*Colors.TEXT)
                        pdf.multi_cell(182, 4, cat_analysis)
                        pdf.set_y(box_y + box_height + 2)
                except Exception:
                    pass

                # 테이블 헤더
                table_y = pdf.get_y()
                pdf.set_fill_color(*cat_light)
                pdf.rect(10, table_y, 190, 6, 'F')

                pdf.set_xy(10, table_y + 1)
                pdf.set_font('AppleSD', '', 7)
                pdf.set_text_color(*cat_color)
                pdf.cell(45, 4, '지표', align='C')
                pdf.cell(28, 4, '기업값', align='C')
                pdf.cell(28, 4, '업종평균', align='C')
                pdf.cell(28, 4, '차이', align='C')
                pdf.cell(28, 4, '순위', align='C')
                pdf.cell(33, 4, '비교', align='C')
                pdf.ln(6)

                # 데이터 행 - 모든 지표 표시 (미공시 포함)
                for i, metric in enumerate(metrics_list):
                    info = metrics_info.get(metric, {})
                    is_missing = not info or info.get('company') is None

                    row_y = pdf.get_y()
                    # 미공시는 연한 회색 배경
                    if is_missing:
                        row_color = (245, 245, 245)
                    else:
                        row_color = (255, 255, 255) if i % 2 == 0 else (250, 251, 252)
                    pdf.set_fill_color(*row_color)
                    pdf.rect(10, row_y, 190, 8, 'F')

                    company_val = info.get('company') if info else None
                    mean_val = info.get('industry_mean') if info else None
                    pct = info.get('percentile', 50) if info else None

                    diff = (company_val - mean_val) if company_val is not None and mean_val is not None else None
                    diff_color = Colors.SUCCESS if diff is not None and diff >= 0 else Colors.DANGER if diff is not None and diff < 0 else Colors.MUTED

                    pdf.set_xy(10, row_y + 1.5)
                    pdf.set_font('AppleSD', '', 7)

                    # 지표명 (미공시는 회색)
                    if is_missing:
                        pdf.set_fill_color(*Colors.MUTED)
                    else:
                        pdf.set_fill_color(*cat_color)
                    pdf.rect(10, row_y, 2, 8, 'F')
                    pdf.set_text_color(*Colors.DARK if not is_missing else Colors.MUTED)
                    pdf.cell(45, 5, f'  {metric}', align='L')

                    if is_missing:
                        # 미공시 표시
                        pdf.set_text_color(*Colors.MUTED)
                        pdf.cell(28, 5, '-', align='C')
                        pdf.cell(28, 5, '-', align='C')
                        pdf.cell(28, 5, '-', align='C')
                        pdf.cell(28, 5, '미공시', align='C')
                    else:
                        # 기업값
                        pdf.set_text_color(*Colors.DARK)
                        val_str = f"{company_val:.1f}%" if company_val is not None else '-'
                        pdf.cell(28, 5, val_str, align='C')

                        # 업종평균
                        pdf.set_text_color(*Colors.MUTED)
                        mean_str = f"{mean_val:.1f}%" if mean_val is not None else '-'
                        pdf.cell(28, 5, mean_str, align='C')

                        # 차이
                        pdf.set_text_color(*diff_color)
                        diff_str = f"{diff:+.1f}%p" if diff is not None else '-'
                        pdf.cell(28, 5, diff_str, align='C')

                        # 순위
                        rank_color = Colors.SUCCESS if pct is not None and pct <= 30 else Colors.DANGER if pct is not None and pct >= 70 else Colors.MUTED
                        pdf.set_text_color(*rank_color)
                        pdf.cell(28, 5, f"상위 {pct:.0f}%" if pct is not None else '-', align='C')

                    # 비교 바 (데이터 있을 때만)
                    bar_x = pdf.get_x() + 3
                    bar_w = 25
                    bar_h = 4

                    pdf.set_fill_color(*Colors.LIGHT)
                    pdf.rect(bar_x, row_y + 2, bar_w, bar_h, 'F')

                    if not is_missing and pct is not None:
                        fill_w = ((100 - pct) / 100) * bar_w
                        pdf.set_fill_color(*cat_color)
                        pdf.rect(bar_x, row_y + 2, fill_w, bar_h, 'F')

                    pdf.ln(8)

                pdf.ln(2)

        pdf.ln(3)

        # 강점/약점 카드 (2열)
        if strengths or weaknesses:
            col_width = 93

            # 강점 카드
            if strengths:
                card_y = pdf.get_y()
                pdf.set_fill_color(240, 255, 245)
                pdf.rect(10, card_y, col_width, 35, 'F')
                pdf.set_fill_color(*Colors.SUCCESS)
                pdf.rect(10, card_y, col_width, 3, 'F')

                pdf.set_xy(15, card_y + 5)
                pdf.set_font('AppleSD', '', 9)
                pdf.set_text_color(*Colors.SUCCESS)
                pdf.cell(0, 5, '강점 (업종 내 상위권)')

                for idx, s in enumerate(strengths[:3]):
                    pdf.set_xy(15, card_y + 12 + idx * 7)
                    pdf.set_font('AppleSD', '', 8)
                    pdf.set_text_color(*Colors.TEXT)
                    pdf.cell(0, 5, f"▲ {s['metric']}: 상위 {s['percentile']:.0f}%")

            # 약점 카드
            if weaknesses:
                pdf.set_fill_color(255, 245, 245)
                pdf.rect(10 + col_width + 4, card_y, col_width, 35, 'F')
                pdf.set_fill_color(*Colors.DANGER)
                pdf.rect(10 + col_width + 4, card_y, col_width, 3, 'F')

                pdf.set_xy(10 + col_width + 9, card_y + 5)
                pdf.set_font('AppleSD', '', 9)
                pdf.set_text_color(*Colors.DANGER)
                pdf.cell(0, 5, '약점 (업종 내 하위권)')

                for idx, w in enumerate(weaknesses[:3]):
                    pdf.set_xy(10 + col_width + 9, card_y + 12 + idx * 7)
                    pdf.set_font('AppleSD', '', 8)
                    pdf.set_text_color(*Colors.TEXT)
                    pdf.cell(0, 5, f"▼ {w['metric']}: 하위 {100-w['percentile']:.0f}%")

            pdf.set_y(card_y + 38)

    # =========================================================================
    # AI 예측 요약
    # =========================================================================

    def _add_ai_prediction_summary(self, pdf: PDFReport, ai_pred: Dict):
        """AI 예측 요약 섹션 (개선된 디자인)"""
        pdf.section_header('4', 'AI 예측 분석')

        summary = ai_pred.get('summary', {})
        all_predictions = ai_pred.get('all_predictions', {})

        # 미공시 지표 확인
        all_metrics = []
        for metrics in TARGET_METRICS.values():
            all_metrics.extend(metrics)
        missing_metrics = [m for m in all_metrics if not all_predictions.get(m, {}).get('predicted')]

        # 미공시 데이터 면책 문구
        if missing_metrics:
            disclaimer_y = pdf.get_y()
            pdf.set_fill_color(255, 248, 240)  # 연한 주황 배경
            pdf.rect(10, disclaimer_y, 190, 12, 'F')
            pdf.set_fill_color(*Colors.WARNING)
            pdf.rect(10, disclaimer_y, 3, 12, 'F')

            pdf.set_xy(15, disclaimer_y + 2)
            pdf.set_font('AppleSD', '', 7)
            pdf.set_text_color(*Colors.WARNING)
            pdf.cell(5, 4, '⚠')
            pdf.set_text_color(*Colors.TEXT)
            missing_str = ', '.join(missing_metrics[:5])
            if len(missing_metrics) > 5:
                missing_str += f' 외 {len(missing_metrics) - 5}개'
            pdf.multi_cell(170, 4, f'데이터 미공시 지표: {missing_str}. 일부 지표는 기업의 데이터 공시 여부에 따라 예측 결과가 제공되지 않을 수 있습니다.')
            pdf.ln(2)

        # 개선/악화 카운트
        improving_count = summary.get('improving_count', 0)
        declining_count = summary.get('declining_count', 0)
        stable_count = summary.get('stable_count', 0)

        # 전망 헤더 박스 (회색 - 비교 업종과 동일)
        header_y = pdf.get_y()
        pdf.set_fill_color(240, 240, 240)
        pdf.rect(10, header_y, 190, 20, 'F')

        pdf.set_xy(15, header_y + 3)
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(0, 5, '다음 분기 전망')

        # 개선/유지/악화 표시 (큰 글씨로, 아이콘 포함)
        pdf.set_xy(15, header_y + 9)
        pdf.set_font('AppleSD', '', 14)
        pdf.set_text_color(*Colors.SUCCESS)
        pdf.cell(55, 8, f'▲ 개선 {improving_count}')
        pdf.set_text_color(*Colors.WARNING)
        pdf.cell(55, 8, f'→ 유지 {stable_count}')
        pdf.set_text_color(*Colors.DANGER)
        pdf.cell(55, 8, f'▼ 악화 {declining_count}')

        pdf.set_y(header_y + 24)

        # 카테고리별 전망 카드 (카테고리 고유 색상 사용)
        category_outlook = summary.get('category_outlook', {})
        card_width = 36
        card_x = 10
        card_y = pdf.get_y()

        for category, cat_info in category_outlook.items():
            cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))

            # 카드 배경 (카테고리 연한 색상)
            pdf.set_fill_color(*cat_light)
            pdf.rect(card_x, card_y, card_width, 24, 'F')
            pdf.set_fill_color(*cat_color)
            pdf.rect(card_x, card_y, card_width, 3, 'F')

            # 카테고리명
            pdf.set_xy(card_x + 2, card_y + 5)
            pdf.set_font('AppleSD', '', 8)
            pdf.set_text_color(*cat_color)
            pdf.cell(card_width - 4, 4, category, align='C')

            # 개선/유지/악화 (카드 안에 표시)
            improving = cat_info.get('improving', 0)
            declining = cat_info.get('declining', 0)
            stable_cat = len([m for m in TARGET_METRICS.get(category, [])]) - improving - declining

            pdf.set_xy(card_x + 2, card_y + 12)
            pdf.set_font('AppleSD', '', 7)
            cell_w = (card_width - 4) / 3
            pdf.set_text_color(*Colors.SUCCESS)
            pdf.cell(cell_w, 4, f"▲{improving}", align='C')
            pdf.set_text_color(*Colors.WARNING)
            pdf.cell(cell_w, 4, f"→{stable_cat}", align='C')
            pdf.set_text_color(*Colors.DANGER)
            pdf.cell(cell_w, 4, f"▼{declining}", align='C')

            card_x += card_width + 2

        pdf.set_y(card_y + 28)

        # 카테고리별 지표 테이블 (카테고리 고유 색상)
        for category, metrics in TARGET_METRICS.items():
            if pdf.get_y() > 220:
                pdf.add_page()

            cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))

            # 카테고리 헤더
            header_y = pdf.get_y()
            pdf.set_fill_color(*cat_color)
            pdf.rect(10, header_y, 190, 8, 'F')

            pdf.set_xy(12, header_y + 2)
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(0, 4, category)
            pdf.ln(10)

            # 지표 테이블 (심플 그리드 스타일) - 모든 지표 표시
            for metric in metrics:
                pred = all_predictions.get(metric, {})
                cur = pred.get('current') if pred else None
                prd = pred.get('predicted') if pred else None
                change = pred.get('change', 0) if pred else None
                direction = pred.get('direction', 'stable') if pred else 'stable'

                # 데이터 미공시 여부 확인
                is_missing = (prd is None)

                if is_missing:
                    dir_color = Colors.MUTED
                    dir_icon = '-'
                else:
                    dir_color = Colors.SUCCESS if direction == 'improving' else Colors.DANGER if direction == 'declining' else Colors.WARNING
                    dir_icon = '▲' if direction == 'improving' else '▼' if direction == 'declining' else '→'

                row_y = pdf.get_y()
                row_h = 10

                # 행 배경 (미공시는 연한 회색)
                if is_missing:
                    pdf.set_fill_color(245, 245, 245)
                else:
                    pdf.set_fill_color(250, 250, 250)
                pdf.rect(10, row_y, 190, row_h, 'F')

                # 그리드 선
                pdf.set_draw_color(230, 230, 230)
                pdf.rect(10, row_y, 190, row_h, 'D')

                # 지표명
                pdf.set_xy(12, row_y + 2)
                pdf.set_font('AppleSD', '', 8)
                pdf.set_text_color(*Colors.DARK if not is_missing else Colors.MUTED)
                pdf.cell(38, 6, metric)

                # 구분선
                pdf.line(50, row_y, 50, row_y + row_h)

                # 현재값 (None이면 '-' 표시)
                pdf.set_xy(52, row_y + 2)
                pdf.set_text_color(*Colors.MUTED)
                pdf.set_font('AppleSD', '', 7)
                cur_str = f'현재 {cur:.1f}%' if cur is not None else '현재 -'
                pdf.cell(30, 6, cur_str)

                # 구분선
                pdf.line(82, row_y, 82, row_y + row_h)

                # 예측값 (미공시면 "미공시" 표시)
                pdf.set_xy(84, row_y + 2)
                pdf.set_font('AppleSD', '', 7)
                if is_missing:
                    pdf.set_text_color(*Colors.MUTED)
                    pdf.cell(30, 6, '미공시')
                else:
                    pdf.set_text_color(*Colors.DARK)
                    pdf.cell(30, 6, f'예측 {prd:.1f}%')

                # 구분선
                pdf.line(114, row_y, 114, row_y + row_h)

                # 변화량 (색상 강조) - 데이터 없으면 '-'
                pdf.set_xy(116, row_y + 1)
                pdf.set_font('AppleSD', '', 9)
                pdf.set_text_color(*dir_color)
                if not is_missing and cur is not None and change is not None:
                    change_str = f'{dir_icon} {change:+.1f}%p'
                else:
                    change_str = '-'
                    pdf.set_text_color(*Colors.MUTED)
                pdf.cell(30, 8, change_str, align='C')

                # 구분선
                pdf.line(146, row_y, 146, row_y + row_h)

                # 변화 바 (오른쪽)
                bar_x = 150
                bar_w = 46
                bar_h = 5
                bar_y = row_y + 2.5

                pdf.set_fill_color(*Colors.LIGHT)
                pdf.rect(bar_x, bar_y, bar_w, bar_h, 'F')

                # 변화 방향에 따른 바 (데이터 있을 때만)
                if not is_missing and cur is not None and change is not None and change != 0:
                    mid_x = bar_x + bar_w / 2
                    change_w = min(abs(change) / 10 * (bar_w / 2), bar_w / 2)
                    pdf.set_fill_color(*dir_color)
                    if change > 0:
                        pdf.rect(mid_x, bar_y, change_w, bar_h, 'F')
                    else:
                        pdf.rect(mid_x - change_w, bar_y, change_w, bar_h, 'F')

                    # 중앙선
                    pdf.set_draw_color(*Colors.MUTED)
                    pdf.line(mid_x, bar_y - 1, mid_x, bar_y + bar_h + 1)

                pdf.set_y(row_y + row_h)

            pdf.ln(5)

    # =========================================================================
    # 카테고리별 XAI + LLM 분석 (핵심 차별점)
    # =========================================================================

    def _add_category_xai_analysis(self, pdf: PDFReport, category_analysis: Dict, ai_pred: Dict):
        """카테고리별 XAI + LLM 분석 (개선된 디자인 + SHAP 시각화)"""
        pdf.add_page()
        pdf.section_header('5', 'XAI 분석 (AI 예측 근거)')

        all_predictions = ai_pred.get('all_predictions', {})

        # 미공시 지표 확인
        all_metrics = []
        for metrics in TARGET_METRICS.values():
            all_metrics.extend(metrics)
        missing_metrics = [m for m in all_metrics if not all_predictions.get(m, {}).get('predicted')]

        # 헤더 설명 박스
        header_y = pdf.get_y()
        pdf.set_fill_color(245, 248, 252)
        pdf.rect(10, header_y, 190, 18, 'F')
        pdf.set_fill_color(*Colors.PRIMARY)
        pdf.rect(10, header_y, 3, 18, 'F')

        pdf.set_xy(15, header_y + 2)
        pdf.set_font('AppleSD', '', 8)
        pdf.set_text_color(*Colors.PRIMARY)
        pdf.cell(0, 4, 'SHAP (SHapley Additive exPlanations) 분석이란?', ln=True)

        pdf.set_x(15)
        pdf.set_font('AppleSD', '', 7)
        pdf.set_text_color(*Colors.TEXT)
        pdf.multi_cell(180, 4, "AI 모델이 예측할 때 어떤 요인이 얼마나 기여했는지 수치화한 결과입니다. 양수(+)는 예측값 상승, 음수(-)는 하락에 기여합니다.")
        pdf.ln(2)

        # 미공시 데이터 면책 문구
        if missing_metrics:
            disclaimer_y = pdf.get_y()
            pdf.set_fill_color(255, 248, 240)
            pdf.rect(10, disclaimer_y, 190, 10, 'F')
            pdf.set_fill_color(*Colors.WARNING)
            pdf.rect(10, disclaimer_y, 3, 10, 'F')

            pdf.set_xy(15, disclaimer_y + 2)
            pdf.set_font('AppleSD', '', 7)
            pdf.set_text_color(*Colors.WARNING)
            pdf.cell(5, 4, '⚠')
            pdf.set_text_color(*Colors.TEXT)
            pdf.cell(170, 4, f'일부 지표({len(missing_metrics)}개)는 데이터 미공시로 XAI 분석 결과가 제공되지 않습니다.')
            pdf.ln(4)

        outlook_colors = {
            'positive': Colors.SUCCESS,
            'negative': Colors.DANGER,
            'neutral': Colors.MUTED
        }
        outlook_text = {
            'positive': '긍정적',
            'negative': '부정적',
            'neutral': '중립'
        }

        for category, analysis in category_analysis.items():
            if pdf.get_y() > 180:
                pdf.add_page()

            # 카테고리 헤더 (카테고리 고유 색상 사용)
            outlook = analysis.get('outlook', 'neutral')
            cat_color, cat_light = CATEGORY_COLORS.get(category, (Colors.PRIMARY, Colors.PRIMARY_LIGHT))

            header_y = pdf.get_y()
            pdf.set_fill_color(*cat_color)
            pdf.rect(10, header_y, 190, 10, 'F')

            pdf.set_xy(12, header_y + 2)
            pdf.set_font('AppleSD', '', 11)
            pdf.set_text_color(*Colors.WHITE)
            pdf.cell(150, 6, category)

            # 전망 뱃지
            outlook_badge_color = outlook_colors.get(outlook, Colors.MUTED)
            pdf.set_fill_color(*outlook_badge_color)
            pdf.set_font('AppleSD', '', 8)
            pdf.cell(30, 6, outlook_text.get(outlook, '중립'), fill=True, align='C')
            pdf.ln(12)

            # 카테고리 종합 분석 박스 - 텍스트 길이에 맞게 높이 계산
            summary = analysis.get('summary', '')
            if summary:
                box_y = pdf.get_y()
                pdf.set_fill_color(*cat_light)
                pdf.set_font('AppleSD', '', 8)
                # 정확한 높이 계산
                char_per_line = 82
                actual_lines = 1
                for line in summary.split('\n'):
                    actual_lines += max(1, len(line) / char_per_line)
                box_height = int(actual_lines * 4) + 4

                pdf.rect(10, box_y, 190, box_height, 'F')
                pdf.set_fill_color(*cat_color)
                pdf.rect(10, box_y, 3, box_height, 'F')
                pdf.set_xy(15, box_y + 2)
                pdf.set_text_color(*Colors.TEXT)
                pdf.multi_cell(182, 4, summary)
                pdf.set_y(box_y + box_height + 2)

            # 핵심 메시지
            key_messages = analysis.get('key_messages', [])
            if key_messages:
                for msg in key_messages[:2]:
                    pdf.set_x(12)
                    pdf.set_font('AppleSD', '', 8)
                    pdf.set_text_color(*cat_color)
                    pdf.cell(4, 5, '●')
                    pdf.set_text_color(*Colors.TEXT)
                    pdf.multi_cell(184, 5, msg)
                pdf.ln(2)

            # 지표별 상세 분석 (카드 형태) - 모든 지표 표시
            metrics_analysis = analysis.get('metrics', {})
            all_predictions = ai_pred.get('all_predictions', {})

            # TARGET_METRICS에서 해당 카테고리의 모든 지표 순회
            for metric in TARGET_METRICS.get(category, []):
                metric_data = metrics_analysis.get(metric, {})
                # 카드 높이(58) + 여유 공간 확보
                if pdf.get_y() > 210:
                    pdf.add_page()

                pred = all_predictions.get(metric, {})
                cur = pred.get('current') if pred else None
                prd = pred.get('predicted') if pred else None
                direction = pred.get('direction', 'stable') if pred else 'stable'
                shap_analysis = pred.get('shap_analysis', {}) if pred else {}

                # 데이터 미공시 여부
                is_missing = (prd is None)

                # 방향 색상
                if is_missing:
                    dir_color = Colors.MUTED
                    dir_icon = '-'
                elif direction == 'improving':
                    dir_color = Colors.SUCCESS
                    dir_icon = '▲'
                elif direction == 'declining':
                    dir_color = Colors.DANGER
                    dir_icon = '▼'
                else:
                    dir_color = Colors.MUTED
                    dir_icon = '→'

                # 지표 카드 높이 (미공시면 작게)
                card_y = pdf.get_y()
                card_height = 25 if is_missing else 72  # 미공시는 간단하게
                pdf.set_fill_color(255, 255, 255) if not is_missing else pdf.set_fill_color(250, 250, 250)
                pdf.rect(10, card_y, 190, card_height, 'F')
                pdf.set_fill_color(*cat_color if not is_missing else Colors.MUTED)
                pdf.rect(10, card_y, 3, card_height, 'F')

                # 지표명 + 예측값
                pdf.set_xy(15, card_y + 2)
                pdf.set_font('AppleSD', '', 10)
                pdf.set_text_color(*cat_color if not is_missing else Colors.MUTED)
                pdf.cell(40, 6, metric)

                if is_missing:
                    # 미공시 표시
                    pdf.set_text_color(*Colors.MUTED)
                    pdf.set_font('AppleSD', '', 9)
                    pdf.cell(80, 6, '데이터 미공시')
                    pdf.set_xy(15, card_y + 12)
                    pdf.set_font('AppleSD', '', 7)
                    pdf.set_text_color(*Colors.MUTED)
                    pdf.cell(180, 4, '해당 지표의 데이터가 공시되지 않아 예측 결과를 제공할 수 없습니다.')
                else:
                    pdf.set_text_color(*dir_color)
                    pdf.set_font('AppleSD', '', 11)
                    cur_str = f'{cur:.1f}%' if cur is not None else '-'
                    pdf.cell(80, 6, f'{dir_icon} {cur_str} → {prd:.1f}%')

                    # LLM 인사이트
                    insight = metric_data.get('insight', '')
                    if insight:
                        pdf.set_xy(15, card_y + 10)
                        pdf.set_font('AppleSD', '', 7)
                        pdf.set_text_color(*Colors.TEXT)
                        pdf.multi_cell(180, 4, insight[:150])

                    # SHAP Waterfall Chart (top/bottom 5개)
                    # shap_analysis에서 직접 가져옴 (예측 결과에서)
                    pos_factors = shap_analysis.get('positive_factors', [])[:5] if shap_analysis else []
                    neg_factors = shap_analysis.get('negative_factors', [])[:5] if shap_analysis else []
                    base_value = shap_analysis.get('base_value', cur) if shap_analysis else cur

                    # Waterfall 또는 바 차트 (SHAP 데이터 있을 때)
                    if base_value is not None and (pos_factors or neg_factors):
                        self._draw_shap_waterfall(pdf, base_value, prd,
                                                  pos_factors, neg_factors,
                                                  15, card_y + 22, 180, 48, cat_color)

                pdf.set_y(card_y + card_height + 3)

            pdf.ln(3)

    def _draw_shap_waterfall(self, pdf: PDFReport, base_value: float, predicted: float,
                              pos_factors: List, neg_factors: List,
                              x: float, y: float, width: float, height: float,
                              cat_color: Tuple = None):
        """SHAP Waterfall Chart - base_value에서 예측값까지의 누적 차트"""
        cat_color = cat_color or Colors.PRIMARY

        # 모든 요인 합치기 (양수 먼저, 음수 나중) - 5개씩
        all_factors = []
        for f in pos_factors[:5]:
            all_factors.append((f.get('feature', ''), f.get('shap_value', 0)))
        for f in neg_factors[:5]:
            all_factors.append((f.get('feature', ''), f.get('shap_value', 0)))

        if not all_factors:
            return

        # 표시된 요인들의 합계
        shown_sum = sum(f[1] for f in all_factors)
        # 기타 요인 (예측값 - base_value - 표시된 요인 합계)
        other_contribution = predicted - base_value - shown_sum

        # 기타 요인이 유의미하면 추가 (절대값 0.01 이상)
        has_other = abs(other_contribution) >= 0.01

        # 차트 영역
        chart_x = x + 60
        chart_w = width - 65
        row_count = len(all_factors) + 2 + (1 if has_other else 0)  # base + factors + (other) + final
        row_h = height / row_count

        # 값 범위 계산 (기타 요인 포함)
        cumulative_vals = [base_value]
        cumsum = base_value
        for f in all_factors:
            cumsum += f[1]
            cumulative_vals.append(cumsum)
        if has_other:
            cumulative_vals.append(cumsum + other_contribution)
        cumulative_vals.append(predicted)

        min_val = min(cumulative_vals)
        max_val = max(cumulative_vals)
        val_range = max_val - min_val if max_val != min_val else 1

        def val_to_x(val):
            return chart_x + ((val - min_val) / val_range) * chart_w

        current_val = base_value
        row_y = y

        # Base Value 행
        pdf.set_xy(x, row_y)
        pdf.set_font('AppleSD', '', 6)
        pdf.set_text_color(*Colors.MUTED)
        pdf.cell(58, row_h, 'Base (평균)', align='R')

        base_x = val_to_x(base_value)
        pdf.set_fill_color(*Colors.PRIMARY_LIGHT)
        pdf.rect(base_x - 1, row_y + 1, 3, row_h - 2, 'F')
        pdf.set_xy(base_x + 4, row_y)
        pdf.set_font('AppleSD', '', 5)
        pdf.set_text_color(*Colors.PRIMARY)
        pdf.cell(15, row_h, f'{base_value:.1f}')

        row_y += row_h

        # 각 피쳐 기여도
        for feature, shap_val in all_factors:
            feature_desc = get_feature_description(feature)
            if len(feature_desc) > 25:
                feature_desc = feature_desc[:23] + '..'

            pdf.set_xy(x, row_y)
            pdf.set_font('AppleSD', '', 5)
            pdf.set_text_color(*Colors.TEXT)
            pdf.cell(58, row_h, feature_desc, align='R')

            # 누적 바 그리기
            start_x = val_to_x(current_val)
            new_val = current_val + shap_val
            end_x = val_to_x(new_val)

            bar_color = Colors.SUCCESS if shap_val > 0 else Colors.DANGER

            # 연결선
            pdf.set_draw_color(220, 220, 220)
            pdf.line(start_x, row_y - 1, start_x, row_y + row_h / 2)

            # 바
            bar_x = min(start_x, end_x)
            bar_w = abs(end_x - start_x)
            pdf.set_fill_color(*bar_color)
            pdf.rect(bar_x, row_y + 1, max(bar_w, 1), row_h - 2, 'F')

            # 값 표시
            pdf.set_xy(end_x + 2, row_y)
            pdf.set_font('AppleSD', '', 5)
            pdf.set_text_color(*bar_color)
            pdf.cell(12, row_h, f'{shap_val:+.2f}')

            current_val = new_val
            row_y += row_h

        # 기타 요인 행 (갭 메꾸기)
        if has_other:
            pdf.set_xy(x, row_y)
            pdf.set_font('AppleSD', '', 5)
            pdf.set_text_color(*Colors.MUTED)
            pdf.cell(58, row_h, '기타 요인', align='R')

            start_x = val_to_x(current_val)
            new_val = current_val + other_contribution
            end_x = val_to_x(new_val)

            bar_color = Colors.SUCCESS if other_contribution > 0 else Colors.DANGER

            # 연결선
            pdf.set_draw_color(220, 220, 220)
            pdf.line(start_x, row_y - 1, start_x, row_y + row_h / 2)

            # 바 (점선 스타일로 다르게 표시)
            bar_x = min(start_x, end_x)
            bar_w = abs(end_x - start_x)
            # 연한 색상으로 표시
            light_color = (200, 230, 200) if other_contribution > 0 else (230, 200, 200)
            pdf.set_fill_color(*light_color)
            pdf.rect(bar_x, row_y + 1, max(bar_w, 1), row_h - 2, 'F')

            # 값 표시
            pdf.set_xy(end_x + 2, row_y)
            pdf.set_font('AppleSD', '', 5)
            pdf.set_text_color(*bar_color)
            pdf.cell(12, row_h, f'{other_contribution:+.2f}')

            current_val = new_val
            row_y += row_h

        # Final Prediction 행
        pdf.set_xy(x, row_y)
        pdf.set_font('AppleSD', '', 6)
        pdf.set_text_color(*Colors.DARK)
        pdf.cell(58, row_h, '예측값', align='R')

        # 연결선 (기타 요인에서 예측값으로)
        pdf.set_draw_color(220, 220, 220)
        pdf.line(val_to_x(current_val), row_y - 1, val_to_x(current_val), row_y + row_h / 2)

        pred_x = val_to_x(predicted)
        pdf.set_fill_color(*Colors.PRIMARY)
        pdf.rect(pred_x - 2, row_y + 1, 5, row_h - 2, 'F')
        pdf.set_xy(pred_x + 5, row_y)
        pdf.set_font('AppleSD', '', 6)
        pdf.set_text_color(*Colors.PRIMARY)
        pdf.cell(15, row_h, f'{predicted:.1f}%')

    def _draw_confidence_gauge(self, pdf: PDFReport, r2: float, confidence: str,
                                x: float, y: float, size: float = 15):
        """모델 신뢰도 게이지 (R² 기반)"""
        # 배경 원호
        center_x = x + size
        center_y = y + size

        # R² 기반 각도 (0~1 → 0~180도)
        angle = min(r2, 1.0) * 180

        # 색상
        if confidence == 'high':
            color = Colors.SUCCESS
        elif confidence == 'medium':
            color = Colors.WARNING
        else:
            color = Colors.DANGER

        # 배경 반원
        pdf.set_fill_color(*Colors.LIGHT)
        for a in range(180, 361, 5):
            rad = math.radians(a)
            px = center_x + size * math.cos(rad)
            py = center_y - size * math.sin(rad)

        # 값 반원 (다각형으로 근사)
        pdf.set_fill_color(*color)

        # 중앙 텍스트
        pdf.set_xy(x, y + size - 3)
        pdf.set_font('AppleSD', '', 8)
        pdf.set_text_color(*color)
        pdf.cell(size * 2, 6, f'R²={r2:.2f}', align='C')

        pdf.set_xy(x, y + size + 3)
        pdf.set_font('AppleSD', '', 6)
        pdf.set_text_color(*Colors.MUTED)
        pdf.cell(size * 2, 4, confidence, align='C')

    def _draw_shap_bar_chart(self, pdf: PDFReport, pos_factors: List, neg_factors: List,
                              x: float, y: float, width: float, height: float):
        """SHAP 기여도 바 차트 (양방향)"""
        all_factors = []
        for f in pos_factors:
            all_factors.append((f.get('feature', ''), f.get('shap_value', 0), 'pos'))
        for f in neg_factors:
            all_factors.append((f.get('feature', ''), f.get('shap_value', 0), 'neg'))

        if not all_factors:
            return

        # SHAP 값 범위 계산
        all_shap = [abs(f[1]) for f in all_factors]
        max_shap = max(all_shap) if all_shap else 1

        # 바 차트 영역
        bar_area_x = x + 55
        bar_area_w = width - 60
        bar_h = 3
        row_h = height / max(len(all_factors), 1)

        # 중앙선 (0점) + 라벨
        center_x = bar_area_x + bar_area_w / 2
        pdf.set_draw_color(180, 180, 180)
        pdf.set_line_width(0.3)
        pdf.line(center_x, y - 2, center_x, y + height)
        pdf.set_line_width(0.2)

        # 축 라벨
        pdf.set_font('AppleSD', '', 4)
        pdf.set_text_color(*Colors.MUTED)
        pdf.set_xy(bar_area_x, y - 4)
        pdf.cell(bar_area_w / 2, 3, '(-) 하락 기여', align='C')
        pdf.cell(bar_area_w / 2, 3, '(+) 상승 기여', align='C')

        # 각 요인별 바
        for idx, (feature, shap_val, factor_type) in enumerate(all_factors[:5]):
            row_y = y + idx * row_h

            # 피쳐 설명
            feature_desc = get_feature_description(feature)
            if len(feature_desc) > 18:
                feature_desc = feature_desc[:16] + '..'

            pdf.set_xy(x, row_y)
            pdf.set_font('AppleSD', '', 5)
            pdf.set_text_color(*Colors.TEXT)
            pdf.cell(53, row_h, feature_desc, align='R')

            # 바 그리기
            bar_w = (abs(shap_val) / max_shap) * (bar_area_w / 2) * 0.85

            if factor_type == 'pos':
                pdf.set_fill_color(*Colors.SUCCESS)
                pdf.rect(center_x + 1, row_y + (row_h - bar_h) / 2, bar_w, bar_h, 'F')
                pdf.set_xy(center_x + bar_w + 3, row_y)
                pdf.set_text_color(*Colors.SUCCESS)
                pdf.set_font('AppleSD', '', 5)
                pdf.cell(12, row_h, f'+{shap_val:.2f}')
            else:
                pdf.set_fill_color(*Colors.DANGER)
                pdf.rect(center_x - bar_w - 1, row_y + (row_h - bar_h) / 2, bar_w, bar_h, 'F')
                pdf.set_xy(center_x - bar_w - 15, row_y)
                pdf.set_text_color(*Colors.DANGER)
                pdf.set_font('AppleSD', '', 5)
                pdf.cell(12, row_h, f'{shap_val:.2f}', align='R')

                pdf.ln(2)

            pdf.ln(5)

    # =========================================================================
    # 종합 의견
    # =========================================================================

    def _add_comprehensive_opinion(self, pdf: PDFReport, opinion: Dict):
        """
        통합 종합의견 섹션 (내러티브 구조)

        한줄요약 → 현황분석 → 추세분석 → AI전망 → 주시포인트 → 결론
        """
        pdf.section_header('6', '종합 의견')

        # 1. 한줄 요약 (강조 박스)
        headline = opinion.get('headline', '')
        if headline:
            # 배경 박스
            box_y = pdf.get_y()
            pdf.set_fill_color(*Colors.PRIMARY_LIGHT)
            pdf.rect(10, box_y, 190, 14, 'F')

            # 텍스트
            pdf.set_xy(15, box_y + 3)
            pdf.set_font('AppleSD', '', 11)
            pdf.set_text_color(*Colors.PRIMARY_DARK)
            pdf.multi_cell(180, 6, headline)
            pdf.set_y(box_y + 16)

        pdf.ln(3)

        # 2. 현황 분석
        analysis = opinion.get('analysis', '')
        if analysis:
            pdf.subsection_title('현황 분석')
            pdf.body_text(analysis)
            pdf.ln(2)

        # 3. 추세 분석
        trend_text = opinion.get('trend', '')
        if trend_text:
            pdf.subsection_title('추세 분석')
            pdf.body_text(trend_text)
            pdf.ln(2)

        # 4. AI 전망
        forecast = opinion.get('forecast', '')
        if forecast:
            pdf.subsection_title('AI 전망')
            pdf.body_text(forecast)
            pdf.ln(2)

        # 5. 주시 포인트
        watch_points = opinion.get('watch_points', [])
        if watch_points:
            pdf.subsection_title('주시 포인트')
            for i, point in enumerate(watch_points[:3], 1):
                pdf.set_font('AppleSD', '', 9)
                # 아이콘 색상 (1: 주의, 2: 기회, 3: 확인)
                if i == 1:
                    icon_color = Colors.WARNING
                elif i == 2:
                    icon_color = Colors.SUCCESS
                else:
                    icon_color = Colors.PRIMARY

                pdf.set_text_color(*icon_color)
                pdf.cell(5, 5, '▶')
                pdf.set_text_color(*Colors.TEXT)
                pdf.cell(0, 5, f' {point}', ln=True)
            pdf.ln(2)

        # 6. 결론 (강조 박스)
        conclusion = opinion.get('conclusion', '')
        if conclusion:
            # 배경 박스
            box_y = pdf.get_y()
            pdf.set_fill_color(*Colors.LIGHT)
            pdf.set_draw_color(*Colors.PRIMARY)
            pdf.set_line_width(0.5)
            pdf.rect(10, box_y, 190, 18, 'FD')
            pdf.set_line_width(0.2)

            # 레이블
            pdf.set_xy(15, box_y + 2)
            pdf.set_font('AppleSD', '', 8)
            pdf.set_text_color(*Colors.PRIMARY)
            pdf.cell(0, 4, '결론', ln=True)

            # 본문
            pdf.set_x(15)
            pdf.set_font('AppleSD', '', 10)
            pdf.set_text_color(*Colors.DARK)
            pdf.multi_cell(180, 5, conclusion[:200])

            pdf.set_y(box_y + 22)

    # =========================================================================
    # 별첨: 분석 방법론
    # =========================================================================

    def _add_methodology_appendix(self, pdf: PDFReport):
        """
        별첨: 분석 방법론

        등급 산정 기준, AI 모델 성능, 지표 설명 등
        보고서에서 사용된 방법론을 설명합니다.
        """
        from .config import (
            GRADE_THRESHOLDS, GRADE_SCORES, MODEL_R2, MODEL_CONFIDENCE,
            TARGET_METRICS, METRIC_DIRECTION, METRIC_DESCRIPTION
        )

        pdf.section_header('별첨', '분석 방법론')

        # ---------------------------------------------------------------------
        # 1. 등급 산정 기준
        # ---------------------------------------------------------------------
        pdf.subsection_title('1. 등급 산정 기준')

        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.TEXT)
        pdf.multi_cell(0, 5, (
            "본 보고서의 등급은 동일 업종 내 상대적 위치(백분위)를 기준으로 산정됩니다. "
            "각 지표별로 업종 내 순위를 계산한 후, 아래 기준에 따라 등급을 부여합니다."
        ))
        pdf.ln(3)

        # 등급 테이블
        table_data = [
            ('등급', '백분위 기준', '점수', '해석'),
            ('A+', '상위 10% 이내', '6점', '매우 우수'),
            ('A', '상위 25% 이내', '5점', '우수'),
            ('B+', '상위 40% 이내', '4점', '양호'),
            ('B', '상위 60% 이내', '3점', '보통'),
            ('C', '상위 80% 이내', '2점', '주의'),
            ('F', '하위 20%', '1점', '위험'),
        ]

        col_widths = [20, 50, 25, 40]
        row_height = 7

        for i, row in enumerate(table_data):
            if i == 0:
                # 헤더
                pdf.set_fill_color(*Colors.DARK)
                pdf.set_text_color(*Colors.WHITE)
                pdf.set_font('AppleSD', '', 9)
            else:
                pdf.set_fill_color(*Colors.LIGHT if i % 2 == 0 else Colors.WHITE)
                pdf.set_text_color(*Colors.TEXT)
                pdf.set_font('AppleSD', '', 9)

            x_start = 25
            for j, (cell, width) in enumerate(zip(row, col_widths)):
                pdf.set_xy(x_start, pdf.get_y())
                pdf.cell(width, row_height, cell, border=0, fill=True, align='C')
                x_start += width
            pdf.ln(row_height)

        pdf.ln(3)
        pdf.set_font('AppleSD', '', 8)
        pdf.set_text_color(*Colors.MUTED)
        pdf.multi_cell(0, 4, (
            "* 종합 등급은 13개 지표 점수의 평균으로 산출됩니다. "
            "결측 지표는 계산에서 제외됩니다."
        ))
        pdf.ln(5)

        # ---------------------------------------------------------------------
        # 2. AI 모델 성능
        # ---------------------------------------------------------------------
        pdf.subsection_title('2. AI 예측 모델 성능')

        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.TEXT)
        pdf.multi_cell(0, 5, (
            "본 보고서의 AI 예측은 XGBoost 앙상블 모델을 사용합니다. "
            "각 지표별로 개별 모델을 학습하였으며, R² (결정계수)를 기준으로 예측 신뢰도를 평가합니다."
        ))
        pdf.ln(3)

        # 신뢰도 기준 설명
        pdf.set_font('AppleSD', '', 8)
        pdf.set_text_color(*Colors.MUTED)
        pdf.multi_cell(0, 4, (
            f"신뢰도 기준: 높음 (R² ≥ {MODEL_CONFIDENCE['high']}), "
            f"중간 ({MODEL_CONFIDENCE['medium']} ≤ R² < {MODEL_CONFIDENCE['high']}), "
            f"낮음 (R² < {MODEL_CONFIDENCE['medium']})"
        ))
        pdf.ln(3)

        # 모델 성능 테이블
        # 카테고리별로 그룹화
        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.TEXT)

        for category, metrics in TARGET_METRICS.items():
            # 카테고리 헤더
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.PRIMARY)
            pdf.cell(0, 6, f"■ {category}", ln=True)

            # 지표별 R² 표시
            for metric in metrics:
                r2 = MODEL_R2.get(metric, 0)
                direction = METRIC_DIRECTION.get(metric, 'higher')
                dir_text = '↑' if direction == 'higher' else '↓'

                # 신뢰도 레벨
                if r2 >= MODEL_CONFIDENCE['high']:
                    confidence = '높음'
                    conf_color = Colors.SUCCESS
                elif r2 >= MODEL_CONFIDENCE['medium']:
                    confidence = '중간'
                    conf_color = Colors.WARNING
                else:
                    confidence = '낮음'
                    conf_color = Colors.DANGER

                pdf.set_x(20)
                pdf.set_font('AppleSD', '', 8)
                pdf.set_text_color(*Colors.TEXT)
                pdf.cell(55, 5, f"{metric} {dir_text}")

                pdf.set_text_color(*Colors.MUTED)
                pdf.cell(25, 5, f"R² = {r2:.4f}")

                pdf.set_text_color(*conf_color)
                pdf.cell(20, 5, f"({confidence})", ln=True)

            pdf.ln(1)

        pdf.ln(3)

        # ---------------------------------------------------------------------
        # 3. 분석 지표 설명
        # ---------------------------------------------------------------------
        # 페이지 여백 확인
        if pdf.get_y() > 200:
            pdf.add_page()

        pdf.subsection_title('3. 분석 지표 설명')

        pdf.set_font('AppleSD', '', 9)
        pdf.set_text_color(*Colors.TEXT)
        pdf.multi_cell(0, 5, (
            "본 보고서에서 분석하는 13개 재무지표의 정의와 해석 방법입니다."
        ))
        pdf.ln(3)

        for category, metrics in TARGET_METRICS.items():
            # 카테고리 헤더
            pdf.set_font('AppleSD', '', 9)
            pdf.set_text_color(*Colors.PRIMARY)
            pdf.cell(0, 6, f"■ {category}", ln=True)

            for metric in metrics:
                description = METRIC_DESCRIPTION.get(metric, '')
                direction = METRIC_DIRECTION.get(metric, 'higher')
                dir_text = '(높을수록 좋음)' if direction == 'higher' else '(낮을수록 좋음)'

                pdf.set_x(20)
                pdf.set_font('AppleSD', '', 8)
                pdf.set_text_color(*Colors.DARK)
                pdf.cell(50, 5, f"• {metric}")

                pdf.set_text_color(*Colors.MUTED)
                pdf.cell(25, 5, dir_text, ln=True)

                if description:
                    pdf.set_x(25)
                    pdf.set_font('AppleSD', '', 7)
                    pdf.set_text_color(*Colors.TEXT)
                    # 설명이 너무 길면 잘라서 표시
                    desc_short = description[:80] + ('...' if len(description) > 80 else '')
                    pdf.cell(0, 4, desc_short, ln=True)

            pdf.ln(2)

        # ---------------------------------------------------------------------
        # 4. 면책 조항
        # ---------------------------------------------------------------------
        pdf.ln(5)
        pdf.set_draw_color(*Colors.MUTED)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(3)

        pdf.set_font('AppleSD', '', 7)
        pdf.set_text_color(*Colors.MUTED)
        pdf.multi_cell(0, 4, (
            "본 보고서는 공시된 재무정보를 기반으로 AI 모델이 자동 생성한 참고 자료입니다. "
            "투자 권유나 재무 자문을 목적으로 하지 않으며, 투자 결정에 대한 책임은 투자자 본인에게 있습니다. "
            "AI 예측은 과거 데이터를 기반으로 하며, 실제 미래 성과와 다를 수 있습니다. "
            "정확한 분석을 위해서는 전문가 상담을 권장합니다."
        ))


def generate_pdf_report(company_code: str, output_dir: Optional[str] = None) -> str:
    generator = PDFReportGenerator()
    return generator.generate(company_code, output_dir)
