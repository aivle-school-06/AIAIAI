# AI Server 아키텍처 문서

## 개요

KOSPI/KOSDAQ 상장기업 재무지표 예측 및 분석 보고서 생성을 위한 AI 서버입니다.

- **프레임워크**: FastAPI
- **ML 모델**: XGBoost (13개 재무지표 예측)
- **배포**: Azure Container Apps (Docker)

---

## 주요 기능

### 1️⃣ 데이터 파이프라인 (분기별)
- 새로운 분기 데이터 업로드 및 처리
- 예측용/학습용 데이터 최신화
- 13개 주요 지표 계산 후 백엔드 전달

### 2️⃣ 기업 분석 서비스
- 예측값 + SHAP 분석 제공
- PDF 보고서 생성
- LLM 기반 인사이트

### 3️⃣ 모델 모니터링 (관리자용)
- 예측 vs 실제 비교
- 모델 성능 피드백
- 재학습 트리거

### 4️⃣ 비정형 데이터 처리 (추후)
- 뉴스/공시 데이터 NLP 처리
- 감성 분석

---

## 폴더 구조

```
ai-server/
├── app/
│   ├── main.py                 # FastAPI 앱 진입점
│   ├── config.py               # 환경설정
│   │
│   ├── api/v1/                 # API 엔드포인트
│   │   ├── data.py             # 데이터 파이프라인
│   │   ├── analysis.py         # 분석 요청
│   │   ├── admin.py            # 관리자 기능
│   │   └── health.py           # 헬스체크
│   │
│   ├── services/               # 비즈니스 로직
│   │   ├── data_pipeline.py    # 데이터 처리
│   │   ├── analysis_service.py # 분석 서비스
│   │   └── monitoring_service.py # 모니터링
│   │
│   ├── core/                   # 핵심 ML/분석 모듈
│   │   ├── data_loader.py
│   │   ├── predictor.py
│   │   ├── report_generator.py
│   │   ├── pdf_generator.py
│   │   └── llm_opinion.py
│   │
│   ├── models/                 # Pydantic 스키마
│   └── utils/                  # 유틸리티
│
├── data/                       # 데이터 저장소
├── ml_models/                  # 학습된 모델 파일
├── reports/                    # 생성된 PDF
├── tests/                      # 테스트
├── docs/                       # 문서
├── Dockerfile
└── requirements.txt
```

---

## API 엔드포인트

### 데이터 파이프라인 (`/api/v1/data`)
| Method | Endpoint | 설명 |
|--------|----------|------|
| POST | `/upload` | 분기 데이터 업로드 |
| POST | `/process` | 데이터 처리 시작 |
| GET | `/status/{job_id}` | 처리 상태 확인 |
| GET | `/metrics/{corp_code}` | 지표 조회 |

### 분석 (`/api/v1/analysis`)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/{corp_code}` | 기업 분석 데이터 |
| GET | `/{corp_code}/report` | PDF 보고서 |
| GET | `/{corp_code}/predict` | 예측값만 |
| GET | `/batch` | 일괄 조회 |

### 관리자 (`/api/v1/admin`)
| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/model/performance` | 성능 현황 |
| POST | `/model/retrain` | 재학습 |
| GET | `/model/feedback/{quarter}` | 분기별 피드백 |

---

## 기술 스택

- **FastAPI**: 비동기 웹 프레임워크
- **XGBoost**: 재무지표 예측 모델
- **SHAP**: 예측 설명 (XAI)
- **OpenAI GPT**: LLM 분석
- **FPDF2**: PDF 생성
- **Pydantic**: 데이터 검증
- **Docker**: 컨테이너화
- **Azure Container Apps**: 배포

---

## 환경변수

```env
# API Keys
OPENAI_API_KEY=your_openai_key

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Paths
DATA_PATH=/app/data
MODEL_PATH=/app/ml_models
REPORT_PATH=/app/reports

# Cache
CACHE_TTL=3600
```

---

## 수정 이력

| 날짜 | 내용 | 작성자 |
|------|------|--------|
| 2026-02-02 | 초기 아키텍처 설계 | - |

