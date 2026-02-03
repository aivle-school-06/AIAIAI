# TODO - AI Server 개발 체크리스트

## 완료된 작업 ✅

- [x] 프로젝트 구조 생성
- [x] FastAPI 기본 설정 (main.py, config.py)
- [x] API 엔드포인트 설계
  - [x] 헬스체크 API
  - [x] 데이터 파이프라인 API
  - [x] 분석 API
  - [x] 관리자 API
- [x] Pydantic 스키마 정의
- [x] 서비스 레이어 구조
- [x] Dockerfile 작성
- [x] docker-compose.yml 작성
- [x] 문서화 (README, ARCHITECTURE)

### Phase 1: Core 모듈 이식 ✅ (완료)

- [x] `core/constants.py` - 지표 설정 이식
- [x] `core/interfaces.py` - 인터페이스/프로토콜 정의
- [x] `core/data_loader.py` - 기존 data_loader 이식
- [x] `core/predictor.py` - 기존 ai_predictor 이식
- [x] `dependencies.py` - 의존성 주입 설정
- [x] `exceptions.py` - 커스텀 예외 클래스

### Phase 2: 서비스 로직 구현 ✅ (완료)

- [x] AnalysisService 실제 구현
  - [x] 기업 목록 조회
  - [x] 기업 분석 (예측 + SHAP)
  - [x] 업종 비교
  - [x] 일괄 예측
- [x] API ↔ 서비스 ↔ Core 연동 완료
- [x] 전역 예외 핸들러 구현
- [x] 요청 로깅 미들웨어 구현

---

## 진행 예정 작업 📋

### Phase 2-1: 추가 Core 모듈 (선택)

- [ ] `core/grade_calculator.py` - 등급 계산기 이식
- [ ] `core/report_generator.py` - 보고서 생성 이식
- [ ] `core/pdf_generator.py` - PDF 생성 이식
- [ ] `core/llm_opinion.py` - LLM 분석 이식
- [ ] 폰트 파일 복사

### Phase 3: 테스트

- [ ] API 테스트 작성 (pytest)
- [ ] 서비스 테스트 작성
- [ ] Core 모듈 테스트 작성
- [ ] 통합 테스트

### Phase 4: 배포

- [ ] Azure Container Registry 설정
- [ ] Azure Container Apps 설정
- [ ] GitHub Actions CI/CD 설정
- [ ] 환경변수 설정 (Azure)
- [ ] 도메인/SSL 설정

### Phase 5: 백엔드 연동

- [ ] API 스펙 협의
- [ ] 데이터 전달 로직 구현
- [ ] 에러 처리 및 재시도 로직
- [ ] 로깅 및 모니터링

---

## 현재 구조

```
ai-server/
├── app/
│   ├── main.py                 # FastAPI 앱 + 미들웨어 + 예외핸들러
│   ├── config.py               # 환경 설정 (Pydantic Settings)
│   ├── dependencies.py         # 의존성 주입 (Container, Depends)
│   ├── exceptions.py           # 커스텀 예외 클래스
│   ├── api/
│   │   ├── router.py           # API 라우터 통합
│   │   └── v1/
│   │       ├── health.py       # 헬스체크
│   │       ├── analysis.py     # 분석 API (★ 핵심)
│   │       ├── data.py         # 데이터 파이프라인
│   │       └── admin.py        # 관리자
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py        # 상수 (지표, 등급 기준)
│   │   ├── interfaces.py       # 인터페이스 (Protocol)
│   │   ├── data_loader.py      # 데이터 로딩 (★)
│   │   └── predictor.py        # XGBoost 예측 + SHAP (★)
│   ├── services/
│   │   ├── analysis_service.py # 분석 비즈니스 로직 (★)
│   │   ├── data_pipeline.py    # 데이터 파이프라인
│   │   └── monitoring_service.py
│   └── models/
│       ├── request.py          # 요청 스키마
│       └── response.py         # 응답 스키마
├── data/, ml_models/, reports/, tests/
├── Dockerfile, docker-compose.yml
└── docs/
```

## 주요 API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/v1/health` | 서버 상태 확인 |
| GET | `/api/v1/analysis/companies` | 기업 목록 (필터, 페이지네이션) |
| GET | `/api/v1/analysis/industries` | 업종 목록 |
| GET | `/api/v1/analysis/{code}` | 기업 종합 분석 |
| GET | `/api/v1/analysis/{code}/predict` | 예측 결과 |
| GET | `/api/v1/analysis/{code}/shap/{metric}` | SHAP 분석 |
| GET | `/api/v1/analysis/{code}/historical` | 과거 데이터 |
| POST | `/api/v1/analysis/batch` | 일괄 예측 |

## 참고사항

### 기존 모듈 위치 (원본 - 건드리지 않음)
```
/Users/guna_bb/Desktop/BigBig/backend/report/
├── data_loader.py
├── ai_predictor.py
├── grade_calculator.py
├── report_generator.py
├── pdf_generator.py
├── llm_opinion.py
├── config.py
└── fonts/
```

### 데이터 위치
```
/Users/guna_bb/Desktop/BigBig/data/processed/  # 전처리된 데이터
/Users/guna_bb/Desktop/BigBig/models/XGBoost/outputs/  # 학습된 모델
```

---

## 수정 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-02 | 초기 TODO 작성 |
| 2026-02-02 | Phase 1 Core 모듈 이식 완료 |
| 2026-02-02 | Phase 2 서비스 로직 구현 완료 |
| 2026-02-02 | FastAPI 베스트 프랙티스 적용 (DI, 예외처리, 미들웨어) |
