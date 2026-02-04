# AI Financial Analysis Server

KOSPI/KOSDAQ 상장기업 재무지표 예측 및 분석 보고서 생성 API 서버

## 주요 기능

1. **데이터 파이프라인** - 분기 데이터 처리 및 13개 재무지표 계산
2. **예측 분석** - XGBoost 모델 기반 재무지표 예측 + SHAP 분석
3. **보고서 생성** - PDF 형태의 상세 분석 보고서
4. **모델 모니터링** - 예측 성능 피드백 및 관리

## 기술 스택

- **Framework**: FastAPI
- **ML**: XGBoost, SHAP, scikit-learn
- **LLM**: OpenAI GPT
- **PDF**: FPDF2
- **Deploy**: Docker, Azure App Service

## 빠른 시작

### 1. 환경 설정

```bash
# 저장소 이동
cd ai-server

# 환경변수 설정
cp .env.example .env
# .env 파일에서 OPENAI_API_KEY 설정

# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
# 개발 모드
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 또는 Docker 사용
docker-compose up -d
```

### 3. API 문서

서버 실행 후 브라우저에서:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 엔드포인트

### 데이터 파이프라인
```
POST /api/v1/data/upload      - 분기 데이터 업로드
POST /api/v1/data/process     - 데이터 처리 시작
GET  /api/v1/data/status/{id} - 처리 상태 확인
GET  /api/v1/data/metrics/{code} - 지표 조회
```

### 분석
```
GET /api/v1/analysis/{code}        - 기업 분석
GET /api/v1/analysis/{code}/report - PDF 보고서
GET /api/v1/analysis/{code}/predict - 예측값만
```

### 관리자
```
GET  /api/v1/admin/model/performance - 모델 성능
GET  /api/v1/admin/model/feedback/{quarter} - 분기별 피드백
POST /api/v1/admin/cache/refresh - 캐시 갱신
```

## 프로젝트 구조

```
ai-server/
├── app/
│   ├── main.py           # FastAPI 앱
│   ├── config.py         # 설정
│   ├── api/v1/           # API 엔드포인트
│   ├── services/         # 비즈니스 로직
│   ├── core/             # ML 모듈
│   └── models/           # Pydantic 스키마
├── data/                 # 데이터
├── ml_models/            # 학습된 모델
├── reports/              # 생성된 PDF
├── Dockerfile
└── docker-compose.yml
```

## 배포 (Azure App Service)

### 자동 배포 (GitHub Actions)

main 브랜치에 푸시하면 자동으로 배포됩니다.

```bash
git push origin main
```

### 수동 배포

```bash
# Azure CLI 로그인
az login

# Container Registry에 이미지 빌드 & 푸시
az acr build --registry bigbigacr --image ai-server:latest .

# App Service에 새 이미지 배포
az webapp config container set \
  --name bigbig-ai-server \
  --resource-group bigbig-rg \
  --container-image-name bigbigacr.azurecr.io/ai-server:latest

# 앱 재시작
az webapp restart --name bigbig-ai-server --resource-group bigbig-rg
```

### 배포 URL

- **API**: https://bigbig-ai-server.azurewebsites.net
- **Swagger**: https://bigbig-ai-server.azurewebsites.net/docs
- **Health**: https://bigbig-ai-server.azurewebsites.net/api/v1/health

## 개발 가이드

### 새 API 추가

1. `app/api/v1/` 에 라우터 파일 생성
2. `app/api/router.py` 에 등록
3. `app/services/` 에 비즈니스 로직 구현

### 테스트

```bash
pytest tests/ -v
```

## 라이선스

MIT
