#!/bin/bash
# =============================================================================
# Azure 전체 배포 스크립트
# =============================================================================
# 사용법: ./scripts/deploy-azure.sh
#
# 사전 준비:
# 1. Azure CLI 설치: brew install azure-cli
# 2. Docker 설치 및 실행
# 3. az login 실행
# =============================================================================

set -e

# -----------------------------------------------------------------------------
# 설정 (필요시 수정)
# -----------------------------------------------------------------------------
RESOURCE_GROUP="bigbig-rg"
LOCATION="koreacentral"
STORAGE_ACCOUNT="bigbigaistorage"
STORAGE_CONTAINER="ai-server-assets"
ACR_NAME="bigbigacr"
APP_NAME="ai-server"
ENVIRONMENT_NAME="bigbig-env"

# 색상
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=============================================="
echo "Azure 전체 배포 시작"
echo -e "==============================================${NC}"

# -----------------------------------------------------------------------------
# 1. Azure 로그인 확인
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[1/7] Azure 로그인 확인...${NC}"
if ! az account show > /dev/null 2>&1; then
    echo "Azure 로그인이 필요합니다."
    az login
fi
echo -e "${GREEN}✓ 로그인 완료${NC}"

# -----------------------------------------------------------------------------
# 2. 리소스 그룹 생성
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[2/7] 리소스 그룹 생성...${NC}"
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION \
    --output none
echo -e "${GREEN}✓ 리소스 그룹: $RESOURCE_GROUP${NC}"

# -----------------------------------------------------------------------------
# 3. Storage Account 생성 및 데이터 업로드
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[3/7] Storage Account 설정...${NC}"

# Storage Account 생성
az storage account create \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --sku Standard_LRS \
    --output none 2>/dev/null || echo "Storage account already exists"

# Container 생성
az storage container create \
    --name $STORAGE_CONTAINER \
    --account-name $STORAGE_ACCOUNT \
    --output none 2>/dev/null || echo "Container already exists"

# Connection String 가져오기
STORAGE_CONN_STR=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query connectionString -o tsv)

echo -e "${GREEN}✓ Storage Account: $STORAGE_ACCOUNT${NC}"

# 데이터 파일 업로드
echo "  데이터 파일 업로드 중..."
if [ -f "./data/processed/03_dataset_final.csv" ]; then
    az storage blob upload \
        --account-name $STORAGE_ACCOUNT \
        --container-name $STORAGE_CONTAINER \
        --file "./data/processed/03_dataset_final.csv" \
        --name "data/03_dataset_final.csv" \
        --overwrite \
        --output none
    echo "  ✓ 03_dataset_final.csv"
fi

if [ -f "./data/processed/03_feature_cols.json" ]; then
    az storage blob upload \
        --account-name $STORAGE_ACCOUNT \
        --container-name $STORAGE_CONTAINER \
        --file "./data/processed/03_feature_cols.json" \
        --name "data/03_feature_cols.json" \
        --overwrite \
        --output none
    echo "  ✓ 03_feature_cols.json"
fi

# 모델 파일 업로드
echo "  모델 파일 업로드 중..."
for f in ./ml_models/*.joblib; do
    if [ -f "$f" ]; then
        filename=$(basename "$f")
        az storage blob upload \
            --account-name $STORAGE_ACCOUNT \
            --container-name $STORAGE_CONTAINER \
            --file "$f" \
            --name "models/$filename" \
            --overwrite \
            --output none
        echo "  ✓ $filename"
    fi
done

echo -e "${GREEN}✓ 데이터/모델 업로드 완료${NC}"

# -----------------------------------------------------------------------------
# 4. Container Registry 생성 및 이미지 빌드
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[4/7] Container Registry 설정...${NC}"

# ACR 생성
az acr create \
    --name $ACR_NAME \
    --resource-group $RESOURCE_GROUP \
    --sku Basic \
    --admin-enabled true \
    --output none 2>/dev/null || echo "ACR already exists"

ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer -o tsv)

# ACR Tasks로 클라우드에서 빌드 (로컬 Docker 불필요)
echo "  클라우드에서 Docker 이미지 빌드 중... (ACR Tasks)"
az acr build \
    --registry $ACR_NAME \
    --image $APP_NAME:latest \
    --file Dockerfile \
    .

echo -e "${GREEN}✓ 이미지 빌드 완료: $ACR_LOGIN_SERVER/$APP_NAME:latest${NC}"

# -----------------------------------------------------------------------------
# 5. Container Apps 환경 생성
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[5/7] Container Apps 환경 생성...${NC}"

# Container Apps 확장 설치
az extension add --name containerapp --upgrade --yes 2>/dev/null || true

# 환경 생성
az containerapp env create \
    --name $ENVIRONMENT_NAME \
    --resource-group $RESOURCE_GROUP \
    --location $LOCATION \
    --output none 2>/dev/null || echo "Environment already exists"

echo -e "${GREEN}✓ 환경: $ENVIRONMENT_NAME${NC}"

# -----------------------------------------------------------------------------
# 6. Container App 배포
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[6/7] Container App 배포...${NC}"

# ACR 자격 증명
ACR_USERNAME=$(az acr credential show --name $ACR_NAME --query username -o tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query "passwords[0].value" -o tsv)

# OPENAI_API_KEY 확인
if [ -z "$OPENAI_API_KEY" ]; then
    echo -e "${RED}⚠ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.${NC}"
    echo "  export OPENAI_API_KEY='your-key' 후 다시 실행하세요."
    OPENAI_API_KEY="not-set"
fi

# Container App 생성/업데이트
az containerapp create \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --environment $ENVIRONMENT_NAME \
    --image $ACR_LOGIN_SERVER/$APP_NAME:latest \
    --registry-server $ACR_LOGIN_SERVER \
    --registry-username $ACR_USERNAME \
    --registry-password $ACR_PASSWORD \
    --target-port 8000 \
    --ingress external \
    --cpu 1 \
    --memory 2Gi \
    --min-replicas 0 \
    --max-replicas 3 \
    --env-vars \
        AZURE_STORAGE_CONNECTION_STRING="$STORAGE_CONN_STR" \
        AZURE_STORAGE_CONTAINER="$STORAGE_CONTAINER" \
        OPENAI_API_KEY="$OPENAI_API_KEY" \
    --output none 2>/dev/null || \
az containerapp update \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --image $ACR_LOGIN_SERVER/$APP_NAME:latest \
    --output none

echo -e "${GREEN}✓ Container App 배포 완료${NC}"

# -----------------------------------------------------------------------------
# 7. 배포 결과 확인
# -----------------------------------------------------------------------------
echo -e "\n${YELLOW}[7/7] 배포 결과 확인...${NC}"

APP_URL=$(az containerapp show \
    --name $APP_NAME \
    --resource-group $RESOURCE_GROUP \
    --query "properties.configuration.ingress.fqdn" -o tsv)

echo -e "\n${GREEN}=============================================="
echo "배포 완료!"
echo "=============================================="
echo -e "앱 URL: https://$APP_URL"
echo -e "Health: https://$APP_URL/api/v1/health"
echo -e "Swagger: https://$APP_URL/docs"
echo -e "==============================================${NC}"

# Health check
echo -e "\n서버 상태 확인 중... (최대 60초 대기)"
for i in {1..12}; do
    if curl -s "https://$APP_URL/api/v1/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ 서버 정상 작동!${NC}"
        curl -s "https://$APP_URL/api/v1/health" | python3 -m json.tool
        exit 0
    fi
    echo "  대기 중... ($i/12)"
    sleep 5
done

echo -e "${YELLOW}⚠ 서버 시작 중... 잠시 후 다시 확인하세요.${NC}"
