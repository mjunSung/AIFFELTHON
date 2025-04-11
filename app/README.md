# Folder Structure
```
├── api                         # FastAPI 핵심 기능 모듈
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
├── config                      # 환경 설정 관련 파일
│   └── settings.py
├── core                        # 시스템 로직 및 핵심 기능 참조
│   └── system.py
├── data                        # 데이터 관련 리소스
│   ├── data_loader.py
│   └── embedding.pkl
├── docker                      # Docker 설정 파일
│   ├── backend.Dockerfile
│   └── frontend.Dockerfile
├── models                      # RAG 기반 모델 정의
│   ├── base_rag.py
│   ├── persona.py
│   └── template.py
├── pipelines                   # RAG 파이프라인 구성
│   └── factory.py
├── ui                          # 단순 UI 컴포넌트(not used)
│   └── greetings.py
├── utils                       # 유틸리티 및 DB 연결 관리 도구
│   ├── helper.py
│   ├── ISBNMergingRetriever.py
│   └── milvus_manager.py
├── web                         # Streamlit 기반 프론트엔드 앱
│    └── app.py
├── requirements.txt            # 프로젝트 의존성 목록
├── docker-compose.yml          # 전체 서비스 도커 컴포즈 설정
└── make_db.py                  # 초기 DB 생성 및 스키마 정의 스크립트
```


# How to Start?

### 1. .env 파일 설정 (config/.env에 저장)
```
echo "NCP_CLOVASTUDIO_API_KEY=<your_api_key>" >> config/.env
echo "NCP_CLOVASTUDIO_API_URL=<URL>" >> config/.env
echo "MILVUS_HOST = <hostname>" >> config/.env
echo "MILVUS_PORT = <port>" >> config/.env
echo "MILVUS_COLLECTION_NAME = <collection_name>" >> config/.env
echo "ES_URL = <ES_URL>" >> config/.env
echo "ES_INDEX_NAME = <Index_name>" >> config/.env
```
----
### 2. Docker-Compose 실행
```
docker-compose up --build
```