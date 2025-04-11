from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.schemas import ChatRequest, ChatResponse
from api.dependencies import get_rag_system
from core.system import RAGSystem
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Chat API",
    description="도서 추천 RAG 시스템 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HealthCheckResponse(BaseModel):
    status: str
    version: str


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """서비스 상태 확인 엔드포인트"""
    return {
        "status": "OK",
        "version": "1.0.0"
    }


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    rag_system: RAGSystem = Depends(get_rag_system)
):
    """
    채팅 요청 처리 엔드포인트
    - message: 사용자 입력 메시지
    - session_id: 세션 식별자
    - persona: 페르소나 선택 (문학/과학/일반)
    """
    try:
        logger.info(
            f"New request - Session: {request.session_id}, Persona: {request.persona}")

        # 세션별 파이프라인 획득
        pipeline = rag_system.get_session_pipeline(
            request.session_id,
            request.persona
        )

        # 메시지 처리
        response_text = await pipeline.process_message(request.message)

        return ChatResponse(
            response=response_text,
            session_id=request.session_id
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail="Service unavailable")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
