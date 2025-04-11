from typing import Dict
from config.settings import load_env_variables
from data.data_loader import DataLoader
from utils.helper import initialize_embeddings
from utils.milvus_manager import MilvusVectorManager
from pipelines.factory import PipelineFactory
from models.base_rag import BaseRAGPipeline

class RAGSystem:
    def __init__(self):
        self._initialized = False
        self.retriever = None
        self.embedding_model = None
        self.milvus_manager = None
        self.vectorstore = None
        self.session: Dict[str, BaseRAGPipeline] = {}   # 세션별 파이프라인 저장

    def initialize(self):
        if self._initialized:
            return

        # 1. 환경 변수 로드 (ClovaX API 키 등)
        milvus_host, milvus_port, collection_name, es_url, es_index_name = load_env_variables()
        # 2. 데이터 로드 및 임베딩 모델 로드
        loader = DataLoader(embedding_file="data/embedding.pkl")
        self.documents = loader.get_documents()
        self.embedding_model = initialize_embeddings()

        # 3. Milvus 연결 설정
        self.milvus_manager = MilvusVectorManager(
            embedding_model=self.embedding_model,
            collection_name=collection_name,
            host=milvus_host,
            port=int(milvus_port),
            auto_id=True
        )
        self.milvus_manager.initialize_collection()
        self.retriever, self.es_store = self.milvus_manager.create_retriever(loader, es_url, es_index_name)
        self.vectorstore = self.milvus_manager.get_vectorstore()
        self._initialized = True
        
    def get_session_pipeline(self, session_id: str, persona_choice: str):
        """세션별 파이프라인 생성 또는 반환"""
        if session_id not in self.session:
            persona_map = {"문학": "Literature",
                           "과학": "Science",
                           "일반": "General"}
            mapped_choice = persona_map.get(persona_choice, "General")
            persona = PipelineFactory(
                embedding_model=self.embedding_model,
                vectorstore=self.vectorstore,
                es_store = self.es_store,
                retriever=self.retriever,
                documents=self.documents,
            ).create_pipeline(mapped_choice)
            self.session[session_id] = persona
        return self.session[session_id]

# 인스턴스 생성
rag_system = RAGSystem()
rag_system.initialize()
