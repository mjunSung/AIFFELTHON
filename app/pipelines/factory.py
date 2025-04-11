from utils.helper import get_llm
from models.persona import LiteratureRAGPipeline
from models.persona import ScienceRAGPipeline
from models.persona import GeneralRAGPipeline

class PipelineFactory:
    def __init__(self, embedding_model, vectorstore, es_store, retriever, documents):
        self.llm = get_llm()
        self.embedding_model = embedding_model
        self.vectorstore = vectorstore
        self.es_store = es_store
        self.retriever = retriever
        self.documents = documents
        self.persona = None
        self._pipelines = {
            "Literature": self._create_literature_pipeline,
            "Science": self._create_science_pipeline,
            "General": self._create_general_pipeline
        }

    def create_pipeline(self, choice):
        """선택에 따라 적절한 페르소나 생성"""        
        persona = self._pipelines.get(choice, self._create_general_pipeline)
        self.persona = persona()
        return self.persona

    def _create_literature_pipeline(self):
        return LiteratureRAGPipeline(
            llm=self.llm,
            embeddings=self.embedding_model,
            vectorstore=self.vectorstore,
            es_store=self.es_store,
            retriever=self.retriever,
            documents=self.documents,
        )

    def _create_science_pipeline(self):
        return ScienceRAGPipeline(
            llm=self.llm,
            embeddings=self.embedding_model,
            vectorstore=self.vectorstore,
            es_store=self.es_store,
            retriever=self.retriever,
            documents=self.documents,
        )

    def _create_general_pipeline(self):
        return GeneralRAGPipeline(
            llm=self.llm,
            embeddings=self.embedding_model,
            vectorstore=self.vectorstore,
            es_store=self.es_store,
            retriever=self.retriever,
            documents=self.documents,
        )