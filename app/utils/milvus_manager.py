from pymilvus import connections
from langchain_milvus.vectorstores import Milvus
from langchain_elasticsearch import ElasticsearchRetriever
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore
from langchain.retrievers import EnsembleRetriever

from utils.ISBNMergingRetriever import ISBNMergingRetriever

class MilvusVectorManager:
    def __init__(self, embedding_model, collection_name, host, port, auto_id):  
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        self.host = host
        self.port = port
        # self.connection_args = {"host": host, "port": port}
        self.auto_id = auto_id
        
    def initialize_collection(self):
        """DB 연결"""
        connections.connect(alias="default", host=self.host, port=int(self.port))
        self.vectorstore = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.host, "port": self.port, "uri": "http://standalone:19530"},
            auto_id=self.auto_id
        )
    
    def get_vectorstore(self):
        return self.vectorstore

    def create_retriever(self, loader, es_url, es_index_name):
        """검색기 생성"""
        es_client = Elasticsearch(hosts=[es_url], request_timeout=120)
        if es_client.indices.exists(index=es_index_name):
            es_client.indices.delete(index=es_index_name, ignore=[400, 404])
        es_store = ElasticsearchStore(
            index_name=es_index_name,
            es_connection=es_client,
            strategy=ElasticsearchStore.ExactRetrievalStrategy(),
        )
        texts = loader.get_texts()
        metadatas = loader.get_metadatas()
        actions = [
            {"_index": es_index_name,
             "_source": {"text": texts[i], "metadata": metadatas[i]},}
            for i in range(len(texts))
        ]
        indexed_count, errors = helpers.bulk(es_client, actions, raise_on_error=False)
        if errors:
            raise ConnectionError("Elastic error!")
        es_client.indices.refresh(index=es_index_name)
        if "es_client" not in locals() or not es_client.ping():
            raise ConnectionError("Elasticsearch client is not connected.")
        
        sparse_retriever = ElasticsearchRetriever(
            es_client=es_client,
            index_name=es_index_name,
            body_func=lambda query: {
                "size": 5,
                "query": {"match": {"text": {"query": query}}},
            },
            content_field="text",
            metadata_field="metadata",
        )
        dense_retriever = self.vectorstore.as_retriever(search_kwargs={"k": 5})
        hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever], weights=[0.5, 0.5], c=60)
        return ISBNMergingRetriever(base_retriever=hybrid_retriever), es_store