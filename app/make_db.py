import os
import pickle
import pandas as pd
from config.settings import load_env_variables
from langchain.docstore.document import Document
from langchain_community.embeddings import ClovaXEmbeddings
from langchain_community.chat_models import ChatClovaX
from pymilvus import connections, utility
from langchain_community.vectorstores.milvus import Milvus
from langchain.schema import Document
from elasticsearch import Elasticsearch, helpers
from langchain_elasticsearch import ElasticsearchStore

milvus_host, milvus_port, collection_name, es_url, es_index_name = load_env_variables()

ncp_embeddings = ClovaXEmbeddings(model="bge-m3")
llm_clova = ChatClovaX(model="HCX-003", max_tokens=2048)


embedding_file = r"/app/data/embedding.pkl"
# embedding_file = r'../data/embedding.pkl'

if not os.path.exists(embedding_file):
    raise FileNotFoundError(
        f"{os.listdir('/app/data/embedding.pkl')}"
    )
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        saved_data = pickle.load(f)
    all_text_embedding_pairs = [(v["text"], v["embedding"])
                                for v in saved_data.values() if "text" in v and "embedding" in v]
    all_metadata_list = [v["metadata"]
                            for v in saved_data.values() if "metadata" in v]
    if len(all_text_embedding_pairs) != len(all_metadata_list):
        min_len = min(len(all_text_embedding_pairs),
                        len(all_metadata_list))
        all_text_embedding_pairs = all_text_embedding_pairs[:min_len]
        all_metadata_list = all_metadata_list[:min_len]

metadata_mapping = {
    "ISBN": "ISBN",
    "페이지": "page",
    "가격": "price",
    "제목": "title",
    "부제": "subtitle",
    "저자": "author",
    "분류": "category",
    "저자소개": "author_intro",
    "책소개": "book_intro",
    "목차": "table_of_contents",
    "출판사리뷰": "publisher_review",
    "추천사": "recommendation",
    "발행자": "publisher",
    "표지": "book_cover",
}


def clean_metadata(meta):
    cleaned = {}
    target_keys = list(metadata_mapping.values())
    original_to_target = {
        k_orig: k_target for k_orig, k_target in metadata_mapping.items()
    }
    for target_key in target_keys:
        original_key = next(
            (k for k, v in original_to_target.items() if v == target_key), None
        )
        value = (
            meta.get(original_key)
            if original_key and original_key in meta
            else meta.get(target_key)
        )
        if pd.isna(value):
            cleaned[target_key] = 0 if target_key in ["page", "price"] else ""
        elif target_key == "ISBN":
            try:
                str_value = str(value).strip()
                cleaned[target_key] = (
                    str_value[:-2] if str_value.endswith(".0") else str_value
                )
            except Exception as e:
                cleaned[target_key] = str(value).strip()
        elif target_key == "subtitle":
            cleaned[target_key] = str(value)
        elif target_key in ["page", "price"]:
            try:
                cleaned[target_key] = int(float(value))
            except (ValueError, TypeError):
                cleaned[target_key] = 0
        else:
            cleaned[target_key] = str(value)
    return cleaned


documents = []
for i, (pair, meta) in enumerate(zip(all_text_embedding_pairs, all_metadata_list)):
    cleaned_meta = clean_metadata(meta)
    documents.append(Document(page_content=pair[0], metadata=cleaned_meta))

texts = [doc.page_content for doc in documents]
embeds = [pair[1] for pair in all_text_embedding_pairs[: len(documents)]]
metadatas = [doc.metadata for doc in documents]

# Milvus Vector DB
temp_conn_alias = "utility_check_conn"
connections.connect(alias=temp_conn_alias, host=milvus_host, port=int(milvus_port))

if utility.has_collection(collection_name, using=temp_conn_alias):
    utility.drop_collection(collection_name, using=temp_conn_alias)

if connections.get_connection_addr(temp_conn_alias):
    connections.disconnect(temp_conn_alias)


vectorstore = Milvus(
    embedding_function=ncp_embeddings,
    collection_name=collection_name,
    connection_args={"host": milvus_host, "port": milvus_port},
    auto_id=True,
)

original_embed_documents = ClovaXEmbeddings.embed_documents

def precomputed_embed_documents(cls, input_texts):
    if len(input_texts) == len(texts) and all(
        t1 == t2 for t1, t2 in zip(input_texts, texts)
    ):
        return embeds
    else:
        return original_embed_documents.__func__(cls, input_texts)

ClovaXEmbeddings.embed_documents = classmethod(precomputed_embed_documents)
vectorstore.add_texts(texts=texts, metadatas=metadatas)
ClovaXEmbeddings.embed_documents = original_embed_documents

# Elasticsearch store 설정
es_client = Elasticsearch(hosts=[es_url], request_timeout=120)
if not es_client.ping():
    raise ConnectionError("Elasticsearch 연결 실패. 서버 상태 및 URL 확인 필요.")
if es_client.indices.exists(index=es_index_name):
    es_client.indices.delete(index=es_index_name, ignore=[400, 404])

es_store = ElasticsearchStore(
    index_name=es_index_name,
    es_connection=es_client,
    strategy=ElasticsearchStore.ExactRetrievalStrategy(),
)

actions = [
    {
        "_index": es_index_name,
        "_source": {"text": texts[i], "metadata": metadatas[i]},
    }
    for i in range(len(texts))
]
indexed_count, errors = helpers.bulk(es_client, actions, raise_on_error=False)

if errors:
    raise
es_client.indices.refresh(index=es_index_name)