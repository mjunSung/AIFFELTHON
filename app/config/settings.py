import os
from dotenv import load_dotenv

def load_env_variables():
    load_dotenv()
    os.environ["NCP_CLOVASTUDIO_API_KEY"] = os.getenv("NCP_CLOVASTUDIO_API_KEY")
    os.environ["NCP_CLOVASTUDIO_API_URL"] = os.getenv(
        "NCP_CLOVASTUDIO_API_URL", "https://clovastudio.stream.ntruss.com/")
    return (
        os.getenv("MILVUS_HOST", "standalone"),
        os.getenv("MILVUS_PORT"),
        os.getenv("MILVUS_COLLECTION_NAME"),
        os.getenv("ES_URL"),
        os.getenv("ES_INDEX_NAME"),
    )
