from langchain_community.chat_models import ChatClovaX
from langchain_community.embeddings import ClovaXEmbeddings
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def initialize_embeddings():
    return ClovaXEmbeddings(model="bge-m3")

def get_llm():
    return ChatClovaX(model="HCX-003", max_tokens=2048)


def is_similar_question(new_emb, prev_embeds, threshold=0.65):
    if not prev_embeds:
        return False
    sim_scores = cosine_similarity([new_emb], prev_embeds)[0]
    max_score = np.max(sim_scores)
    return max_score > threshold

def extract_field(text, field_name):
    pattern = rf"^\s*{re.escape(field_name)}\s*[:：]\s*(.*?)\s*$"
    lines = text.splitlines()
    for line in lines:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def extract_metadata_field(doc, field, default = "N/A"):
    """
    Document의 다양한 metadata 구조에서 필드 값을 추출 (예: title, ISBN 등).
    """
    candidates = []
    candidates.append(doc.metadata)
    if "metadata" in doc.metadata and isinstance(doc.metadata["metadata"], dict):
        candidates.append(doc.metadata["metadata"])
        inner = doc.metadata["metadata"]
        if "metadata" in inner and isinstance(inner["metadata"], dict):
            candidates.append(inner["metadata"])
    if "_source" in doc.metadata and isinstance(doc.metadata["_source"], dict):
        source_meta = doc.metadata["_source"].get("metadata", {})
        if isinstance(source_meta, dict):
            candidates.append(source_meta)
    for meta in candidates:
        if field in meta:
            return (meta[field].strip() if isinstance(meta[field], str) else str(meta[field]))
    return default