import pickle
import os
import pandas as pd
from langchain.docstore.document import Document

class DataLoader:
    def __init__(self, embedding_file):
        self.embedding_file = embedding_file
        self._metadata_mapping = {
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
        self.documents = []
        
    def _load_file(self):
        if os.path.exists(self.embedding_file):
            with open(self.embedding_file, "rb") as f:
                data = pickle.load(f)
                all_text_embedding_pairs = [(v["text"], v["embedding"])
                                            for v in data.values() if "text" in v and "embedding" in v]
                all_metadata_list = [v["metadata"]
                                    for v in data.values() if "metadata" in v]
                min_len = min(len(all_text_embedding_pairs), len(all_metadata_list))
                all_text_embedding_pairs = all_text_embedding_pairs[:min_len]
                all_metadata_list = all_metadata_list[:min_len]
        else:
            raise FileNotFoundError(f"임베딩 파일을 찾을 수 없습니다: {self.embedding_file}")
        return all_text_embedding_pairs, all_metadata_list
       
    def _clean_metadata(self, meta):
        cleaned = {}
        target_keys = list(self._metadata_mapping.values())
        original_to_target = {
            k_orig: k_target for k_orig, k_target in self._metadata_mapping.items()
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
                str_value = str(value).strip()
                cleaned[target_key] = (str_value[:-2] if str_value.endswith(".0") else str_value)
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
    
    def _create_documents(self):
        all_text_embedding_pairs, all_metadata_list = self._load_file()
        for i, (pair, meta) in enumerate(zip(all_text_embedding_pairs, all_metadata_list)):
            cleaned_meta = self._clean_metadata(meta)
            self.documents.append(
                Document(page_content=pair[0], metadata=cleaned_meta))
            
    def get_documents(self):
        if not self.documents:
            self._create_documents()
        return self.documents
        
    def get_texts(self):
        if not self.documents:
            self._create_documents()
        return [doc.page_content for doc in self.documents]

    def get_metadatas(self):
        if not self.documents:
            self._create_documents()
        return [doc.metadata for doc in self.documents]    