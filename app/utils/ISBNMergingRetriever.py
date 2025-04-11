from typing import List, Dict, Optional, Any
from collections import defaultdict
from langchain.schema import BaseRetriever
from langchain.docstore.document import Document

class ISBNMergingRetriever(BaseRetriever):
    """
    검색된 Document 리스트를 ISBN 기준으로 그룹화하고,
    같은 ISBN을 가진 문서들의 page_content를 병합하며,
    그룹 내에서 가장 정보가 많은 메타데이터를 대표로 사용합니다.
    """
    base_retriever: BaseRetriever

    def _extract_isbn(self, doc: Document) -> Optional[str]:
        for key in ["ISBN", "isbn"]:
            if key in doc.metadata:
                isbn_val = doc.metadata[key]
                return str(isbn_val).replace(".0", "").strip() if isbn_val else None
        inner_meta = doc.metadata.get("metadata", {})
        if isinstance(inner_meta, dict):
            for key in ["ISBN", "isbn"]:
                if key in inner_meta:
                    isbn_val = inner_meta[key]
                    return (
                        str(isbn_val).replace(".0", "").strip()
                        if isbn_val
                        else None
                    )
            inner_inner_meta = inner_meta.get("metadata", {})
            if isinstance(inner_inner_meta, dict):
                for key in ["ISBN", "isbn"]:
                    if key in inner_inner_meta:
                        isbn_val = inner_inner_meta[key]
                        return (
                            str(isbn_val).replace(".0", "").strip()
                            if isbn_val
                            else None
                        )
        source_meta = doc.metadata.get("_source", {}).get("metadata", {})
        if isinstance(source_meta, dict):
            for key in ["ISBN", "isbn"]:
                if key in source_meta:
                    isbn_val = source_meta[key]
                    return (
                        str(isbn_val).replace(".0", "").strip()
                        if isbn_val
                        else None
                    )
        return None

    def _get_best_metadata(self, doc_list: List[Document]) -> Dict[str, Any]:
        best_meta = {}
        max_score = -1
        required_keys = {"title", "author", "ISBN"}
        for doc in doc_list:
            current_meta = doc.metadata
            score = 0
            keys_lower = {k.lower() for k in current_meta.keys()}
            required_keys_lower = {rk.lower() for rk in required_keys}
            score += sum(
                1
                for req_key in required_keys_lower
                if req_key in keys_lower
                and current_meta.get(
                    next((k for k in current_meta if k.lower() == req_key), None)
                )
            )
            score += len(current_meta) * 0.1
            if score > max_score:
                max_score = score
                best_meta = current_meta
        if best_meta:
            pass
        else:
            if doc_list:
                best_meta = doc_list[0].metadata
        return dict(best_meta)

    def _merge_documents_by_isbn(self, docs: List[Document], is_async=False) -> List[Document]:
        grouped = defaultdict(list)
        merged_docs = []
        for idx, doc in enumerate(docs):
            isbn = self._extract_isbn(doc)
            if isbn:
                grouped[isbn].append(doc)
        for isbn, doc_list in grouped.items():
            merged_meta = self._get_best_metadata(doc_list)
            combined_text = "\n\n---\n\n".join(
                d.page_content for d in doc_list if d.page_content
            ).strip()
            if combined_text:
                merged_docs.append(Document(page_content=combined_text, metadata=merged_meta))
        return merged_docs

    def _get_relevant_documents(self, query: str) -> List[Document]:
        try:
            docs = self.base_retriever.get_relevant_documents(query)
        except Exception as e:
            docs = []
        return self._merge_documents_by_isbn(docs, is_async=False)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        try:
            docs = await self.base_retriever.aget_relevant_documents(query)
        except Exception as e:
            docs = []
        return self._merge_documents_by_isbn(docs, is_async=True)
