from typing import List, Dict, Optional
from langchain.chains import LLMChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import re
import json
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper import extract_field, is_similar_question, extract_metadata_field
from models.template import get_general_role

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class BaseRAGPipeline:
    def __init__(self, config, llm, embeddings, vectorstore, es_store, retriever, documents):
        self.config = config  # config에 페르소나별 템플릿 및 가중치 포함
        self.llm = llm
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.es_store = es_store
        self.retriever = retriever
        self.documents = documents
        self.MIN_INFO_LENGTH = 10
        self.previous_additional_question_embeddings=[]
        self.user_history = []
        self.llm_history = []
        self.user_preferences = self._initialize_preferences()
        self.preferences_text = "수집된 선호도 없음"
        self.preference_update_count = 0
        self.last_recommendations = []
        self.last_action = None
        
        # 페르소나별 프롬프트 체인 생성
        self.extract_pref_chain = LLMChain(
            llm=self.llm, prompt=self.config["pref_extraction_template"])
        self.decision_chain = LLMChain(
            llm=self.llm, prompt=self.config.get("decision_template", None))
        self.final_query_generation_chain = LLMChain(
            llm=self.llm, prompt=self.config["final_query_template"])
        self.refine_chain = LLMChain(
            llm=self.llm, prompt=self.config["refine_template"])
        self.query_expansion_chain = LLMChain(
            llm=self.llm, prompt=self.config["expansion_template"])
        self.re_ranking_chain = LLMChain(
            llm=self.llm, prompt=self.config.get("re_ranking_template", None))
        self.hyde_generation_chain = LLMChain(
            llm=self.llm, prompt=self.config.get("hyde_generation_template", None))
        self.hyde_keyword_chain = LLMChain(
            llm=self.llm, prompt=self.config.get("hyde_keyword_template", None))
        self.recommendation_intro_chain = LLMChain(
            llm=self.llm, prompt=self.config["recommendation_intro_template"])
        self.recommendation_outro_chain = LLMChain(
            llm=self.llm, prompt=self.config["recommendation_outro_template"])

    async def process_message(self, user_query: str) -> str:
        """웹 요청 처리용 단일 메시지 처리 메서드"""
        self._validate_query(user_query)
        self._update_query_history(f"사용자: {user_query}")
        
        answer = await self.process_query(user_query)
        self._update_query_history(f"챗봇: {answer}")

        return answer
        #return self._format_response(answer)

    def _validate_query(self, query: str):
        """입력 검증"""
        if len(query) > 500:
            raise ValueError("Query too long")
        if not query.strip():
            raise ValueError("Empty query")

    def _update_query_history(self, entry: str):
        """히스토리 관리 (최대 30개 유지)"""
        self.user_history.append(entry)
        if len(self.user_history) > 30:
            self.user_history.pop(0)

    def _format_response(self, raw_answer: str) -> str:
        """응답 포맷팅"""
        cleaned = re.sub(r"\n+", "\n", raw_answer)
        return cleaned.strip()
    
    def _initialize_preferences(self) -> Dict[str, List[str]]:
        return {
            "title": [],
            "author": [],
            "category": [],
            "author_intro": [],
            "book_intro": [],
            "table_of_contents": [],
            "purpose": [],
            "implicit info": [],
        }
        
    async def _async_invoke(self, chain, vars_dict, step_name):
        """LLMChain을 비동기로 실행하고 로깅"""
        try:
            log_vars = {
                k: (v[:100] + "..." if isinstance(v, str) and len(v) > 100 else v)
                for k, v in vars_dict.items()
            }
            logger.debug(f"[{step_name}] Chain 호출 시작. 입력 변수 일부: {log_vars}")
            result = await asyncio.to_thread(chain.invoke, vars_dict)
            result_text = result.get("text", "")
            log_result = (
                result_text[:200] +
                "..." if len(result_text) > 200 else result_text
            )
            logger.debug(f"[{step_name}] Chain 호출 완료. 결과 일부: {log_result}")
            return result
        except Exception as e:
            logger.error(f"[{step_name}] Chain 호출 중 예외 발생: {e}", exc_info=True)
            return {"text": ""}
        
    async def _async_invoke_llm(self, prompt, step_name):
        """LLM 직접 호출을 비동기로 실행하고 로깅"""
        try:
            log_prompt = prompt[:200] + "..." if len(prompt) > 200 else prompt
            logger.debug(f"[{step_name}] LLM 호출 시작. 프롬프트 일부: {log_prompt}")
            response = await asyncio.to_thread(self.llm.invoke, prompt)
            if hasattr(response, "content"):
                result_text = response.content.strip()
            elif isinstance(response, str):
                result_text = response.strip()
            else:
                result_text = str(response).strip()
            log_result = (
                result_text[:200] +
                "..." if len(result_text) > 200 else result_text)
            logger.debug(f"[{step_name}] LLM 호출 완료. 응답 일부: {log_result}")
            return result_text
        except Exception as e:
            logger.error(f"[{step_name}] LLM 호출 중 예외 발생: {e}", exc_info=True)
            return ""

    def robust_parse_decision_response(self, response_text):
        action = None
        additional_question = ""
        action_match = re.search(r'행동\s*[:：]\s*"?([^"\n]+)"?', response_text)
        if action_match:
            action = action_match.group(1).strip().lower()
            if action not in ["추천", "추가 질문"]:
                logger.warning(
                    f"알 수 없는 행동 값 파싱됨: '{action}'. '추가 질문'으로 처리."
                )
                action = "추가 질문"
        follow_match = re.search(
            r'추가\s*질문\s*[:：]\s*"?(.+)"?', response_text, re.DOTALL
        )
        if follow_match:
            additional_question = follow_match.group(1).strip()
        if action == "추가 질문" and not additional_question:
            additional_question = "어떤 점이 궁금하신가요? 또는 어떤 책을 찾으시나요?"
            logger.warning(
                f"행동은 '추가 질문'이나 질문 내용 없음. 기본 질문 사용: '{additional_question}'"
            )
        if not action:
            logger.warning(
                f"행동 결정 파싱 실패: '{response_text}'. 기본 '추가 질문'으로 처리."
            )
            action = "추가 질문"
            if not additional_question:
                additional_question = "요청을 이해하기 어려웠습니다. 어떤 책을 찾으시는지 다시 말씀해주시겠어요?"
        logger.info(
            f"행동 결정 파싱 결과: 행동='{action}', 추가 질문='{additional_question[:50]}...'"
        )
        return action, additional_question

    async def update_preferences_from_input(self, user_input: str) -> None:
        logger.info(f"사용자 입력에서 선호도 추출 시작: '{user_input[:100]}...'")
        extract_result = await self._async_invoke(
            self.extract_pref_chain, {"text": user_input}, "선호도 추출"
        )
        extracted_text = extract_result.get("text", "{}")
        extracted_prefs: Dict[str, List[str]] = {}
        try:
            json_match = re.search(r"\{.*\}", extracted_text, re.DOTALL)
            if json_match:
                extracted_prefs_raw = json.loads(json_match.group(0))
                defined_keys = self.user_preferences.keys()
                for key, value in extracted_prefs_raw.items():
                    if key in defined_keys:
                        vals_to_add = []
                        if isinstance(value, list):
                            vals_to_add = [
                                str(item).strip()
                                for item in value
                                if item and str(item).strip()
                            ]
                        elif isinstance(value, str) and value.strip():
                            vals_to_add = [value.strip()]
                        if vals_to_add:
                            extracted_prefs[key] = vals_to_add
                    else:
                        logger.warning(
                            f"추출된 선호도 키 '{key}'가 정의된 형식에 없음. 무시."
                        )
            else:
                logger.warning(
                    f"선호도 추출 결과에서 JSON 객체를 찾을 수 없음: {extracted_text}"
                )
        except json.JSONDecodeError as e:
            logger.error(
                f"선호도 추출 결과 JSON 파싱 실패: {e}. 원본 텍스트: {extracted_text}"
            )
        except Exception as e:
            logger.error(f"선호도 추출/처리 중 예외 발생: {e}", exc_info=True)
        if not extracted_prefs:
            logger.info("새로 추출된 유효한 선호도 정보가 없음")
            return
        updated_something = False
        for key, new_values in extracted_prefs.items():
            existing_values_set = set(self.user_preferences.get(key, []))
            added_values = [
                v for v in new_values if v not in existing_values_set]
            if added_values:
                self.user_preferences[key].extend(added_values)
                updated_something = True
                logger.info(f"선호도 업데이트됨 [{key}]: {self.user_preferences[key]}")
        if updated_something:
            self.preference_update_count += 1
            logger.info(
                f"선호도 업데이트 완료. 누적 업데이트 횟수: {self.preference_update_count}"
            )
            self._update_preferences_text()
        else:
            logger.info("기존 선호도에서 변경된 내용 없음.")

    def _update_preferences_text(self):
        pref_items = []
        display_key_map = {
            "title": "관련 제목",
            "author": "선호 저자",
            "category": "선호 장르/분류",
            "author_intro": "저자 관련 요구",
            "book_intro": "내용 관련 요구",
            "table_of_contents": "목차/키워드 요구",
            "purpose": "독서 목적",
            "implicit info": "기타 희망 사항/분위기",
        }
        for key, values in self.user_preferences.items():
            if values:
                display_key = display_key_map.get(key, key)
                pref_items.append(f"- {display_key}: {', '.join(values)}")
        self.preferences_text = (
            "\n".join(pref_items) if pref_items else "수집된 선호도 없음"
        )
        logger.debug(f"업데이트된 선호도 요약 텍스트:\n{self.preferences_text}")

    async def get_final_query(self, current_user_query: str) -> str:
        logger.info("최종 검색 쿼리 생성 시작")
        persona_info = self.config.get("persona_info", "기본 정보")
        final_query_vars = {
            "history": "\n".join(self.user_history[-5:] + self.llm_history[-5:]),
            "query": current_user_query,
            "persona_info": persona_info,
            "preferences": self.preferences_text,
        }
        result_gen = await self._async_invoke(
            self.final_query_generation_chain, final_query_vars, "선호도 종합 쿼리 생성"
        )
        generated_query = result_gen.get("text", "").strip()
        logger.info(f"LLM 생성 쿼리 (정제 전): '{generated_query}'")
        query_to_use = generated_query
        if generated_query:
            refine_result = await self._async_invoke(
                self.refine_chain, {"query": generated_query}, "쿼리 정제"
            )
            refined_query = refine_result.get("text", "").strip().strip('"')
            logger.info(f"정제된 쿼리: '{refined_query}'")
            negative_keywords = [
                "없",
                "못",
                "않",
                "오류",
                "잘못",
                "알 수 없",
                "죄송",
                "필요",
            ]
            is_invalid_refinement = (
                not refined_query
                or len(refined_query) < 3
                or any(keyword in refined_query for keyword in negative_keywords)
                or "{" in refined_query
                or "}" in refined_query
                or refined_query.lower() == generated_query.lower()
            )
            if is_invalid_refinement:
                logger.warning(
                    f"정제된 쿼리('{refined_query}')가 유효하지 않아 정제 전 쿼리('{generated_query}') 사용."
                )
            else:
                query_to_use = refined_query
        else:
            logger.warning("선호도 종합 쿼리 생성 실패. 원본 사용자 쿼리 사용.")
            query_to_use = current_user_query
        if not query_to_use or len(query_to_use) < 3:
            logger.warning(
                f"최종 결정된 쿼리('{query_to_use}')가 너무 짧거나 비어있어 원본 사용자 쿼리('{current_user_query}') 사용."
            )
            query_to_use = current_user_query
        logger.info(f"최종 결정된 검색 쿼리: '{query_to_use}'")
        return query_to_use

    async def _summarize_chunk_with_llm(self, text: str) -> str:
        if not text or len(text.strip()) < self.MIN_INFO_LENGTH:
            return "요약할 정보가 충분하지 않습니다."
        max_len = 4000
        truncated_text = text[:max_len].strip()
        if not truncated_text:
            return "요약할 정보가 없습니다."
        prompt = f"""
        다음 책 정보를 2~3문장으로 핵심 내용만 요약해줘.  
        말투는 너무 딱딱하지 않고, 사용자에게 설명하듯 **자연스럽고 친근한 어조**로 해줘.  
        예를 들어 "~있다" 대신 "~있어요", "~이다" 대신 "~예요"처럼 말해줘.:\n\n{truncated_text}\n\n요약:"
        """
        summary = await self._async_invoke_llm(prompt, "청크 요약")
        if (
            not summary
            or len(summary) < 10
            or "요약할 정보가" in summary
            or "죄송" in summary
            or "모르겠" in summary
        ):
            fallback_summary = text[:300].strip(
            ) + ("..." if len(text) > 300 else "")
            logger.warning(
                f"LLM 요약 실패 또는 부적절. Fallback 요약 사용: '{fallback_summary[:100]}...'"
            )
            return (
                fallback_summary
                if fallback_summary
                else "별도의 상세 정보가 충분치 않습니다."
            )
        return summary

    def _merge_documents_by_isbn(self, isbn: str) -> Optional[Document]:
        if not isbn:
            logger.warning("ISBN 없이 문서 병합 시도됨.")
            return None
        docs_for_isbn = [
            doc
            for doc in self.documents
            if str(doc.metadata.get("ISBN", "")).strip() == str(isbn).strip()
        ]
        if not docs_for_isbn:
            logger.warning(
                f"ISBN '{isbn}'에 해당하는 문서를 마스터 목록에서 찾을 수 없음."
            )
            return None
        combined_text = "\n\n---\n\n".join(
            doc.page_content for doc in docs_for_isbn if doc.page_content
        ).strip()
        merged_meta = dict(docs_for_isbn[0].metadata)
        logger.debug(
            f"ISBN '{isbn}' 문서 병합 완료 (전체 원본 기준). 병합된 청크 수: {len(docs_for_isbn)}, 총 텍스트 길이: {len(combined_text)}"
        )
        return Document(page_content=combined_text, metadata=merged_meta)

    async def _embedding_rerank_documents(
        self, query: str, documents: List[Document]
    ) -> List[Document]:
        if not documents:
            logger.info("리랭킹할 문서가 없습니다.")
            return []
        embedding_tasks = {
            "main": asyncio.to_thread(self.embeddings.embed_query, query)
        }
        if self.user_preferences.get("author"):
            embedding_tasks["author"] = asyncio.to_thread(
                self.embeddings.embed_query, self.user_preferences["author"][0]
            )
        if self.user_preferences.get("title"):
            embedding_tasks["title"] = asyncio.to_thread(
                self.embeddings.embed_query, self.user_preferences["title"][0]
            )
        if self.user_preferences.get("category"):
            embedding_tasks["category"] = asyncio.to_thread(
                self.embeddings.embed_query, self.user_preferences["category"][0]
            )
        query_embeddings = await asyncio.gather(*embedding_tasks.values())
        query_embedding_map = dict(
            zip(embedding_tasks.keys(), query_embeddings))
        # 페르소나별 retrieval_weights 사용 (없으면 기본값)
        weights = self.config.get(
            "retrieval_weights",
            {"main": 0.1, "author": 0.3, "title": 0.1, "category": 0.5},
        )
        scored_docs = []
        for doc in documents:
            score_map: Dict[str, float] = {}
            real_meta = doc.metadata.get("metadata", doc.metadata)
            doc_embedding = await asyncio.to_thread(
                self.embeddings.embed_query, doc.page_content
            )
            score_map["main"] = cosine_similarity(
                [query_embedding_map["main"]], [doc_embedding]
            )[0][0]
            author_text = real_meta.get("author", "")
            if author_text and "author" in query_embedding_map:
                author_emb = await asyncio.to_thread(
                    self.embeddings.embed_query, author_text
                )
                score_map["author"] = cosine_similarity(
                    [query_embedding_map["author"]], [author_emb]
                )[0][0]
            title_text = real_meta.get("title", "")
            if title_text and "title" in query_embedding_map:
                title_emb = await asyncio.to_thread(
                    self.embeddings.embed_query, title_text
                )
                score_map["title"] = cosine_similarity(
                    [query_embedding_map["title"]], [title_emb]
                )[0][0]
            category_text = real_meta.get("category", "")
            if category_text and "category" in query_embedding_map:
                category_emb = await asyncio.to_thread(
                    self.embeddings.embed_query, category_text
                )
                score_map["category"] = cosine_similarity(
                    [query_embedding_map["category"]], [category_emb]
                )[0][0]
            valid_keys = score_map.keys()
            total_weight = sum(weights[k] for k in valid_keys)
            final_score = sum(
                (weights[k] / total_weight) * score_map[k] for k in valid_keys
            )
            scored_docs.append((doc, final_score))
        sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)
        logger.info("리랭킹 완료 (정규화 가중치 방식)")
        for i, (doc, score) in enumerate(sorted_docs):
            title = extract_metadata_field(doc, "title", default="제목 없음")
            isbn = extract_metadata_field(doc, "ISBN", default="N/A")
            logger.info(f"{i+1}. 제목: {title} | ISBN: {isbn} | 점수: {round(score, 4)}")

        reranked_docs = [doc for doc, score in sorted_docs if score >= 0.5]
        logger.info("리랭킹 완료 (0.5 이상 필터 적용됨)")
        return reranked_docs

    async def _expand_query(self, query: str) -> List[str]:
        logger.info(f"검색 쿼리 확장 시작: '{query}'")
        expansion_result = await self._async_invoke(
            self.query_expansion_chain, {"query": query}, "검색 쿼리 확장"
        )
        expanded_queries_text = expansion_result.get("text", "").strip()
        expanded_queries = []
        for line in expanded_queries_text.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            query_part = re.sub(r"^\d+\.\s*", "", line_stripped).strip()
            if query_part and query_part.lower() != query.lower():
                expanded_queries.append(query_part)
        final_expanded_queries = expanded_queries[:3]
        logger.info(
            f"생성된 확장 쿼리 ({len(final_expanded_queries)}개): {final_expanded_queries}"
        )
        return final_expanded_queries

    async def _retrieve_documents(
        self, query: str, use_hyde: bool = False
    ) -> List[Document]:
        retrieval_query = query
        hyde_summary = ""
        if use_hyde and self.user_preferences.get("implicit info"):
            logger.info("HyDE 활성화: 가상 문서 요약 및 키워드 추출 시도")
            hyde_result = await self._async_invoke(
                self.hyde_generation_chain, {"query": query}, "HyDE 가상 문서 생성")
            hyde_summary = hyde_result.get("text", "").strip()
            if hyde_summary:
                logger.info(f"생성된 가상 문서 요약: '{hyde_summary[:200]}...'")
                keyword_result = await self._async_invoke(
                    self.hyde_keyword_chain,
                    {"hyde_summary": hyde_summary},
                    "HyDE 키워드 추출",)
                hyde_keywords_text = keyword_result.get("text", "").strip()
                hyde_keywords = [k.strip() for k in hyde_keywords_text.split(",") if k.strip()]
                if hyde_keywords:
                    logger.info(f"추출된 HyDE 키워드: {hyde_keywords}")
                    retrieval_query = f"{query} {' '.join(hyde_keywords)}"
                    logger.info(f"HyDE 적용된 검색 쿼리: '{retrieval_query}'")
                else:
                    logger.warning("HyDE 요약에서 키워드를 추출하지 못했습니다. 원본 쿼리 사용.")
            else:
                logger.warning("HyDE 가상 문서 요약 생성 실패. 원본 쿼리 사용.")
        elif use_hyde:
            logger.info("HyDE 조건 미충족 ('implicit info' 없음). 일반 쿼리 검색 수행.")
        else:
            logger.info("일반 쿼리 검색 수행.")
        logger.info(f"최종 리트리버 호출 시작 (쿼리: '{retrieval_query[:100]}...')")
        try:
            retrieved_docs = await self.retriever.aget_relevant_documents(
                retrieval_query
            )
            logger.info(f"검색된 문서 수 (ISBN 병합 완료됨): {len(retrieved_docs)}")
        except Exception as e:
            logger.error(f"문서 검색 실패 (쿼리: '{retrieval_query}'): {e}", exc_info=True)
            retrieved_docs = []
        return retrieved_docs

    async def _generate_recommendations(self, final_query: str) -> str:
        logger.info(f"추천 생성 시작. 최종 쿼리: '{final_query}'")
        use_hyde = False
        use_expansion = False
        has_author = bool(self.user_preferences.get("author"))
        has_title = bool(self.user_preferences.get("title"))
        has_implicit = bool(self.user_preferences.get("implicit info"))
        if not has_author and not has_title and has_implicit:
            use_hyde = True
            use_expansion = True
        initial_docs = await self._retrieve_documents(final_query, use_hyde=use_hyde)
        all_docs_to_consider = list(initial_docs)
        processed_queries = {final_query.lower()}
        if use_expansion:
            expanded_queries = await self._expand_query(final_query)
            expansion_tasks = []
            if expanded_queries:
                logger.info(f"확장 쿼리 ({len(expanded_queries)}개)로 추가 검색 수행...")
                for eq in expanded_queries:
                    eq_lower = eq.lower()
                    if eq_lower not in processed_queries:
                        expansion_tasks.append(self._retrieve_documents(eq, use_hyde=False))
                        processed_queries.add(eq_lower)
                if expansion_tasks:
                    expansion_results = await asyncio.gather(*expansion_tasks)
                    existing_isbns = {
                        doc.metadata.get("ISBN")
                        for doc in all_docs_to_consider
                        if doc.metadata.get("ISBN")
                    }
                    for docs_list in expansion_results:
                        for doc in docs_list:
                            isbn = doc.metadata.get("ISBN")
                            if isbn and isbn not in existing_isbns:
                                all_docs_to_consider.append(doc)
                                existing_isbns.add(isbn)
        logger.info(f"초기 및 확장 검색 후 고려할 총 고유 문서 수: {len(all_docs_to_consider)}")

        for i, doc in enumerate(all_docs_to_consider):
            metadata_content = doc.metadata if doc.metadata else {}
            logger.debug(f"문서 {i+1} Metadata: {json.dumps(metadata_content, indent=2, ensure_ascii=False)}")
            try:
                isbn_check = metadata_content.get("ISBN", "ISBN 키 없음")
                title_check = metadata_content.get("title", "Title 키 없음")
                logger.debug(
                    f"  -> 확인: ISBN='{isbn_check}', Title='{title_check}'")
            except Exception as e:
                logger.error(f"  -> 메타데이터 접근 오류 발생: {e}")
        logger.debug("--- 리랭킹 입력 데이터 확인 완료 ---")
        if not all_docs_to_consider:
            logger.warning("검색 및 확장 결과 문서를 찾지 못했습니다. 추천 생성 불가.")
            return "죄송합니다, 해당 조건에 맞는 책을 찾지 못했습니다. 다른 조건으로 다시 시도해 보시겠어요?"
        logger.info("리랭킹 수행...")
        ranked_docs = await self._embedding_rerank_documents(
            final_query, all_docs_to_consider
        )
        if not ranked_docs:
            logger.warning("리랭킹 결과 유효한 문서가 없습니다. 추천 생성 불가.")
            self.last_recommendations = []  # 후속 질문 방지
            return "죄송합니다, 요청하신 조건에 맞는 책을 찾지 못했습니다. 다른 조건으로 다시 시도해 보시겠어요?"
        top_docs = ranked_docs[:3]
        logger.info(f"최종 추천 후보 문서 수: {len(top_docs)}")
        recommendations = []
        self.last_recommendations = []
        processed_isbns = set()
        for rank, doc in enumerate(top_docs):
            isbn = extract_metadata_field(doc, "ISBN")
            title = extract_metadata_field(doc, "title", default="제목 없음")
            logger.debug(
                f"추천 후보 처리 중 (Rank {rank+1}): ISBN={isbn}, Title={title}"
            )
            if not isbn or isbn == "N/A":
                logger.warning(
                    f"Rank {rank+1} 추천 후보에 ISBN 없음. 건너뜀. Title: {title}"
                )
                continue
            if isbn in processed_isbns:
                logger.warning(f"중복 ISBN '{isbn}' 추천 목록에 이미 존재. 건너뜀.")
                continue
            full_merged_doc = self._merge_documents_by_isbn(isbn)
            if not full_merged_doc:
                logger.error(
                    f"치명적 오류: 리랭킹된 문서(ISBN: {isbn})의 전체 정보를 병합하지 못했습니다. 추천 목록에서 제외."
                )
                continue
            self.last_recommendations.append(full_merged_doc)
            processed_isbns.add(isbn)
            metadata = full_merged_doc.metadata
            title = metadata.get("title", "제목 정보 없음")
            author = metadata.get("author", "저자 정보 없음")
            book_cover = metadata.get("book_cover", "표지 정보 없음")
            book_intro = extract_field(full_merged_doc.page_content, "책소개")
            publisher_review = extract_field(
                full_merged_doc.page_content, "출판사리뷰")
            recommendation_field = extract_field(
                full_merged_doc.page_content, "추천사")
            text_for_summary = ""
            if book_intro and len(book_intro.strip()) >= self.MIN_INFO_LENGTH:
                text_for_summary = book_intro
            elif publisher_review and len(publisher_review.strip()) >= self.MIN_INFO_LENGTH:
                text_for_summary = publisher_review
            elif (
                recommendation_field
                and len(recommendation_field.strip()) >= self.MIN_INFO_LENGTH
            ):
                text_for_summary = recommendation_field
            else:
                page_content_cleaned = re.sub(
                    r"^\s*.*?\s*[:：]\s*",
                    "",
                    full_merged_doc.page_content,
                    flags=re.MULTILINE,
                ).strip()
                text_for_summary = page_content_cleaned[:500]
            if text_for_summary and len(text_for_summary.strip()) >= self.MIN_INFO_LENGTH:
                summary = await self._summarize_chunk_with_llm(text_for_summary)
            else:
                summary = "책에 대한 상세 설명이 부족합니다."
            recommendation_text = f"{rank+1}. \r\n표지: {book_cover}\r\n\n제목: {title}\r\n저자: {author}\r\n- 추천 이유: {summary}"
            recommendations.append(recommendation_text)
            logger.debug(f"추천 문구 생성됨: {title}")
        if self.last_recommendations:
            rec_titles = [
                d.metadata.get("title", "N/A") for d in self.last_recommendations
            ]
            logger.info(
                f"'_generate_recommendations' 종료. last_recommendations 업데이트 ({len(self.last_recommendations)}개): {rec_titles}"
            )
        else:
            logger.warning(
                "'_generate_recommendations' 종료. 최종 추천 목록(last_recommendations)이 비어있음!"
            )
        if not recommendations:
            return "추천할 만한 책을 찾지 못했습니다. 조건을 바꿔서 다시 질문해 주시겠어요?"
        persona_info_map = {
            "Literature": "감성, 현재 기분, 선호하는 문학 장르 및 작가 스타일",
            "Science": "독자의 지식 수준(초심자/전문가), 관심 기술 분야, 문제 해결 목표",
            "General": "선호 장르, 책을 찾는 이유, 원하는 분위기나 난이도",
        }
        intro_vars = {
            "query": final_query,
            "persona_info": persona_info_map.get(self.config.get("persona", "General")),
        }
        intro_result = await self._async_invoke(
            self.recommendation_intro_chain, intro_vars, "추천 인삿말 생성"
        )
        recommendation_intro = intro_result.get("text", "").strip()
        if not recommendation_intro:
            recommendation_intro = "이런 책들은 어떠세요?"

        # 추천 아웃트로 생성
        outro_vars = {
            "query": final_query,
            "persona_info": persona_info_map.get(self.config.get("persona", "General")),
        }
        try:
            outro_result = await self._async_invoke(
                self.recommendation_outro_chain, outro_vars, "추천 마무리 멘트 생성"
            )
            recommendation_outro = outro_result.get("text", "").strip()
            if not recommendation_outro or len(recommendation_outro) < 3:
                recommendation_outro = "마음에 드는 책이 있으셨으면 좋겠어요 :)"
        except Exception as e:
            logger.warning(f"추천 아웃트로 생성 실패: {e}")
            recommendation_outro = "마음에 드는 책이 있으셨으면 좋겠어요 :)"

        # 최종 답변 조립
        final_answer = (
            recommendation_intro
            + "\n\n"
            + "\n\n".join(recommendations)
            + "\n\n"
            + recommendation_outro
        )
        logger.info("추천 응답 생성 완료.")
        return final_answer

    async def handle_followup_query(self, followup_query: str) -> tuple[bool, str]:
        logger.info(f"후속 질문 처리 시작. Query: '{followup_query}'")
        if not self.last_recommendations:
            logger.error(
                "handle_followup_query 진입 오류: last_recommendations가 비어 있음!"
            )
            return False, ""
        rec_info = []
        for i, doc in enumerate(self.last_recommendations):
            title = doc.metadata.get("title", "제목 없음")
            isbn = doc.metadata.get("ISBN", "NO_ISBN")
            snippet = await self._summarize_chunk_with_llm(doc.page_content[:500])
            rec_info.append(
                f"{i+1}. 제목: {title}, ISBN: {isbn}\n   요약: {snippet}")
        rec_info_str = "\n".join(rec_info)
        prompt = f"""이전에 다음 책들을 추천했습니다:
{rec_info_str}

사용자의 후속 질문은 다음과 같습니다: "{followup_query}"

이 질문이 위 추천 목록과 관련된 후속 질문인지, 아니면 완전히 새로운 질문인지 판단하고, 후속 질문이라면 그 의도를 분석하여 다음 JSON 형식 중 **하나만** 출력해라. 새로운 질문이면 {{"action": "새 질문", "ISBN": null, "query": null}} 형식으로 출력하라. **절대로 다른 설명이나 대화 없이 오직 JSON 객체 하나만 출력해야 한다.**
[후속 질문 의도 분류 및 JSON 형식]
- 특정 책 상세 정보 요청: {{"action": "상세", "ISBN": "<요청된 책의 ISBN>", "query": "{followup_query}"}}
- 특정 책과 유사한 책 추천 요청: {{"action": "유사", "ISBN": "<기준 책의 ISBN>", "query": "<유사성 관련 사용자 언급>"}}
- 추천된 책들 비교 요청: {{"action": "비교", "ISBN": "<비교 대상 ISBN 목록 (쉼표 구분)>", "query": "{followup_query}"}}
- 추천 결과에 대한 피드백/불만: {{"action": "피드백", "ISBN": null, "query": "{followup_query}"}}
- 완전히 새로운 질문: {{"action": "새 질문", "ISBN": null, "query": null}}
[분석 결과 (JSON 객체만 출력)]
"""
        try:
            result_text = await self._async_invoke_llm(prompt, "후속 질문 의도 분석")
            logger.debug(f"후속 질문 의도 분석 LLM 원본 응답: {result_text}")
            analysis_result = None
            json_match = re.search(r"\{.*\}", result_text, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    logger.error(
                        f"후속 질문 의도 분석 결과 JSON 파싱 실패 (JSON 형식 오류): {result_text}"
                    )
                    analysis_result = {"action": "새 질문"}
            else:
                if (
                    "새 질문" in result_text
                    or '"action": "새 질문"' in result_text.replace(" ", "")
                ):
                    logger.warning(
                        f"후속 질문 의도 분석 JSON 객체 미발견, '새 질문' 패턴 감지: {result_text}"
                    )
                    analysis_result = {"action": "새 질문"}
                else:
                    logger.error(
                        f"후속 질문 의도 분석 결과 JSON 파싱 실패 및 '새 질문' 패턴 미감지: {result_text}"
                    )
                    return False, ""
            action = analysis_result.get("action", "새 질문")
            isbn_str = analysis_result.get("ISBN", "")
            query_part = analysis_result.get("query", followup_query)
            logger.info(
                f"후속 질문 분석 결과: action='{action}', ISBN='{isbn_str}', query='{query_part[:50]}...'"
            )
            if action == "새 질문":
                return False, ""
            isbn_list = (
                [s.strip() for s in str(isbn_str).split(",") if s.strip()]
                if isbn_str
                else []
            )
            target_isbn = isbn_list[0] if isbn_list else None
            if action == "상세":
                if not target_isbn:
                    return (
                        True,
                        "어떤 책에 대해 더 알고 싶으신지 알려주시겠어요? (예: 첫 번째 책 또는 책 제목)",
                    )
                target_doc = next(
                    (
                        doc
                        for doc in self.last_recommendations
                        if str(doc.metadata.get("ISBN", "")).strip()
                        == str(target_isbn).strip()
                    ),
                    None,
                )
                if target_doc:
                    logger.debug(
                        f"상세 정보 요청: ISBN '{target_isbn}' 문서 찾음: Title='{target_doc.metadata.get('title')}'"
                    )
                    detail_prompt = f"""다음은 사용자가 문의한 '{target_doc.metadata.get("title", "해당 책")}'에 대한 정보입니다. 이 정보를 바탕으로 사용자 질문 "{query_part}"에 답하거나, 특별한 질문이 없다면 책에 대해 자연스럽게 더 자세히 설명해주세요.
[책 정보 요약]
제목: {target_doc.metadata.get("title", "정보 없음")}
저자: {target_doc.metadata.get("author", "정보 없음")}
분류: {target_doc.metadata.get("category", "정보 없음")}
페이지: {target_doc.metadata.get("page", "정보 없음")} 쪽
가격: {target_doc.metadata.get("price", "정보 없음")} 원
[책 소개 및 내용 (일부)]
{target_doc.page_content[:3000]}...
[답변 또는 상세 설명]
"""
                    detailed_info = await self._async_invoke_llm(
                        detail_prompt, "후속 상세 설명 생성"
                    )
                    return True, (
                        detailed_info
                        if detailed_info
                        else "죄송합니다, 요청하신 내용에 대한 추가 정보를 제공하기 어렵습니다."
                    )
                else:
                    available_isbns = [
                        d.metadata.get("ISBN") for d in self.last_recommendations
                    ]
                    return (
                        True,
                        f"죄송합니다, 요청하신 ISBN({target_isbn})의 책 정보를 찾을 수 없습니다. 현재 추천된 책들의 ISBN은 다음과 같습니다: {available_isbns}",
                    )
            elif action == "유사":
                if not target_isbn:
                    return (
                        True,
                        "어떤 책과 유사한 책을 찾으시는지 알려주시겠어요? (예: 두 번째 책 같은 스타일)",
                    )
                base_doc = next(
                    (
                        doc
                        for doc in self.last_recommendations
                        if str(doc.metadata.get("ISBN", "")).strip()
                        == str(target_isbn).strip()
                    ),
                    None,
                )
                if base_doc:
                    logger.debug(
                        f"유사 책 요청: 기준 ISBN '{target_isbn}' 문서 찾음: Title='{base_doc.metadata.get('title')}'"
                    )
                    base_title = base_doc.metadata.get("title", "")
                    base_author = base_doc.metadata.get("author", "")
                    base_category = base_doc.metadata.get("category", "")
                    similarity_aspects = (
                        f"{base_category} 장르"
                        if base_category
                        else f"'{base_title}'와 비슷한"
                    )
                    if base_author:
                        similarity_aspects += f" {base_author} 작가 스타일"
                    user_refinement = (
                        f" 그리고 '{query_part}' 특징을 가진"
                        if query_part and query_part != followup_query
                        else ""
                    )
                    new_query = f"{similarity_aspects}{user_refinement} 책 추천"
                    logger.info(f"유사 책 추천을 위한 새 쿼리 생성: {new_query}")
                    recommendation_result = await self._generate_recommendations(
                        new_query
                    )
                    return (
                        True,
                        f"네, '{base_title}'와(과) 비슷한 다른 책을 찾아볼게요.\n\n"
                        + recommendation_result,
                    )
                else:
                    available_isbns = [
                        d.metadata.get("ISBN") for d in self.last_recommendations
                    ]
                    return (
                        True,
                        f"죄송합니다, 기준이 되는 ISBN({target_isbn})의 책 정보를 찾을 수 없습니다. 현재 추천된 책들의 ISBN은 다음과 같습니다: {available_isbns}",
                    )
            elif action == "비교":
                if len(isbn_list) < 2:
                    return (
                        True,
                        "비교할 책을 두 권 이상 알려주시겠어요? (예: 첫 번째랑 세 번째 책 비교해주세요)",
                    )
                valid_comparison_isbns = [
                    isbn
                    for isbn in isbn_list
                    if any(
                        str(d.metadata.get("ISBN", "")
                            ).strip() == str(isbn).strip()
                        for d in self.last_recommendations
                    )
                ]
                if len(valid_comparison_isbns) < 2:
                    available_isbns = [
                        d.metadata.get("ISBN") for d in self.last_recommendations
                    ]
                    return (
                        True,
                        f"죄송합니다, 비교 요청하신 책({isbn_list}) 중 일부를 찾을 수 없거나 유효하지 않습니다. 현재 추천된 책 ISBN: {available_isbns}",
                    )
                comparison_result = await self._handle_comparison(
                    query_part, valid_comparison_isbns
                )
                return True, comparison_result
            elif action == "피드백":
                logger.info(f"사용자 피드백/불만 처리 시도: '{query_part}'")
                feedback_based_query = f"이전 추천에 대해 '{query_part}' 라는 피드백이 있었습니다. 이 점을 고려하여 다른 책을 추천해주세요."
                logger.info(f"피드백 기반 새 쿼리 생성: {feedback_based_query}")
                recommendation_result = await self._generate_recommendations(
                    feedback_based_query
                )
                return (
                    True,
                    "피드백 감사합니다. 말씀해주신 점을 바탕으로 다른 책을 찾아보겠습니다.\n\n"
                    + recommendation_result,
                )
            else:
                logger.warning(
                    f"처리되지 않은 후속 질문 action: {action}. 새 질문으로 간주."
                )
                return False, ""
        except Exception as e:
            logger.error(f"후속 질문 처리 중 예외 발생: {e}", exc_info=True)
            return True, "후속 질문 처리 중 오류가 발생했습니다. 다시 시도해 주세요."

    async def _handle_comparison(self, query_part: str, isbn_list: List[str]) -> str:
        logger.info(
            f"도서 비교 시작. ISBN 목록: {isbn_list}, 비교 관점: '{query_part}'"
        )
        comparison_docs_info = []
        for isbn in isbn_list:
            doc = next(
                (
                    d
                    for d in self.last_recommendations
                    if str(d.metadata.get("ISBN", "")).strip() == str(isbn).strip()
                ),
                None,
            )
            if doc:
                metadata = doc.metadata
                summary = await self._summarize_chunk_with_llm(doc.page_content[:2000])
                comparison_docs_info.append(
                    {
                        "title": metadata.get("title", "제목 없음"),
                        "author": metadata.get("author", "저자 없음"),
                        "summary": summary,
                        "category": metadata.get("category", "분류 없음"),
                        "page": metadata.get("page", "페이지 수 없음"),
                        "price": metadata.get("price", "가격 정보 없음"),
                        "isbn": isbn,
                    }
                )
            else:
                logger.error(
                    f"비교 오류: 유효성 검사 후에도 ISBN '{isbn}' 문서를 last_recommendations에서 찾지 못함."
                )
        if len(comparison_docs_info) < 2:
            logger.warning(
                f"비교 가능한 문서가 2개 미만입니다 (찾은 문서 수: {len(comparison_docs_info)})."
            )
            return "비교할 책 정보를 충분히 찾지 못했습니다."
        comparison_prompt_template = """다음은 사용자가 비교를 요청한 책들의 정보입니다. 사용자의 비교 요청 관점인 "{{ query }}"에 특히 초점을 맞춰 이 책들을 명확하게 비교 설명해주세요. 각 책의 주요 특징, 장르, 내용 스타일, 난이도, 분량, 가격 등 관련 정보를 활용하고, 어떤 독자에게 더 적합할지 등을 포함하여 답변하는 것이 좋습니다. 자연스러운 문장으로 설명해주세요.
[비교 대상 책 정보]
{% for doc in summaries %}
--- 책 {{ loop.index }} (ISBN: {{ doc.isbn }}) ---
제목: {{ doc.title }}
저자: {{ doc.author }}
분류: {{ doc.category }}
페이지 수: {{ doc.page }}
가격: {{ doc.price }} 원
요약: {{ doc.summary }}
{% endfor %}
[사용자 비교 요청 관점/질문]
"{{ query }}"
[비교 설명]
"""
        try:
            comp_template = PromptTemplate(
                template=comparison_prompt_template,
                input_variables=["summaries", "query"],
                template_format="jinja2",
            )
            rendered_prompt = comp_template.render(
                summaries=comparison_docs_info, query=query_part
            )
            logger.debug(
                f"도서 비교 프롬프트 생성 완료 (일부):\n{rendered_prompt[:500]}..."
            )
            comparison_result = await self._async_invoke_llm(
                rendered_prompt, "도서 비교 설명 생성"
            )
            if (
                not comparison_result
                or len(comparison_result) < 20
                or "죄송" in comparison_result
                or "모르겠" in comparison_result
            ):
                logger.warning("LLM 기반 도서 비교 설명 생성 실패 또는 결과 부적절.")
                return "죄송합니다, 요청하신 책들을 비교 설명하는 데 어려움이 있습니다."
            logger.info("도서 비교 설명 생성 완료.")
            return comparison_result
        except Exception as e:
            logger.error(f"도서 비교 설명 생성 중 예외 발생: {e}", exc_info=True)
            return "죄송합니다, 책들을 비교하는 중 오류가 발생했습니다."

    async def process_query(
        self, user_query: str, force_recommendation: bool = False
    ) -> str:
        logger.info(f"=== 새로운 사용자 쿼리 처리 시작: '{user_query}' ===")
        self.user_history.append(f"사용자: {user_query}")
        await self.update_preferences_from_input(user_query)
        is_potential_followup = self.last_action == "추천" and self.last_recommendations
        if is_potential_followup:
            logger.info("이전 추천에 대한 후속 질문 가능성 확인 중...")
            handled, followup_output = await self.handle_followup_query(user_query)
            if handled:
                logger.info("후속 질문 처리 완료.")
                self.llm_history.append(f"챗봇: {followup_output}")
                return followup_output
            else:
                logger.info("후속 질문이 아님. 일반 질문 처리 로직으로 진행합니다.")
        action: Optional[str] = None
        additional_question: str = ""
        if not force_recommendation:
            logger.info("행동 결정 요청 (추천 vs 추가 질문)")
            meaningful_prefs_count = sum(
                1 for k, v in self.user_preferences.items() if v and k != "title"
            )
            should_recommend_heuristically = (
                meaningful_prefs_count >= 1 or self.preference_update_count >= 2
            )
            if not should_recommend_heuristically and self.preference_update_count < 2:
                logger.info(
                    f"선호도 부족 ({meaningful_prefs_count}개 / {self.preference_update_count}번 업데이트). '추가 질문' 강제 실행."
                )
                action = "추가 질문"
                additional_question = "어떤 종류의 책을 찾으시는지 좀 더 자세히 말씀해주시겠어요? (예: 구체적인 카테고리, 특정 작가, 책의 수준/분위기 등)"
            else:
                prompt_vars = {
                    "history": "\n".join(
                        self.user_history[-5:] + self.llm_history[-5:]
                    ),
                    "query": user_query,
                    "preferences": self.preferences_text,
                    "role_instructions": self.config.get(
                        "role_instructions", get_general_role()
                    ),
                }
                try:
                    decision_result = await self._async_invoke(
                        self.decision_chain, prompt_vars, "행동 결정"
                    )
                    decision_text = decision_result.get("text", "").strip()
                    action, additional_question_llm = (
                        self.robust_parse_decision_response(decision_text)
                    )
                    if action == "추가 질문" and additional_question_llm:
                        additional_question = additional_question_llm
                except Exception as e:
                    logger.error(
                        f"행동 결정 LLM 호출 실패: {e}. '추가 질문'으로 안전하게 진행.",
                        exc_info=True,
                    )
                    action = "추가 질문"
                    additional_question = "요청을 이해하는 데 어려움이 있었습니다. 어떤 책을 찾으시는지 다시 말씀해주시겠어요?"
        else:
            logger.info("강제 추천 모드 활성화. 행동='추천'")
            action = "추천"
        response = ""
        if action == "추가 질문":
            logger.info("행동: 추가 질문")
            self.last_action = "추가 질문"
            if additional_question:
                try:
                    add_q_emb = await asyncio.to_thread(
                        self.embeddings.embed_query, additional_question
                    )
                    if is_similar_question(
                        add_q_emb,
                        self.previous_additional_question_embeddings,
                        threshold=0.90,
                    ):
                        logger.warning(
                            "이전과 매우 유사한 추가 질문 생성됨. 추천 강제 시도."
                        )
                        return await self.process_query(
                            user_query, force_recommendation=True
                        )
                    else:
                        self.previous_additional_question_embeddings.append(
                            add_q_emb)
                        if len(self.previous_additional_question_embeddings) > 5:
                            self.previous_additional_question_embeddings.pop(0)
                        response = additional_question
                except Exception as e:
                    logger.error(f"추가 질문 임베딩 또는 유사도 비교 중 오류: {e}")
                    response = additional_question
            else:
                logger.warning("추가 질문 행동 결정되었으나 질문 내용 없음. 추천 시도.")
                action = "추천"
        if action == "추천":
            logger.info("행동: 추천")
            self.last_action = "추천"
            final_query = await self.get_final_query(user_query)
            if not final_query:
                logger.error("최종 검색 쿼리 생성 실패. 추천 불가.")
                self.last_action = None
                response = "죄송합니다, 검색어를 만드는 데 실패했습니다. 다시 질문해 주시겠어요?"
            else:
                recommendation_result = await self._generate_recommendations(
                    final_query
                )
                response = recommendation_result
                if (
                    not self.last_recommendations
                    and "죄송합니다" not in response
                    and "찾지 못했습니다" not in response
                ):
                    logger.warning(
                        "추천 생성 과정 완료 후 self.last_recommendations가 비어있으나, 응답은 성공 메시지 형태임."
                    )
                    self.last_action = None
        if response:
            self.llm_history.append(f"챗봇: {response}")
            logger.info(
                f"챗봇 응답 생성 완료 (Action: {self.last_action}). 응답 일부: {response[:200]}..."
            )
            return response
        else:
            logger.error(f"최종 응답 생성 실패 (Action: {action}).")
            self.last_action = None
            fallback_msg = (
                "죄송합니다, 요청을 처리하는 중 문제가 발생했습니다. 다시 시도해주세요."
            )
            self.llm_history.append(f"챗봇: {fallback_msg}")
            return fallback_msg