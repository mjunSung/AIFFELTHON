from models.base_rag import BaseRAGPipeline
from models.template import *

class LiteratureRAGPipeline(BaseRAGPipeline):
    def __init__(self, llm, embeddings, vectorstore, es_store, retriever, documents):
        config = {
            "persona": "Literature",
            "role_instructions": get_literature_role(),
            "pref_extraction_template": get_extract_pref_prompt_v2(),  # literature_pref_template,
            "consolidate_pref_template": get_consolidate_pref_prompt(),
            "decision_template": get_decision_prompt_template(),
            "final_query_template": get_literature_final_query_template(),
            "refine_template": get_literature_refine_template(),
            "expansion_template": get_literature_expansion_template(),
            "re_ranking_template": get_re_ranking_prompt(),
            "hyde_generation_template": get_hyde_generation_prompt(),
            "hyde_keyword_template": get_hyde_keyword_prompt(),
            "retrieval_weights": {
                "main": 0.1,
                "author": 0.5,
                "title": 0.1,
                "category": 0.3,
            },
            "persona_info": "감성, 현재 기분, 선호하는 문학 장르 및 작가 스타일",
            "recommendation_intro_template": get_recommendation_intro_prompt(),
            "recommendation_outro_template": get_recommendation_outro_prompt(),
        }
        super().__init__(config, llm, embeddings, vectorstore, es_store, retriever, documents)


class ScienceRAGPipeline(BaseRAGPipeline):
    def __init__(self, llm, embeddings, vectorstore, es_store, retriever, documents):
        config = {
            "persona": "Science",
            "role_instructions": get_science_role(),
            "pref_extraction_template": get_extract_pref_prompt_v2(),  # science_pref_template,
            "consolidate_pref_template": get_consolidate_pref_prompt(),
            "decision_template": get_decision_prompt_template(),
            "final_query_template": get_science_final_query_template(),
            "refine_template": get_science_refine_template(),
            "expansion_template": get_science_expansion_template(),
            "re_ranking_template": get_re_ranking_prompt(),
            "hyde_generation_template": get_hyde_generation_prompt(),
            "hyde_keyword_template": get_hyde_keyword_prompt(),
            "retrieval_weights": {
                "main": 0.2,
                "author": 0.1,
                "title": 0.3,
                "category": 0.4,
            },
            "persona_info": "정확, 논리, 최신 기술 동향 및 전문 지식을 반영",
            "recommendation_intro_template": get_recommendation_intro_prompt(),
            "recommendation_outro_template": get_recommendation_outro_prompt(),
        }
        super().__init__(config, llm, embeddings, vectorstore, es_store, retriever, documents)


class GeneralRAGPipeline(BaseRAGPipeline):
    def __init__(self, llm, embeddings, vectorstore, es_store, retriever, documents):
        config = {
            "persona": "General",
            "role_instructions": get_general_role(),
            "pref_extraction_template": get_extract_pref_prompt_v2(),  # general_pref_template,
            "consolidate_pref_template": get_consolidate_pref_prompt(),
            "decision_template": get_decision_prompt_template(),
            "final_query_template": get_general_final_query_template(),
            "refine_template": get_general_refine_template(),
            "expansion_template": get_general_expansion_template(),
            "re_ranking_template": get_re_ranking_prompt(),
            "hyde_generation_template": get_hyde_generation_prompt(),
            "hyde_keyword_template": get_hyde_keyword_prompt(),
            "retrieval_weights": {
                "main": 0.25,
                "author": 0.25,
                "title": 0.25,
                "category": 0.25,
            },
            "persona_info": "친절, 균형잡힌 정보, 사용자의 선호와 분위기를 반영",
            "recommendation_intro_template": get_recommendation_intro_prompt(),
            "recommendation_outro_template": get_recommendation_outro_prompt(),
        }
        super().__init__(config, llm, embeddings, vectorstore, es_store, retriever, documents)
