# Chameleon : AIFFELthon repo

## RAG 기법을 적용한 사용자 맞춤형 도서 큐레이션

## 👥Member


| 이름   | 직책  | 역할 |
|--------|-------|------|
| 성명준 | 팀장  | 데이터 전처리<br>RAG 고도화 |
| 김용석 | 팀원  | 프로젝트 방향 설정<br>평가지표 수립 |
| 김희찬 | 팀원  | 패키징<br>데모 제작 |
| 박세희 | 팀원  | 데이터 전처리<br>평가지표 수립 |
| 양지웅 | 팀원  | 데이터 전처리<br>RAG 고도화 |
---
## System Pipeline
![image](https://github.com/user-attachments/assets/c2505642-8a82-4c47-9cb1-4440c211024e)
![image](https://github.com/user-attachments/assets/b9ba75ff-251c-41c2-be33-32a83d92a4bd)


---
## 사용 모델
- Embedding Model : ClovaXEmbeddings(model="bge-m3")
- LLM(Chat Model) : ChatClovaX(model="HCX-003", max_tokens=2048)

## 데이터 처리 방식
- API 활용 데이터 보충 및 N/A 값 처리
- Prompt-based Data Transformation(Chain-of-Density)

---
## Metrics
### 평가 모델
| Model   | RAG                            | Method | VDB  |
|---------|--------------------------------|--------|------|
| Model1  | NaiveRAG                       | Score  | v1   |
| Model2  | ModularRAG                     | Action | v1   |
| Model3  | ModularRAG + Hybrid Search     | Action | v2   |

### Model3 페르소나별 RAGAS Faithfulness, Answer Relevancy 점수 비교
| 페르소나     | Faithfulness | Answer Relevancy |
|--------------|--------------|------------------|
| General      | 0.880        | 0.749            |
| Literature   | 0.835        | 0.740            |
| Science      | 0.844        | 0.718            |
| Aggregated   | 0.889        | 0.750            |

   
### LLM Evaluation 점수 비교
일반 페르소나 모델별 점수 비교 (*CremaAI: Yes24 도서 큐레이션 상용 AI 챗봇)
| 평가 항목               | Model1 | Model2 | Model3 | CremaAI |
|------------------------|--------|--------|--------|---------|
| 사용자의도반영성         | 4.17   | 4.78   | 4.96   | 5.00    |
| 추천도서적합성           | 3.82   | 4.50   | 4.74   | 4.90    |
| 대화흐름자연스러움       | 4.15   | 4.62   | 4.76   | 4.95    |
| 추천이유설명력           | 4.48   | 4.52   | 4.71   | 4.10    |
         
A/B Test 결과 (우수 응답 선택)
| 비교 항목               | Response A (Model3) | Response B (비교 모델) |
|------------------------|---------------------|------------------------|
| Model3 vs Model1       | 47                  | 13                     |
| Model3 vs Model2       | 45                  | 15                     |
| Model3 vs CremaAI      | 34                  | 26                     |

## Folders
- Rag_system : 사용 모델
- app : Demo dokcer files
- eda_preprocessing : 데이터 EDA 및 전처리 
