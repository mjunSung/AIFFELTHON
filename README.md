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
![image](https://github.com/user-attachments/assets/0435f383-e87c-4b51-9a51-6d2be5302203)

---
## 사용 모델
- Embedding Model : ClovaXEmbeddings(model="bge-m3")
- LLM(Chat Model) : ChatClovaX(model="HCX-003", max_tokens=2048)

## 데이터 처리 방식
- API 활용 데이터 보충 및 N/A 값 처리
- Prompt-based Data Transformation(Chain-of-Density)

---
## Metrics



## Folders
- Rag_system : 사용 모델
- app : Demo dokcer files
- eda_preprocessing : 데이터 EDA 및 전처리 
