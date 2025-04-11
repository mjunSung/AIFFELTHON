import streamlit as st
import requests
import uuid

API_URL = "http://backend:8000/chat"


def get_welcome_message(persona):
    messages = {
        "문학": "📖 안녕하세요! 감성적이고 문학적인 도서 추천 챗봇입니다. 어떤 책을 읽고 싶으신가요?",
        "과학": "🔬 안녕하십니까. 정확하고 논리적인 과학/기술 도서 추천 챗봇입니다. 관심 있는 분야에 대해 편하게 이야기해 주세요.",
        "일반": "📚 안녕하세요! 친절하고 신뢰할 수 있는 범용 도서 추천 챗봇입니다. 어떤 책을 찾으시나요?"
    }
    return messages.get(persona, messages["일반"])


def initialize_session():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'previous_persona' not in st.session_state:
        st.session_state.previous_persona = None
    if 'is_loading' not in st.session_state:
        st.session_state.is_loading = False


def display_chat_history():
    """채팅 기록 표시"""
    st.markdown("### 📚 Chameleon 도서 추천 챗봇")
    with st.container(height=1200):
        for msg in st.session_state.chat_history:
            role = "user" if msg["role"] == "user" else "assistant"
            with st.chat_message(role):
                content = msg["content"]
                lines = content.split('\n')

                for line in lines:
                    # 이미지 URL이 포함된 줄 처리
                    if line.startswith("표지"):
                        image_url = line.split("표지: ")[1].strip()
                        st.image(image_url, caption='책 표지', width=200)
                    # 일반 텍스트 처리
                    elif line.startswith("제목"):
                        title = line.split("제목: ")[1].strip()
                        st.markdown(f"#### 제목: {title}")
                    else:
                        st.markdown(line)


def handle_user_input():
    if user_input := st.chat_input("메시지를 입력하세요..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        # 임시 응답 중 메시지 추가
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "⏳ 모델이 응답 중입니다..."
        })
        st.session_state.is_loading = True
        st.rerun()


def process_model_response():
    if st.session_state.get('is_loading', False):
        try:
            # 마지막 사용자 메시지를 기반으로 API 호출
            last_user_msg = next(
                msg["content"]
                for msg in reversed(st.session_state.chat_history)
                if msg["role"] == "user"
            )

            response = requests.post(
                API_URL,
                json={
                    "message": last_user_msg,
                    "session_id": st.session_state.session_id,
                    "persona": st.session_state.selected_persona
                }
            )

            # 응답 대기 메시지 제거
            if st.session_state.chat_history[-1]["content"] == "⏳ 모델이 응답 중입니다...":
                st.session_state.chat_history.pop()

            # 응답 내용 추가
            if response.status_code == 200:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.json()["response"]
                })
            else:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ API 오류: {response.text}"
                })

        except Exception as e:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"⚠️ 서버 연결 실패: {str(e)}"
            })

        st.session_state.is_loading = False
        st.rerun()


def main():
    st.set_page_config(page_title="도서 추천 챗봇", layout="wide")
    initialize_session()

    # 사이드바
    with st.sidebar:
        st.header("설정")
        new_persona = st.selectbox(
            "챗봇 성격 선택",
            ("문학", "과학", "일반"),
            index=2
        )
        if st.button("대화 기록 및 세션 초기화"):
            st.session_state.chat_history = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": get_welcome_message(new_persona)
            })
            st.rerun()

    # 페르소나 변경 시 초기화
    if 'selected_persona' not in st.session_state or st.session_state.selected_persona != new_persona:
        st.session_state.selected_persona = new_persona
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": get_welcome_message(new_persona)
        })
        st.rerun()

    display_chat_history()
    handle_user_input()
    process_model_response()


if __name__ == "__main__":
    main()
