import streamlit as st
import requests
import uuid
API_URL = "http://backend:8000/chat"


def get_welcome_message(persona):
    """페르소나에 맞는 환영 메시지 반환"""
    messages = {
        "문학": "📖 안녕하세요! 감성적이고 문학적인 도서 추천 챗봇입니다. 어떤 책을 읽고 싶으신가요?",
        "과학": "🔬 안녕하십니까. 정확하고 논리적인 과학/기술 도서 추천 챗봇입니다. 관심 있는 분야에 대해 편하게 이야기해 주세요.",
        "일반": "📚 안녕하세요! 친절하고 신뢰할 수 있는 범용 도서 추천 챗봇입니다. 어떤 책을 찾으시나요?"
    }
    return messages.get(persona, messages["일반"])


def initialize_session():
    """세션 상태 초기화"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'previous_persona' not in st.session_state:
        st.session_state.previous_persona = None


def display_chat_history():
    """채팅 기록 표시"""
    st.markdown("### 📚 도서 추천 챗봇")
    with st.container(height=1000):
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
    """사용자 입력 처리"""
    if user_input := st.chat_input("메시지를 입력하세요..."):
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        try:
            response = requests.post(
                API_URL,
                json={
                    "message": user_input,
                    "session_id": st.session_state.session_id,
                    "persona": st.session_state.selected_persona
                }
            )
            if response.status_code == 200:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response.json()["response"]
                })
                st.rerun()
            else:
                st.error(f"API 오류: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"서버 연결 실패: {str(e)}")


def main():
    st.set_page_config(page_title="도서 추천 챗봇", layout="wide")
    initialize_session()

    # 사이드바 설정
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

    # 페르소나 변경 감지 및 세션 초기화
    if 'selected_persona' not in st.session_state or st.session_state.selected_persona != new_persona:
        st.session_state.selected_persona = new_persona
        # st.session_state.chat_history = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": get_welcome_message(new_persona)
        })
        st.rerun()

    display_chat_history()
    handle_user_input()


if __name__ == "__main__":
    main()
