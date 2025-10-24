import google.generativeai as genai
import os


class CounselTherapist:
    def __init__(self, api_key=None):
        """심리 상담자 역할을 수행하는 챗봇"""
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API 키를 설정해주세요. (환경변수 GEMINI_API_KEY)")

        genai.configure(api_key=api_key)

        # Therapist 역할 프롬프트를 단순하게 유지
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction="너는 내담자를 공감적으로 돕는 전문 심리 상담자 역할을 연기해야 해. 공감하고 열린 질문을 짧고 자연스럽게 건네."
        )

        # 대화 세션 시작
        self.chat = self.model.start_chat(history=[])

    def say(self, message):
        """내담자의 메시지에 응답"""
        response = self.chat.send_message(message)
        return response.text
