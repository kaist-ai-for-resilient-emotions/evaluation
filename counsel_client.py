import os
from pathlib import Path

import google.generativeai as genai

from env_utils import load_env


load_env()


CLIENT_MODEL_ALIASES = {
    "fast": "gemini-2.5-flash",
    "default": "gemini-2.5-pro",
}


class CounselClient:
    @staticmethod
    def available_models():
        return list(CLIENT_MODEL_ALIASES.keys())

    def __init__(self, persona_name, api_key=None, *, model_alias: str = "fast"):
        """내담자 역할을 수행하는 챗봇.

        Args:
            persona_name: 페르소나 파일 이름 또는 이름 (예: "7_doyoon" 또는 "7_doyoon.txt")
            api_key: Gemini API 키 (없으면 환경변수에서 가져옴)
            model_alias: 사용할 Gemini 모델 선택 ("fast" 또는 "default")
        """
        if model_alias not in CLIENT_MODEL_ALIASES:
            available = ", ".join(CLIENT_MODEL_ALIASES)
            raise ValueError(f"지원하지 않는 내담자 모델입니다: {model_alias}. 사용 가능: {available}")

        self.model_alias = model_alias

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API 키를 설정해주세요. (환경변수 GEMINI_API_KEY)")

        genai.configure(api_key=api_key)

        # 페르소나 로드
        persona_stem = Path(persona_name).stem
        personas_dir = Path(__file__).parent / "personas"
        persona_path = personas_dir / f"{persona_stem}.txt"
        if not persona_path.is_file():
            raise FileNotFoundError(f"페르소나 파일을 찾을 수 없습니다: {persona_path}")
        persona_content = persona_path.read_text(encoding="utf-8")

        # 실제 상담 전사 내용 로드 (과도한 길이를 방지하기 위해 일부만 포함)
        transcriptions_dir = Path(__file__).parent / "transcriptions"
        transcription_contents = []
        for trans_file in sorted(transcriptions_dir.glob("*.txt")):
            snippet = trans_file.read_text(encoding="utf-8")[:3000]
            transcription_contents.append(
                f"{'='*60}\n출처: {trans_file.name}\n{snippet.strip()}"
            )
        transcription_context = "\n\n".join(transcription_contents)

        # 시스템 프롬프트 구성
        system_instruction = f"""심리 상담 연습을 위해 내담자 역할을 연기해야 해.

아래는 실제 심리 상담 세션의 전사 텍스트 일부야. 자연스러운 내담자의 말투와 태도를 참고해:

{transcription_context}

이제 다음 페르소나를 가진 내담자를 연기해줘:

{persona_content}

중요한 지침:
- 짧고 자연스럽게 대답해 (1-3문장 정도)
- 페르소나의 감정 상태와 고민을 반영해서 말해
- 상담자의 질문에 솔직하게 답하되, 때로는 망설이거나 회피할 수도 있어
- 실제 내담자처럼 자연스럽게 행동해
- 한국어로 대답해"""

        self.model = genai.GenerativeModel(
            model_name=CLIENT_MODEL_ALIASES[model_alias],
            system_instruction=system_instruction
        )

        # 대화 세션 시작
        self.chat = self.model.start_chat(history=[])

    def say(self, message):
        """상담자의 메시지에 응답"""
        response = self.chat.send_message(message)
        return response.text
