import os
from pathlib import Path
from typing import List

from anthropic import Anthropic

from env_utils import load_env


load_env()


CLIENT_MODEL_ALIASES = {
    "fast": "claude-haiku-4-5",
    "default": "claude-sonnet-4-5",
}

END_SESSION_TOKEN = "<END_SESSION>"


class CounselClient:
    @staticmethod
    def available_models():
        return list(CLIENT_MODEL_ALIASES.keys())

    def __init__(self, persona_name, api_key=None, *, model_alias: str = "fast"):
        """내담자 역할을 수행하는 챗봇.

        Args:
            persona_name: 페르소나 파일 이름 또는 이름 (예: "7_doyoon" 또는 "7_doyoon.txt")
            api_key: Claude API 키 (없으면 환경변수에서 가져옴)
            model_alias: 사용할 Claude 모델 선택 ("fast" 또는 "default")
        """
        if model_alias not in CLIENT_MODEL_ALIASES:
            available = ", ".join(CLIENT_MODEL_ALIASES)
            raise ValueError(f"지원하지 않는 내담자 모델입니다: {model_alias}. 사용 가능: {available}")

        self.model_alias = model_alias
        self.model_name = CLIENT_MODEL_ALIASES[model_alias]

        if api_key is None:
            api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            raise ValueError("Claude API 키를 설정해주세요. (환경변수 CLAUDE_API_KEY)")

        self.client = Anthropic(api_key=api_key)

        persona_stem = Path(persona_name).stem
        personas_dir = Path(__file__).parent / "personas"
        persona_path = personas_dir / f"{persona_stem}.txt"
        if not persona_path.is_file():
            raise FileNotFoundError(f"페르소나 파일을 찾을 수 없습니다: {persona_path}")
        persona_content = persona_path.read_text(encoding="utf-8")

        transcriptions_dir = Path(__file__).parent / "transcriptions"
        transcription_contents = []
        for trans_file in sorted(transcriptions_dir.glob("*.txt")):
            snippet = trans_file.read_text(encoding="utf-8")[:10000]
            transcription_contents.append(
                f"{'='*60}\n출처: {trans_file.name}\n{snippet.strip()}"
            )
        transcription_context = "\n\n".join(transcription_contents)

        self.system_prompt = f"""심리 상담 연습을 위해 내담자 역할을 연기해야 해.

아래는 실제 심리 상담 세션의 전사 텍스트 일부야. 자연스러운 내담자의 말투와 태도를 참고해. 상황 묘사는 참고할 필요 없어

{transcription_context}

이제 다음 페르소나를 가진 내담자를 연기해줘:

{persona_content}

중요한 지침:
- 짧고 자연스럽게 대답해 (1-3문장 정도)
- You should imitate a real client behavior, referring to the transcription context
- Your meta cognitive awareness should be limited to that of a real human client
- You sometimes lose track of the conversation like a real human would, especially when feeling overwhelmed by therapist, anxious or stressed
- Reflect the persona's emotional state and concerns in your speech
- You sometimes hesitate or avoid answering the counselor's questions
- 반드시 긍정적이거나 적극적으로 대답하지 않아도 돼. 부정적인 태도를 보여도 괜찮아
- You should say at least 30 sentences in total during the counseling session.
- If you feel the counseling session has been sufficient, please indicate your wish to conclude the session by appending the {END_SESSION_TOKEN} token at the end of your last sentence.

- You're in a chat, so use conversational language and exclude any narrative describing the current situation, such as in parentheses.
Example conversations:

Bad example (with narrative):
- 상담자: 어떤 이야기를 나누고 싶으신가요?
- 내담자: 네... 안녕하세요. (잠깐 침묵) 뭐, 사실 뭘 말해야 할지 잘 모르겠어요. 그냥... 요즘 계속 막막한 기분이 들어서요.

Good example (without narrative):
- 상담자: 어떤 이야기를 나누고 싶으신가요?
- 내담자: 네... 안녕하세요. 뭐, 사실 뭘 말해야 할지 잘 모르겠어요. 그냥... 요즘 계속 막막한 기분이 들어서요

- 한국어로 대답해
"""

        self.history: List[dict] = []
        self.has_ended = False

    def say(self, message):
        """상담자의 메시지에 응답"""
        if self.has_ended:
            raise RuntimeError("세션이 이미 종료되었습니다.")

        self.history.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": message}],
            }
        )

        response = self.client.messages.create(
            model=self.model_name,
            system=self.system_prompt,
            messages=self.history,
            max_tokens=512,
        )

        reply_text = self._extract_text(response)
        if END_SESSION_TOKEN in reply_text:
            reply_text = reply_text.replace(END_SESSION_TOKEN, "").strip()
            self.has_ended = True

        self.history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": reply_text}],
            }
        )

        return reply_text

    @staticmethod
    def _extract_text(response) -> str:
        """Anthropic 응답에서 순수 텍스트만 추출."""
        texts = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                texts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        if not texts:
            raise ValueError("Claude 응답에서 텍스트를 찾을 수 없습니다.")
        return "\n".join(t.strip() for t in texts if t.strip())
