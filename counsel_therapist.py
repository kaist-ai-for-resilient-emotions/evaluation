import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, TypeVar

import google.generativeai as genai

from env_utils import load_env

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None


load_env()


T = TypeVar("T", bound="CounselTherapist")


class CounselTherapist(ABC):
    """상담자 역할을 수행하는 모델들의 공통 인터페이스."""

    alias: Optional[str] = None
    _registry: Dict[str, Type["CounselTherapist"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        alias = getattr(cls, "alias", None)
        if alias:
            CounselTherapist._registry[alias] = cls

    @classmethod
    def available_models(cls) -> List[str]:
        """등록된 상담자 모델 목록을 반환."""
        return sorted(cls._registry.keys())

    @classmethod
    def create(cls: Type[T], model_alias: str, **kwargs) -> T:
        """alias에 따라 상담자 인스턴스를 생성."""
        try:
            subclass = cls._registry[model_alias]
        except KeyError as exc:
            available = ", ".join(cls.available_models()) or "없음"
            raise ValueError(
                f"지원하지 않는 상담자 모델입니다: {model_alias}. 사용 가능: {available}"
            ) from exc
        return subclass(**kwargs)

    @abstractmethod
    def say(self, message: str) -> str:
        """내담자의 메시지를 보고 상담자의 응답을 생성."""


class GeminiCounselTherapist(CounselTherapist):
    """Gemini Pro 2.5 기반 상담자."""

    alias = "gemini-default"

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API 키를 설정해주세요. (환경변수 GEMINI_API_KEY)")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=(
                "너는 내담자를 공감적으로 돕는 전문 심리 상담자 역할을 연기해야 해. "
                "공감하고 열린 질문을 짧고 자연스럽게 건네."
            ),
        )
        self.chat = self.model.start_chat(history=[])

    def say(self, message: str) -> str:
        response = self.chat.send_message(message)
        return response.text


class GeminiFastCounselTherapist(GeminiCounselTherapist):
    """Gemini 2.5 Fast 모델을 사용하는 상담자."""

    alias = "gemini-fast"

    def __init__(self, api_key: Optional[str] = None):
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("Gemini API 키를 설정해주세요. (환경변수 GEMINI_API_KEY)")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=(
                "너는 내담자를 공감적으로 돕는 전문 심리 상담자 역할을 연기해야 해. "
                "공감하고 열린 질문을 짧고 자연스럽게 건네."
            ),
        )
        self.chat = self.model.start_chat(history=[])

class GPTSafetyCounselTherapist(CounselTherapist):
    """OpenAI Responses API를 이용한 GPT 기반 상담자."""

    alias = "gpt-safety"
    _PROMPT_REF = {
        "id": "pmpt_68ad656b5cfc819597e147cf74fb77e707e439745a36603f",
        "version": "6",
    }
    _INCLUDE = [
        "reasoning.encrypted_content",
        "web_search_call.action.sources",
    ]

    def __init__(self, openai_api_key: Optional[str] = None):
        if OpenAI is None:
            raise ImportError("openai 패키지가 설치되지 않았습니다. 'pip install openai' 후 다시 시도해주세요.")

        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API 키를 설정해주세요. (환경변수 OPENAI_API_KEY)")

        self.client = OpenAI(api_key=api_key)
        self.history: List[Dict[str, str]] = []

    def say(self, message: str) -> str:
        self.history.append({"role": "user", "content": message})
        response = self.client.responses.create(
            prompt=self._PROMPT_REF,
            input=self.history,
            reasoning={},
            store=True,
            include=self._INCLUDE,
        )

        assistant_text = self._extract_openai_text(response)
        self.history.append({"role": "assistant", "content": assistant_text})
        return assistant_text

    @staticmethod
    def _extract_openai_text(response) -> str:
        """OpenAI Responses API 응답에서 텍스트만 추출."""
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()

        output = getattr(response, "output", None)
        if output:
            for item in output:
                contents = getattr(item, "content", []) or []
                for content in contents:
                    text = getattr(content, "text", None)
                    if text:
                        return text.strip()
                    if isinstance(content, dict):
                        text_val = content.get("text") or content.get("content")
                        if isinstance(text_val, str):
                            return text_val.strip()

        raise ValueError("OpenAI 응답에서 텍스트를 찾을 수 없습니다.")
