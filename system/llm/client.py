"""
Responsible for communicating with the Groq API (OpenAI-compatible).
"""

import os

from huggingface_hub import InferenceClient
from dotenv import load_dotenv

from system.llm.prompt import build_prompt
from system.profile.schema import UserProfile

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-20b"


class LLMClient:
    """
    Client for communicating with the Groq API via its OpenAI-compatible endpoint.
    The free tier allows 30 req/min, 1000 req/day, 8k tokens/min and 200k tokens/day.
    """

    def __init__(self):
        token = os.getenv("GROQ_API_KEY")
        if not token:
            raise ValueError("'GROQ_API_KEY' not found. Check .env file or Space secrets.")
        self._client = InferenceClient(
            base_url="https://api.groq.com/openai/v1",
            api_key=token,
        )

    def generate_response(
        self,
        user_message: str,
        memory_context: list[str] | None = None,
        model: str = DEFAULT_MODEL,
        max_new_tokens: int = 512,
        profile: UserProfile | None = None,
    ) -> str:
        prompt = build_prompt(user_message, memory_context, profile)

        response = self._client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=0.7,
        )

        return response.choices[0].message.content.strip()  # type: ignore
    