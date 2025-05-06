import ollama
from ollama import ChatResponse
from typing import Optional
import dotenv


dotenv.load_dotenv()

class LLM:
    def __init__(
        self, 
        model_type: str = "llama3.1",
        temperature: float = 0.4,
        max_tokens: int = 1024,
        top_p: float = 0.4,
    ):
        self._model_type = model_type
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def chat(self, messages: list[dict]) -> ChatResponse:
        try:
            response = ollama.chat(
                model=self._model_type,
                messages=messages,
                options={
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
            }
        )
        except Exception as e:
            response = None

        return response.message.content
    
    def construct_messages(self, sys_msg: str, user_msg: str) -> list[dict]:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages
