import ollama
from ollama import ChatResponse
from typing import Optional
import dotenv


dotenv.load_dotenv()

class LLM:
    def __init__(
        self, 
        model_type: str = "llama3.1",
        temperature: float = 1,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ):
        self._model_type = model_type
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def _update_config(
        self, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        if temperature is not None:
            self._temperature = temperature
        if max_tokens is not None:
            self._max_tokens = max_tokens
        if top_p is not None:
            self._top_p = top_p

    def chat(self, messages: list[dict]) -> ChatResponse:
        response = ollama.chat(
            model=self._model_type,
            messages=messages,
            options={
                "temperature": self._temperature,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
            }
        )
    
        return response
