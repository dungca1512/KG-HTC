from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from typing import Optional
import os
import dotenv


dotenv.load_dotenv()

class LLM:
    def __init__(
        self, 
        temperature: float = 1,
        max_tokens: int = 1024,
        top_p: float = 1.0,
    ):
        self._client = AzureOpenAI(
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["API_KEY"],
        )
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

    def chat(self, messages: list[dict]) -> ChatCompletion:
        response = self._client.chat.completions.create(
            model=os.getenv("DEPLOYMENT_NAME"),
            messages=messages,
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            top_p=self._top_p,
        )

        return response
