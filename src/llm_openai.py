from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
import dotenv


dotenv.load_dotenv()

class LLM:
    def __init__(
        self, 
        temperature: float = 0.4,
        max_tokens: int = 1024,
        top_p: float = 0.4,
        model: str = "gpt-3.5-turbo"
    ):
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p
        self._model = model

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
            response = response.choices[0].message.content
        except Exception as e:
            print(f"OpenAI API Error: {e}")
            response = None

        return response
    
    def construct_messages(self, sys_msg: str, user_msg: str) -> list[dict]:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages