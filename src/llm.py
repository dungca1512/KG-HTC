from openai import OpenAI
from openai.types.chat import ChatCompletion
import os
import dotenv


dotenv.load_dotenv()

class LLM:
    def __init__(self, temperature: float = 0.4, max_tokens: int = 1024, top_p: float = 0.4):
        self._client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]  # OpenAI API key
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def chat(self, messages: list[dict]) -> str:
        try:
            response = self._client.chat.completions.create(
                model="gpt-4",  # Hoặc "gpt-4"
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
            return response.choices[0].message.content
        except Exception as e:
            return None

        return response
    
    def construct_messages(self, sys_msg: str, user_msg: str) -> list[dict]:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages
