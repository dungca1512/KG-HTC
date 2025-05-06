from openai import AzureOpenAI
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
    ):
        self._client = AzureOpenAI(
            api_version=os.environ["API_VERSION"],
            azure_endpoint=os.environ["AZURE_ENDPOINT"],
            api_key=os.environ["API_KEY"],
        )
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._top_p = top_p

    def chat(self, messages: list[dict]) -> ChatCompletion:
        try:
            response = self._client.chat.completions.create(
                model=os.getenv("DEPLOYMENT_NAME"),
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                top_p=self._top_p,
            )
            response = response.choices[0].message.content
        except Exception as e:
            response = None

        return response
    
    def construct_messages(self, sys_msg: str, user_msg: str) -> list[dict]:
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": user_msg},
        ]

        return messages
