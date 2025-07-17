import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Lấy API key
api_key = os.getenv("OPENAI_API_KEY")
print(f"API Key (first 10 chars): {api_key[:10]}..." if api_key else "No API key found")

# Test API
try:
    client = openai.OpenAI(api_key=api_key)
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Say hello"}],
        max_tokens=10
    )
    
    print("✅ API test successful!")
    print(f"Response: {response.choices[0].message.content}")
    
except Exception as e:
    print(f"❌ API test failed: {e}")
    print(f"Error type: {type(e)}") 