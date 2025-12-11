
import google.generativeai as genai

from config import get_config

config = get_config()
api_key = config.ai.gemini_api_key

if not api_key:
    print("No API key found")
    exit(1)

genai.configure(api_key=api_key)

print("Listing available models:")
for m in genai.list_models():
    if 'generateContent' in m.supported_generation_methods:
        print(m.name)
