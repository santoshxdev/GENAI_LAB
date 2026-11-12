import os

from dotenv import load_dotenv

from llm_client import GroqProvider, OllamaProvider, LLMClient
from usage_logger import UsageLogger

load_dotenv(override=True)

usage_logger = UsageLogger()

try:
    provider_name = input("Choose provider (groq/ollama): ").strip().lower()

    if provider_name == "groq":
        provider = GroqProvider()

    elif provider_name == "ollama":
        provider = OllamaProvider("llama3.2")

    else:
        raise ValueError("Invalid provider selected")

    client = LLMClient(provider, usage_logger)

    prompt = input("Enter your question: ")
    answer = client.generate(prompt)

    print("\nLLM Response:")
    print(answer)

finally:
    usage_logger.close()