import os

from dotenv import load_dotenv

from llm_client import (
    GroqProvider,
    OllamaProvider,
    LLMClient
)

# Load environment variables
load_dotenv(override=True)

# Choose provider
provider_name = input(
    "Choose provider (groq/ollama): "
).strip().lower()


if provider_name == "groq":

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError("Groq API key not found in .env")

    provider = GroqProvider(api_key)


elif provider_name == "ollama":

    provider = OllamaProvider("llama3.2")


else:

    raise ValueError("Invalid provider selected")


# Create common client
client = LLMClient(provider)

# Get user question
prompt = input("Enter your question: ")

# Generate response
answer = client.generate(prompt)

print("\nLLM Response:")
print(answer)