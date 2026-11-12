import os
from abc import ABC, abstractmethod

import requests
from openai import OpenAI


# Common Interface
class LLMProvider(ABC):

    @abstractmethod
    def generate(self, prompt):
        pass


# Groq Cloud Provider (uses Groq's OpenAI-compatible endpoint)
class GroqProvider(LLMProvider):

    def __init__(self, api_key=None, model=None):
        # Model priority: argument > GROQ_MODEL env var > default
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

        self.client = OpenAI(
            api_key=api_key or os.getenv("GROQ_API_KEY"),
            base_url="https://api.groq.com/openai/v1"
        )

    def generate(self, prompt):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response.choices[0].message.content or ""


# Ollama Local Provider
class OllamaProvider(LLMProvider):

    def __init__(self, model="llama3.2"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        response.raise_for_status()

        return response.json()["response"]


# Common Client
class LLMClient:

    def __init__(self, provider):
        self.provider = provider

    def generate(self, prompt):
        return self.provider.generate(prompt)