import os
import time
from dotenv import load_dotenv
from openai import OpenAI, InternalServerError, RateLimitError, APIConnectionError

# Step 1: Load environment variables
load_dotenv()

# Step 2: Get API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Step 3: Create Groq client (OpenAI-compatible endpoint)
# max_retries=0 so the retry loop below is the only one in play
client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1",
    timeout=60.0,
    max_retries=0
)

# Step 4: Define models (primary first, fallback second)
models = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant"
]

prompt = "What is Artificial Intelligence? Explain in simple words."

try:

    for model in models:

        print(f"\nTrying model: {model}")

        for attempt in range(3):

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}]
                )

                print("\nLLM Response:")
                print(response.choices[0].message.content)

                break

            except (InternalServerError, RateLimitError, APIConnectionError) as e:

                print(f"Server busy. Attempt {attempt + 1}/3")

                if attempt < 2:
                    time.sleep(2 ** (attempt + 1))
                else:
                    print("Switching to another model...")

        else:
            continue

        break

    else:
        print("All models are currently unavailable. Try again later.")

finally:
    client.close()