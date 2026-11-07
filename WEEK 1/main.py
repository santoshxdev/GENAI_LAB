import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

# Step 1: Load environment variables
load_dotenv()

# Step 2: Get API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env file")

# Step 3: Create Gemini client
client = genai.Client(api_key=api_key)

# Step 4: Define models
models = [
    "gemini-3.8-flash",
    "gemini-3.5-flash-lite"
]

prompt = "What is Artificial Intelligence? Explain in simple words."

try:

    for model in models:

        print(f"\nTrying model: {model}")

        for attempt in range(3):

            try:
                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                print("\nLLM Response:")
                print(response.text)

                break

            except errors.ServerError as e:

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