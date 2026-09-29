import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


load_dotenv(Path(__file__).resolve().parent / ".env")


def llm_generate(user_prompt):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to the repository's local .env file."
        )

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
        )
        response = client.chat.completions.create(
            model="nvidia/nemotron-3-ultra-550b-a55b:free",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a professional translator. Translate the user's input "
                        "into Simplified Chinese, preserving its meaning and tone."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
        )
    except OpenAIError as exc:
        raise RuntimeError(
            f"OpenRouter request failed ({type(exc).__name__})."
        ) from None

    choices = response.choices or []
    translation = choices[0].message.content if choices and choices[0].message else None
    if not translation:
        raise RuntimeError("OpenRouter returned an empty translation.")
    return translation


def main():
    parser = argparse.ArgumentParser(description="Translate text into Simplified Chinese.")
    parser.add_argument("text", help="Text to translate")
    args = parser.parse_args()

    try:
        print(llm_generate(args.text))
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())