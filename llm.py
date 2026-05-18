import os
import sys
import re
from dotenv import load_dotenv

load_dotenv()


def generate_answer(context: str, query: str, max_words: int = 200, expand: bool = False) -> dict:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"ok": False, "error": "invalid_key", "message": "GROQ_API_KEY not set in .env file."}

    def _clean(text: str, max_words: int) -> str:
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'\*+', '', text).strip()
        words = text.split()
        if len(words) <= max_words:
            return text
        truncated = ' '.join(words[:max_words])
        last_end = max(truncated.rfind('.'), truncated.rfind('!'), truncated.rfind('?'))
        return truncated[:last_end + 1].strip() if last_end > 0 else truncated

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        if expand:
            instr = (
                f"Expand into a clear, detailed explanation up to {max_words} words. "
                "Focus on practical guidance, plain text only, no markdown symbols, and end with a complete sentence."
            )
        else:
            instr = (
                f"Answer helpfully in at most {max_words} words. "
                "Plain text only, no markdown symbols, end with a complete sentence. "
                "Give the direct answer first, then practical steps, then one short next-step suggestion. "
                "Never tell the farmer to consult local experts — you are the expert. Give full advice directly."
            )

        models = [
            os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            "llama-3.3-70b-versatile",
        ]

        last_error = None
        for model_name in models:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": f"{query}\n\n{instr}"}
                    ],
                    max_tokens=min(max_words * 3, 420),
                    temperature=0.35,
                )
                answer = response.choices[0].message.content.strip()
                print(f"[LLM DEBUG] Groq response received successfully from {model_name}", file=sys.stderr)
                return {"ok": True, "text": _clean(answer, max_words), "model": model_name}
            except Exception as model_error:
                last_error = model_error
                lowered_model_error = str(model_error).lower()
                if "rate limit" in lowered_model_error or "429" in lowered_model_error:
                    raise model_error
                if "model" in lowered_model_error and ("decommissioned" in lowered_model_error or "not found" in lowered_model_error):
                    continue
                raise

        raise last_error or RuntimeError("No model response received.")

    except Exception as e:
        err = str(e)
        print(f"[LLM ERROR] Groq error: {err}", file=sys.stderr)
        lowered = err.lower()
        if 'rate limit' in lowered or '429' in lowered:
            return {
                "ok": False,
                "error": "rate_limit",
                "message": f"Groq rate limit reached: {err}",
                "retry_after": 10
            }
        if 'invalid' in lowered or 'api key' in lowered or 'auth' in lowered:
            return {"ok": False, "error": "invalid_key", "message": f"Invalid Groq API key: {err}"}
        return {"ok": False, "error": "api_error", "message": err}
