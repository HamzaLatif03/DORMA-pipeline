"""
Generate a conversational reply from the user's transcript.
Tries OpenAI first, then Gemini (GEMINI_API_KEY) if OpenAI fails (e.g. quota). Static fallback last.
Reply text is then spoken by ElevenLabs TTS.
"""
import os
import re
import logging
import random
import time
from pathlib import Path

from dotenv import load_dotenv
# Load .env from project root so API keys are set before any provider runs
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)
if not _env_path.exists():
    logging.warning("reply_api: .env not found at %s (API keys may be missing)", _env_path)

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = 60
# Give Gemini up to 60s to respond (e.g. when rate-limited or slow)
GEMINI_TIMEOUT = 60
# Keep Gemini replies short for faster response and TTS; limit input to avoid overflow/crash
GEMINI_MAX_OUTPUT_TOKENS = 50
# Only send a few words to the model to avoid overflow and long runs
MAX_INPUT_WORDS = int(os.environ.get("REPLY_MAX_INPUT_WORDS", "12"))
MAX_INPUT_CHARS = int(os.environ.get("REPLY_MAX_INPUT_CHARS", "120"))

SINGLE_LINE_FALLBACKS = [
    "Got it.",
    "Sure, I hear you.",
    "Okay.",
    "Noted.",
    "Thanks for that.",
    "Understood.",
]

DEFAULT_SYSTEM = "You are a helpful voice assistant. Reply in one short sentence. Do not repeat the user's words."


def _truncate_to_last_n_words(text: str, n: int, max_chars: int = 0) -> str:
    """Keep only the last n words and optionally cap total length to avoid Gemini overflow."""
    if n <= 0 or not text:
        return text.strip()
    words = text.split()
    if len(words) > n:
        text = " ".join(words[-n:])
    if max_chars > 0 and len(text) > max_chars:
        text = text[-max_chars:].strip()
        if text and text[0] not in " \t":
            first_space = text.find(" ")
            if first_space > 0:
                text = text[first_space + 1:]
    return text.strip()


def get_conversational_reply(user_message: str, system_prompt: str | None = None) -> str:
    """
    Return a reply: try OpenAI first, then Gemini (GEMINI_API_KEY / GEMINI_MODEL), then static fallback.
    Only the last ~10 seconds of speech (REPLY_MAX_INPUT_WORDS) are sent to the model for faster response.
    """
    raw = (user_message or "").strip()
    if not raw:
        return "I didn't catch that."

    user_message = _truncate_to_last_n_words(raw, MAX_INPUT_WORDS, MAX_INPUT_CHARS)
    print(f"[REPLY_API] get_conversational_reply (last {MAX_INPUT_WORDS} words, max {MAX_INPUT_CHARS} chars):", repr(user_message[:80]))
    system = (system_prompt or os.environ.get("REPLY_SYSTEM_PROMPT", DEFAULT_SYSTEM))[:500]

    # 1) Try OpenAI
    reply = _reply_with_openai(user_message, system)
    if reply and not _is_echo(reply, user_message, system):
        print("[REPLY_API] Using OPENAI response:", repr(reply[:200]))
        logger.info("OpenAI response: %s", reply[:120])
        return reply

    # 2) Try Gemini (e.g. when OpenAI has no quota)
    reply = _reply_with_gemini(user_message, system)
    if reply and not _is_echo(reply, user_message, system):
        print("[REPLY_API] Using GEMINI response:", repr(reply[:200]))
        logger.info("Gemini response: %s", reply[:120])
        return reply

    # 3) Static fallback
    if not reply:
        print("[REPLY_API] OpenAI and Gemini failed or empty; using static fallback.")
        logger.warning("OpenAI and Gemini failed or empty; using static fallback.")
    else:
        print("[REPLY_API] Reply was echo; using static fallback.")
    last = _single_line_fallback(user_message)
    print("[REPLY_API] Using STATIC fallback:", repr(last))
    return last


def _reply_with_openai(user_message: str, system: str) -> str:
    """Call OpenAI API with a long timeout so we give it time to respond."""
    key = (os.environ.get("OPENAI_API_KEY") or "").strip().strip('"')
    if not key:
        print("[REPLY_API] OPENAI_API_KEY not set")
        return ""
    try:
        print(f"[REPLY_API] Calling OpenAI API (timeout={OPENAI_TIMEOUT}s)...")
        from openai import OpenAI
        client = OpenAI(api_key=key, timeout=OPENAI_TIMEOUT)
        r = client.chat.completions.create(
            model=os.environ.get("REPLY_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            max_tokens=200,
            temperature=0.7,
        )
        reply = (r.choices[0].message.content or "").strip()
        print("[REPLY_API] OpenAI raw response:", repr(reply[:400]) if reply else "None/empty")
        return reply if reply else ""
    except Exception as e:
        print("[REPLY_API] OpenAI FAILED:", e)
        logger.warning("OpenAI failed: %s", e)
        return ""


# Fallbacks when primary returns 429. Use current v1beta model IDs (gemini-1.5-* often 404).
GEMINI_FALLBACK_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]


def _is_gemini_429(e: Exception) -> bool:
    s = str(e).upper()
    return "429" in s or "RESOURCE_EXHAUSTED" in s or "QUOTA" in s


def _retry_delay_seconds(e: Exception) -> float:
    """Parse 'Please retry in X.XXs' from Gemini error; allow up to 60s so Gemini can answer."""
    err = str(e)
    m = re.search(r"[Pp]lease\s+retry\s+in\s+(\d+(?:\.\d+)?)\s*s", err)
    if m:
        return min(float(m.group(1)), float(GEMINI_TIMEOUT))  # cap at 60s
    return 5.0  # default wait


def _call_gemini_once(key: str, model_name: str, prompt: str) -> str:
    """Single Gemini call; returns reply text or raises. Short max_output_tokens for faster reply."""
    try:
        from google import genai
        try:
            from google.genai import types
            http_opts = types.HttpOptions(timeout=int(GEMINI_TIMEOUT * 1000))  # ms
            client = genai.Client(api_key=key, http_options=http_opts)
            config = types.GenerateContentConfig(max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS)
            response = client.models.generate_content(model=model_name, contents=prompt, config=config)
        except (ImportError, AttributeError):
            client = genai.Client(api_key=key)
            try:
                config = types.GenerateContentConfig(max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS)
                response = client.models.generate_content(model=model_name, contents=prompt, config=config)
            except (NameError, AttributeError):
                response = client.models.generate_content(model=model_name, contents=prompt)
        return (getattr(response, "text", None) or "").strip()
    except (ImportError, AttributeError):
        import google.generativeai as genai_legacy
        genai_legacy.configure(api_key=key)
        model = genai_legacy.GenerativeModel(model_name)
        try:
            gen_config = getattr(genai_legacy.types, "GenerationConfig", None) or {}
            if callable(gen_config):
                gen_config = gen_config(max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS)
            else:
                gen_config = {"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS}
            response = model.generate_content(prompt, generation_config=gen_config)
        except Exception:
            response = model.generate_content(prompt)
        return (response.text or "").strip()


def _reply_with_gemini(user_message: str, system: str) -> str:
    """Use Gemini (GEMINI_API_KEY, GEMINI_MODEL) when OpenAI is unavailable (e.g. quota)."""
    key = (os.environ.get("GEMINI_API_KEY") or "").strip().strip('"')
    if not key:
        print("[REPLY_API] GEMINI_API_KEY not set")
        return ""
    primary = (os.environ.get("GEMINI_MODEL") or "gemini-2.0-flash-lite").strip().strip('"')
    models_to_try = [primary] + [m for m in GEMINI_FALLBACK_MODELS if m != primary]
    short_instruction = "Reply in one or two short sentences so the voice assistant can respond quickly."
    prompt = f"{system}\n{short_instruction}\n\nUser: {user_message}\n\nAssistant:"

    for model_name in models_to_try:
        for attempt in range(2):  # initial + one retry after delay
            try:
                reply = _call_gemini_once(key, model_name, prompt)
                if reply:
                    return reply
            except Exception as e:
                if _is_gemini_429(e) and attempt == 0:
                    delay = _retry_delay_seconds(e)
                    print(f"[REPLY_API] Gemini 429 (model={model_name}), retry in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                print(f"[REPLY_API] Gemini FAILED ({model_name}):", e)
                logger.warning("Gemini failed (%s): %s", model_name, e)
                break
    return ""


def _is_echo(reply: str, user_message: str, system: str) -> bool:
    """True only if reply is clearly just the user's words echoed (not a real response)."""
    if not reply or not user_message:
        return True
    r = reply.strip().lower()
    u = user_message.strip().lower()
    # Exact duplicate
    if r == u:
        return True
    # Reply is only slightly longer than user and starts with user text (echo)
    if len(r) < len(u) + 30 and r.startswith(u[: min(50, len(u))]):
        return True
    # Reply is mostly the system prompt
    if system and len(r) < 200 and system.strip().lower()[:80] in r:
        return True
    return False


def _single_line_fallback(user_message: str) -> str:
    """Last resort: single short line (no API)."""
    return random.choice(SINGLE_LINE_FALLBACKS)
