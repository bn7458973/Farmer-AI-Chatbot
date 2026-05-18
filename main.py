
from asr import voice_to_text
from translate import detect_language, translate_to_english, translate_to_target
from rag import retrieve_context
from llm import generate_answer
import subprocess
import sys


audio_file = "farmer_query.wav"

print("Converting voice to text...")
text = voice_to_text(audio_file)
print("Detected Text:", text)

if not text:
    print("No speech detected. Exiting.")
    sys.exit(1)

print("Detecting language...")
lang = detect_language(text)
print("Detected language:", lang)

if lang != "en" and lang != "unknown":
    print("Translating to English...")
    english_text = translate_to_english(text)
else:
    english_text = text
print("English Query:", english_text)

print("Retrieving context...")
context = retrieve_context(english_text)
print("Retrieved Context:", context)

print("Generating AI response...")
answer = generate_answer(context, english_text)
print("Raw Answer:", answer)

# If we got an error about missing/invalid API key, show it and exit
if isinstance(answer, str) and ("Invalid OpenAI API key" in answer or answer.startswith("Error: OPENAI_API_KEY")):
    print(answer)
    sys.exit(1)

if lang != "en" and lang != "unknown":
    print(f"Translating answer back to {lang}...")
    answer_translated = translate_to_target(answer, lang)
else:
    answer_translated = answer

print("Final Answer:", answer_translated)

# Simple TTS using macOS `say`
try:
    print("Speaking reply...")
    subprocess.run(["say", answer_translated])
except Exception as e:
    print("TTS failed:", e)
