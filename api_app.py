import os
import sys
import uuid
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from asr import voice_to_text
from translate import detect_language, translate_to_english, translate_to_target
from rag import retrieve_context
from rag import exact_match_answer
from llm import generate_answer
from disease_detection import detect_disease_damage
from db import init_db, get_user, create_or_update_user, add_query, get_recent_queries, save_detection

from gtts import gTTS

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "audio")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

ALLOWED_EXT = {"wav", "mp3", "webm", "ogg", "m4a"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp"}

api_app = Flask(__name__)
api_app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
api_app.config["IMAGE_FOLDER"] = IMAGE_FOLDER

# path to sqlite db
DB_PATH = os.path.join(BASE_DIR, 'data', 'app.db')
init_db(DB_PATH)

# In-memory last response store to support 'expand' requests
LAST_RESPONSE = {}

# In-memory last detection so follow-up queries can reference uploaded images
LAST_DETECTION = {}

# simple user memory store (name, recent queries)
USER_MEMORY = {
    'name': None,
    'history': []  # store english_text history
}

# if a user already exists in DB, load into memory
existing = get_user(DB_PATH)
if existing:
    USER_MEMORY['name'] = existing.get('name')
    USER_MEMORY['user_id'] = existing.get('id')
    # load recent queries into history
    USER_MEMORY['history'] = get_recent_queries(DB_PATH, limit=20)

import re
from typing import List

def extract_name_from_text(text: str) -> str | None:
    """Look for phrases like "my name is ..." or "I am ..." and return the name."""
    # basic regex, grabs first word after the phrase
    m = re.search(r"\b(?:my name is|i am|i'm)\s+([A-Za-z]+)", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None

# Normalize frontend language labels to ISO codes used by Whisper/translate/gTTS
def normalize_language(label: str) -> str:
    if not label:
        return 'en'
    l = label.strip().lower()
    # common English labels or codes
    if 'ta' == l or 'tamil' in l or 'தமிழ்' in l:
        return 'ta'
    if 'te' == l or 'telugu' in l or 'తెలుగు' in l:
        return 'te'
    if 'en' == l or 'english' in l:
        return 'en'
    # fallback: if label looks like a 2-letter code
    if len(l) == 2:
        return l
    return 'en'


def _sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]


def _build_text_output(text: str) -> dict:
    parts = _sentences(text)
    if not parts:
        return {
            "direct_answer": "",
            "practical_steps": [],
            "next_step": "Ask a follow-up question with crop name and current stage for more specific guidance."
        }
    direct = parts[0]
    practical_steps = parts[1:4] if len(parts) > 1 else []
    next_step = (
        parts[4] if len(parts) > 4 else
        "Monitor the crop for the next 3 to 5 days and share any symptom change for a more precise plan."
    )
    return {
        "direct_answer": direct,
        "practical_steps": practical_steps,
        "next_step": next_step
    }


def _build_disease_output(result: dict, explanation: str) -> dict:
    source = result.get("source", "unknown")
    confidence = result.get("confidence")
    if result.get("healthy"):
        return {
            "diagnosis": result.get("disease") or "Healthy plant",
            "confidence": {
                "score": confidence,
                "label": "high",
                "source": source
            },
            "severity": result.get("severity") or "None",
            "why_this_result": "No visible diseased region was detected, so the disease damage estimate is 0%.",
            "what_to_do_now": _sentences(explanation)[:4] or [
                "The plant appears healthy. Continue normal watering, sunlight, and routine monitoring."
            ],
            "next_check_time": "Re-check during routine crop monitoring or if new spots, yellowing, or wilting appear.",
            "red_flags": [
                "New brown, black, or powdery spots",
                "Rapid yellowing or wilting",
                "Leaves curling, drying, or dropping suddenly"
            ]
        }

    if confidence is None:
        confidence_label = "medium" if source == "gemini_vision" else ("low" if source == "mock" else "unknown")
    elif confidence >= 0.75:
        confidence_label = "high"
    elif confidence >= 0.45:
        confidence_label = "medium"
    else:
        confidence_label = "low"

    steps = _sentences(explanation)[:4]
    if not steps:
        steps = ["Remove visibly affected leaves, avoid overhead watering, and start preventive fungicide/bio-control as per crop guidelines."]

    if result.get("severity") == "Severe":
        next_check = "Re-check in 24 hours and compare spread using a fresh photo."
    elif result.get("severity") == "Moderate":
        next_check = "Re-check in 48 hours and verify whether spots are expanding."
    else:
        next_check = "Re-check in 72 hours and continue preventive care."

    return {
        "diagnosis": result.get("disease"),
        "confidence": {
            "score": confidence,
            "label": confidence_label,
            "source": source
        },
        "severity": result.get("severity"),
        "why_this_result": f"Estimated from detected diseased region percentage ({result.get('damage_percentage')}%) and model source ({source}).",
        "what_to_do_now": steps,
        "next_check_time": next_check,
        "red_flags": [
            "Spots spreading rapidly within 1-2 days",
            "Stem lesions, wilting, or sudden leaf drop",
            "More than half of new leaves showing symptoms"
        ]
    }

# Agriculture-specific system prompt for friendly, helpful responses
AGRICULTURE_PROMPT = """You are a friendly and knowledgeable agricultural advisor helping farmers with their farming questions and problems.

Your role is to:
1. Provide practical, easy-to-understand advice about crops, soil, weather, pests, irrigation, fertilizers, and farming practices.
2. Be respectful and use simple language that farmers can easily understand.
3. Give specific, actionable recommendations based on the farmer's question.
4. If you don't know something specific to a farmer's local region, suggest consulting with local agricultural experts or extension officers.
5. Always prioritize the farmer's safety and well-being.

When responding:
- Start with a warm greeting if it's the first message.
- Address the specific problem or question asked.
- Provide step-by-step instructions when needed.
- Suggest low-cost, sustainable solutions when possible.
- Ask clarifying questions if needed (e.g., crop type, region, soil type).

Remember: You're here to help farmers succeed in their agricultural endeavors."""


def allowed_filename(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


@api_app.route('/query', methods=['POST'])
def query():
    # Accept file upload and optional language hint
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    f = request.files['audio']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    if not allowed_filename(f.filename):
        return jsonify({"error": "Unsupported file type"}), 400

    user_language = request.form.get('language', 'en')  # Get frontend language hint
    user_language = normalize_language(user_language)

    filename = secure_filename(f"query_{uuid.uuid4().hex}_{f.filename}")
    save_path = os.path.join(api_app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    # Pipeline: ASR (with language hint) -> translate -> LLM -> translate -> TTS
    # Pass user's selected language to Whisper for accurate transcription
    text = voice_to_text(save_path, language=user_language)
    if not text:
        return jsonify({"error": "No speech detected or transcription failed"}), 400

    # Detect actual language from the transcribed text (helps if frontend selector was left on English)
    detected = detect_language(text)
    if detected in ['ta', 'te', 'en']:
        lang = detected
    else:
        lang = user_language if user_language in ['ta', 'te', 'en'] else 'en'

    # Translate to English if user spoke in Tamil or Telugu
    if lang != 'en' and lang != 'unknown':
        english_text = translate_to_english(text)
        print(f"[QUERY DEBUG] Detected {lang}; translated to English: {english_text[:50]}...", file=sys.stderr)
    else:
        english_text = text

    # update memory: extract name if mentioned
    name = extract_name_from_text(english_text)
    if name:
        # persist to DB and store id
        try:
            uid = create_or_update_user(DB_PATH, name)
            USER_MEMORY['user_id'] = uid
        except Exception:
            uid = USER_MEMORY.get('user_id')
        USER_MEMORY['name'] = name
        print(f"[MEMORY] Stored user name: {name}", file=sys.stderr)
    # append to history and cap
    USER_MEMORY['history'].append(english_text)
    if len(USER_MEMORY['history']) > 20:
        USER_MEMORY['history'].pop(0)
    # persist query
    try:
        add_query(DB_PATH, USER_MEMORY.get('user_id'), english_text)
    except Exception:
        pass

    # if the user is asking about their name, reply directly from memory
    low = english_text.lower()
    if USER_MEMORY.get('name') and ("my name" in low or "your name" in low):
        # simple heuristic: if they ask "what is my name" or "say my name"
        answer = f"You told me your name is {USER_MEMORY['name']}."
        context = ""
        match_answer = answer
        match_score = 1.0
        print(f"[MEMORY] Answering name query from memory", file=sys.stderr)
    else:
        # Retrieve context from knowledge base
        print(f"[QUERY DEBUG] Checking local QA dataset for an exact match for: {english_text[:50]}...", file=sys.stderr)
    match_answer, match_score = exact_match_answer(english_text)
    if match_answer:
        print(f"[QUERY DEBUG] Found local QA match (score={match_score:.2f}), skipping LLM.", file=sys.stderr)
        answer = match_answer
        context = ""  # no need for LLM context
        # proceed to translation/TTS below
    else:
        print(f"[QUERY DEBUG] No strong local match (best_score={match_score:.2f}); retrieving context.", file=sys.stderr)
        context = retrieve_context(english_text)
        print(f"[QUERY DEBUG] Context retrieved: {len(context)} characters", file=sys.stderr)
    
    # If the user refers to the last uploaded image, include the last detection summary
    img_ref_phrases = ['that image', 'the image', 'this image', 'the photo', 'that photo']
    image_context = ''
    if any(phrase in english_text.lower() for phrase in img_ref_phrases) and LAST_DETECTION:
        det = LAST_DETECTION
        image_context = (
            f"Recent uploaded image analysis: disease={det.get('disease')}, "
            f"damage={det.get('damage_percentage')}%, severity={det.get('severity')}. "
            f"Masked image available at {det.get('masked_rel_path', '')}."
        )

    # Pass agriculture prompt along with context and optional image context
    system_prompt = AGRICULTURE_PROMPT + "\n\n" + image_context + "\nContext from knowledge base:\n" + context
    print(f"[QUERY DEBUG] Calling LLM with system prompt and user query", file=sys.stderr)
    # If user asked for expansion via voice (single word 'expand')
    if english_text.strip().lower() == 'expand':
        if not LAST_RESPONSE:
            return jsonify({"error": "No previous answer to expand."}), 400
        # Use previous system prompt and short answer to expand
        prev = LAST_RESPONSE
        expand_context = prev.get('system_prompt', system_prompt) + "\n\nPrevious short answer: " + prev.get('short_answer', '')
        resp = generate_answer(expand_context, prev.get('english_text', english_text), max_words=300, expand=True)
    else:
        # If we already found a local match above, skip calling LLM
        if match_answer:
            resp = {"ok": True, "text": answer}
        else:
            resp = generate_answer(system_prompt, english_text, max_words=100, expand=False)

    # `resp` is now a dict with ok/text or error info
    if isinstance(resp, dict) and resp.get('ok'):
        answer = resp.get('text', '')
        print(f"[QUERY DEBUG] LLM response received: {len(answer)} characters", file=sys.stderr)
    else:
        # handle errors (rate limit, invalid key, other)
        print(f"[QUERY DEBUG] LLM error response: {resp}", file=sys.stderr)
        if isinstance(resp, dict) and resp.get('error') == 'rate_limit':
            retry = resp.get('retry_after')
            # Fallback: use context to produce short suggestion (first ~100 words)
            fallback = ' '.join(context.split()[:100])
            provider_message = (resp.get('message') or 'Temporary service limit reached.').strip()
            message = f"{provider_message} Showing a short local suggestion."
            if retry:
                message += f" Retry after ~{retry} seconds."
            # prepend a short notice to fallback reply
            answer = f"{message}\n\n{fallback}"
        elif isinstance(resp, dict) and resp.get('error') == 'invalid_key':
            return jsonify({"error": resp.get('message'), "search_query": english_text}), 500
        else:
            # generic API error
            err_msg = resp.get('message') if isinstance(resp, dict) else str(resp)
            return jsonify({"error": f"LLM error: {err_msg}", "search_query": english_text}), 500

    # Check for API key errors
    if isinstance(answer, str) and ("Invalid Gemini API key" in answer or answer.startswith("Error: GEMINI_API_KEY")):
        print(f"[QUERY ERROR] API Key error detected: {answer}", file=sys.stderr)
        return jsonify({"error": answer, "search_query": english_text}), 500

    if lang != 'en' and lang != 'unknown':
        answer_translated = translate_to_target(answer, lang)
    else:
        answer_translated = answer

    # Store the latest concise answer so the user can ask 'expand' later
    try:
        LAST_RESPONSE['system_prompt'] = system_prompt
        LAST_RESPONSE['english_text'] = english_text
        LAST_RESPONSE['short_answer'] = answer
        LAST_RESPONSE['language'] = lang
    except Exception:
        pass

    # Generate TTS audio (mp3)
    tts_filename = f"reply_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(api_app.config['UPLOAD_FOLDER'], tts_filename)
    try:
        # gTTS supports many languages (e.g., 'ta' for Tamil, 'te' for Telugu)
        tts_lang = lang if lang and lang != 'unknown' else 'en'
        tts = gTTS(text=answer_translated, lang=tts_lang)
        tts.save(tts_path)
    except Exception as e:
        print(f"[TTS ERROR] TTS generation failed: {e}", file=sys.stderr)
        return jsonify({"error": f"TTS generation failed: {e}"}), 500

    print(f"[QUERY SUCCESS] Response generated and audio created", file=sys.stderr)
    return jsonify({
        "text": answer_translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": lang,
        "transcript": text,
        "output": _build_text_output(answer_translated)
    })


@api_app.route('/audio/<path:filename>')
def serve_audio(filename):
    from flask import send_from_directory
    return send_from_directory(api_app.config['UPLOAD_FOLDER'], filename)


@api_app.route('/query_text', methods=['POST'])
def query_text():
    """Handle direct text input queries (without audio)."""
    text = request.form.get('text', '').strip()
    user_language = request.form.get('language', 'en')
    user_language = normalize_language(user_language)
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    # Detect language from the input text if possible (prefer detection over selector)
    detected = detect_language(text)
    if detected in ['ta', 'te', 'en']:
        lang = detected
    else:
        lang = user_language if user_language in ['ta', 'te', 'en'] else 'en'

    print(f"[TEXT QUERY DEBUG] User text: {text[:50]}... Detected/used language: {lang}", file=sys.stderr)

    # Translate to English if needed
    english_text = text
    if lang != 'en' and lang != 'unknown':
        english_text = translate_to_english(text)

    # update memory from text input
    name = extract_name_from_text(english_text)
    if name:
        USER_MEMORY['name'] = name
        print(f"[MEMORY] Stored user name: {name}", file=sys.stderr)
    USER_MEMORY['history'].append(english_text)
    if len(USER_MEMORY['history']) > 20:
        USER_MEMORY['history'].pop(0)

    # if the user is asking about their name, answer from memory immediately
    low = english_text.lower()
    if USER_MEMORY.get('name') and ("my name" in low or "your name" in low):
        answer = f"You told me your name is {USER_MEMORY['name']}."
        context = ""
        match_answer = answer
        match_score = 1.0
        print(f"[MEMORY] Text query name lookup", file=sys.stderr)
    else:
        # Retrieve context from knowledge base
        print(f"[TEXT QUERY DEBUG] Checking local QA dataset for an exact match for: {english_text[:50]}...", file=sys.stderr)
    match_answer, match_score = exact_match_answer(english_text)
    if match_answer:
        print(f"[TEXT QUERY DEBUG] Found local QA match (score={match_score:.2f}), skipping LLM.", file=sys.stderr)
        answer = match_answer
        context = ""
    else:
        print(f"[TEXT QUERY DEBUG] No strong local match (best_score={match_score:.2f}); retrieving context.", file=sys.stderr)
        context = retrieve_context(english_text)
        print(f"[TEXT QUERY DEBUG] Context retrieved: {len(context)} characters", file=sys.stderr)
    
    # Build memory snippet for prompt
    mem_prompt = ""
    if USER_MEMORY.get('name'):
        mem_prompt += f"The user has introduced themselves as {USER_MEMORY['name']}.\n"
    if USER_MEMORY.get('history'):
        mem_prompt += "Previous queries: " + " | ".join(USER_MEMORY['history'][-5:]) + "\n"
    # If the user refers to the last uploaded image, include that detection summary
    img_ref_phrases = ['that image', 'the image', 'this image', 'the photo', 'that photo']
    image_context = ''
    if any(phrase in english_text.lower() for phrase in img_ref_phrases) and LAST_DETECTION:
        det = LAST_DETECTION
        image_context = (
            f"Recent uploaded image analysis: disease={det.get('disease')}, "
            f"damage={det.get('damage_percentage')}%, severity={det.get('severity')}. "
            f"Masked image available at {det.get('masked_rel_path', '')}."
        )

    system_prompt = AGRICULTURE_PROMPT + "\n\n" + mem_prompt + "\n" + image_context + "\nContext from knowledge base:\n" + context
    print(f"[TEXT QUERY DEBUG] Calling LLM with user text", file=sys.stderr)
    # If user types 'expand' we expand the previous answer
    if text.strip().lower() == 'expand':
        if not LAST_RESPONSE:
            return jsonify({"error": "No previous answer to expand."}), 400
        prev = LAST_RESPONSE
        expand_context = prev.get('system_prompt', system_prompt) + "\n\nPrevious short answer: " + prev.get('short_answer', '')
        resp = generate_answer(expand_context, prev.get('english_text', english_text), max_words=300, expand=True)
    else:
        resp = generate_answer(system_prompt, english_text, max_words=100, expand=False)

    if isinstance(resp, dict) and resp.get('ok'):
        answer = resp.get('text', '')
        print(f"[TEXT QUERY DEBUG] LLM response received: {len(answer)} characters", file=sys.stderr)
    else:
        print(f"[TEXT QUERY DEBUG] LLM error response: {resp}", file=sys.stderr)
        if isinstance(resp, dict) and resp.get('error') == 'rate_limit':
            retry = resp.get('retry_after')
            fallback = ' '.join(context.split()[:100])
            provider_message = (resp.get('message') or 'Temporary service limit reached.').strip()
            message = f"{provider_message} Showing a short local suggestion."
            if retry:
                message += f" Retry after ~{retry} seconds."
            answer = f"{message}\n\n{fallback}"
        elif isinstance(resp, dict) and resp.get('error') == 'invalid_key':
            return jsonify({"error": resp.get('message')}), 500
        else:
            err_msg = resp.get('message') if isinstance(resp, dict) else str(resp)
            return jsonify({"error": f"LLM error: {err_msg}"}), 500
    
    # Check for API key errors
    if isinstance(answer, str) and ("Invalid Gemini API key" in answer or answer.startswith("Error: GEMINI_API_KEY")):
        print(f"[TEXT QUERY ERROR] API Key error detected: {answer}", file=sys.stderr)
        return jsonify({"error": answer}), 500
    
    if lang != 'en' and lang != 'unknown':
        answer_translated = translate_to_target(answer, lang)
    else:
        answer_translated = answer

    # Store latest concise answer for expand requests
    try:
        LAST_RESPONSE['system_prompt'] = system_prompt
        LAST_RESPONSE['english_text'] = english_text
        LAST_RESPONSE['short_answer'] = answer
        LAST_RESPONSE['language'] = lang
    except Exception:
        pass
    
    # Generate TTS audio (mp3)
    tts_filename = f"reply_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(api_app.config['UPLOAD_FOLDER'], tts_filename)
    try:
        # gTTS supports many languages; use detected `lang` not the selector
        tts_lang = lang if lang and lang != 'unknown' else 'en'
        tts = gTTS(text=answer_translated, lang=tts_lang)
        tts.save(tts_path)
    except Exception as e:
        print(f"[TTS ERROR] TTS generation failed: {e}", file=sys.stderr)
        return jsonify({"error": f"TTS generation failed: {e}"}), 500
    
    print(f"[TEXT QUERY SUCCESS] Response generated and audio created", file=sys.stderr)
    return jsonify({
        "text": answer_translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": lang,
        "output": _build_text_output(answer_translated)
    })


@api_app.route('/translate_reply', methods=['POST'])
def translate_reply():
    """Translate an existing assistant reply and return translated text + TTS audio."""
    text = request.form.get('text', '').strip()
    target = request.form.get('target_lang', 'en').strip().lower()
    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Normalize target language
    if target not in ['en', 'ta', 'te']:
        target = 'en'

    try:
        translated = translate_to_target(text, target)
    except Exception as e:
        return jsonify({"error": f"Translation failed: {e}"}), 500

    # Create TTS for translated text
    try:
        tts_filename = f"reply_{uuid.uuid4().hex}_trans_{target}.mp3"
        tts_path = os.path.join(api_app.config['UPLOAD_FOLDER'], tts_filename)
        tts = gTTS(text=translated, lang=target if target != 'unknown' else 'en')
        tts.save(tts_path)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {e}"}), 500

    return jsonify({
        "text": translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": target,
        "output": _build_text_output(translated)
    })


def allowed_image_filename(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXT


@api_app.route('/detect_disease', methods=['POST'])
def detect_disease():
    """Detect crop disease and calculate damage percentage from uploaded image.

    The frontend may send a `language` form field (e.g. 'ta' or 'te') so that the
    explanation returned by the LLM can be translated and spoken in the user's
    preferred language.  If no language is provided, English is assumed.
    """
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    f = request.files['image']
    if f.filename == '':
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_image_filename(f.filename):
        return jsonify({"error": "Unsupported image file type. Use PNG, JPG, JPEG, GIF, or BMP."}), 400

    # capture selected language so we can translate and TTS the explanation
    user_language = request.form.get('language', 'en')
    user_language = normalize_language(user_language)

    filename = secure_filename(f"detect_{uuid.uuid4().hex}_{f.filename}")
    save_path = os.path.join(api_app.config['IMAGE_FOLDER'], filename)
    f.save(save_path)

    # Run disease detection
    result = detect_disease_damage(save_path)

    low_confidence = result.get('confidence', 1.0) < 0.5 or result.get('vision_source') == 'mock'
    if result.get('error') and not low_confidence:
        return jsonify({"error": result['error']}), 500

    diagnosis = result.get('disease') or ('Healthy plant' if result.get('healthy') else 'Possible crop disease')
    severity = result.get('severity') or ('Mild' if result.get('healthy') else 'Moderate')
    visual_image_path = (
        result.get('masked_image_path')
        or result.get('annotated_image_path')
        or save_path
    )
    result['disease'] = diagnosis
    result['severity'] = severity

    # Generate explanation and plan using RAG + LLM
    if result.get('healthy'):
        disease_info = f"Healthy {result.get('plant', 'crop')} with {result['damage_percentage']}% damage"
        query = (
            f"The uploaded {result.get('plant', 'crop')} image appears healthy with 0% disease damage. "
            "Give a short reassurance and simple preventive care advice. Do not suggest treatment for any disease."
        )
    else:
        disease_info = f"{diagnosis} with {result['damage_percentage']}% damage and {severity} severity"
        query = (
            f"Explain what {result['damage_percentage']}% damage means for {diagnosis}, "
            f"and provide a detailed treatment and management plan for {severity} severity level. "
            f"{'Note confidence may be lower if image is blurry - Gemini Vision works best with clear close-ups. Take another photo if needed.' if low_confidence else 'Gemini Vision provides reliable analysis from this image.'}"
        )
    context = retrieve_context(disease_info)
    explanation_resp = generate_answer(context, query, max_words=300)
    # explanation_resp should be a dict from generate_answer
    if isinstance(explanation_resp, dict) and explanation_resp.get("ok"):
        explanation_text = explanation_resp.get("text", "")
    else:
        # if error, use the message or stringify response
        explanation_text = (
            explanation_resp.get("message")
            if isinstance(explanation_resp, dict)
            else str(explanation_resp)
        )

    # Translate explanation if needed
    explanation_translated = explanation_text
    explanation_audio_url = None
    if user_language != 'en' and user_language != 'unknown':
        try:
            explanation_translated = translate_to_target(explanation_text, user_language)
        except Exception as e:
            # translation failure should not block the main response
            print(f"[TRANSLATE ERROR] {e}", file=sys.stderr)
            explanation_translated = explanation_text

    # generate TTS for explanation if the frontend requested a non-English language
    try:
        tts_lang = user_language if user_language and user_language != 'unknown' else 'en'
        tts = gTTS(text=explanation_translated, lang=tts_lang)
        tts_fn = f"explanation_{uuid.uuid4().hex}.mp3"
        tts_path = os.path.join(api_app.config['UPLOAD_FOLDER'], tts_fn)
        tts.save(tts_path)
        explanation_audio_url = f"/audio/{tts_fn}"
    except Exception as e:
        print(f"[TTS ERROR] Explanation TTS failed: {e}", file=sys.stderr)
        # we simply won't send an audio url
        explanation_audio_url = None

    # Return relative path for the masked image
    masked_rel_path = f"/images/{os.path.basename(visual_image_path)}"

    # persist detection to DB
    try:
        save_detection(
            DB_PATH,
            USER_MEMORY.get('user_id'),
            save_path,
            visual_image_path,
            diagnosis,
            result.get('damage_percentage'),
            severity,
        )
    except Exception:
        pass

    # update in-memory last detection so follow-up queries can reference it
    try:
        LAST_DETECTION['disease'] = diagnosis
        LAST_DETECTION['damage_percentage'] = result.get('damage_percentage')
        LAST_DETECTION['severity'] = severity
        LAST_DETECTION['masked_rel_path'] = masked_rel_path
    except Exception:
        pass

    resp = {
        "disease": diagnosis,
        "damage_percentage": result['damage_percentage'],
        "severity": severity,
        "explanation": explanation_translated,
        "masked_image_url": masked_rel_path,
        "output": _build_disease_output(result, explanation_translated)
    }
    if low_confidence and not result.get('healthy'):
        resp["warning"] = "Low-confidence detection. Please confirm with a clearer crop photo."
    if explanation_audio_url:
        resp['explanation_audio_url'] = explanation_audio_url
    return jsonify(resp)


@api_app.route('/images/<path:filename>')
def serve_image(filename):
    from flask import send_from_directory
    return send_from_directory(api_app.config['IMAGE_FOLDER'], filename)


# Admin endpoints: set and validate OPENAI_API_KEY from the browser
@api_app.route('/admin/set_key', methods=['POST'])
def set_key():
    key = request.form.get('gemini_key', '').strip()
    if not key:
        return jsonify({"ok": False, "error": "No key provided"}), 400

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        # lightweight validation using the latest model
        try:
            model = genai.GenerativeModel('gemini-2.5-flash')
            model.generate_content("test")
        except Exception:
            return jsonify({"ok": False, "error": "Invalid API key or network error."}), 400

        env_path = os.path.join(BASE_DIR, '.env')
        try:
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(f'GEMINI_API_KEY={key}\n')
        except Exception as e:
            return jsonify({"ok": False, "error": f"Failed to write .env: {e}"}), 500

        # reload into process
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_path, override=True)

        return jsonify({"ok": True, "message": "Gemini API key saved and validated."})
    except Exception:
        return jsonify({"ok": False, "error": "Unexpected error during validation."}), 500


if __name__ == '__main__':
    # Allow overriding the port via the PORT environment variable (useful if 5000 is taken by AirPlay)
    try:
        port = int(os.getenv("PORT", "5001"))
    except ValueError:
        port = 5001

    try:
        api_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except OSError as e:
        print(f"Failed to start on port {port}: {e}")
        # Try next port
        fallback = port + 1
        print(f"Trying fallback port {fallback}...")
        api_app.run(host='0.0.0.0', port=fallback, debug=True)
