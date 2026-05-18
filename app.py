import os
import sys
import uuid
from flask import Flask, request, jsonify, send_from_directory, render_template, redirect, url_for, session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from asr import voice_to_text
from translate import detect_language, translate_to_english, translate_to_target
from rag import retrieve_context
from rag import exact_match_answer
from llm import generate_answer
from disease_detection import detect_disease_damage
from db import init_db, get_user, create_or_update_user, add_query, get_recent_queries, save_detection, register_user, login_user, get_user_history, save_message, get_chat_messages, create_session, get_sessions, delete_session_messages, get_session_by_id, update_session_title, update_user_profile

from gtts import gTTS
from functools import wraps

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "audio")
IMAGE_FOLDER = os.path.join(BASE_DIR, "static", "images")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

ALLOWED_EXT = {"wav", "mp3", "webm", "ogg", "m4a"}
ALLOWED_IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "bmp"}

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'farmer-ai-secret-2024')
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["IMAGE_FOLDER"] = IMAGE_FOLDER

# path to sqlite db
DB_PATH = os.path.join(BASE_DIR, 'data', 'app.db')
init_db(DB_PATH)

# simple user memory store for identity only — conversation memory is session-scoped
USER_MEMORY = {
    'name': None,
    'user_id': None,
}

# Per-session ephemeral state so new chat starts fresh while existing chats keep context.
SESSION_STATE = {}

# if a user already exists in DB, load into memory
existing = get_user(DB_PATH)
if existing:
    USER_MEMORY['name'] = existing.get('name')
    USER_MEMORY['user_id'] = existing.get('id')

import re


def clean_text(text: str) -> str:
    """Strip markdown bold/italic markers and 'consult local expert' phrases."""
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
    text = re.sub(r'\*(.+?)\*', r'\1', text)
    text = re.sub(r'\*+', '', text)
    # Remove any sentence that suggests consulting local/extension experts
    sentences = re.split(r'(?<=[.!?])\s+', text)
    filtered = [
        s for s in sentences
        if not re.search(
            r'consult|extension officer|local expert|local agricultural|seek.*advice|contact.*expert|local.*guidance',
            s, re.IGNORECASE
        )
    ]
    return ' '.join(filtered).strip()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


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


def determine_session_title(user_text: str, answer_text: str = None) -> str:
    """Generate a meaningful session title from user text and answer fallback."""
    if not user_text:
        user_text = ''
    txt = user_text.strip()
    ans = (answer_text or '').strip()
    greetings = {
        'hi', 'hello', 'hey', 'hai', 'hii', 'hello there', 'hey there',
        'good morning', 'good afternoon', 'good evening'
    }

    # if user typed just a greeting or a tiny token, prefer answer summary
    if txt.lower() in greetings or len(txt.split()) <= 2:
        candidate = ans or txt
    else:
        candidate = txt

    candidate = ' '.join((candidate or 'New Chat').split())
    if len(candidate) > 60:
        candidate = candidate[:57].rstrip() + '...'
    return candidate or 'New Chat'


def is_greeting_title(title: str) -> bool:
    low = (title or '').strip().lower()
    return low in {
        'hi', 'hello', 'hey', 'hai', 'hii', 'new chat'
    }

# Agriculture-specific system prompt for friendly, helpful responses
AGRICULTURE_PROMPT = """You are a friendly, warm and knowledgeable agricultural advisor helping farmers.

Your role:
1. Give practical, easy-to-understand advice about crops, soil, pests, irrigation, fertilizers and farming.
2. Use simple language farmers can understand. Be warm, encouraging and supportive.
3. When a crop disease image has been analyzed, always use that information to answer follow-up questions.
4. Never say you don't know about an uploaded image — if image analysis data is provided in context, use it fully.
5. Give specific step-by-step treatment plans, prevention tips and next steps.
6. Do NOT use markdown symbols like **, *, or bullet dashes. Write in plain friendly sentences.
7. Always end your response with an encouraging note or offer to help further.
8. NEVER tell the farmer to consult a local expert or extension officer. You ARE the expert. Always give the best, most complete advice yourself.
9. Never say phrases like 'consult your local agricultural experts', 'contact extension officers', 'seek local guidance' or anything similar. Give the full answer directly.
10. Treat each chat session separately. Only use memory from the current chat session.
11. If the farmer starts a new chat with a greeting like hello or hi, respond freshly and do not mention old topics unless they are in the current chat history.
12. Make the answer specific and useful, not generic. When relevant, include timing, quantity, sequence, and warning signs to watch for."""


def allowed_filename(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT


def _request_session_id() -> int | None:
    raw = request.form.get('session_id') or request.args.get('session_id')
    if not raw:
        return session.get('chat_session_id')
    try:
        return int(raw)
    except (TypeError, ValueError):
        return session.get('chat_session_id')


def _get_session_state(session_id: int | None) -> dict:
    if not session_id:
        return {}
    return SESSION_STATE.setdefault(session_id, {})


def _set_active_session(session_id: int | None) -> int | None:
    user_id = session.get('user_id')
    if not user_id or not session_id:
        session.pop('chat_session_id', None)
        return None
    existing_session = get_session_by_id(DB_PATH, user_id, session_id)
    if not existing_session:
        session.pop('chat_session_id', None)
        return None
    session['chat_session_id'] = session_id
    return session_id


def _ensure_active_session(first_user_text: str = "New Chat", answer_text: str = "") -> int | None:
    user_id = session.get('user_id') or USER_MEMORY.get('user_id')
    if not user_id:
        return None
    session_id = _set_active_session(_request_session_id())
    if session_id:
        return session_id
    title = determine_session_title(first_user_text, answer_text)
    session_id = create_session(DB_PATH, user_id, title)
    session['chat_session_id'] = session_id
    return session_id


def _session_messages_for_prompt(user_id: int | None, session_id: int | None, limit: int = 10) -> list[dict]:
    if not user_id or not session_id:
        return []
    messages = get_chat_messages(DB_PATH, user_id, session_id=session_id, limit=limit)
    return messages[-limit:]


def _extract_session_name(messages: list[dict]) -> str | None:
    for message in reversed(messages):
        if message.get('role') != 'user':
            continue
        name = extract_name_from_text(message.get('content', ''))
        if name:
            return name
    return None


def _build_session_memory_prompt(messages: list[dict]) -> str:
    if not messages:
        return "Current chat memory: this is a fresh chat with no earlier messages.\n"
    transcript_lines = []
    for message in messages[-8:]:
        role = "Farmer" if message.get('role') == 'user' else "Advisor"
        transcript_lines.append(f"{role}: {message.get('content', '').strip()}")
    return "Current chat memory from this session only:\n" + "\n".join(transcript_lines) + "\n"


def _save_chat_messages(session_id: int | None, user_text: str, answer_text: str, audio_url: str | None = None):
    user_id = session.get('user_id') or USER_MEMORY.get('user_id')
    if not user_id or not session_id:
        return

    existing_session = get_session_by_id(DB_PATH, user_id, session_id)
    if existing_session and is_greeting_title(existing_session.get('title')):
        session_title = determine_session_title(user_text, answer_text)
        if session_title and not is_greeting_title(session_title):
            update_session_title(DB_PATH, user_id, session_id, session_title)

    save_message(DB_PATH, user_id, 'user', user_text, session_id=session_id)
    save_message(DB_PATH, user_id, 'assistant', answer_text, audio_url, session_id=session_id)


@app.route('/')
@login_required
def index():
    return render_template('index.html', username=session.get('username'), name=session.get('name'))


@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user_id' in session:
        return redirect('/')
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            user = login_user(DB_PATH, username, password)
            if user:
                session['user_id'] = user['id']
                session['username'] = user['username']
                session['name'] = user['name']
                session['location'] = user.get('location', '')
                USER_MEMORY['name'] = user['name']
                USER_MEMORY['user_id'] = user['id']
                return redirect('/')
            else:
                error = 'Invalid username or password.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if 'user_id' in session:
        return redirect('/')
    error = None
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm', '').strip()
        if not name or not username or not password or not confirm:
            error = 'All fields are required.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        else:
            result = register_user(DB_PATH, name, username, password)
            if result['ok']:
                return redirect('/login')
            else:
                error = result['error']
    return render_template('register.html', error=error)


@app.route('/api/profile')
@login_required
def api_get_profile():
    return jsonify({
        'name': session.get('name', ''),
        'username': session.get('username', ''),
        'location': session.get('location', '')
    })


@app.route('/api/profile/update', methods=['POST'])
@login_required
def api_update_profile():
    name = request.form.get('name', '').strip()
    location = request.form.get('location', '').strip()
    if not name:
        return jsonify({'ok': False, 'error': 'Name cannot be empty'}), 400
    user_id = session.get('user_id')
    update_user_profile(DB_PATH, user_id, name, location)
    session['name'] = name
    session['location'] = location
    USER_MEMORY['name'] = name
    return jsonify({'ok': True, 'name': name, 'location': location})


@app.route('/logout')
def logout():
    session.clear()
    USER_MEMORY['name'] = None
    USER_MEMORY['user_id'] = None
    return redirect('/login')


@app.route('/api/history')
@login_required
def api_history():
    user_id = session.get('user_id')
    sessions = get_sessions(DB_PATH, user_id, limit=30)
    return jsonify(sessions)


@app.route('/api/messages')
@login_required
def api_messages():
    user_id = session.get('user_id')
    session_id = request.args.get('session_id', type=int) or session.get('chat_session_id')
    if not session_id:
        return jsonify([])
    _set_active_session(session_id)
    messages = get_chat_messages(DB_PATH, user_id, session_id=session_id)
    return jsonify(messages)


@app.route('/api/current_session')
@login_required
def api_current_session():
    return jsonify({'session_id': session.get('chat_session_id')})


@app.route('/api/select_session', methods=['POST'])
@login_required
def api_select_session():
    session_id = request.form.get('session_id', type=int)
    active = _set_active_session(session_id)
    if not active:
        return jsonify({'ok': False, 'error': 'Session not found'}), 404
    return jsonify({'ok': True, 'session_id': active})


@app.route('/api/clear_messages', methods=['POST'])
@login_required
def api_clear_messages():
    # New Chat should keep history/session records while starting fresh chat window.
    # Do not delete previous sessions from history; just reset active session scope.
    previous_session_id = session.pop('chat_session_id', None)
    if previous_session_id:
        SESSION_STATE.pop(previous_session_id, None)
    return jsonify({'ok': True})


@app.route('/query', methods=['POST'])
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
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
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

    session_id = _ensure_active_session(text)
    session_messages = _session_messages_for_prompt(session.get('user_id'), session_id)
    session_memory_prompt = _build_session_memory_prompt(session_messages)
    session_name = _extract_session_name(session_messages)
    session_state = _get_session_state(session_id)

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
    # persist query
    try:
        add_query(DB_PATH, USER_MEMORY.get('user_id'), english_text)
    except Exception:
        pass

    # if the user is asking about their name, reply directly from memory
    low = english_text.lower()
    remembered_name = name or session_name
    if remembered_name and ("my name" in low or "your name" in low):
        answer = f"You told me your name is {remembered_name}."
        context = ""
        match_answer = answer
        match_score = 1.0
        print(f"[MEMORY] Answering name query from memory", file=sys.stderr)
    else:
        print(f"[QUERY DEBUG] Checking local QA dataset for an exact match for: {english_text[:50]}...", file=sys.stderr)
        match_answer, match_score = exact_match_answer(english_text)
        if match_answer:
            print(f"[QUERY DEBUG] Found local QA match (score={match_score:.2f}), skipping LLM.", file=sys.stderr)
            answer = match_answer
            context = ""
        else:
            print(f"[QUERY DEBUG] No strong local match (best_score={match_score:.2f}); retrieving context.", file=sys.stderr)
            context = retrieve_context(english_text)
            print(f"[QUERY DEBUG] Context retrieved: {len(context)} characters", file=sys.stderr)
    
    # If the user refers to the last uploaded image, include the full detection summary
    img_ref_phrases = [
        'that image', 'the image', 'this image', 'the photo', 'that photo',
        'my plant', 'my crop', 'my leaf', 'the leaf', 'the plant', 'the crop',
        'what disease', 'what is wrong', 'what happened', 'tell me more',
        'more details', 'what should i do', 'how to treat', 'how to fix',
        'the disease', 'that disease', 'detected disease', 'uploaded'
    ]
    image_context = ''
    if session_state.get('last_detection_summary') and any(phrase in english_text.lower() for phrase in img_ref_phrases):
        image_context = (
            f"IMPORTANT - The farmer already uploaded a crop image and here is the full analysis result: "
            f"{session_state['last_detection_summary']} "
            f"Use this information to answer the farmer's follow-up question in a friendly, helpful and detailed way."
        )

    # Pass agriculture prompt along with context and optional image context
    identity_prompt = f"The farmer introduced themselves as {remembered_name}.\n" if remembered_name else ""
    system_prompt = AGRICULTURE_PROMPT + "\n\n" + identity_prompt + session_memory_prompt + image_context + "\nContext from knowledge base:\n" + context
    print(f"[QUERY DEBUG] Calling LLM with system prompt and user query", file=sys.stderr)
    # If user asked for expansion via voice (single word 'expand')
    if english_text.strip().lower() == 'expand':
        if not session_state.get('last_response'):
            return jsonify({"error": "No previous answer to expand."}), 400
        # Use previous system prompt and short answer to expand
        prev = session_state['last_response']
        expand_context = prev.get('system_prompt', system_prompt) + "\n\nPrevious short answer: " + prev.get('short_answer', '')
        resp = generate_answer(expand_context, prev.get('english_text', english_text), max_words=300, expand=True)
    else:
        # If we already found a local match above, skip calling LLM
        if match_answer:
            resp = {"ok": True, "text": answer}
        else:
            resp = generate_answer(system_prompt, english_text, max_words=200, expand=False)
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
        answer_translated = clean_text(translate_to_target(answer, lang))
    else:
        answer_translated = clean_text(answer)

    # Store the latest concise answer so the user can ask 'expand' later
    try:
        session_state['last_response'] = {
            'system_prompt': system_prompt,
            'english_text': english_text,
            'short_answer': answer,
            'language': lang,
        }
    except Exception:
        pass

    # Generate TTS audio (mp3)
    tts_filename = f"reply_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(app.config['UPLOAD_FOLDER'], tts_filename)
    try:
        # gTTS supports many languages (e.g., 'ta' for Tamil, 'te' for Telugu)
        tts_lang = lang if lang and lang != 'unknown' else 'en'
        tts = gTTS(text=answer_translated, lang=tts_lang)
        tts.save(tts_path)
    except Exception as e:
        print(f"[TTS ERROR] TTS generation failed: {e}", file=sys.stderr)
        return jsonify({"error": f"TTS generation failed: {e}"}), 500

    print(f"[QUERY SUCCESS] Response generated and audio created", file=sys.stderr)
    # Save both messages to DB under current session
    try:
        _save_chat_messages(session_id, text, answer_translated, f"/audio/{tts_filename}")
    except Exception as e:
        print(f"[SESSION SAVE ERROR] {e}", file=sys.stderr)
    return jsonify({
        "text": answer_translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": lang,
        "transcript": text
    })


@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/query_text', methods=['POST'])
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

    session_id = _ensure_active_session(text)
    session_messages = _session_messages_for_prompt(session.get('user_id'), session_id)
    session_memory_prompt = _build_session_memory_prompt(session_messages)
    session_name = _extract_session_name(session_messages)
    session_state = _get_session_state(session_id)

    # update memory from text input
    name = extract_name_from_text(english_text)
    if name:
        USER_MEMORY['name'] = name
        print(f"[MEMORY] Stored user name: {name}", file=sys.stderr)

    # if the user is asking about their name, answer from memory immediately
    low = english_text.lower()
    remembered_name = name or session_name
    if remembered_name and ("my name" in low or "your name" in low):
        answer = f"You told me your name is {remembered_name}."
        context = ""
        match_answer = answer
        match_score = 1.0
        print(f"[MEMORY] Text query name lookup", file=sys.stderr)
    else:
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
    
    # If the user refers to the last uploaded image, include the full detection summary
    img_ref_phrases = [
        'that image', 'the image', 'this image', 'the photo', 'that photo',
        'my plant', 'my crop', 'my leaf', 'the leaf', 'the plant', 'the crop',
        'what disease', 'what is wrong', 'what happened', 'tell me more',
        'more details', 'what should i do', 'how to treat', 'how to fix',
        'the disease', 'that disease', 'detected disease', 'uploaded'
    ]
    image_context = ''
    if session_state.get('last_detection_summary') and any(phrase in english_text.lower() for phrase in img_ref_phrases):
        image_context = (
            f"IMPORTANT - The farmer already uploaded a crop image and here is the full analysis result: "
            f"{session_state['last_detection_summary']} "
            f"Use this information to answer the farmer's follow-up question in a friendly, helpful and detailed way."
        )

    identity_prompt = f"The farmer introduced themselves as {remembered_name}.\n" if remembered_name else ""
    system_prompt = AGRICULTURE_PROMPT + "\n\n" + identity_prompt + session_memory_prompt + image_context + "\nContext from knowledge base:\n" + context
    print(f"[TEXT QUERY DEBUG] Calling LLM with user text", file=sys.stderr)
    # If user types 'expand' we expand the previous answer
    if text.strip().lower() == 'expand':
        if not session_state.get('last_response'):
            return jsonify({"error": "No previous answer to expand."}), 400
        prev = session_state['last_response']
        expand_context = prev.get('system_prompt', system_prompt) + "\n\nPrevious short answer: " + prev.get('short_answer', '')
        resp = generate_answer(expand_context, prev.get('english_text', english_text), max_words=300, expand=True)
    else:
        resp = generate_answer(system_prompt, english_text, max_words=200, expand=False)

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
        answer_translated = clean_text(translate_to_target(answer, lang))
    else:
        answer_translated = clean_text(answer)

    # Store latest concise answer for expand requests
    try:
        session_state['last_response'] = {
            'system_prompt': system_prompt,
            'english_text': english_text,
            'short_answer': answer,
            'language': lang,
        }
    except Exception:
        pass
    
    # Generate TTS audio (mp3)
    tts_filename = f"reply_{uuid.uuid4().hex}.mp3"
    tts_path = os.path.join(app.config['UPLOAD_FOLDER'], tts_filename)
    try:
        # gTTS supports many languages; use detected `lang` not the selector
        tts_lang = lang if lang and lang != 'unknown' else 'en'
        tts = gTTS(text=answer_translated, lang=tts_lang)
        tts.save(tts_path)
    except Exception as e:
        print(f"[TTS ERROR] TTS generation failed: {e}", file=sys.stderr)
        return jsonify({"error": f"TTS generation failed: {e}"}), 500
    
    print(f"[TEXT QUERY SUCCESS] Response generated and audio created", file=sys.stderr)
    # Save both messages to DB under current session
    try:
        _save_chat_messages(session_id, text, answer_translated, f"/audio/{tts_filename}")
    except Exception as e:
        print(f"[SESSION SAVE ERROR] {e}", file=sys.stderr)
    return jsonify({
        "text": answer_translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": lang
    })


@app.route('/translate_reply', methods=['POST'])
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
        tts_path = os.path.join(app.config['UPLOAD_FOLDER'], tts_filename)
        tts = gTTS(text=translated, lang=target if target != 'unknown' else 'en')
        tts.save(tts_path)
    except Exception as e:
        return jsonify({"error": f"TTS generation failed: {e}"}), 500

    return jsonify({
        "text": translated,
        "audio_url": f"/audio/{tts_filename}",
        "detected_language": target
    })


def allowed_image_filename(filename):
    return "." in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXT


@app.route('/detect_disease', methods=['POST'])
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
    save_path = os.path.join(app.config['IMAGE_FOLDER'], filename)
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

    # Generate explanation using a friendly, detailed prompt
    session_id = _ensure_active_session("Analyze crop image")
    session_state = _get_session_state(session_id)
    if result.get('healthy'):
        disease_info = f"Healthy {result.get('plant', 'crop')} with {result['damage_percentage']}% damage"
    else:
        disease_info = f"{diagnosis} with {result['damage_percentage']}% damage and {severity} severity"
    context = retrieve_context(disease_info)
    query = (
        f"A farmer uploaded a crop image. The AI detected: {diagnosis} with {result['damage_percentage']}% damage "
        f"{'Healthy' if result.get('healthy') else severity} severity level. "
        f"Please respond in a warm, friendly and helpful way. "
        f"1. Explain clearly what {diagnosis} is and what causes it. "
        f"2. Explain what {result['damage_percentage']}% damage means for the crop and how serious it is. "
        f"3. Give step-by-step treatment and prevention steps the farmer should take right now. "
        f"4. Give suggestions on what to do next to protect the remaining crop. "
        f"{'Note the analysis uses Gemini Vision AI - for best results use clear close-up photos of affected leaves. ' if low_confidence else 'This AI analysis uses advanced Gemini Vision for reliable detection.'}"
        f"Use simple language a farmer can understand. Do not use bullet symbols or markdown."
    )
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
    explanation_translated = clean_text(explanation_text)
    explanation_audio_url = None
    if user_language != 'en' and user_language != 'unknown':
        try:
            explanation_translated = clean_text(translate_to_target(explanation_text, user_language))
        except Exception as e:
            # translation failure should not block the main response
            print(f"[TRANSLATE ERROR] {e}", file=sys.stderr)
            explanation_translated = explanation_text

    # generate TTS for explanation if the frontend requested a non-English language
    try:
        tts_lang = user_language if user_language and user_language != 'unknown' else 'en'
        tts = gTTS(text=explanation_translated, lang=tts_lang)
        tts_fn = f"explanation_{uuid.uuid4().hex}.mp3"
        tts_path = os.path.join(app.config['UPLOAD_FOLDER'], tts_fn)
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
        session_state['last_detection'] = {
            'disease': diagnosis,
            'damage_percentage': result.get('damage_percentage'),
            'severity': severity,
            'masked_rel_path': masked_rel_path,
            'explanation': explanation_text,
        }
        session_state['last_detection_summary'] = (
            f"The farmer uploaded a crop image. Gemini Vision identified: "
            f"Disease = {diagnosis}, "
            f"Damage = {result.get('damage_percentage')}%, "
            f"Severity = {severity}. "
            f"Full explanation: {explanation_text}"
        )
    except Exception:
        pass

    try:
        _save_chat_messages(
            session_id,
            "Analyze my crop image",
    f"Plant: {result.get('plant', 'Crop')}. {'Healthy' if result.get('healthy') else f'Disease: {result["disease"]}. Damage: {result["damage_percentage"]}%. Severity: {result["severity"]}'}. {explanation_translated}",
            explanation_audio_url,
        )
    except Exception as e:
        print(f"[SESSION SAVE ERROR] {e}", file=sys.stderr)

    resp = {
        "disease": diagnosis,
        "damage_percentage": result['damage_percentage'],
        "severity": severity,
        "explanation": explanation_translated,
        "masked_image_url": masked_rel_path
    }
    if low_confidence:
        resp["warning"] = "Low-confidence detection. Please confirm with a clearer crop photo."
    if explanation_audio_url:
        resp['explanation_audio_url'] = explanation_audio_url
    return jsonify(resp)


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(app.config['IMAGE_FOLDER'], filename)


# Admin endpoints: set and validate OPENAI_API_KEY from the browser
@app.route('/admin', methods=['GET'])
def admin_page():
    return render_template('admin.html')


@app.route('/admin/set_key', methods=['POST'])
def set_key():
    key = request.form.get('gemini_key', '').strip()
    if not key:
        return jsonify({"ok": False, "error": "No key provided"}), 400

    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        # lightweight validation using the latest model
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
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
        port = int(os.getenv("PORT", "5000"))
    except ValueError:
        port = 5000

    try:
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    except OSError as e:
        print(f"Failed to start on port {port}: {e}")
        # Try next port
        fallback = port + 1
        print(f"Trying fallback port {fallback}...")
        app.run(host='0.0.0.0', port=fallback, debug=True)
