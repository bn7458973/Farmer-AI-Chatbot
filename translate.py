from deep_translator import GoogleTranslator


def detect_language(text: str) -> str:
    if not text:
        return "unknown"
    # Script-based detection for Tamil and Telugu (fast, reliable)
    for ch in text:
        code = ord(ch)
        if 0x0B80 <= code <= 0x0BFF:
            return 'ta'
        if 0x0C00 <= code <= 0x0C7F:
            return 'te'
    # Fallback: try to detect via translation attempt
    try:
        from deep_translator import single_detection
        lang = single_detection(text, api_key=None)
        return lang if lang else 'en'
    except Exception:
        return 'en'


def translate_to_english(text: str) -> str:
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception:
        return text


def translate_to_target(text: str, target_lang: str) -> str:
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
    except Exception:
        return text
