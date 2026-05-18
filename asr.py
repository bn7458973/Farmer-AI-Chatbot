import shutil
import os
import subprocess
import whisper


def record_audio(output_path: str, duration: int = 5) -> bool:
    if shutil.which("ffmpeg") is None:
        print("ASR error: ffmpeg not found. Install with: brew install ffmpeg")
        return False
    try:
        print(f"Recording {duration} seconds from microphone to {output_path}...")
        # macOS avfoundation; default device ":0". Adjust device index if needed.
        subprocess.run(["ffmpeg", "-y", "-f", "avfoundation", "-i", ":0", "-t", str(duration), output_path], check=True)
        print("Recording complete.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ASR error (recording): {e}")
        return False


def voice_to_text(audio_path: str, language: str = None) -> str:
    if not os.path.exists(audio_path):
        print(f"Audio file '{audio_path}' not found. Attempting to record from mic...")
        ok = record_audio(audio_path)
        if not ok:
            return ""
    if shutil.which("ffmpeg") is None:
        print("ASR error: ffmpeg not found. Install with: brew install ffmpeg")
        return ""
    try:
        # large-v3 is the most accurate Whisper model, especially for Tamil and Telugu
        model = whisper.load_model("large-v3")
        transcribe_kwargs = {
            "audio": audio_path,
            "beam_size": 5,          # better accuracy vs greedy
            "best_of": 5,
            "temperature": 0.0,      # deterministic — reduces hallucinations
            "condition_on_previous_text": False,
        }
        if language and language != 'unknown':
            transcribe_kwargs["language"] = language
        result = model.transcribe(**transcribe_kwargs)
        text = result.get("text", "").strip()
        detected_lang = result.get("language", "unknown")
        print(f"[ASR DEBUG] Transcribed text: {text[:80]}... Language: {detected_lang}")
        return text
    except Exception as e:
        print(f"ASR error: {e}")
        return ""
