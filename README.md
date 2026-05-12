# Farmer-AI-Chatbot
A comprehensive AI-powered chatbot for farmers featuring plant disease detection through computer vision, voice-based queries via ASR, multilingual translation, and intelligent Q&amp;A using RAG and LLM technologies. Built with web and desktop interfaces, supporting farmers with crop health analysis and agricultural knowledge



Farmer AI Advisor - Multilingual Voice Chatbot

A friendly, intelligent AI chatbot that helps farmers with agriculture advice in Tamil, English, and Telugu.

Features:
- 🌾 Voice input in Tamil, English, or Telugu
- 🤖 Smart responses powered by Gemini AI and RAG
- 🔊 Voice replies in the farmer's language
- 💬 Modern ChatGPT-like interface with message bubbles
- 🗣️ Real-time speech-to-text transcription using Whisper
- 📚 Agriculture-specific knowledge base (crops, soil, pests, water management)
- 🌍 Multilingual translation and detection
- 📷 Camera-based crop disease detection with damage percentage calculation

Pipeline:
Voice Input (Mic) → ASR (Whisper) → Language Detection → Translation (if needed) → LLM + RAG → Translate back → TTS Voice Reply
Camera Input → Image Segmentation (YOLOv8) → Disease Detection → Damage % Calculation → Results Display

Tech Stack:
- Frontend: HTML5, CSS3, JavaScript (Web Audio API for mic, Camera API for photos)
- Backend: Flask (Python)
- Speech: Whisper (OpenAI) for ASR, gTTS for TTS
- Language: googletrans for translation
- NLP: sentence-transformers + FAISS for semantic search (RAG)
- LLM: Google Gemini AI
- Computer Vision: YOLOv8 Segmentation (Ultralytics) for disease detection

Setup Instructions:

1. Clone or download the project.

2. Create a virtual environment:
   python3 -m venv .venv
   source .venv/bin/activate

3. Install dependencies:
   pip install -r requirements.txt

4. Set up your OpenAI API key:
   Option A (Environment):
   export OPENAI_API_KEY="sk-..."
   
   Option B (.env file in project root):
   Create .env file with:
   OPENAI_API_KEY=sk-...

5. Run the web app:
   PORT=5001 python app.py
   
   (Or just: python app.py if port 5000 is available)

6. Open your browser:
   http://localhost:5001

7. How to use:
   - Select your preferred language (Tamil, English, or Telugu) from the dropdown.
   - Click the microphone button to start recording.
   - Speak clearly in your selected language about your farming question.
   - Stop recording and wait for the AI response.
   - The chatbot will reply in your language with both text and voice.

Example Questions (in any language):
- "How do I grow better tomatoes?"
- "What fertilizer should I use for rice?"
- "How to save water in my farm?"
- "My crops have yellow leaves, what should I do?"
- "When should I plant wheat?"
- "Tell me about drip irrigation"

How to run (CLI mode - for debugging):
- Install dependencies: pip install -r requirements.txt
- python main.py (expects farmer_query.wav file or auto-records)

Knowledge Base:
The chatbot has comprehensive agricultural knowledge covering:
- Crop Management: Rice, wheat, vegetables, and more
- Soil Health: Testing, organic farming, nutrient management
- Water Management: Irrigation techniques, water-saving tips
- Pest & Disease Control: IPM, common pests, organic solutions
- Fertilizer Use: Organic and chemical options
- Seasonal Planting: Kharif, Rabi, and summer crops
- Climate-Sensitive Practices: Drought, flood, heat management
- Government Schemes: Agricultural subsidies and crop insurance

Language Support:
- English: Full support
- Tamil (தமிழ்): Speech recognition, translation, TTS
- Telugu (తెలుగు): Speech recognition, translation, TTS

Notes:
- First run may be slow (Whisper model download ~1.4 GB).
- For production, use a proper WSGI server (gunicorn) instead of Flask debug mode.
- Ensure microphone permissions are granted in your browser.
- For high-quality Tamil/Telugu TTS, consider integrating Coqui TTS in the future.

Troubleshooting:
- "Port 5000 already in use": Set PORT=5001 or another port before running.
- "Microphone access denied": Check browser permissions and allow microphone access.
- "No speech detected": Speak clearly, ensure audio is working, increase recording duration.
- "Invalid API key": Check that your OPENAI_API_KEY is correct and has access to GPT-3.5-turbo.
- "FAISS error": The system will fall back to keyword-based search; no action needed.

Crop Disease Detection:
- Click the camera button to take a photo of your crop
- The system uses YOLOv8 segmentation to detect diseased areas
- Returns disease name, damage percentage, and severity level
- Shows highlighted image of affected regions
- Currently uses mock detection for demo; train a real model for production

Future Improvements:
- Fine-tuned model specifically for agriculture domain
- Multi-turn conversation memory
- User feedback and model improvement loop
- Integration with weather APIs
- Enhanced crop disease image recognition with trained models
- Integration with local agricultural extension services

License: Open source
Support: For issues or feature requests, please contact the development team.
