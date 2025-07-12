# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# --- Voice & Language ---
LANGUAGE_CODE = "tr-TR"
SAMPLE_RATE = 16000  # Rate for STT and live audio processing

# --- API & Services ---
# Make sure 'google-tts-key.json' is in the same directory.
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "google-tts-key.json"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Asterisk ARI Configs ---
ARI_USER = 'ai'
ARI_PASSWORD = 'ai_secret'
ARI_HOST = 'localhost'
ARI_PORT = 8088
ARI_APP = 'aiagent'
BASE_URL = f'http://{ARI_HOST}:{ARI_PORT}/ari'

# --- Paths & Naming ---
LIVE_RECORDING_PATH = "/var/spool/asterisk/recording"
TTS_SOUND_FILE_PATH = "/var/lib/asterisk/sounds/en/ai_agent_response.wav"
TTS_SOUND_ID = "ai_agent_response" # The name Asterisk uses to play the sound

# --- Call Behavior ---
# The endpoint to call when the script starts
OUTBOUND_ENDPOINT = "PJSIP/7001" 
# The context in extensions.conf that routes to Stasis(aiagent)
DIAL_CONTEXT = "ai-survey" 
CALLER_ID = "AI Bot"


# --- Supabase Config ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")