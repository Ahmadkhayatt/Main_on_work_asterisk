# ==============================================================================
# AI VOICE AGENT - FLASK BACKEND
# ==============================================================================
# This single file contains the complete backend application, combining all
# previous scripts (main, audio, conversation, services, database, config)
# into a professional, production-ready Flask server.
#
# To Run:
# 1. Install dependencies:
#    pip install Flask Flask-SocketIO Flask-Cors websocket-client requests supabase google-cloud-speech google-cloud-texttospeech google-generativeai python-dotenv webrtcvad
# 2. Run this file:
#    python your_app_name.py
# ==============================================================================

import os
import threading
import queue
import time
import json
import logging
import pwd
import grp
import wave
from functools import wraps
from datetime import datetime, timezone

# --- Third-Party Imports ---
import requests
import websocket
import supabase
import google.generativeai as genai
from google.cloud import speech, texttospeech
from requests.auth import HTTPBasicAuth
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from flask_cors import CORS
from dotenv import load_dotenv

# ==============================================================================
# 1. INITIALIZATION AND CONFIGURATION
# ==============================================================================

# --- Load Environment Variables ---
load_dotenv()

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app)  # Allow requests from our React frontend
socketio = SocketIO(app, cors_allowed_origins="*")

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Application Configuration ---
class Config:
    # Voice & Language
    LANGUAGE_CODE = "tr-TR"
    SAMPLE_RATE = 16000

    # API & Services
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "google-tts-key.json")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    # Asterisk ARI
    ARI_USER = os.getenv("ARI_USER", 'ai')
    ARI_PASSWORD = os.getenv("ARI_PASSWORD", 'ai_secret')
    ARI_HOST = os.getenv("ARI_HOST", 'localhost')
    ARI_PORT = os.getenv("ARI_PORT", 8088)
    ARI_APP = os.getenv("ARI_APP", 'aiagent')
    BASE_URL = f'http://{ARI_HOST}:{ARI_PORT}/ari'
    WEBSOCKET_URL = f'ws://{ARI_HOST}:{ARI_PORT}/ari/events?app={ARI_APP}&api_key={ARI_USER}:{ARI_PASSWORD}'

    # Paths & Naming
    LIVE_RECORDING_PATH = "/var/spool/asterisk/recording"
    TTS_SOUND_FILE_PATH = "/var/lib/asterisk/sounds/en/ai_agent_response.wav"
    TTS_SOUND_ID = "ai_agent_response"

    # Call Behavior
    OUTBOUND_ENDPOINT = os.getenv("OUTBOUND_ENDPOINT", "PJSIP/7001")
    DIAL_CONTEXT = os.getenv("DIAL_CONTEXT", "ai-survey")
    CALLER_ID = os.getenv("CALLER_ID", "AI Bot")

# Set Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_APPLICATION_CREDENTIALS

# ==============================================================================
# 2. GLOBAL SERVICES AND STATE MANAGEMENT
# ==============================================================================

# --- Thread-safe dictionary to manage active calls ---
active_calls = {}
active_calls_lock = threading.Lock()

# --- Initialize Supabase Client ---
try:
    supabase_client = supabase.create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    logging.info("✅ Supabase client initialized successfully.")
except Exception as e:
    supabase_client = None
    logging.error(f"❌ Failed to initialize Supabase client: {e}")

# --- Initialize AI Model ---
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192}
    )
    with open('Prompt.txt', "r", encoding="utf-8") as file:
        SYSTEM_PROMPT = file.read()
    logging.info("✅ Gemini Model and prompt loaded successfully.")
except Exception as e:
    GEMINI_MODEL = None
    SYSTEM_PROMPT = None
    logging.fatal(f"Could not initialize GenerativeAI. Check API Key and prompt file. Error: {e}")

# ==============================================================================
# 3. HELPER FUNCTIONS AND DECORATORS
# ==============================================================================

def broadcast_log(log_type, message):
    """Logs to console and broadcasts to all connected UI clients."""
    log_entry = f"[{log_type}] {message}"
    
    if log_type == "ERROR" or log_type == "FATAL":
        logging.error(log_entry)
    else:
        logging.info(log_entry)
        
    socketio.emit('LOG_UPDATE', {'type': log_type, 'message': message})

def send_ari_request(method, url, **kwargs):
    """A centralized function for making authenticated requests to the ARI."""
    try:
        response = requests.request(
            method,
            url,
            auth=HTTPBasicAuth(Config.ARI_USER, Config.ARI_PASSWORD),
            timeout=5,
            **kwargs
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        broadcast_log("ERROR", f"ARI request {method} {url} failed: {e}")
        if e.response is not None:
            broadcast_log("ERROR", f"ARI Response: {e.response.text}")
        return None

# ==============================================================================
# 4. CORE APPLICATION LOGIC (from audio.py, services.py, etc.)
# ==============================================================================

class AsteriskLiveAudioStreamer(threading.Thread):
    """Tails a live recording file and pushes audio chunks into consumer queues."""
    def __init__(self, recording_path, consumer_queues):
        super().__init__()
        self.recording_path = recording_path
        self.consumer_queues = consumer_queues
        self._stop_event = threading.Event()
        self.chunk_size = int(Config.SAMPLE_RATE * 30 / 1000) * 2

    def run(self):
        timeout_seconds = 5
        start_time = time.time()
        while not os.path.exists(self.recording_path) and not self._stop_event.is_set():
            if time.time() - start_time > timeout_seconds:
                broadcast_log("FATAL", f"Timed out waiting for recording file: {self.recording_path}")
                return
            time.sleep(0.1)

        broadcast_log("INFO", f"File found! Starting to stream audio from {self.recording_path}")
        try:
            with open(self.recording_path, "rb") as f:
                while not self._stop_event.is_set():
                    chunk = f.read(self.chunk_size)
                    if chunk:
                        for q in self.consumer_queues:
                            q.put(chunk)
                    else:
                        time.sleep(0.01)
        except Exception as e:
            broadcast_log("ERROR", f"Error during audio streaming from file: {e}")
        finally:
            broadcast_log("DEBUG", f"Audio streamer for {self.recording_path} stopped.")

    def stop(self):
        self._stop_event.set()

class GoogleStreamer:
    """Manages Google's Streaming Speech-to-Text from an audio queue."""
    def __init__(self, audio_queue):
        self.audio_queue = audio_queue
        self.client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=Config.SAMPLE_RATE,
            language_code=Config.LANGUAGE_CODE,
            enable_automatic_punctuation=False,
            model="latest_long"
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config,
            interim_results=False
        )
        self._closed = threading.Event()

    def _generator(self):
        while not self._closed.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                if chunk is None: return
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def listen(self, silence_timeout=7):
        broadcast_log("DEBUG", "Listening for user input via Google STT...")
        responses = self.client.streaming_recognize(
            config=self.streaming_config, requests=self._generator()
        )
        last_speech_time = time.time()
        for response in responses:
            if not response.results:
                if time.time() - last_speech_time > silence_timeout:
                    broadcast_log("DEBUG", "Silence timeout reached.")
                    break
                continue
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                broadcast_log("INFO", f"STT result: '{transcript}'")
                return transcript
        return None

    def close(self):
        self._closed.set()
        self.audio_queue.put(None)

def speak_and_prepare_for_asterisk(text):
    """Generates speech, saves it as a WAV file, and sets permissions."""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(language_code=Config.LANGUAGE_CODE, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=8000)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        with wave.open(Config.TTS_SOUND_FILE_PATH, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(response.audio_content)

        a_uid = pwd.getpwnam("asterisk").pw_uid
        a_gid = grp.getgrnam("asterisk").gr_gid
        os.chown(Config.TTS_SOUND_FILE_PATH, a_uid, a_gid)
        os.chmod(Config.TTS_SOUND_FILE_PATH, 0o644)

        broadcast_log("DEBUG", f"TTS WAV saved and permissions set: {Config.TTS_SOUND_FILE_PATH}")
        return Config.TTS_SOUND_ID
    except Exception as e:
        broadcast_log("ERROR", f"TTS generation failed: {e}")
        return None

def log_call_to_db(caller_id, duration, transcript):
    """Inserts a new record into the call_logs table in Supabase."""
    if not supabase_client:
        broadcast_log("ERROR", "Supabase client not available. Cannot log call.")
        return None
    try:
        broadcast_log("INFO", f"Logging call from {caller_id} to database...")
        data, count = supabase_client.from_('call_logs').insert({
            'caller_id': caller_id,
            'call_duration_seconds': duration,
            'transcript': transcript
        }).execute()
        broadcast_log("INFO", "Call log saved successfully to Supabase.")
        return data
    except Exception as e:
        broadcast_log("ERROR", f"Database logging failed: {e}")
        return None

# ==============================================================================
# 5. CONVERSATION AND THREADING LOGIC
# ==============================================================================

def run_stt_listener(stt_client, result_queue, stop_event):
    """Runs in a thread to listen for utterances and put them on a queue."""
    broadcast_log("DEBUG", "[STT Thread] Starting to listen...")
    while not stop_event.is_set():
        transcript = stt_client.listen()
        if transcript:
            result_queue.put(transcript)
    broadcast_log("DEBUG", "[STT Thread] Stopped.")

def interact_with_user(channel_id, snoop_channel_id, recording_file, caller_id):
    """Handles the main conversation flow for a single call."""
    if not GEMINI_MODEL or not SYSTEM_PROMPT:
        broadcast_log("FATAL", "Cannot start interaction because AI model failed to initialize.")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{snoop_channel_id}")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")
        return

    start_time = time.time()
    conversation_history = []
    
    stt_queue = queue.Queue()
    stt_stop_event = threading.Event()

    with active_calls_lock:
        stt_result_queue = active_calls.get(channel_id)
    
    if not stt_result_queue:
        broadcast_log("ERROR", f"Could not find result queue for channel {channel_id}.")
        return

    audio_streamer = AsteriskLiveAudioStreamer(recording_file, [stt_queue])
    stt_client = GoogleStreamer(stt_queue)
    audio_streamer.start()

    stt_thread = threading.Thread(target=run_stt_listener, args=(stt_client, stt_result_queue, stt_stop_event), daemon=True)
    stt_thread.start()

    chat_session = GEMINI_MODEL.start_chat(history=[{"role": "user", "parts": [SYSTEM_PROMPT]}])
    stop_words = ["exit", "kapat", "bitir"]

    try:
        while True:
            try:
                user_text = stt_result_queue.get(timeout=3600)
                if user_text == "HANGUP_EVENT":
                    broadcast_log("INFO", "Hangup signal received. Ending interaction loop.")
                    break
                
                socketio.emit('TRANSCRIPT_UPDATE', {'source': 'User', 'text': user_text})
                conversation_history.append(f"User: {user_text}")

                if any(word in user_text.lower() for word in stop_words):
                    broadcast_log("INFO", "Ending call based on user stop word.")
                    break

                response = chat_session.send_message(user_text)
                reply = response.text.strip()
                broadcast_log("INFO", f"AI reply: {reply}")
                socketio.emit('TRANSCRIPT_UPDATE', {'source': 'AI', 'text': reply})
                conversation_history.append(f"AI: {reply}")

                media_id = speak_and_prepare_for_asterisk(reply)
                if media_id:
                    play_response = send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/play", params={"media": f"sound:{media_id}"})
                    # Interruption logic can be added here if needed

                if any(phrase in reply.lower() for phrase in ["anketimiz sona erdi"]):
                    broadcast_log("INFO", "Ending call based on AI closing phrase.")
                    break

            except queue.Empty:
                broadcast_log("INFO", "Timed out waiting for user input. Exiting.")
                break
    except Exception as e:
        broadcast_log("ERROR", f"Error in interaction loop for {channel_id}: {e}")
    finally:
        broadcast_log("INFO", f"Cleaning up resources for channel {channel_id}")
        
        duration = round(time.time() - start_time)
        full_transcript = "\n".join(conversation_history)
        log_call_to_db(caller_id=caller_id, duration=duration, transcript=full_transcript)

        stt_stop_event.set()
        stt_client.close()
        audio_streamer.stop()
        stt_thread.join()
        audio_streamer.join()

        send_ari_request("delete", f"{Config.BASE_URL}/channels/{snoop_channel_id}")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")
        
        if os.path.exists(recording_file):
            try:
                os.remove(recording_file)
                broadcast_log("DEBUG", f"Deleted recording file: {recording_file}")
            except OSError as e:
                broadcast_log("ERROR", f"Error deleting recording file: {e}")

# ==============================================================================
# 6. ASTERISK REAL-TIME EVENT HANDLER (ARI WebSocket)
# ==============================================================================

def ari_event_handler():
    """Handles incoming events from the Asterisk ARI WebSocket."""
    ws_url = Config.WEBSOCKET_URL
    
    def on_message(ws, message):
        event = json.loads(message)
        event_type = event.get('type')
        # Only log the important events to prevent noise
        if event_type in ['StasisStart', 'StasisEnd']:
            broadcast_log("EVENT", f"ARI Event: {event_type} for channel {event.get('channel', {}).get('id')}")

        if event_type == 'StasisStart':
            # ... (StasisStart logic remains the same) ...
            channel = event['channel']
            channel_id = channel['id']
            caller_id = channel['caller']['number']
            
            if 'snoop' in channel.get('name', '').lower():
                broadcast_log("DEBUG", f"Ignoring StasisStart for snoop channel {channel_id}")
                return

            broadcast_log("INFO", f"Channel {channel_id} from {caller_id} entered Stasis.")
            socketio.emit('CALL_START', {'callerId': caller_id, 'callId': channel_id})

            with active_calls_lock:
                active_calls[channel_id] = queue.Queue()

            send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/answer")
            
            snoop_params = {"app": Config.ARI_APP, "spy": "in"}
            snoop_response = send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/snoop", params=snoop_params)
            
            if not snoop_response:
                send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")
                return

            snoop_channel_id = snoop_response.json()['id']
            broadcast_log("INFO", f"Snoop channel {snoop_channel_id} created.")

            recording_name = f"live_rec_{channel_id}"
            send_ari_request("post", f"{Config.BASE_URL}/channels/{snoop_channel_id}/record", params={"name": recording_name, "format": "sln16", "ifExists": "overwrite"})
            broadcast_log("INFO", f"Recording started on snoop channel {snoop_channel_id}.")

            slin_path = os.path.join(Config.LIVE_RECORDING_PATH, f"{recording_name}.sln16")
            threading.Thread(target=interact_with_user, args=(channel_id, snoop_channel_id, slin_path, caller_id), daemon=True).start()


        elif event_type == 'StasisEnd':
            channel_id = event['channel']['id']
            broadcast_log("INFO", f"Call {channel_id} hung up.")
            
            # MODIFICATION: Emit CALL_END with the ID of the channel that ended.
            # This is the crucial change that tells the frontend WHICH call has ended.
            socketio.emit('CALL_END', {'callId': channel_id})
            
            with active_calls_lock:
                queue_to_signal = active_calls.pop(channel_id, None)
            if queue_to_signal:
                queue_to_signal.put("HANGUP_EVENT")

    def on_error(ws, error):
        broadcast_log("ERROR", f"ARI WebSocket Error: {error}")

    def on_close(ws, close_status_code, close_msg):
        broadcast_log("INFO", "ARI WebSocket closed. Reconnecting...")
        time.sleep(5)
        start_ari_connection() # Reconnect

    def on_open(ws):
        broadcast_log("INFO", "✅ WebSocket connected to ARI")

    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()

def start_ari_connection():
    """Starts the ARI WebSocket connection in a background thread."""
    broadcast_log("INFO", "Starting ARI WebSocket listener...")
    ari_thread = threading.Thread(target=ari_event_handler, daemon=True)
    ari_thread.start()

# ==============================================================================
# 7. FLASK ROUTES AND MAIN EXECUTION
# ==============================================================================

@app.route('/')
def index():
    return "AI Voice Agent Backend is running."

@app.route('/api/call', methods=['POST'])
def make_call():
    """API endpoint for the UI to initiate an outbound call."""
    data = request.json
    phone_number = data.get('phoneNumber')
    if not phone_number:
        return jsonify({'error': 'phoneNumber is required'}), 400

    broadcast_log("INFO", f"Received API request to call {phone_number}")
    
    endpoint = f"PJSIP/{phone_number}"
    
    call_data = {
        "endpoint": endpoint,
        "extension": "s",
        "context": Config.DIAL_CONTEXT,
        "priority": "1",
        "app": Config.ARI_APP,
        "callerId": Config.CALLER_ID
    }
    response = send_ari_request("post", f'{Config.BASE_URL}/channels', data=call_data)
    
    # MODIFICATION: Return the actual channelId from ARI to the frontend.
    if response:
        channel_info = response.json()
        return jsonify({
            'message': f'Call initiated to {phone_number}',
            'channelId': channel_info.get('id')
        }), 200
    else:
        return jsonify({'error': 'Failed to initiate call via ARI'}), 500

# --- API Endpoint for Dashboard Statistics ---
@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Calculates and returns key statistics from the database."""
    if not supabase_client:
        return jsonify({"error": "Database client not initialized"}), 500

    try:
        # 1. Get total calls today
        today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_response = supabase_client.from_('call_logs').select('*', count='exact').gte('created_at', today_start).execute()
        total_calls_today = today_response.count

        # 2. Get total calls this month
        month_start = datetime.now(timezone.utc).replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_response = supabase_client.from_('call_logs').select('*', count='exact').gte('created_at', month_start).execute()
        total_calls_month = month_response.count

        # 3. Calculate average duration
        all_calls_response = supabase_client.from_('call_logs').select('call_duration_seconds').execute()
        durations = [c['call_duration_seconds'] for c in all_calls_response.data if c['call_duration_seconds'] is not None]
        
        if durations:
            avg_seconds = sum(durations) / len(durations)
            avg_minutes = int(avg_seconds // 60)
            avg_remainder_seconds = int(avg_seconds % 60)
            average_duration = f"{avg_minutes}:{avg_remainder_seconds:02d}"
        else:
            average_duration = "0:00"

        stats = {
            "totalCallsToday": total_calls_today,
            "totalCallsMonth": total_calls_month,
            "averageDuration": average_duration
        }
        return jsonify(stats), 200

    except Exception as e:
        broadcast_log("ERROR", f"Failed to fetch stats from database: {e}")
        return jsonify({"error": "Failed to fetch stats"}), 500


@socketio.on('connect')
def handle_connect():
    broadcast_log("INFO", f"UI client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    broadcast_log("INFO", f"UI client disconnected: {request.sid}")

if __name__ == "__main__":
    if not all([Config.GEMINI_API_KEY, Config.SUPABASE_URL, Config.SUPABASE_KEY]):
        logging.fatal("FATAL: Missing one or more required environment variables (GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY).")
    else:
        start_ari_connection()
        broadcast_log("INFO", "Starting Flask-SocketIO server...")
        socketio.run(app, host='0.0.0.0', port=8000, allow_unsafe_werkzeug=True)
