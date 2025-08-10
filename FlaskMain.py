# ==============================================================================
# AI VOICE AGENT - FLASK BACKEND (with hangup reasons & robust CALL_END)
# ==============================================================================

import os
import re
import threading
import queue
import time
import json
import logging
import pwd
import grp
import wave
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
# 1) INIT & CONFIG
# ==============================================================================

load_dotenv()

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


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
    ARI_USER = os.getenv("ARI_USER", "ai")
    ARI_PASSWORD = os.getenv("ARI_PASSWORD", "ai_secret")
    ARI_HOST = os.getenv("ARI_HOST", "localhost")
    ARI_PORT = int(os.getenv("ARI_PORT", "8088"))
    ARI_APP = os.getenv("ARI_APP", "aiagent")
    BASE_URL = f"http://{ARI_HOST}:{ARI_PORT}/ari"
    WEBSOCKET_URL = f"ws://{ARI_HOST}:{ARI_PORT}/ari/events?app={ARI_APP}&api_key={ARI_USER}:{ARI_PASSWORD}"

    # Paths & Naming
    LIVE_RECORDING_PATH = "/var/spool/asterisk/recording"
    TTS_SOUND_FILE_PATH = "/var/lib/asterisk/sounds/en/ai_agent_response.wav"
    TTS_SOUND_ID = "ai_agent_response"

    # Call Behavior
    OUTBOUND_ENDPOINT = os.getenv("OUTBOUND_ENDPOINT", "PJSIP/7001")
    DIAL_CONTEXT = os.getenv("DIAL_CONTEXT", "ai-survey")
    CALLER_ID = os.getenv("CALLER_ID", "AI Bot")


# For Google TTS/STT
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_APPLICATION_CREDENTIALS

# ==============================================================================
# 2) GLOBALS
# ==============================================================================

# Active call -> queue for STT results
active_calls = {}
active_calls_lock = threading.Lock()

# Track hangup cause per channel & avoid duplicate CALL_END emits
CALL_END_META = {}       # channel_id -> {"cause": int|None, "cause_txt": str|None}
CALL_END_EMITTED = set() # channel_ids that we already emitted CALL_END for

# Supabase
try:
    supabase_client = supabase.create_client(Config.SUPABASE_URL, Config.SUPABASE_KEY)
    logging.info("✅ Supabase client initialized successfully.")
except Exception as e:
    supabase_client = None
    logging.error(f"❌ Failed to initialize Supabase client: {e}")

# Gemini
try:
    genai.configure(api_key=Config.GEMINI_API_KEY)
    GEMINI_MODEL = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config={"temperature": 1, "top_p": 0.95, "top_k": 40, "max_output_tokens": 8192},
    )
    with open("Prompt.txt", "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
    logging.info("✅ Gemini Model and prompt loaded successfully.")
except Exception as e:
    GEMINI_MODEL = None
    SYSTEM_PROMPT = None
    logging.fatal(f"Could not initialize GenerativeAI. Check API key and Prompt.txt. Error: {e}")

SURVEY_TABLE = "survey_results"  # change if your table differs

# ==============================================================================
# 3) HELPERS
# ==============================================================================

def broadcast_log(log_type: str, message: str):
    """Log & emit to UI."""
    log_entry = f"[{log_type}] {message}"
    if log_type in ("ERROR", "FATAL"):
        logging.error(log_entry)
    else:
        logging.info(log_entry)
    try:
        socketio.emit("LOG_UPDATE", {"type": log_type, "message": message})
    except Exception:
        pass


def send_ari_request(method: str, url: str, **kwargs):
    """Authenticated request to ARI, with basic error logs."""
    try:
        resp = requests.request(
            method, url, auth=HTTPBasicAuth(Config.ARI_USER, Config.ARI_PASSWORD), timeout=5, **kwargs
        )
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        broadcast_log("ERROR", f"ARI {method} {url} failed: {e}")
        if getattr(e, "response", None) is not None:
            try:
                broadcast_log("ERROR", f"ARI Response: {e.response.text}")
            except Exception:
                pass
        return None


def resolve_caller_id_from_event(event: dict) -> str:
    """
    Best-effort caller id for outbound calls:
      1) channel.caller.number
      2) event.args[0] (we pass dialed number via appArgs)
      3) CALLERID(num)
      4) parse from channel.name (e.g., PJSIP/7001-xxx -> 7001)
      5) 'Unknown'
    """
    channel = event.get("channel", {}) or {}
    channel_id = channel.get("id")
    cid = (channel.get("caller") or {}).get("number") or ""

    if not cid:
        args = event.get("args") or []
        cid = (args[0] if args else "") or ""

    if not cid and channel_id:
        var_resp = send_ari_request(
            "get", f"{Config.BASE_URL}/channels/{channel_id}/variable", params={"variable": "CALLERID(num)"}
        )
        try:
            cid = (var_resp.json().get("value") if var_resp else "") or ""
        except Exception:
            cid = ""

    if not cid:
        name = channel.get("name") or ""
        m = re.search(r"/([^-/]+)", name)
        if m:
            cid = m.group(1)

    return cid or "Unknown"


def _map_cause_to_reason(cause: int | None, cause_txt: str | None) -> str:
    """Map Q.850 cause/cause_txt to a friendly reason."""
    ct = (cause_txt or "").lower()
    if cause in (16,) or "normal" in ct:
        return "completed"
    if cause in (17,) or "busy" in ct:
        return "busy"
    if cause in (21,) or "rejected" in ct or "declined" in ct:
        return "rejected"
    if cause in (19,) or "no answer" in ct:
        return "no-answer"
    if cause in (34, 41, 42) or "congestion" in ct or "network" in ct:
        return "failed"
    return "failed"


def _emit_call_end(channel_id: str):
    """Emit CALL_END once with best available reason/cause."""
    if not channel_id or channel_id in CALL_END_EMITTED:
        return
    meta = CALL_END_META.get(channel_id) or {}
    cause = meta.get("cause")
    cause_txt = meta.get("cause_txt")
    reason = _map_cause_to_reason(cause, cause_txt)
    socketio.emit("CALL_END", {"callId": channel_id, "cause": cause, "causeText": cause_txt, "reason": reason})
    CALL_END_EMITTED.add(channel_id)
    broadcast_log("EVENT", f"CALL_END {channel_id} reason={reason} cause={cause} text={cause_txt}")


def log_call_to_db(caller_id: str, duration: int, transcript: str):
    """Insert a call into call_logs."""
    if not supabase_client:
        broadcast_log("ERROR", "Supabase client not available. Cannot log call.")
        return None
    try:
        cid = caller_id if caller_id and caller_id != "EMPTY" else None
        resp = supabase_client.from_("call_logs").insert(
            {"caller_id": cid, "call_duration_seconds": duration, "transcript": transcript}
        ).execute()
        broadcast_log("INFO", "Call log saved to Supabase.")
        return resp.data
    except Exception as e:
        broadcast_log("ERROR", f"DB logging failed: {e}")
        return None

# ==============================================================================
# 4) AUDIO / STT / TTS
# ==============================================================================

class AsteriskLiveAudioStreamer(threading.Thread):
    """Tail the SLIN16 recording file and push audio chunks to queues."""

    def __init__(self, recording_path: str, consumer_queues: list[queue.Queue]):
        super().__init__()
        self.recording_path = recording_path
        self.consumer_queues = consumer_queues
        self._stop_event = threading.Event()
        self.chunk_size = int(Config.SAMPLE_RATE * 30 / 1000) * 2  # 30ms 16-bit mono

    def run(self):
        timeout_seconds = 5
        start_time = time.time()
        while not os.path.exists(self.recording_path) and not self._stop_event.is_set():
            if time.time() - start_time > timeout_seconds:
                broadcast_log("FATAL", f"Timed out waiting for recording: {self.recording_path}")
                return
            time.sleep(0.1)

        broadcast_log("INFO", f"Streaming audio from {self.recording_path}")
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
            broadcast_log("ERROR", f"Audio streaming error: {e}")
        finally:
            broadcast_log("DEBUG", f"Audio streamer stopped for {self.recording_path}")

    def stop(self):
        self._stop_event.set()


class GoogleStreamer:
    """Google Streaming STT over an audio queue."""

    def __init__(self, audio_queue: queue.Queue):
        self.audio_queue = audio_queue
        self.client = speech.SpeechClient()
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=Config.SAMPLE_RATE,
            language_code=Config.LANGUAGE_CODE,
            enable_automatic_punctuation=False,
            model="latest_long",
        )
        self.streaming_config = speech.StreamingRecognitionConfig(config=recognition_config, interim_results=False)
        self._closed = threading.Event()

    def _generator(self):
        while not self._closed.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                if chunk is None:
                    return
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def listen(self, silence_timeout=7):
        broadcast_log("DEBUG", "Listening via Google STT…")
        responses = self.client.streaming_recognize(config=self.streaming_config, requests=self._generator())
        last_speech_time = time.time()
        for response in responses:
            if not response.results:
                if time.time() - last_speech_time > silence_timeout:
                    broadcast_log("DEBUG", "Silence timeout.")
                    break
                continue
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                broadcast_log("INFO", f"STT: '{transcript}'")
                return transcript
        return None

    def close(self):
        self._closed.set()
        self.audio_queue.put(None)


def speak_and_prepare_for_asterisk(text: str):
    """TTS to WAV with correct owner/permissions for Asterisk playback."""
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=Config.LANGUAGE_CODE, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=8000)
        response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)

        os.makedirs(os.path.dirname(Config.TTS_SOUND_FILE_PATH), exist_ok=True)
        with wave.open(Config.TTS_SOUND_FILE_PATH, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            wf.writeframes(response.audio_content)

        try:
            a_uid = pwd.getpwnam("asterisk").pw_uid
            a_gid = grp.getgrnam("asterisk").gr_gid
            os.chown(Config.TTS_SOUND_FILE_PATH, a_uid, a_gid)
            os.chmod(Config.TTS_SOUND_FILE_PATH, 0o644)
        except Exception as e:
            broadcast_log("ERROR", f"File ownership/perm set failed: {e}")

        broadcast_log("DEBUG", f"TTS saved: {Config.TTS_SOUND_FILE_PATH}")
        return Config.TTS_SOUND_ID
    except Exception as e:
        broadcast_log("ERROR", f"TTS generation failed: {e}")
        return None

# ==============================================================================
# 5) CONVERSATION LOOP
# ==============================================================================

def run_stt_listener(stt_client: GoogleStreamer, result_queue: "queue.Queue[str]", stop_event: threading.Event):
    broadcast_log("DEBUG", "[STT Thread] start")
    while not stop_event.is_set():
        transcript = stt_client.listen()
        if transcript:
            result_queue.put(transcript)
    broadcast_log("DEBUG", "[STT Thread] stop")


def interact_with_user(channel_id: str, snoop_channel_id: str, recording_file: str, caller_id: str):
    """Main per-call loop."""
    if not GEMINI_MODEL or not SYSTEM_PROMPT:
        broadcast_log("FATAL", "AI not initialized.")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{snoop_channel_id}")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")
        return

    start_time = time.time()
    history = []

    stt_queue = queue.Queue()
    stt_stop = threading.Event()

    with active_calls_lock:
        stt_result_queue = active_calls.get(channel_id)

    if not stt_result_queue:
        broadcast_log("ERROR", f"No result queue for {channel_id}")
        return

    audio_streamer = AsteriskLiveAudioStreamer(recording_file, [stt_queue])
    stt_client = GoogleStreamer(stt_queue)
    audio_streamer.start()

    stt_thread = threading.Thread(target=run_stt_listener, args=(stt_client, stt_result_queue, stt_stop), daemon=True)
    stt_thread.start()

    chat = GEMINI_MODEL.start_chat(history=[{"role": "user", "parts": [SYSTEM_PROMPT]}])
    stop_words = ["exit", "kapat", "bitir"]

    try:
        while True:
            try:
                user_text = stt_result_queue.get(timeout=3600)
                if user_text == "HANGUP_EVENT":
                    broadcast_log("INFO", "Hangup signal. Ending call loop.")
                    break

                socketio.emit("TRANSCRIPT_UPDATE", {"source": "User", "text": user_text})
                history.append(f"User: {user_text}")

                if any(w in user_text.lower() for w in stop_words):
                    broadcast_log("INFO", "User stop word. Ending call.")
                    break

                response = chat.send_message(user_text)
                reply = (response.text or "").strip()
                socketio.emit("TRANSCRIPT_UPDATE", {"source": "AI", "text": reply})
                history.append(f"AI: {reply}")

                media_id = speak_and_prepare_for_asterisk(reply)
                if media_id:
                    send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/play", params={"media": f"sound:{media_id}"})

                if "anketimiz sona erdi" in reply.lower():
                    broadcast_log("INFO", "AI said to end the survey. Ending call.")
                    break

            except queue.Empty:
                broadcast_log("INFO", "No user input (timeout). Ending call.")
                break
    except Exception as e:
        broadcast_log("ERROR", f"Interaction loop error for {channel_id}: {e}")
    finally:
        duration = int(round(time.time() - start_time))
        transcript = "\n".join(history)
        log_call_to_db(caller_id=caller_id, duration=duration, transcript=transcript)

        stt_stop.set()
        stt_client.close()
        audio_streamer.stop()
        stt_thread.join()
        audio_streamer.join()

        send_ari_request("delete", f"{Config.BASE_URL}/channels/{snoop_channel_id}")
        send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")

        if os.path.exists(recording_file):
            try:
                os.remove(recording_file)
                broadcast_log("DEBUG", f"Deleted {recording_file}")
            except OSError as e:
                broadcast_log("ERROR", f"File delete failed: {e}")

# ==============================================================================
# 6) ARI EVENTS
# ==============================================================================

def ari_event_handler():
    """Listen to ARI WebSocket and handle events."""
    ws_url = Config.WEBSOCKET_URL

    def on_message(ws, message):
        event = json.loads(message)
        event_type = event.get("type")

        if event_type in ("StasisStart", "StasisEnd"):
            broadcast_log("EVENT", f"{event_type} for {event.get('channel', {}).get('id')}")

        # ----- Start of call (ignore snoop) -----
        if event_type == "StasisStart":
            channel = event["channel"]
            channel_id = channel["id"]

            if "snoop" in (channel.get("name") or "").lower():
                broadcast_log("DEBUG", f"Ignoring StasisStart for snoop {channel_id}")
                return

            caller_id = resolve_caller_id_from_event(event)
            broadcast_log("INFO", f"Channel {channel_id} from {caller_id} entered Stasis.")
            socketio.emit("CALL_START", {"callId": channel_id, "callerId": caller_id})

            with active_calls_lock:
                active_calls[channel_id] = queue.Queue()

            # Answer real leg
            send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/answer")

            # Snoop & record
            snoop_params = {"app": Config.ARI_APP, "spy": "in"}
            snoop_response = send_ari_request("post", f"{Config.BASE_URL}/channels/{channel_id}/snoop", params=snoop_params)
            if not snoop_response:
                send_ari_request("delete", f"{Config.BASE_URL}/channels/{channel_id}")
                return

            snoop_id = snoop_response.json()["id"]
            broadcast_log("INFO", f"Snoop channel {snoop_id} created.")

            recording_name = f"live_rec_{channel_id}"
            send_ari_request(
                "post",
                f"{Config.BASE_URL}/channels/{snoop_id}/record",
                params={"name": recording_name, "format": "sln16", "ifExists": "overwrite"},
            )
            broadcast_log("INFO", f"Recording started on snoop channel {snoop_id}.")

            slin_path = os.path.join(Config.LIVE_RECORDING_PATH, f"{recording_name}.sln16")
            threading.Thread(
                target=interact_with_user,
                args=(channel_id, snoop_id, slin_path, caller_id),
                daemon=True,
            ).start()
            return

        # ----- NEW: capture hangup cause early (decline, busy, etc.) -----
        if event_type == "ChannelHangupRequest":
            ch = event.get("channel", {}) or {}
            channel_id = ch.get("id")
            name = (ch.get("name") or "").lower()
            if channel_id and "snoop" not in name:
                cause = event.get("cause")
                cause_txt = event.get("cause_txt")
                CALL_END_META[channel_id] = {"cause": cause, "cause_txt": cause_txt}
                broadcast_log("EVENT", f"HangupRequest for {channel_id}: cause={cause} text={cause_txt}")
                # emit immediately so FE can react without waiting for StasisEnd
                _emit_call_end(channel_id)
            return

        # ----- Stasis end (ignore snoop); ensure cleanup -----
        if event_type == "StasisEnd":
            ch = event.get("channel", {}) or {}
            channel_id = ch.get("id")
            name = (ch.get("name") or "").lower()
            if "snoop" in name:
                broadcast_log("DEBUG", f"Ignoring StasisEnd for snoop {channel_id}")
                return

            broadcast_log("INFO", f"Call {channel_id} hung up (StasisEnd).")
            _emit_call_end(channel_id)  # no-op if already emitted

            with active_calls_lock:
                q_to_signal = active_calls.pop(channel_id, None)
            if q_to_signal:
                q_to_signal.put("HANGUP_EVENT")

            # cleanup for this channel
            CALL_END_META.pop(channel_id, None)
            CALL_END_EMITTED.discard(channel_id)
            return

        # ----- Optional fallback -----
        if event_type == "ChannelDestroyed":
            ch = event.get("channel", {}) or {}
            channel_id = ch.get("id")
            name = (ch.get("name") or "").lower()
            if channel_id and "snoop" not in name:
                broadcast_log("EVENT", f"ChannelDestroyed for {channel_id}")
                _emit_call_end(channel_id)
                CALL_END_META.pop(channel_id, None)
                CALL_END_EMITTED.discard(channel_id)
            return

    def on_error(ws, error):
        broadcast_log("ERROR", f"ARI WebSocket error: {error}")

    def on_close(ws, close_status_code, close_msg):
        broadcast_log("INFO", "ARI WebSocket closed. Reconnecting in 5s…")
        time.sleep(5)
        start_ari_connection()

    def on_open(ws):
        broadcast_log("INFO", "✅ Connected to ARI WebSocket")

    ws = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()


def start_ari_connection():
    broadcast_log("INFO", "Starting ARI WebSocket listener…")
    threading.Thread(target=ari_event_handler, daemon=True).start()

# ==============================================================================
# 7) FLASK ROUTES
# ==============================================================================

@app.route("/")
def index():
    return "AI Voice Agent Backend is running."


@app.route("/api/call", methods=["POST"])
def make_call():
    """Originate an outbound call."""
    data = request.json or {}
    phone_number = data.get("phoneNumber")
    if not phone_number:
        return jsonify({"error": "phoneNumber is required"}), 400

    broadcast_log("INFO", f"API request to call {phone_number}")

    endpoint = f"PJSIP/{phone_number}"
    call_data = {
        "endpoint": endpoint,
        "extension": "s",
        "context": Config.DIAL_CONTEXT,
        "priority": "1",
        "app": Config.ARI_APP,
        "appArgs": str(phone_number),  # pass dialed number → Stasis args
        "callerId": Config.CALLER_ID,  # name; numeric often blank in ARI originates
    }
    response = send_ari_request("post", f"{Config.BASE_URL}/channels", data=call_data)
    if response:
        info = response.json()
        return jsonify({"message": f"Call initiated to {phone_number}", "channelId": info.get("id")}), 200
    return jsonify({"error": "Failed to initiate call via ARI"}), 500


def _parse_iso(val: str | None):
    if not val:
        return None
    try:
        s = val
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if not dt.tzinfo:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


@app.route("/api/reports", methods=["GET"])
def get_reports():
    """
    Returns:
      { callHistory: [...], surveyResponses: [...] }
    Optional query params: from, to (ISO); page (1), pageSize (100, max 500)
    """
    if not supabase_client:
        return jsonify({"error": "Database client not initialized"}), 500
    try:
        from_param = request.args.get("from")
        to_param = request.args.get("to")
        page = max(int(request.args.get("page", 1)), 1)
        page_size = min(max(int(request.args.get("pageSize", 100)), 1), 500)

        dt_from = _parse_iso(from_param)
        dt_to = _parse_iso(to_param)

        q = supabase_client.from_("call_logs").select("*", count="exact")
        if dt_from:
            q = q.gte("created_at", dt_from.isoformat())
        if dt_to:
            q = q.lte("created_at", dt_to.isoformat())
        q = q.order("created_at", desc=True)

        offset = (page - 1) * page_size
        q = q.range(offset, offset + page_size - 1)

        calls_resp = q.execute()
        calls = calls_resp.data or []
        call_ids = [c["id"] for c in calls]

        surveys = []
        if call_ids:
            surveys_resp = (
                supabase_client.from_(SURVEY_TABLE)
                .select("*")
                .in_("call_log_id", call_ids)
                .order("created_at", desc=True)
                .execute()
            )
            surveys = surveys_resp.data or []

        return jsonify({"callHistory": calls, "surveyResponses": surveys}), 200
    except Exception as e:
        broadcast_log("ERROR", f"Failed to fetch reports: {e}")
        return jsonify({"error": "Failed to fetch reports"}), 500


@app.route("/api/stats", methods=["GET"])
def get_stats():
    """Simple stats: today, this month, avg duration."""
    if not supabase_client:
        return jsonify({"error": "Database client not initialized"}), 500
    try:
        now_utc = datetime.now(timezone.utc)
        today_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        month_start = now_utc.replace(day=1, hour=0, minute=0, second=0, microsecond=0).isoformat()

        today_resp = supabase_client.from_("call_logs").select("id", count="exact").gte("created_at", today_start).execute()
        month_resp = supabase_client.from_("call_logs").select("id", count="exact").gte("created_at", month_start).execute()

        total_calls_today = today_resp.count or 0
        total_calls_month = month_resp.count or 0

        all_calls = supabase_client.from_("call_logs").select("call_duration_seconds").execute()
        durations = [c["call_duration_seconds"] for c in (all_calls.data or []) if c.get("call_duration_seconds") is not None]
        if durations:
            avg = sum(durations) / len(durations)
            average_duration = f"{int(avg // 60)}:{int(avg % 60):02d}"
        else:
            average_duration = "0:00"

        return jsonify(
            {"totalCallsToday": total_calls_today, "totalCallsMonth": total_calls_month, "averageDuration": average_duration}
        ), 200
    except Exception as e:
        broadcast_log("ERROR", f"Failed to fetch stats: {e}")
        return jsonify({"error": "Failed to fetch stats"}), 500

# ==============================================================================
# 8) SOCKET.IO LIFECYCLE
# ==============================================================================

@socketio.on("connect")
def handle_connect():
    broadcast_log("INFO", f"UI connected: {request.sid}")

@socketio.on("disconnect")
def handle_disconnect():
    broadcast_log("INFO", f"UI disconnected: {request.sid}")

# ==============================================================================
# 9) MAIN
# ==============================================================================

if __name__ == "__main__":
    if not all([Config.GEMINI_API_KEY, Config.SUPABASE_URL, Config.SUPABASE_KEY]):
        logging.fatal("FATAL: Missing required env vars (GEMINI_API_KEY, SUPABASE_URL, SUPABASE_KEY).")
    else:
        start_ari_connection()
        broadcast_log("INFO", "Starting Flask-SocketIO server…")
        socketio.run(app, host="0.0.0.0", port=8000, allow_unsafe_werkzeug=True)
