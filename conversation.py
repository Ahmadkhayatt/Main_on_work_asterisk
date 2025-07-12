# conversation.py
import threading
import queue
import requests
import os
from requests.auth import HTTPBasicAuth
import time 
from database import log_call
import config
# 👇 --- MODIFIED IMPORT ---
from services import initialize_ai_model, speak_and_prepare_for_asterisk
from audio import AsteriskLiveAudioStreamer, GoogleStreamer

# 👇 --- INITIALIZE SERVICES AND CHECK FOR SUCCESS ---
model, prompt = initialize_ai_model()

def run_stt_listener(stt_client, result_queue, stop_event):
    """Runs in a thread to listen for utterances and put them on a queue."""
    print("  [STT Thread] Starting to listen...")
    while not stop_event.is_set():
        transcript = stt_client.listen()
        if transcript:
            print(f"  [STT Thread] Detected transcript: '{transcript}'. Placing in queue.")
            result_queue.put(transcript)
    print("  [STT Thread] Stopped.")


def interact_with_user(channel_id, snoop_channel_id, recording_file, caller_id):
    """Handles the main conversation flow with STT-based interruption."""
    # 👇 --- ADDED CHECK ---
    if not model or not prompt:
        print("❌ Cannot start interaction because AI model failed to initialize.")
        # Clean up and exit the thread immediately
        requests.delete(f"{config.BASE_URL}/channels/{snoop_channel_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
        requests.delete(f"{config.BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
        return
        # --- New variables to store conversation data ---
    start_time = time.time()
    conversation_history = []
    print(f"🚀 Interaction starting for channel {channel_id} from {caller_id}.")
    
    stt_queue = queue.Queue()
    stt_result_queue = queue.Queue()
    stt_stop_event = threading.Event()

    audio_streamer = AsteriskLiveAudioStreamer(recording_file, [stt_queue])
    stt_client = GoogleStreamer(stt_queue)
    
    audio_streamer.start()

    stt_thread = threading.Thread(
        target=run_stt_listener,
        args=(stt_client, stt_result_queue, stt_stop_event)
    )
    stt_thread.start()

    chat_session = model.start_chat(history=[{"role": "user", "parts": [prompt]}])
    stop_words = ["exit", "kapat", "bitir"]

    try:
        while True:
            try:
                user_text = stt_result_queue.get(timeout=3600)
                print(f"📝 User said: {user_text}")
                conversation_history.append(f"User: {user_text}") # <--- Save user's text
            except queue.Empty:
                print(" timed out waiting for user input. Exiting.")
                break

            print(f"📝 User said: {user_text}")
            if any(word in user_text.lower() for word in stop_words):
                print("📴 Ending call based on stop word.")
                break

            response = chat_session.send_message(user_text)
            reply = response.text.strip()
            print(f"🤖 Gemini: {reply}")
            conversation_history.append(f"AI: {reply}") # <--- Save AI's reply
            print("reply added to database OwO")

            media_id = speak_and_prepare_for_asterisk(reply)
            if not media_id:
                continue

            play_response = requests.post(
                f"{config.BASE_URL}/channels/{channel_id}/play",
                params={"media": f"sound:{media_id}"},
                auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD)
            )

            if 200 <= play_response.status_code < 300:
                playback_id = play_response.json().get('id')
                print(f"  [INFO] Playback {playback_id} started. Monitoring for interruption...")

                interrupted = False
                while True:
                    try:
                        interrupt_text = stt_result_queue.get(timeout=0.1)
                        print(f"  [INTERRUPT] New STT result received. Stopping playback.")
                        requests.delete(f"{config.BASE_URL}/playbacks/{playback_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
                        stt_result_queue.put(interrupt_text) # Put text back for next loop
                        interrupted = True
                        break
                    except queue.Empty:
                        pass # No interruption

                    status_resp = requests.get(f"{config.BASE_URL}/playbacks/{playback_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
                    if status_resp.status_code == 404:
                        print("  [INFO] Playback finished naturally.")
                        break
                
                if interrupted:
                    continue

            if any(phrase in reply.lower() for phrase in ["anketimiz sona erdi"]):
                print("📴 Ending call based on Gemini response.")
                break

    except Exception as e:
        print(f"❌ An error occurred in the interaction loop: {e}")
    finally:
        print(f"🧹 Cleaning up resources for channel {channel_id}")
               # 1. Calculate duration
        end_time = time.time()
        duration = round(end_time - start_time)
        
        # 2. Format the transcript
        full_transcript = "\n".join(conversation_history)
        
        # 3. Call our database function
        log_call(caller_id=caller_id, duration=duration, transcript=full_transcript)

        stt_stop_event.set()
        stt_client.close()
        audio_streamer.stop()
        stt_thread.join()
        audio_streamer.join()

        requests.delete(f"{config.BASE_URL}/channels/{snoop_channel_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
        requests.delete(f"{config.BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))

        if os.path.exists(recording_file):
            try:
                os.remove(recording_file)
                print(f"🗑️ Deleted recording file: {recording_file}")
            except OSError as e:
                print(f"   Error deleting recording file: {e}")