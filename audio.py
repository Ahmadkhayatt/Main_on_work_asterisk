# audio.py
import threading
import queue
import time
import os
import webrtcvad
from google.cloud import speech
import config

class AsteriskLiveAudioStreamer(threading.Thread):
    """Tails a live recording file and pushes audio chunks into consumer queues."""
    def __init__(self, recording_path, consumer_queues):
        super().__init__()
        self.recording_path = recording_path
        self.consumer_queues = consumer_queues
        self._stop_event = threading.Event()
        self.chunk_size = int(config.SAMPLE_RATE * 30 / 1000) * 2

    def run(self):
        timeout_seconds = 5
        start_time = time.time()
        while not os.path.exists(self.recording_path) and not self._stop_event.is_set():
            if time.time() - start_time > timeout_seconds:
                print(f"❌ FATAL: Timed out waiting for recording file: {self.recording_path}")
                return
            time.sleep(0.1)

        print(f"✅ File found! Starting to stream audio from {self.recording_path}")
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
            print(f"❌ Error during audio streaming from file: {e}")
        finally:
            print(f"⏹️ Audio streamer for {self.recording_path} stopped.")

    def stop(self):
        self._stop_event.set()


class GoogleStreamer:
    """Manages Google's Streaming Speech-to-Text from an audio queue."""
    def __init__(self, audio_queue, language_code=config.LANGUAGE_CODE):
        self.audio_queue = audio_queue
        self.client = speech.SpeechClient()
        # 👇 --- SOLUTION: Rename the local variable ---
        recognition_config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=config.SAMPLE_RATE, # Now correctly refers to the imported module
            language_code=language_code,
            enable_automatic_punctuation=False,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=recognition_config, # Use the new variable name here too
            interim_results=False
        )
        self._closed = threading.Event()

    def _generator(self):
        while not self._closed.is_set():
            try:
                chunk = self.audio_queue.get(timeout=0.5)
                yield speech.StreamingRecognizeRequest(audio_content=chunk)
            except queue.Empty:
                continue

    def listen(self, silence_timeout=7):
        print("🎤 Listening for user input via Google STT...")
        responses = self.client.streaming_recognize(
            config=self.streaming_config, requests=self._generator()
        )
        last_speech_time = time.time()
        for response in responses:
            if not response.results:
                if time.time() - last_speech_time > silence_timeout:
                    print("...Silence timeout reached.")
                    break
                continue
            result = response.results[0]
            if result.is_final:
                transcript = result.alternatives[0].transcript.strip()
                print(f"🗣️  Final Transcript: {transcript}")
                return transcript
        return None

    def close(self):
        self._closed.set()