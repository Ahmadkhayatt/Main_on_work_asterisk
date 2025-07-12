# services.py
import os
import pwd
import grp
import wave
import google.generativeai as genai
from google.cloud import texttospeech
import config

def initialize_ai_model():
    """
    Initializes the GenerativeAI model and loads the system prompt.
    Returns the model and prompt text, or (None, None) on failure.
    """
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        )
        # Load your prompt file
        with open('PromptNew2.txt', "r", encoding="utf-8") as file:
            prompt_text = file.read()
            
        print("✅ Gemini Model and prompt loaded successfully.")
        return model, prompt_text

    except Exception as e:
        print(f"FATAL: Could not initialize GenerativeAI. Check API Key and prompt file. Error: {e}")
        return None, None

# === THE DEFINITIVE TTS FUNCTION ===
def speak_and_prepare_for_asterisk(text, lang=config.LANGUAGE_CODE):
    """
    Generates speech, saves it as a correctly formatted WAV file, and sets permissions for Asterisk.
    """
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=lang, ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=8000
        )
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        with wave.open(config.TTS_SOUND_FILE_PATH, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(8000)
            wf.writeframes(response.audio_content)

        # Set correct file ownership for Asterisk
        a_uid = pwd.getpwnam("asterisk").pw_uid
        a_gid = grp.getgrnam("asterisk").gr_gid
        os.chown(config.TTS_SOUND_FILE_PATH, a_uid, a_gid)
        os.chmod(config.TTS_SOUND_FILE_PATH, 0o644)

        print(f"  🔊 WAV saved and permissions set: {config.TTS_SOUND_FILE_PATH}")
        return config.TTS_SOUND_ID

    except Exception as e:
        print(f"❌ TTS+WAV error: {e}")
        return None