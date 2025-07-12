
# 🧙‍♂️ Arcane Dialer: Summon the AI Voice Bot with Asterisk

> _"In the ancient realms of SIP and sockets, a wizard conjured a voice from the void... one that listens, speaks, and reasons."_  
> _Powered by Asterisk, Python, and the minds of machines._

---

## 🧭 Project Overview

**Arcane Dialer** is a magical voice bot system that merges the mystical powers of:

- 🌀 **Asterisk** for telephonic summoning
- 🧠 **Large Language Models (LLMs)** for reasoning
- 🗣️ **Speech-to-Text (STT)** to listen
- 🔊 **Text-to-Speech (TTS)** to speak
- 🐍 **Python** as the scripting grimoire

Use it to create:
- Smart IVR bots
- Survey wizards
- Conversational AI familiars

---

## 🧪 Features

- 📞 Accept SIP calls via Asterisk
- 🧞 Speak with the user via TTS
- 👂 Listen to responses via STT
- 🧠 Ask an LLM what to say next
- 🧹 Automatically handle silence, interruptions, or confusion
- 💾 Optional: store transcripts & audio logs

---

## 🧩 Tech Stack

| Magic Component    | Description |
|--------------------|-------------|
| 🧙 Asterisk         | Handles SIP calls |
| 🧪 ARI              | Asterisk REST Interface |
| 🐍 Python           | Core bot logic |
| 🧞 Gemini/DeepSeek  | LLM used for replies |
| 🔊 Google TTS       | Text-to-speech voice |
| 🧏 Whisper / Google STT | Speech-to-text recognition |

---

## ⚙️ Setup & Installation

> Tested on **Ubuntu 22.04** & **Python 3.10+**

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/arcane-dialer.git
cd arcane-dialer

# Step 2: Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Configure your environment
cp .env.example .env
nano .env  # Fill in API keys and Asterisk configs

# Step 5: Start your bot
python summon_bot.py
