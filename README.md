🧙‍♂️✨ The Arcane Dialer: Asterisk + AI Voice Bot Quest

    "In the kingdom of SIP, amidst the stars of VoIP, rose a conjurer of speech and spell... forged in Asterisk and whispered by AI."

📖 Table of Spells (Contents)

    🌀 About the Project

    🛠️ Magical Architecture

    🧩 Summoned Technologies

    🗺️ Installation Ritual

    🧪 How to Summon the Bot

    🧞 Voice of the Oracle (AI Model)

    🛡️ Known Curses & Fixes

    ⚔️ Contribution Rites

    🏰 The Guild of Contributors

    📜 License of the Ancients

🌀 About the Project

The Arcane Dialer is a mystic fusion of Asterisk, Python, and Generative AI.

It listens. It speaks. It thinks.

You can:

    Whisper secrets (STT),

    Hear prophecies (TTS),

    Speak over the spirit (interrupt),

    And command the bot with voice spells (LLM).

This is not just an IVR.
This... is a talking construct of old-world telephony and new-age sorcery.
🛠️ Magical Architecture

User (Hero) ☎️
   ↓
Asterisk (The Summoner) 📞
   ↓
Python ARI (The Ritual Keeper) 🐍
   ↓
Gemini / DeepSeek / LLaMA (The Oracle) 🧠
   ↓
Google / ElevenLabs TTS (The Voice of Ancients) 🔊
   ↓
Whisper / Google STT (The Listener in the Void) 👂

🧩 Summoned Technologies

    🧙 Asterisk – The telephonic sorcerer

    🧪 ARI (Asterisk REST Interface) – Channeling spells

    🐍 Python – The scripting staff

    🗣️ Google Speech / Whisper – The Listener

    🔊 Google TTS / ElevenLabs – The Voice

    🧠 LLM (Gemini / Open Source) – The Oracle

🗺️ Installation Ritual

# Summon the virtual realm
python3 -m venv voice-dungeon
source voice-dungeon/bin/activate

# Install the magical scrolls
pip install -r requirements.txt

# Whisper your secrets to .env
cp .env.example .env
vim .env

    ⚠️ Don't forget to configure your SIP familiars and Asterisk incantations (extensions.conf, ari.conf, http.conf...)

🧪 How to Summon the Bot

python summon_bot.py

When the phone rings... the ritual begins.
🧞 Voice of the Oracle

Choose your oracle:

    🧠 Gemini: Fast and affordable AI priest

    🐲 DeepSeek / LLaMA: Host your own mind-wizard

    🔮 Gemini Pro (Cloud): Vast, wise, but distant

You may pass a prompt_template of your own to change the bot’s personality. Make it a pirate. Or a polite barista. Or a suspicious cat. It listens.
🛡️ Known Curses & Fixes
Curse	Spell to Break
“ARI connection refused”	Asterisk HTTP/ARI not configured
“Bot doesn't hear my voice”	Check STT config, mic, VAD
“Bot speaks over itself”	Enable interrupt detection (auto VAD or DTMF)
“Voice not majestic enough”	Try ElevenLabs TTS with stability=0.1
⚔️ Contribution Rites

All wizards welcome!
Pull Requests must be reviewed by at least one necromancer or two potion brewers.
Use black to format Python spells.
🏰 The Guild of Contributors

    🧙‍♂️ The Founder: @yourusername

    👻 The Listener: Whisper

    🎤 The Speaker: Google / ElevenLabs

    🧠 The Oracle Whisperer: Gemini

📜 License of the Ancients

Released under the MIT License.
Use freely, but don't anger the telephony spirits.
🔮 Final Words

    "This is not the end, but the beginning of a new voice-driven realm.
    May your calls be clear, your bots be kind, and your logs forever detailed."
