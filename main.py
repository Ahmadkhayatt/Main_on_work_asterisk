# main.py
import websocket
import json
import requests
import threading
import os
from requests.auth import HTTPBasicAuth

import config
from conversation import interact_with_user

def originate_call():
    """Initiates an outbound call to the endpoint specified in config."""
    print("📞 Attempting to originate an outbound call...")
    data = {
        "endpoint": config.OUTBOUND_ENDPOINT,
        "extension": "s",
        "context": config.DIAL_CONTEXT,
        "priority": "1",
        "app": config.ARI_APP,
        "callerId": config.CALLER_ID
    }
    try:
        response = requests.post(f'{config.BASE_URL}/channels', data=data, auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
        response.raise_for_status()
        print("✅ Outbound call successfully initiated.")
    except requests.RequestException as e:
        print(f"❌ Call failed: {e}")
        if e.response is not None:
            print(f"Response body: {e.response.text}")

def on_message(ws, message):
    event = json.loads(message)
    event_type = event.get('type')
    print(f"📡 Event: {event_type}")

    if event_type == 'StasisStart':
        channel_id = event['channel']['id']
        caller_id = event['channel']['caller']['number']
        
        print(f"Channel {channel_id} from {caller_id} entered Stasis.")        
        
        requests.post(
            f"{config.BASE_URL}/channels/{channel_id}/answer",
            auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD)
        )

        try:
            snoop_response = requests.post(
                f"{config.BASE_URL}/channels/{channel_id}/snoop",
                params={"app": config.ARI_APP, "spy": "in"},
                auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD)
            )
            snoop_response.raise_for_status()
            snoop_channel_id = snoop_response.json()['id']
            print(f"✅ Snoop channel {snoop_channel_id} created.")
        except requests.RequestException as e:
            print(f"❌ Failed to create snoop channel: {e}")
            requests.delete(f"{config.BASE_URL}/channels/{channel_id}", auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD))
            return

        recording_name = f"live_rec_{channel_id}"
        requests.post(
            f"{config.BASE_URL}/channels/{snoop_channel_id}/record",
            params={"name": recording_name, "format": "sln16", "ifExists": "overwrite"},
            auth=HTTPBasicAuth(config.ARI_USER, config.ARI_PASSWORD)
        )
        print(f"✅ Recording started on snoop channel {snoop_channel_id}.")

        slin_path = os.path.join(config.LIVE_RECORDING_PATH, f"{recording_name}.sln16")
        threading.Thread(
            target=interact_with_user,
            args=(channel_id, snoop_channel_id, slin_path, caller_id),
            daemon=True
        ).start()

    elif event_type == 'StasisEnd':
        print("📞 Call ended and channel destroyed.")

def on_open(ws):
    print("✅ WebSocket connected to ARI")
    originate_call()

def on_error(ws, error):
    print(f"❗ WebSocket Error: {error}")

def on_close(ws, *args):
    print("🔌 WebSocket closed")

if __name__ == "__main__":
    print("--- Starting AI Agent ---")
    ws_url = f'ws://{config.ARI_HOST}:{config.ARI_PORT}/ari/events?app={config.ARI_APP}&api_key={config.ARI_USER}:{config.ARI_PASSWORD}'
    ws = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()