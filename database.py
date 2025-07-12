# database.py
import supabase
import config
import traceback

# 1. Initialize the Supabase client
try:
    client = supabase.create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    print("✅ Supabase client initialized successfully.")
except Exception as e:
    client = None
    print(f"❌ Failed to initialize Supabase client: {e}")

# 2. Create a function to save the call log
def log_call(caller_id, duration, transcript):
    """Inserts a new record into the call_logs table."""
    if not client:
        print("DB ERROR: Supabase client not available. Cannot log call.")
        return None

    try:
        print(f"📝 Logging call from {caller_id} to database...")
        data, count = client.from_('call_logs').insert({
            'caller_id': caller_id,
            'call_duration_seconds': duration,
            'transcript': transcript
        }).execute()
        
        print("✅ Call log saved successfully.")
        return data

    except Exception:
        print("❌ An error occurred while logging the call to the database.")
        # traceback.print_exc() # Uncomment for detailed error logging
        return None