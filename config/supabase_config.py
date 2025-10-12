from supabase import create_client, Client

# import os
# from dotenv import load_dotenv

# load_dotenv()

# supabase_url: str = os.getenv("SUPABASE_URL")
# supabase_key: str = os.getenv("SUPABASE_KEY")

SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImR5eWZrZmpheHhreWludHV1Z2xsIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTA4Mzc5MDgsImV4cCI6MjA2NjQxMzkwOH0.b0pG8wGtHsyhRUv_hv3U9TJQRYMNC6qMYyH5btYdrWk"
SUPABASE_URL="https://dyyfkfjaxxkyintuugll.supabase.co"




supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

