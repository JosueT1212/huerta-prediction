import os
from supabase import create_client, Client

SUPABASE_URL: str = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY: str = os.environ["SUPABASE_SERVICE_KEY"]

service_client: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
