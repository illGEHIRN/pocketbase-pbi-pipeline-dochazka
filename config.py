import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

# URLs, endpoints
BASE_URL = "https://db.dochazka.czs.muni.cz"
# exposed PocketBase API URLs
POCKETBASE_USERS_URL = f"{BASE_URL}/api/collections/users/records"
POCKETBASE_ATTENDANCE_URL = f"{BASE_URL}/api/collections/attendance/records"
POCKETBASE_HOLIDAY_URL = f"{BASE_URL}/api/collections/holidays/records"
POCKETBASE_LOGS_URL = f"{BASE_URL}/api/collections/logs/records"
AUTH_URL = f"{BASE_URL}/api/admins/auth-with-password"

# secrets
load_dotenv()
PB_PASSWORD = os.getenv("PB_PASSWORD")
PB_EMAIL = os.getenv("PB_EMAIL")


# folder paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
LOG_DIR = BASE_DIR / "logs"

# ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)