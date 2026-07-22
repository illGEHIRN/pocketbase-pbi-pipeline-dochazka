import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://db.dochazka.czs.muni.cz"

# auth
load_dotenv()
PB_PASSWORD = os.getenv("PB_PASSWORD")

AUTH_URL = f"{BASE_URL}/api/admins/auth-with-password"
credentials = {"identity": "hoso@czs.muni.cz", "password": PB_PASSWORD}

auth_response = requests.post(AUTH_URL, json=credentials)
# check response
print(f"Auth Status Code: {auth_response.status_code}")

auth_data = auth_response.json()
token = auth_data.get("token")
# check if token received
print("Token received", "YES" if token else "NO")

# authorisation header
headers = {"Authorization": token}

# exposed PocketBase API URLs
POCKETBASE_USERS_URL = f"{BASE_URL}/api/collections/users/records"
POCKETBASE_ATTENDANCE_URL = f"{BASE_URL}/api/collections/attendance/records"

# USERS list JSON fetch
response_users = requests.get(POCKETBASE_USERS_URL, headers=headers)
users_data = response_users.json().get("items", [])

# ATTENDANCE list JSON fetch

# info on total items in DB from JSON
attendance_total_items = (
    requests.get(POCKETBASE_ATTENDANCE_URL, headers=headers).json().get("totalItems")
)
print("Total number of items found in the database:", attendance_total_items)

# pagination while loop
all_attendance_records = []
page = 1
per_page = 500

while (
    True
):  # while True:... is basically infinite loop until we break it with if ... break
    params = {"page": page, "perPage": per_page}

    response = requests.get(POCKETBASE_ATTENDANCE_URL, headers=headers, params=params)
    data = response.json()

    items = data.get("items", [])
    all_attendance_records.extend(items)

    total_pages = data.get("totalPages", 1)
    # print loop to see progress
    print(
        f"Fetched page {page} of total {total_pages} ({len(all_attendance_records)} record total)"
    )

    if page >= total_pages:
        break  # stop the loops once we go through all pages

    page += 1  # the page turner

print(f"\nDone, Total records: {len(all_attendance_records)}")
if attendance_total_items == len(all_attendance_records):
    print("Fetched records equal to total items in the database, SUCCESS!!")
else:
    print(
        f"Missing records compared to all items in the database ({attendance_total_items - len(all_attendance_records)} missing)"
    )


# create pandas data frames
df_users = pd.DataFrame(users_data)
df_attendance = pd.DataFrame(all_attendance_records)


# quick check
print(df_users.head(2))
print(df_attendance.head(1))
len(df_attendance)


# check how one entire JSON record look like!
sample = df_attendance.iloc[0].to_dict()
with open("sample.json", "w", encoding="utf-8") as f:
    json.dump(sample, f, indent=2, ensure_ascii=False)


# save fetched data for now (for development)
# later restructire this .py to be a function, more convenient, more dynamic, better utility

df_users.to_json("JSONs/users_raw.json", orient="records", indent=2)
df_attendance.to_json("JSONs/attendance_raw.json", orient="records", indent=2)
