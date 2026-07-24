import logging
import json
import pandas as pd
import requests
from config import (
    AUTH_URL,
    PB_EMAIL,
    PB_PASSWORD,
    POCKETBASE_ATTENDANCE_URL,
    POCKETBASE_USERS_URL,
    RAW_DATA_DIR,
)

logger = logging.getLogger("ETL_Logger")

class PocketBaseExtractor:
    def __init__(self):
        self.headers = {}
        self.authenticate()

    def authenticate(self):
        """Authentification to commmunicate with PocketBase to obtain Token"""
        logger.info("Auth with PB API")

        credentials = {"identity": PB_EMAIL, "password": PB_PASSWORD}

        try:
            auth_response = requests.post(AUTH_URL, json=credentials, timeout=10)
            print(f"Auth Status Code: {auth_response.status_code}")
            auth_response.raise_for_status()
            token = auth_response.json().get("token")

            self.headers["Authorization"] = token
            logger.info("Auth ran succesfully")

        except requests.RequestException as exception:
            logger.error(f"Auth failed: {exception}")
            raise

    def fetch_users(self) -> list:
        """ Fetch users, no pagination needed """
        logger.info("Fetching users from PB collection 'users'")
        try:
            response_users = requests.get(
                POCKETBASE_USERS_URL,
                headers = self.headers,
                params={"perPage": 200},
                timeout = 15,
            )
            response_users.raise_for_status()
            print(f"Users fetch Status Code: {response_users.status_code}")

            users_data = response_users.json().get("items", [])
            logger.info(f"Fetched {len(users_data)} users")

            return users_data

        except requests.RequestException as exception:
            logger.error(f"Failed to fetch users: {exception}")
            raise

    def fetch_attendance(self) -> list:
        """ Fetch attendance records, pagination loops to extract it all """
        logger.info("Fetching attendance records")

        # info on total items in DB from JSON
        attendence_total_items = (
        requests.get(POCKETBASE_ATTENDANCE_URL, headers=self.headers).json().get("totalItems")
        )
        print("Total number of items found in the database:", attendence_total_items)

        # ATTENDANCE list JSON fetch
        all_attendance_records = []
        page = 1
        per_page = 500
    
        while True: # while True:... is basically infinite loop until we break it with if ... break
            try:           
                params = {"page": page, "perPage": per_page}

                response = requests.get(
                    POCKETBASE_ATTENDANCE_URL, 
                    headers = self.headers, 
                    params = params,
                    timeout = 200
                )

                response.raise_for_status()
                
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

            except requests.RequestException as exception:
                logger.error(f"Failed t fetch attendance on page {page}: {exception}")
                raise

        logger.info(f"Fetched {len(all_attendance_records)} records")
        return all_attendance_records


    def save_raw_json(self, data: list, filename: str):
        """Save data to RAW_DATA_DIR as raw JSON files"""
        output_path = RAW_DATA_DIR / filename
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved raw data to {output_path}")

    # Self running the script, later from main...

    def run_extract(self):
        """Main execution."""
        users = self.fetch_users()
        attendance = self.fetch_attendance()

        self.save_raw_json(users, "users_raw.json")
        self.save_raw_json(attendance, "attendance_raw.json")
