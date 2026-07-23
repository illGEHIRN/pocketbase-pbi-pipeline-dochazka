import json

import numpy as np
import pandas as pd

# load data in JSONs to dataframes
df_users = pd.read_json("JSONs/users_raw.json")
df_attendance = pd.read_json("JSONs/attendance_raw.json")

# explode/flatten the "times" in attendance JSON
df_exploded = df_attendance.explode("times").reset_index(drop=True)
times_df = pd.json_normalize(df_exploded["times"])

df_attendance = pd.concat([df_exploded.drop(columns=["times"]), times_df], axis=1)

# -----------------------------------------------------------------------------------------

# inspection

# sample if the flattening worked, in JSON format
sample_flattened = df_attendance.iloc[0].to_dict()
with open("JSONs/sample_flattened.json", "w", encoding="utf-8") as f:
    json.dump(sample_flattened, f, indent=2, ensure_ascii=False, default=str)


# inspect it in the dataframe column structure...
print(df_attendance[["date", "user", "from", "to", "type", "wage"]].head(5))

# -----------------------------------------------------------------------------------------

# Cleaning, merging types

print(df_attendance["type"].value_counts())

# delete "arrival" type, its obsolete
df_attendance = df_attendance[df_attendance["type"] != "arrival"].copy()

# merge two homeoffice types
df_attendance["type"] = df_attendance["type"].replace(
    {"homeoffice-extraordinary": "homeoffice"}
)

# define categories
working_types = [
    "present",
    "homeoffice",
    "businesstrip-foreign ",
    "businesstrip-domestic",
]

# calculate hours worked
time_from = pd.to_timedelta(df_attendance["from"] + ":00", errors="coerce")
time_to = pd.to_timedelta(df_attendance["to"] + ":00", errors="coerce")

raw_duration = round(((time_to - time_from).dt.total_seconds() / 3600), 1)

# create dummies
is_working = df_attendance["type"].isin(working_types)
is_pause = df_attendance["type"] == "pause"
is_unclosed = df_attendance["from"].notna() & df_attendance["to"].isna()

# working hours (for actual work types)
df_attendance["hours_worked"] = np.where(
    is_working & ~is_unclosed, raw_duration.fillna(0), 0.0
)

# calculate pause hours
df_attendance["hours_break"] = np.where(
    is_pause & ~is_unclosed, raw_duration.fillna(0), 0.0
)

conditions = [is_unclosed & is_working, is_unclosed & is_pause, is_pause, ~is_working]
choices = [
    "Unclosed Work Shift",
    "Unclosed Break",
    "Break / Pause",
    "Non-Working Event",
]

df_attendance["status_flag"] = np.select(
    conditions, choices, default="Standard Completed Shift"
)


# -----------------------------------------------------------------------------------------

# Datetime, Durations Formatting and Cleaning
df_attendance["date"] = pd.to_datetime(
    df_attendance["date"]
).dt.date  # "Date" is the exact day the attendance is occuring not when its set or updated so "Date" with type "Vacation" means that on that specific date a vacation occured


# Filter for records where 'from' exists, but 'to' is missing (NaN/None)
missing_to_with_from = df_attendance[
    df_attendance["from"].notna() & df_attendance["to"].isna()
]

print(f"Total rows where 'from' exists but 'to' is NA: {len(missing_to_with_from)}")
print("\n--- Breakdown by Type ---")
print(missing_to_with_from["type"].value_counts(dropna=False))
