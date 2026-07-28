import logging
import json
import numpy as np
import pandas as pd
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

logger = logging.getLogger("ETL_Logger")

# load data in JSONs to dataframes
df_users = pd.read_json(RAW_DATA_DIR / "users_raw.json")
df_attendance = pd.read_json(RAW_DATA_DIR / "attendance_raw.json")
df_holidays = pd.read_json(RAW_DATA_DIR / "holidays_raw.json")

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

# print(df_attendance["type"].value_counts())

# delete "arrival" type, its obsolete
df_attendance = df_attendance[df_attendance["type"] != "arrival"].copy()
# delete archived users
df_users = df_users[~df_users["archived"]].copy()

# merge two homeoffice types
df_attendance["type"] = df_attendance["type"].replace(
    {"homeoffice-extraordinary": "homeoffice"}
)

# standardize date
# Datetime, Durations Formatting and Cleaning
df_attendance["date"] = pd.to_datetime(
    df_attendance["date"]).dt.date  # "Date" is the exact day the attendance is occuring not when its set or updated so "Date" with type "Vacation" means that on that specific date a vacation occured


# define categories
working_types = [
    "present",
    "homeoffice",
]

businesstrips = [
    "businesstrip-domestic",
    "businesstrip-foreign"
]

# --------------------------------------------------------------------------------------------------------
# autoclose types
#should_autoclose = df_attendance["to"].isna() & ~df_attendance["type"].isin(working_types)
#df_attendance["to"] = np.where(
#    should_autoclose, 
#    "17:00:00",
#    df_attendance["to"]
#)
# --------------------------------------------------------------------------------------------------------

# calculate hours worked
time_from = pd.to_timedelta(df_attendance["from"] + ":00", errors="coerce")
time_to = pd.to_timedelta(df_attendance["to"] + ":00", errors="coerce")

raw_duration = round(((time_to - time_from).dt.total_seconds() / 3600), 1)

# create dummies
is_working = df_attendance["type"].isin(working_types)
is_businesstrip = df_attendance["type"].isin(businesstrips)
is_pause = df_attendance["type"] == "pause"
is_doctor = df_attendance["note"] == "Lékař"
is_training = df_attendance["note"] == "Školení"
is_unclosed = df_attendance["from"].notna() & df_attendance["to"].isna()
is_homeoffice = df_attendance["type"] == "homeoffice"
is_sick = df_attendance["type"] == "sick"

# working hours (for actual work types)
df_attendance["hours_worked"] = np.where(
    is_working & ~is_unclosed,
    raw_duration.fillna(0), 
    0.0
)

# calculate pause hours
df_attendance["hours_paused"] = np.where(
    is_pause & ~is_unclosed,
    raw_duration.fillna(0),
    0.0
)

conditions = [is_unclosed & is_working,
              is_unclosed & is_pause,
              is_pause,
              is_businesstrip,
              ~is_working,
              ]

choices = [
    "Unclosed Work Shift",
    "Unclosed Break",
    "Break / Pause",
    "Business Trip",
    "Non-Working Event",
]

df_attendance["status_flag"] = np.select(
    conditions, 
    choices,
    default="Standard Completed Shift"
)

df_attendance["Doctor"] = is_doctor
df_attendance["Training"] = is_training
df_attendance["SickLeave"] = is_sick

# keys, join 
df_dim_users = df_users.rename(columns={
    "id": "user_id"
})
df_attendance = df_attendance.rename(columns={
    "user": "user_id"
})


# select clean needed cols
attendance_cols = [
    "id", "user_id", "date", "type", "note", "status_flag", "Doctor", 
    "Training", "SickLeave", "from", "to", "hours_worked", "hours_paused",
    "lunch", "manual", "wage"
]

df_attendance = df_attendance[attendance_cols]

# keep only valid type subcategories ("note")
valid_notes = [
    "Schůzka", "Lékař", "Osobní důvody", "Pořádání akce", "Školení", 
    "Doprovod osoby blízké k lékaři", "Stáž v zahraničí (přidáno skriptem)", 
    "Škola", "Doprovod dítěte", "Čerpání zbytku dovolené"
]

# logic to drop invalid pause subcategories ("note"s) and keep all rows with other type than "pause" regardless of "note" status
is_valid_pause = (df_attendance["type"] == "pause") & (df_attendance["note"].isin(valid_notes))
is_other_type  = df_attendance["type"] != "pause"

df_attendance = df_attendance[is_other_type | is_valid_pause].copy()

# Resolve "invalid date" value in 'to' directly from PB API
resolve_invalid_date = df_attendance["to"] == "Invalid Date"
df_attendance["to"] = np.where(
    resolve_invalid_date,
    np.nan,
    df_attendance["to"]
)

# PRIMARY TYPE
# EXPLAINER: There can occur an edge case where a person has both "homeoffice" and "present" types where we have to determine which is dominant, the more hours worked in a specific "type" will be the PRIMARY TYPE

df_attendance["_is_present_entry"] = (df_attendance["type"] == "present") & (
    (df_attendance["hours_worked"] > 0) | (df_attendance["from"].notna())
)
df_attendance["_is_ho_entry"] = (df_attendance["type"] == "homeoffice") & (
    (df_attendance["hours_worked"] > 0) | (df_attendance["from"].notna())
)

present_hrs = np.where(df_attendance["type"] == "present", df_attendance["hours_worked"], 0.0)
ho_hrs = np.where(df_attendance["type"] == "homeoffice", df_attendance["hours_worked"], 0.0)

# 2. Attach temporarily or group directly using pandas.NamedAgg / assign
daily_summary = (
    df_attendance
    .assign(_present=present_hrs, _ho=ho_hrs)
    .groupby(["user_id", "date"], as_index=False)
    .agg(
        present_hrs=("_present", "sum"),
        ho_hrs=("_ho", "sum"),
        has_present_entry=("_is_present_entry", "any"),
        has_ho_entry=("_is_ho_entry", "any"),
        is_sick=("SickLeave", "any")
    )
)

def assign_primary_type(row):
    if row["is_sick"]:
        return "Sick Leave"
    
    # Priority 1: Check completed hours worked
    elif row["present_hrs"] > 0 and row["present_hrs"] >= row["ho_hrs"]:
        return "In-Office"
    elif row["ho_hrs"] > 0 and row["ho_hrs"] > row["present_hrs"]:
        return "Home Office"
    
    # Priority 2: Fallback for Active/Unclosed Shifts today (where hours_worked is 0.0)
    elif row["has_present_entry"]:
        return "In-Office"
    elif row["has_ho_entry"]:
        return "Home Office"
    
    else:
        return "Out of Office"

daily_summary["primary_type"] = daily_summary.apply(assign_primary_type, axis=1)
df_attendance = df_attendance.drop(columns=["_is_present_entry", "_is_ho_entry"]) # clean up helper cols

# merge back
df_attendance = df_attendance.merge(
    daily_summary[["user_id", "date", "primary_type"]],
    on=["user_id", "date"],
    how="left"
)

# select final clean columns for PBI
attendance_cols = [
    "id", "user_id", "date", "type", "primary_type", "note", "status_flag", 
    "Doctor", "Training", "SickLeave", "from", "to", "hours_worked", 
    "hours_paused", "lunch", "manual", "wage"
]
df_attendance = df_attendance[attendance_cols]



# export to csv
df_dim_users.to_csv(PROCESSED_DATA_DIR / "dim_users.csv", index=False, encoding="utf-8-sig")
df_attendance.to_csv(PROCESSED_DATA_DIR / "fact_attendance.csv", index=False, encoding="utf-8-sig")
df_holidays.to_csv(PROCESSED_DATA_DIR / "dim_holidays.csv", index=False, encoding="utf-8-sig")


# upload data to SQL server..... TO DO




#################################################################################################


# Filter for records where 'from' exists, but 'to' is missing (NaN/None)
missing_to_with_from = df_attendance[
    df_attendance["from"].notna() & df_attendance["to"].isna()
]

print(f"\nTotal rows where 'from' exists but 'to' is NA: {len(missing_to_with_from)}")
print("\n--- Breakdown by Type ---")
print(missing_to_with_from["type"].value_counts(dropna=False))



df_attendance["note"].value_counts()
df_attendance["type"].value_counts()