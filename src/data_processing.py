import logging
import json
import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo
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


# calculate hours worked
time_from = pd.to_timedelta(df_attendance["from"] + ":00", errors="coerce")
time_to = pd.to_timedelta(df_attendance["to"] + ":00", errors="coerce")

raw_duration = (time_to - time_from).dt.total_seconds() / 3600

# create dummies
is_working = df_attendance["type"].isin(working_types)
is_businesstrip = df_attendance["type"].isin(businesstrips)
is_pause = df_attendance["type"] == "pause"
is_vacation = df_attendance["type"] == "vacation"
is_doctor = df_attendance["note"] == "Lékař"
is_training = df_attendance["note"] == "Školení"
is_unclosed = df_attendance["from"].notna() & df_attendance["to"].isna()
is_homeoffice = df_attendance["type"] == "homeoffice"
is_sick = df_attendance["type"] == "sick"

# Normalize wage to a real boolean.
df_attendance["wage"] = (
    df_attendance["wage"]
    .fillna(False)
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes", "ano"])
)

is_wage = df_attendance["wage"]

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

# calculate wage hours which may include pauses etc
df_attendance["wage_hours"] = np.where(
    is_wage
    & ~is_working
    & ~is_unclosed,
    raw_duration.fillna(0),
    0.0,
)

# calculate vacation hours
df_attendance["vacation_hours"] = np.where(
    is_vacation & ~is_unclosed,
    raw_duration.fillna(0),
    0.0,
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

# flags
df_attendance["is_doctor"] = is_doctor
df_attendance["is_training"] = is_training
df_attendance["is_sick"] = is_sick
df_attendance["is_businesstrip"] = is_businesstrip

df_attendance["is_vacation"] = is_vacation

df_attendance["is_paid_absence"] = (
    df_attendance["type"].eq("paidabsent")
)

# keys, join 
df_dim_users = df_users.rename(columns={
    "id": "user_id"
}).copy()
df_attendance = df_attendance.rename(columns={
    "user": "user_id"
}).copy()

# handle archived users
df_dim_users["is_currently_active"] = (
    ~df_dim_users["archived"].fillna(False)
)

schedule_parts = (
    df_dim_users["schedule"]
    .fillna("0,0,0,0,0")
    .astype(str)
    .str.split(",", expand=True)
)

# error checks --------------------------------------
if schedule_parts.shape[1] != 5:
    invalid_schedule_rows = df_dim_users.loc[
        schedule_parts.notna().sum(axis=1) != 5,
        ["user_id", "schedule"],
    ]

    raise ValueError(
        "Every schedule must contain exactly five values "
        "(Monday-Friday).\n"
        + invalid_schedule_rows.to_string(index=False)
    )

schedule_parts = schedule_parts.apply(
    pd.to_numeric,
    errors="coerce",
)

if schedule_parts.isna().any().any():
    invalid_schedule_rows = df_dim_users.loc[
        schedule_parts.isna().any(axis=1),
        ["user_id", "schedule"],
    ]

    raise ValueError(
        "Some schedule values could not be converted to numbers:\n"
        + invalid_schedule_rows.to_string(index=False)
    )
# --------------------------------------------------------

schedule_parts.columns = [
    "planned_monday",
    "planned_tuesday",
    "planned_wednesday",
    "planned_thursday",
    "planned_friday",
]

df_dim_users = pd.concat(
    [
        df_dim_users.reset_index(drop=True),
        schedule_parts.reset_index(drop=True),
    ],
    axis=1,
)

schedule_columns = [
    "planned_monday",
    "planned_tuesday",
    "planned_wednesday",
    "planned_thursday",
    "planned_friday",
]

df_dim_users["weekly_planned_hours"] = (
    df_dim_users[schedule_columns].sum(axis=1)
)

# False for schedules such as "0,0,0,0,0".
# These employees will be excluded from schedule-dependent calculations.
df_dim_users["has_calculable_schedule"] = (
    df_dim_users["weekly_planned_hours"] > 0
)

# validation for logging
# does obligation contract match scheduled hours?
# JUST FOR LOG
df_dim_users["expected_weekly_hours_from_obligation"] = (
    pd.to_numeric(
        df_dim_users["obligation"],
        errors="coerce",
    ).fillna(0)
    * 40
)

df_dim_users["schedule_obligation_difference"] = (
    df_dim_users["weekly_planned_hours"]
    - df_dim_users["expected_weekly_hours_from_obligation"]
)

schedule_mismatches = df_dim_users.loc[
    df_dim_users["schedule_obligation_difference"].abs() > 0.01,
    [
        "user_id",
        "obligation",
        "schedule",
        "weekly_planned_hours",
        "expected_weekly_hours_from_obligation",
        "schedule_obligation_difference",
    ],
]

if not schedule_mismatches.empty:
    logger.warning(
        "Schedules not matching contractual obligation:\n%s",
        schedule_mismatches.to_string(index=False),
    )



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

# select final clean columns for PBI
attendance_cols = [
    "id",
    "user_id",
    "date",
    "type",
    "note",
    "status_flag",

    "is_doctor",
    "is_training",
    "is_sick",
    "is_businesstrip",
    "is_vacation",
    "is_paid_absence",

    "from",
    "to",
    "hours_worked",
    "hours_paused",
    "wage_hours",
    "vacation_hours",
    "lunch",
    "manual",
    "wage",
]

df_attendance = df_attendance[attendance_cols]


# standardize lunch to a proper boolean instead of string
df_attendance["lunch"] = (
    df_attendance["lunch"]
    .fillna(False)
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(["true", "1", "yes", "ano"])
)

# Fact table splitting

# first: one row per each attendance time/event entry (basically copy of df_attendance)
df_fact_attendance_event = df_attendance.copy()

# add flags
df_fact_attendance_event["has_present_status"] = (
    df_fact_attendance_event["type"].eq("present")
)

df_fact_attendance_event["has_homeoffice_status"] = (
    df_fact_attendance_event["type"].eq("homeoffice")
)

# further data processing for df_fact_attendance_event table
# create unique key for every exploded event from original JSON (was missing earlier)
df_fact_attendance_event["event_number"] = (
    df_fact_attendance_event.groupby("id", dropna = False).cumcount().add(1)
)

df_fact_attendance_event["attendance_event_key"] = (
    df_fact_attendance_event["id"].astype("string")
    + "-"
    + df_fact_attendance_event["event_number"].astype("string")
)

# additional event flags
df_fact_attendance_event["is_present_entry"] = (
    df_fact_attendance_event["type"].eq("present")
    & (
        df_fact_attendance_event["hours_worked"].gt(0)
        | df_fact_attendance_event["from"].notna()
    )
)

df_fact_attendance_event["is_homeoffice_entry"] = (
    df_fact_attendance_event["type"].eq("homeoffice")
    & (
        df_fact_attendance_event["hours_worked"].gt(0)
        | df_fact_attendance_event["from"].notna()
    )
)


# split working hours based on location (present/HO)
df_fact_attendance_event["office_hours"] = np.where(
    df_fact_attendance_event["type"].eq("present"),
    df_fact_attendance_event["hours_worked"],
    0.0,
)

df_fact_attendance_event["homeoffice_hours"] = np.where(
    df_fact_attendance_event["type"].eq("homeoffice"),
    df_fact_attendance_event["hours_worked"],
    0.0,
)

# 2. fact table: user-day
# one row per user and date that has a recorded event

# aggregates
df_fact_user_day_recorded = (
    df_fact_attendance_event
    .groupby(["user_id", "date"], as_index=False)
    .agg(
        office_hours=("office_hours", "sum"),
        homeoffice_hours=("homeoffice_hours", "sum"),
        pause_hours=("hours_paused", "sum"),
        wage_hours=("wage_hours", "sum"),
        vacation_hours=("vacation_hours", "sum"),

        has_present_entry=("is_present_entry", "any"),
        has_homeoffice_entry=("is_homeoffice_entry", "any"),

        has_sick=("is_sick", "any"),
        has_vacation=("is_vacation", "any"),
        has_paid_absence=("is_paid_absence", "any"),

        has_business_trip=("is_businesstrip", "any"),
        has_doctor=("is_doctor", "any"),
        has_training=("is_training", "any"),

        has_lunch=("lunch", "any"),

        has_present_status=("has_present_status", "any"),
        has_homeoffice_status=("has_homeoffice_status", "any"),

        recorded_event_count=("attendance_event_key", "count"),
    )
)

# total hrs worked aggregate (brutto)
df_fact_user_day_recorded["total_work_hours"] = (
    df_fact_user_day_recorded["office_hours"]
    + df_fact_user_day_recorded["homeoffice_hours"]
)

# official hours worked with lunch pause substracted if present (netto)
df_fact_user_day_recorded["official_hours_worked"] = (
    df_fact_user_day_recorded["total_work_hours"]
    - np.where(
        df_fact_user_day_recorded["has_lunch"],
        0.5,
        0.0,
    )
).clip(lower=0.0)

# flag for mixed location workdays
df_fact_user_day_recorded["is_mixed_location"] = (
    df_fact_user_day_recorded["has_present_entry"]
    & df_fact_user_day_recorded["has_homeoffice_entry"]
)

# employee active ranges, historic
today = pd.Timestamp.now(
    tz=ZoneInfo("Europe/Prague")
).date()

minimum_data_date = df_fact_user_day_recorded["date"].min()

recorded_ranges = (
    df_fact_user_day_recorded
    .groupby("user_id", as_index=False)
    .agg(
        first_recorded_date=("date", "min"),
        last_recorded_date=("date", "max"),
    )
)

df_dim_users = df_dim_users.merge(
    recorded_ranges,
    on="user_id",
    how="left",
)

if "startDate" not in df_dim_users.columns:
    raise ValueError(
        "dim_users does not contain the expected startDate column."
    )

df_dim_users["employment_start_date"] = (
    pd.to_datetime(
        df_dim_users["startDate"],
        errors="coerce",
    ).dt.date
)

df_dim_users["active_from"] = (
    df_dim_users["employment_start_date"]
    .combine_first(df_dim_users["first_recorded_date"])
)

# The API extraction starts in 2024, so do not generate older rows.
df_dim_users["active_from"] = df_dim_users["active_from"].apply(
    lambda value: (
        max(value, minimum_data_date)
        if pd.notna(value)
        else minimum_data_date
    )
)

df_dim_users["active_to"] = today

archived_users = ~df_dim_users["is_currently_active"]

# Approximation until an actual employment-end field is available.
df_dim_users.loc[archived_users, "active_to"] = (
    df_dim_users.loc[archived_users, "last_recorded_date"]
)

# employee-day spine
holiday_dates = set(
    pd.to_datetime(
        df_holidays["date"],
        errors="coerce",
    )
    .dropna()
    .dt.date
)

expected_rows = []

eligible_users = df_dim_users.loc[
    df_dim_users["has_calculable_schedule"]
    & df_dim_users["active_from"].notna()
    & df_dim_users["active_to"].notna()
].copy()

for user in eligible_users.itertuples(index=False):

    weekly_schedule = [
        user.planned_monday,
        user.planned_tuesday,
        user.planned_wednesday,
        user.planned_thursday,
        user.planned_friday,
    ]

    employee_dates = pd.date_range(
        start=user.active_from,
        end=user.active_to,
        freq="D",
    )

    for timestamp in employee_dates:

        date_value = timestamp.date()
        weekday_index = timestamp.weekday()

        # Saturday or Sunday
        if weekday_index >= 5:
            continue

        # Public holiday
        if date_value in holiday_dates:
            continue

        planned_hours = float(
            weekly_schedule[weekday_index]
        )

        # A zero in an otherwise valid schedule means the person
        # is known not to be scheduled that weekday.
        if planned_hours <= 0:
            continue

        expected_rows.append(
            {
                "user_id": user.user_id,
                "date": date_value,
                "planned_hours": planned_hours,
            }
        )

df_expected_user_day = pd.DataFrame(expected_rows)


# final employee day fact
df_fact_user_day = df_expected_user_day.merge(
    df_fact_user_day_recorded,
    on=["user_id", "date"],
    how="outer",
)

df_fact_user_day = df_fact_user_day.merge(
    df_dim_users[
        [
            "user_id",
            "has_calculable_schedule",
            "is_currently_active",
            "planned_monday",
            "planned_tuesday",
            "planned_wednesday",
            "planned_thursday",
            "planned_friday",
        ]
    ],
    on="user_id",
    how="left",
)

df_fact_user_day["weekday_number"] = (
    pd.to_datetime(df_fact_user_day["date"])
    .dt.weekday
    .add(1)
)

derived_planned_hours = np.select(
    [
        df_fact_user_day["weekday_number"].eq(1),
        df_fact_user_day["weekday_number"].eq(2),
        df_fact_user_day["weekday_number"].eq(3),
        df_fact_user_day["weekday_number"].eq(4),
        df_fact_user_day["weekday_number"].eq(5),
    ],
    [
        df_fact_user_day["planned_monday"],
        df_fact_user_day["planned_tuesday"],
        df_fact_user_day["planned_wednesday"],
        df_fact_user_day["planned_thursday"],
        df_fact_user_day["planned_friday"],
    ],
    default=0.0,
)

missing_planned = (
    df_fact_user_day["planned_hours"].isna()
    & df_fact_user_day["has_calculable_schedule"].fillna(False)
)

df_fact_user_day.loc[
    df_fact_user_day["date"].isin(holiday_dates),
    "planned_hours",
] = 0.0

df_fact_user_day.loc[
    missing_planned,
    "planned_hours",
] = derived_planned_hours[missing_planned]

# export
event_columns = [
    "attendance_event_key",
    "id",
    "event_number",
    "user_id",
    "date",
    "type",
    "note",
    "status_flag",
    "from",
    "to",

    "hours_worked",
    "office_hours",
    "homeoffice_hours",
    "hours_paused",
    "wage_hours",
    "vacation_hours",

    "is_doctor",
    "is_training",
    "is_sick",
    "is_businesstrip",
    "is_vacation",
    "is_paid_absence",

    "is_present_entry",
    "is_homeoffice_entry",

    "lunch",
    "manual",
    "wage",
]

df_fact_attendance_event = (
    df_fact_attendance_event[event_columns]
    .sort_values(["date", "user_id", "attendance_event_key"])
    .reset_index(drop=True)
)

numeric_columns = [
    "office_hours",
    "homeoffice_hours",
    "pause_hours",
    "wage_hours",
    "vacation_hours",
    "recorded_event_count",
]

for column in numeric_columns:
    df_fact_user_day[column] = (
        pd.to_numeric(
            df_fact_user_day[column],
            errors="coerce",
        )
        .fillna(0)
    )

boolean_columns = [
    "has_present_entry",
    "has_homeoffice_entry",
    "has_present_status",
    "has_homeoffice_status",
    "has_lunch",
    "has_sick",
    "has_vacation",
    "has_paid_absence",
    "has_business_trip",
    "has_doctor",
    "has_training",
    "is_mixed_location",
]

for column in boolean_columns:
    df_fact_user_day[column] = (
        df_fact_user_day[column]
        .fillna(False)
        .astype(bool)
    )


df_fact_user_day["weekday_number"] = (
    pd.to_datetime(df_fact_user_day["date"])
    .dt.weekday
    .add(1)
)

df_fact_user_day["is_holiday"] = (
    df_fact_user_day["date"].isin(holiday_dates)
)

df_fact_user_day["is_expected_workday"] = (
    df_fact_user_day["planned_hours"].gt(0)
)

df_fact_user_day["has_recorded_event"] = (
    df_fact_user_day["recorded_event_count"] > 0
)

df_fact_user_day["is_missing_record"] = (
    df_fact_user_day["is_expected_workday"]
    & ~df_fact_user_day["has_recorded_event"]
)

df_fact_user_day["total_work_hours"] = (
    df_fact_user_day["office_hours"]
    + df_fact_user_day["homeoffice_hours"]
)

df_fact_user_day["has_lunch"] = (
    df_fact_user_day["has_lunch"]
    .fillna(False)
    .astype(bool)
)

df_fact_user_day["official_hours_worked"] = (
    df_fact_user_day["total_work_hours"]
    + df_fact_user_day["wage_hours"]
    - np.where(
        df_fact_user_day["has_lunch"],
        0.5,
        0.0,
    )
).clip(lower=0.0)

# rounding to the nearest 0.25 hours
df_fact_user_day["official_hours_worked"] = (
    df_fact_user_day["official_hours_worked"] * 4
).round() / 4

df_fact_user_day["is_mixed_location"] = (
    df_fact_user_day["has_present_entry"]
    & df_fact_user_day["has_homeoffice_entry"]
)


# separate classifiactions
def assign_work_location(row: pd.Series) -> str:

    if (
        row["has_present_status"]
        and row["has_homeoffice_status"]
    ):
        if row["homeoffice_hours"] > row["office_hours"]:
            return "Home Office"

        return "In-Office"

    if row["has_homeoffice_status"]:
        return "Home Office"

    if row["has_present_status"]:
        return "In-Office"

    return "No Location Work"


df_fact_user_day["work_location_status"] = (
    df_fact_user_day.apply(
        assign_work_location,
        axis=1,
    )
)

def assign_day_status(row: pd.Series) -> str:
    """
    Assign one mutually exclusive headline status.
    """

    if row["has_sick"]:
        return "Sick"

    if row["has_vacation"]:
        return "Vacation"

    if row["has_business_trip"]:
        return "Business Trip"

    if row["work_location_status"] == "Home Office":
        return "Home Office"

    if row["work_location_status"] == "In-Office":
        return "In-Office"

    if row["has_paid_absence"]:
        return "Paid Absence"

    return "Other / Not In"


df_fact_user_day["day_status"] = (
    df_fact_user_day.apply(
        assign_day_status,
        axis=1,
    )
)

# for introductory cards
def assign_card_status(row: pd.Series) -> str:
    if row["has_present_status"]:
        return "In-Office"

    if row["has_homeoffice_status"]:
        return "Home Office"

    if row["has_business_trip"]:
        return "Business Trip"

    if row["has_sick"]:
        return "Sick"

    if row["has_vacation"]:
        return "Vacation"

    if row["has_paid_absence"]:
        return "Paid Absence"

    return "Other / Not In"


df_fact_user_day["card_status"] = (
    df_fact_user_day.apply(assign_card_status, axis=1)
)

# determine full vacation day
df_fact_user_day["vacation_day_equivalent"] = np.where(
    df_fact_user_day["planned_hours"].gt(0),
    (
        df_fact_user_day["vacation_hours"]
        / df_fact_user_day["planned_hours"]
    ).clip(upper=1),
    np.nan,
)

# finalisaton, columns and validation
user_day_columns = [
    "user_id",
    "date",
    "weekday_number",

    "has_calculable_schedule",
    "is_currently_active",
    "planned_hours",
    "is_expected_workday",
    "is_holiday",

    "office_hours",
    "homeoffice_hours",
    "total_work_hours",
    "has_lunch",
    "official_hours_worked",
    "wage_hours",
    "pause_hours",

    "has_recorded_event",
    "recorded_event_count",
    "is_missing_record",

    "has_present_entry",
    "has_homeoffice_entry",
    "is_mixed_location",

    "has_sick",
    "has_vacation",
    "has_paid_absence",
    "has_business_trip",
    "has_doctor",
    "has_training",

    "vacation_hours",
    "vacation_day_equivalent",
    "has_present_status",
    "has_homeoffice_status",

    "work_location_status",
    "day_status",
    "card_status",
    ]

df_fact_user_day = (
    df_fact_user_day[user_day_columns]
    .sort_values(["date", "user_id"])
    .reset_index(drop=True)
)

# validation
if df_fact_user_day.duplicated(
    ["user_id", "date"]
).any():
    raise ValueError(
        "Duplicate user-date rows found in fact_user_day."
    )

if not df_fact_attendance_event[
    "attendance_event_key"
].is_unique:
    raise ValueError(
        "attendance_event_key is not unique."
    )


# to csv
df_dim_users.to_csv(PROCESSED_DATA_DIR / "dim_users.csv", index=False, encoding="utf-8-sig")
df_fact_attendance_event.to_csv(PROCESSED_DATA_DIR / "fact_attendance_event.csv", index=False, encoding="utf-8-sig")
df_fact_user_day.to_csv(PROCESSED_DATA_DIR / "fact_user_day.csv", index=False, encoding="utf-8-sig")
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









print(df_attendance["type"].value_counts(dropna=False))
print(df_dim_users[["obligation", "schedule"]].head(20).to_string())
print(df_dim_users["schedule"].value_counts(dropna=False))











user_idd = "6grzczss35nwgkh"

check_user = df_dim_users.loc[
    df_dim_users["user_id"].eq(user_idd),
    [
        "user_id",
        "name",
        "schedule",
        "has_calculable_schedule",
        "is_currently_active",
        "active_from",
        "active_to",
    ],
]

print(check_user.to_string(index=False))

check_user_id = check_user["user_id"].iloc[0]

print(
    df_expected_user_day.loc[
        (df_expected_user_day["user_id"] == check_user_id)
        & (df_expected_user_day["date"] == today)
    ].to_string(index=False)
)



print(
    df_fact_user_day.loc[
        df_fact_user_day["date"] == today,
        ["user_id", "date", "weekday_number", "planned_hours"]
    ].to_string(index=False)
)

print("today variable:", today)
print("actual now:", pd.Timestamp.today().date())