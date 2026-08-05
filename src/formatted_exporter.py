from pathlib import Path
from typing import Optional
import logging
import numpy as np
import pandas as pd
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR
)

logger = logging.getLogger("ETL_Logger")

# load data in JSONs to dataframes
df_users = pd.read_csv(PROCESSED_DATA_DIR / "dim_users.csv")
df_user_day = pd.read_csv(PROCESSED_DATA_DIR / "fact_user_day.csv")
df_holidays = pd.read_csv(PROCESSED_DATA_DIR / "dim_holidays.csv")


CZECH_WEEKDAY_ABBR = {
    0: "Po",
    1: "Út",
    2: "St",
    3: "Čt",
    4: "Pá",
    5: "So",
    6: "Ne",
}

def parse_schedule(schedule_value) -> list[float]:
    """ ... """

    if pd.isna(schedule_value):
        raise ValueError("User does not have a defined schedule")

    if isinstance(schedule_value, str):
        values = [
            float(value.strip()) for value in schedule_value.split(",")
        ]
    else:
        raise TypeError(
            f"Unexpected schedule format: {type(schedule_value).__name__}"
        )

    if len(values) != 5:
        raise ValueError(
            f"Schedule has to contain 5 values for each of the wordays, this schedule contains {len(values)}"
        )

    return values


def normalize_date(
    values: pd.Series,
    timezone: str = "Europe/Prague",
) -> pd.Series:
    """
    Convert timestamps to Prague local dates without a time component.
    """
    return (
        pd.to_datetime(
            values,
            errors="coerce",
            utc=True,
        )
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
        .dt.normalize()
    )


def value_is_true(value) -> bool:
    """
    funciton understand boolean values and also strings containing true/false
    """
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "ano",
    }

def build_user_daily_report(
    users: pd.DataFrame,
    user_day: pd.DataFrame,
    holidays: pd.DataFrame,
    selected_user_id: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Vytvoří jeden řádek pro každý kalendářní den ve vybraném období.
    """

    start_date = pd.Timestamp(start_date).normalize()
    end_date = pd.Timestamp(end_date).normalize()

    if start_date > end_date:
        raise ValueError("start_date musí být před end_date.")

    # -------------------------------------------------------
    # 1. Selected user
    # -------------------------------------------------------

    selected_users = users.loc[
        users["user_id"] == selected_user_id
    ]

    if selected_users.empty:
        raise ValueError(
            f"Uživatel {selected_user_id!r} nebyl nalezen."
        )

    if len(selected_users) > 1:
        raise ValueError(
            f"Uživatel {selected_user_id!r} je v tabulce vícekrát."
        )

    selected_user = selected_users.iloc[0]
    schedule = parse_schedule(selected_user["schedule"])

    schedule_by_weekday = {
        0: schedule[0],  # pondělí
        1: schedule[1],  # úterý
        2: schedule[2],  # středa
        3: schedule[3],  # čtvrtek
        4: schedule[4],  # pátek
        5: 0.0,          # sobota
        6: 0.0,          # neděle
    }

    # -------------------------------------------------------
    # 2. List of all days in range
    # -------------------------------------------------------

    report = pd.DataFrame({
        "date": pd.date_range(
            start=start_date,
            end=end_date,
            freq="D",
        )
    })

    report["weekday_number"] = report["date"].dt.weekday

    report["weekday_abbr"] = report[
        "weekday_number"
    ].map(CZECH_WEEKDAY_ABBR)

    report["is_weekend"] = report[
        "weekday_number"
    ].isin([5, 6])

    # -------------------------------------------------------
    # 3. Hours worked
    # -------------------------------------------------------

    worked = user_day.loc[
        user_day["user_id"] == selected_user_id,
        ["date", "official_hours_worked"],
    ].copy()

    worked["date"] = normalize_date(worked["date"])

    worked = worked.loc[
        worked["date"].between(start_date, end_date)
    ]

    worked["official_hours_worked"] = pd.to_numeric(
        worked["official_hours_worked"],
        errors="coerce",
    )

    # Pro případ, že by existovalo více řádků za jeden den.
    worked = (
        worked
        .groupby("date", as_index=False)["official_hours_worked"]
        .sum(min_count=1)
    )

    report = report.merge(
        worked,
        on="date",
        how="left",
    )

    report["official_hours_worked"] = report[
        "official_hours_worked"
    ].fillna(0.0)

    # -------------------------------------------------------
    # 4. Holiday
    # -------------------------------------------------------

    holiday_data = holidays[["date", "name"]].copy()
    holiday_data["date"] = normalize_date(holiday_data["date"])

    holiday_data = holiday_data.loc[
        holiday_data["date"].between(start_date, end_date)
    ]

    # Kdyby bylo pro jeden den více názvů.
    holiday_data = (
        holiday_data
        .groupby("date", as_index=False)["name"]
        .agg(
            lambda names: "; ".join(
                pd.unique(names.dropna().astype(str))
            )
        )
        .rename(columns={"name": "holiday_name"})
    )

    report = report.merge(
        holiday_data,
        on="date",
        how="left",
    )

    report["is_holiday"] = report[
        "holiday_name"
    ].notna()

    # -------------------------------------------------------
    # 5. Schedule
    # -------------------------------------------------------

    report["scheduled_hours"] = (
        report["weekday_number"]
        .map(schedule_by_weekday)
        .astype(float)
    )

    # O svátku je plán 0 hodin.
    report.loc[
        report["is_holiday"],
        "scheduled_hours",
    ] = 0.0

    # True pouze v případě, že má podle rozvrhu skutečně pracovat.
    report["is_expected_workday"] = (
        report["scheduled_hours"] > 0
    )

    # -------------------------------------------------------
    # 6. Day status
    # -------------------------------------------------------

    report["status"] = report[
        "official_hours_worked"
    ].apply(
        lambda hours: (
            "přítomen"
            if hours > 0
            else "žádný záznam"
        )
    )

    # -------------------------------------------------------
    # 7. Deviation/Odchylka
    # -------------------------------------------------------

    report["deviation_hours"] = (
        report["official_hours_worked"]
        - report["scheduled_hours"]
    )

    return report, selected_user



def format_signed_hours(value: float) -> str:
    """
    Formátuje odchylku včetně znaménka.
    Zároveň zabrání zobrazení hodnoty -0.00h.
    """
    value = float(value)

    if abs(value) < 1e-9:
        value = 0.0

    return f"{value:+.2f}h"


def export_report_to_txt(
    report: pd.DataFrame,
    selected_user: pd.Series,
    selected_user_id: str,
    start_date: str,
    end_date: str,
    output_path: str | Path,
) -> str:
    """
    Převede připravený denní report do formátovaného TXT souboru.
    """

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()

    lines = [
        f"Uživatel: {selected_user_id}",
        (
            f"Období: {start.strftime('%d.%m.%Y')} - "
            f"{end.strftime('%d.%m.%Y')}"
        ),
        "",
        f"Jméno: {selected_user['name']}",
        f"Výchozí rozvrh: {selected_user['schedule']}",
        "",
        "",
        "",
        "=== DENNÍ PŘEHLED ===",
        "",
    ]

    for row in report.itertuples(index=False):
        tags = []

        if row.is_holiday:
            tags.append(
                f"[SVÁTEK: {row.holiday_name}]"
            )

        if row.is_weekend:
            tags.append("[VÍKEND]")

        tags_text = ""

        if tags:
            tags_text = " " + " ".join(tags)

        worked_hours = float(row.official_hours_worked)
        scheduled_hours = float(row.scheduled_hours)
        deviation_hours = float(row.deviation_hours)

        lines.append(
            f"{row.date.strftime('%d.%m.%Y')} "
            f"({row.weekday_abbr})"
            f"{tags_text} - "
            f"{row.status} | "
            f"Odpracováno: {worked_hours:.2f}h | "
            f"Rozvrh: {scheduled_hours:.2f}h | "
            f"Odchylka: {format_signed_hours(deviation_hours)}"
        )

    total_worked = float(
        report["official_hours_worked"].sum()
    )

    total_scheduled = float(
        report["scheduled_hours"].sum()
    )

    total_deviation = float(
        report["deviation_hours"].sum()
    )

    lines.extend([
        "",
        "=== CELKOVÝ SOUHRN ===",
        f"Celkem odpracováno: {total_worked:.2f}h",
        f"Celkem podle rozvrhu: {total_scheduled:.2f}h",
        (
            "Celková odchylka: "
            f"{format_signed_hours(total_deviation)}"
        ),
    ])

    report_text = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # utf-8-sig je bezpečné pro české znaky i Windows Notepad.
    output_path.write_text(
        report_text,
        encoding="utf-8-sig",
    )

    return report_text







report, selected_user = build_user_daily_report(
    users=df_users,
    user_day=df_user_day,
    holidays=df_holidays,
    selected_user_id="vnfg855sgn5gm5h",
    start_date="2026-01-01",
    end_date="2026-02-10",
)

print(
    report[
        [
            "date",
            "weekday_abbr",
            "is_weekend",
            "is_holiday",
            "official_hours_worked",
            "scheduled_hours",
            "is_expected_workday",
            "status",
            "deviation_hours",
        ]
    ].to_string(index=False)
)