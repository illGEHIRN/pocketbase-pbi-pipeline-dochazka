import argparse
from pathlib import Path

import pandas as pd

from config import PROCESSED_DATA_DIR
from src.formatted_exporter import (
    build_user_daily_report,
    export_report_to_txt,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export attendance report for one user "
            "and a selected date range."
        )
    )

    parser.add_argument(
        "user_id",
        help="User ID, for example vnfg855sgn5gm5h.",
    )

    parser.add_argument(
        "start_date",
        help="Beginning of the report in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "end_date",
        help="End of the report in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help=(
            "Optional output TXT path. "
            "When omitted, a filename is generated automatically."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    # Load processed ETL outputs.
    df_users = pd.read_csv(
        PROCESSED_DATA_DIR / "dim_users.csv",
        dtype={"user_id": "string"},
    )

    df_user_day = pd.read_csv(
        PROCESSED_DATA_DIR / "fact_user_day.csv",
        dtype={"user_id": "string"},
    )

    df_holidays = pd.read_csv(
        PROCESSED_DATA_DIR / "dim_holidays.csv",
    )

    report, selected_user = build_user_daily_report(
        users=df_users,
        user_day=df_user_day,
        holidays=df_holidays,
        selected_user_id=args.user_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    if args.output is not None:
        output_path = args.output
    else:
        full_name = str(selected_user["name"]).strip()

        # Names are stored as "Surname Firstname"
        surname = full_name.split()[0]

        output_path = (
            PROCESSED_DATA_DIR
            / "attendance_exports"
            / (
                f"{surname}_{args.user_id}_"
                f"{args.start_date}_{args.end_date}.txt"
            )
        )

    export_report_to_txt(
        report=report,
        selected_user=selected_user,
        selected_user_id=args.user_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
    )

    print(f"Report successfully created: {output_path.resolve()}")


if __name__ == "__main__":
    main()