import argparse
from pathlib import Path
import unicodedata

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
        "user_identifier",
        help=(
            "Exact user ID or surname without diacritics, all lowercase,"
            "for example vnfg855sgn5gm5h or boucnik."
        ),
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

def normalize_surname(value: str) -> str:
    """
    Convert a surname to lowercase ASCII without diacritics.

    Examples:
        Boučník -> boucnik
        Šmatera -> smatera
        Dvořáková -> dvorakova
    """
    normalized = unicodedata.normalize(
        "NFKD",
        str(value).strip(),
    )

    without_diacritics = "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )

    # Remove spaces, hyphens, apostrophes and other punctuation.
    return "".join(
        character
        for character in without_diacritics.casefold()
        if character.isalnum()
    )


def resolve_user_id(
    users: pd.DataFrame,
    user_identifier: str,
) -> str:
    """
    Resolve either an exact user_id or a normalized surname
    to one user_id.
    """
    identifier = str(user_identifier).strip()

    # First try an exact user_id match.
    id_matches = users.loc[
        users["user_id"].astype("string").str.casefold()
        == identifier.casefold()
    ]

    if len(id_matches) == 1:
        return str(id_matches.iloc[0]["user_id"])

    if len(id_matches) > 1:
        raise ValueError(
            f"User ID {identifier!r} occurs more than once."
        )

    # Extract the first part of "Surname Firstname".
    surname_keys = (
        users["name"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.split()
        .str[0]
        .map(normalize_surname)
    )

    normalized_identifier = normalize_surname(identifier)

    surname_matches = users.loc[
        surname_keys == normalized_identifier
    ]

    if surname_matches.empty:
        raise ValueError(
            f"No user found for ID or surname {identifier!r}."
        )

    if len(surname_matches) > 1:
        candidates = surname_matches[
            ["name", "user_id"]
        ].to_string(index=False)

        raise ValueError(
            f"More than one user has surname {identifier!r}:\n"
            f"{candidates}\n\n"
            "Use the exact user_id instead."
        )

    return str(surname_matches.iloc[0]["user_id"])

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

    selected_user_id = resolve_user_id(
    users=df_users,
    user_identifier=args.user_identifier,
)

    report, selected_user = build_user_daily_report(
        users=df_users,
        user_day=df_user_day,
        holidays=df_holidays,
        selected_user_id=selected_user_id,
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
                f"{surname}_{selected_user_id}_"
                f"{args.start_date}_{args.end_date}.txt"
            )
        )

    export_report_to_txt(
        report=report,
        selected_user=selected_user,
        selected_user_id=selected_user_id,
        start_date=args.start_date,
        end_date=args.end_date,
        output_path=output_path,
    )

    print(f"Report successfully created: {output_path.resolve()}")


if __name__ == "__main__":
    main()