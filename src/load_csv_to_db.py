"""
Load synthetic ticket dataset from CSV into the SQLite database.

Owner: R2 – Backend/BD (Camilo)
"""

import csv
from pathlib import Path

from db_utils import insert_raw_ticket

DATA_PATH = Path("data/tickets_train.csv")


def load_csv_to_raw() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            insert_raw_ticket(
                client_name=row.get("client_name", "Cliente N/A"),
                project_name=row.get("project_name", "Proyecto N/A"),
                channel=row.get("channel", "email"),
                original_text=row.get("text", ""),
            )
            count += 1

    print(f"✅ {count} tickets loaded into raw_tickets from {DATA_PATH}")


if __name__ == "__main__":
    load_csv_to_raw()
