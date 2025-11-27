import sqlite3
from pathlib import Path

DB_PATH = Path("support_ai.db")
SCHEMA_PATH = Path("sql/schema.sql")

def init_db():
    if not SCHEMA_PATH.exists():
        raise FileNotFoundError(f"Schema file not found: {SCHEMA_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        with SCHEMA_PATH.open("r", encoding="utf-8") as f:
            schema_sql = f.read()
        conn.executescript(schema_sql)

    print(f"✅ Base de datos inicializada en {DB_PATH.resolve()}")

if __name__ == "__main__":
    init_db()
