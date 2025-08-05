"""
ingest_matching_pennies.py
--------------------------
Bulk-ingest Matching-Pennies CSV logs into an SQLite database.

Usage
-----
    python ingest_matching_pennies.py  /data/csv_root   matching_pennies.sqlite
"""

import sys, re, pathlib, sqlite3, itertools
import pandas as pd
from datetime import datetime

###############################################################################
# 1 ── Config
###############################################################################

# Regex that extracts rat-tag and timestamp from the filename
FNAME_REGEX = re.compile(
    r"(?P<rat>P\d+?)_(?P<Y>\d{4})_(?P<m>\d{2})_(?P<d>\d{2})_"
    r"(?P<H>\d{2})_(?P<M>\d{2})_(?P<S>\d{2})\.csv$"
)

# Map CSV column → DB column; tweak once, keep forever
COLUMN_MAP = {
    "trial_number"     : "trial_idx",
    "choice"           : "choice",        # 0 = Left, 1 = Right
    "rewarded"         : "rewarded",      # 0/1
    "nosepoke_ts"      : "nosepoke_ts",
    "feeder_ts"        : "feeder_ts",
    "rt_ms"            : "rt_ms",
    "iti_ms"           : "iti_ms",
    "explored"         : "explored",
    "licks"            : "licks",
    # add more if you need them in the DB
}

CHUNK_SIZE = 50_000            # trials per SQL batch


###############################################################################
# 2 ── Schema helpers
###############################################################################

CREATE_TABLES_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS animal (
    id       INTEGER PRIMARY KEY,
    rat_tag  TEXT UNIQUE
);

CREATE TABLE IF NOT EXISTS session (
    id              INTEGER PRIMARY KEY,
    animal_id       INTEGER NOT NULL,
    session_ts      TEXT NOT NULL,               -- ISO 8601
    hallway         TEXT,
    UNIQUE (animal_id, session_ts),
    FOREIGN KEY (animal_id) REFERENCES animal(id)
);

CREATE TABLE IF NOT EXISTS trial (
    id          INTEGER PRIMARY KEY,
    session_id  INTEGER NOT NULL,
    trial_idx   INTEGER NOT NULL,
    choice      INTEGER,
    rewarded    INTEGER,
    nosepoke_ts REAL,
    feeder_ts   REAL,
    rt_ms       REAL,
    iti_ms      REAL,
    explored    INTEGER,
    licks       INTEGER,
    UNIQUE (session_id, trial_idx),
    FOREIGN KEY (session_id) REFERENCES session(id)
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Create schema and switch to WAL mode for speed."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.executescript(CREATE_TABLES_SQL)
    return conn


###############################################################################
# 3 ── ETL helpers
###############################################################################

def parse_filename(path: pathlib.Path):
    m = FNAME_REGEX.search(path.name)
    if not m:
        raise ValueError(f"Filename not recognised: {path}")
    ts = datetime(
        int(m["Y"]), int(m["m"]), int(m["d"]),
        int(m["H"]), int(m["M"]), int(m["S"])
    )
    return m["rat"], ts.isoformat(sep=" ")


def get_or_create(cursor: sqlite3.Cursor, table: str, column: str, value):
    """Insert value if it doesn't exist and return the PK."""
    cursor.execute(
        f"INSERT INTO {table} ({column}) VALUES (?) "
        f"ON CONFLICT({column}) DO NOTHING",
        (value,)
    )
    cursor.execute(f"SELECT id FROM {table} WHERE {column} = ?", (value,))
    return cursor.fetchone()[0]


def insert_trials(cursor: sqlite3.Cursor, session_id: int, df: pd.DataFrame):
    cols_csv, cols_db = zip(*COLUMN_MAP.items())
    rows = itertools.zip_longest(
        [],                                   # placeholder for session_id
        df.loc[:, cols_csv].itertuples(index=False, name=None),
        fillvalue=session_id
    )
    cursor.executemany(
        f"""INSERT INTO trial
                (session_id, {", ".join(cols_db)})
            VALUES
                ({", ".join("?" * (len(cols_db) + 1))})
            ON CONFLICT(session_id, trial_idx) DO NOTHING
        """,
        rows
    )


###############################################################################
# 4 ── Main ingest loop
###############################################################################

def ingest_csv(conn: sqlite3.Connection, csv_path: pathlib.Path):
    rat_tag, session_ts = parse_filename(csv_path)
    print(f"→  {csv_path.name}  ::  {rat_tag}  {session_ts}")

    with conn:                              # single transaction
        cur = conn.cursor()

        # 1) animal
        animal_id = get_or_create(cur, "animal", "rat_tag", rat_tag)

        # 2) session
        cur.execute(
            """INSERT INTO session (animal_id, session_ts)
               VALUES (?, ?)
               ON CONFLICT(animal_id, session_ts) DO NOTHING""",
            (animal_id, session_ts)
        )
        cur.execute(
            "SELECT id FROM session WHERE animal_id = ? AND session_ts = ?",
            (animal_id, session_ts)
        )
        session_id = cur.fetchone()[0]

        # 3) trials (chunked)
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE):
            # rename & keep only mapped columns
            chunk = (
                chunk
                .rename(columns=COLUMN_MAP)
                .loc[:, list(COLUMN_MAP.values())]
            )
            insert_trials(cur, session_id, chunk)


def main(csv_root: pathlib.Path, db_path: pathlib.Path):
    conn = init_db(db_path)

    csv_files = sorted(csv_root.rglob("*.csv"))
    if not csv_files:
        print("No CSV files found.")
        return

    for f in csv_files:
        try:
            ingest_csv(conn, f)
        except Exception as exc:
            print(f"!!  {f} skipped  ({exc})")


###############################################################################
# CLI entry-point
###############################################################################

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:  python ingest_matching_pennies.py  <csv_dir>  <db.sqlite>")
        sys.exit(1)

    csv_dir = pathlib.Path(sys.argv[1]).expanduser().resolve()
    db_file = pathlib.Path(sys.argv[2]).expanduser().resolve()
    main(csv_dir, db_file)
    print("✓  Ingest complete.")
