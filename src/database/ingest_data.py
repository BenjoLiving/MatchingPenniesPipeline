"""
Ingestion script for rodent trial data.

This module provides functions to build a SQLite database from a collection
of CSV files produced by the operant conditioning system. Each CSV file
corresponds to a single training session for one animal. The filename
encodes both the subject identifier and the start time of the session.

The database schema is intentionally simple and flexible:

* **experiments** – high‑level grouping of animals. Each experiment has a
  unique name. Experiments are created on the fly when encountering
  previously unseen experiment names in the mapping file.

* **treatments** – within an experiment, animals can belong to different
  treatment groups. The mapping file specifies which treatment each
  subject received. Treatments are unique per experiment.

* **animals** – every subject is registered under a particular
  experiment and treatment. An entry in the ``animals`` table ensures
  uniqueness of each animal within an experiment.

* **sessions** – a single CSV file corresponds to a single session. A
  session is uniquely identified by a combination of ``animal_id`` and
  the timestamp parsed from the filename. The ``sessions`` table stores
  metadata about the session, including a reference back to the animal,
  the experiment, and the original filename.

* **trials** – each row of the CSV file is inserted into the ``trials``
  table. Trials reference their parent session via ``session_id``. To
  keep the schema lightweight and future‑proof, only a handful of
  frequently used fields are stored in dedicated columns. The full
  original row is preserved as a JSON blob in the ``data_json`` column.

The ingestion functions are idempotent: if a session has already been
imported, it will be skipped rather than duplicated. Duplicate detection
uses the combination of ``animal_id`` and session timestamp. If an
unknown subject appears in a CSV file and the mapping file does not list
that subject, an exception is raised unless ``skip_unlisted_ids`` is
set to ``True``.

Usage example:

```
python ingest_data.py /path/to/folder mapping.csv --db rat_trials.db
```

The ``mapping.csv`` file must contain at least the columns ``experiment``,
``animal_id`` and ``treatment``. Additional columns are ignored.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List

import pandas as pd


@dataclass
class MappingEntry:
    """Simple container for animal to experiment/treatment mapping."""

    experiment: str
    treatment: str


def parse_filename(filename: str) -> Tuple[str, datetime]:
    """Parse the CSV filename to extract the subject ID and start time.

    Filenames are expected to follow the convention
    ``P<animal_id>_<YYYY>_<MM>_<DD>_<HH>_<MM>_<SS>.csv``. For example
    ``P4677_2024_10_06_12_37_31.csv`` corresponds to subject ``P4677``
    starting at 2024‑10‑06 12:37:31.

    Args:
        filename: Name of the CSV file.

    Returns:
        Tuple of the subject identifier (e.g. ``"P4677"``) and a
        ``datetime`` object representing the start time.

    Raises:
        ValueError: if the filename does not conform to the expected
            pattern.
    """
    base = os.path.basename(filename)
    # Accept filenames of the form
    #   P####_YYYY_MM_DD_HH_MM_SS.csv
    # There should be seven underscore‑separated segments: the subject
    # identifier and six time components (year, month, day, hour,
    # minute, second). The regular expression captures the subject
    # number and each component separately.
    match = re.match(
        r'^P(\d{4})_(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})\.csv$',
        base,
    )
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")
    animal_num = match.group(1)
    # Extract temporal components
    year, month, day, hour, minute, second = map(int, match.groups()[1:])
    dt = datetime(year, month, day, hour, minute, second)
    return f"P{animal_num}", dt


def load_mapping_file(path: str) -> Dict[str, MappingEntry]:
    """Load the experiment and treatment assignments from a mapping file.

    The file can be comma‑separated or tab‑separated. It must contain at
    least three columns: ``experiment``, ``animal_id`` and ``treatment``.
    Values are stripped of leading/trailing whitespace. Additional
    columns are ignored.

    Args:
        path: Path to the mapping file.

    Returns:
        A dictionary keyed by ``animal_id`` with values containing
        ``experiment`` and ``treatment`` assignments.

    Raises:
        ValueError: if the required columns are not present.
    """
    mapping: Dict[str, MappingEntry] = {}
    # Attempt to detect delimiter (comma or tab) using the csv.Sniffer.
    with open(path, newline='') as f:
        sample = f.read(2048)
        f.seek(0)
        # Default dialect is comma; attempt to sniff delimiters
        delim = ','
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',\t')
            delim = dialect.delimiter
        except Exception:
            # fallback: if header contains tabs, assume tab delimiter
            if '\t' in sample.splitlines()[0]:
                delim = '\t'
        reader = csv.DictReader(f, delimiter=delim)
        required = {"experiment", "animal_id", "treatment"}
        if reader.fieldnames is None:
            raise ValueError("Mapping file appears empty or invalid.")
        missing = required - set([h.strip() for h in reader.fieldnames])
        if missing:
            raise ValueError(
                f"Mapping file is missing required columns: {', '.join(sorted(missing))}"
            )
        for row in reader:
            # Normalise keys by stripping whitespace
            # Some mapping files might include extra whitespace around
            # column names. Use row.get rather than direct indexing to
            # avoid KeyError.
            animal = row.get("animal_id")
            experiment = row.get("experiment")
            treatment = row.get("treatment")
            if animal is None or experiment is None or treatment is None:
                continue
            animal = animal.strip()
            experiment = experiment.strip()
            treatment = treatment.strip()
            mapping[animal] = MappingEntry(experiment=experiment, treatment=treatment)
    return mapping


def init_database(conn: sqlite3.Connection) -> None:
    """Initialise database tables if they do not already exist."""
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS treatments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            UNIQUE(experiment_id, name),
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS animals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_id TEXT NOT NULL,
            experiment_id INTEGER NOT NULL,
            treatment_id INTEGER NOT NULL,
            UNIQUE(animal_id, experiment_id),
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
            FOREIGN KEY(treatment_id) REFERENCES treatments(id) ON DELETE CASCADE
        );
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            animal_id INTEGER NOT NULL,
            experiment_id INTEGER NOT NULL,
            session_datetime TEXT NOT NULL,
            file_name TEXT NOT NULL,
            ses_num INTEGER NOT NULL,
            UNIQUE(animal_id, session_datetime),
            FOREIGN KEY(animal_id) REFERENCES animals(id) ON DELETE CASCADE,
            FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
        );
        """
    )
    # If the sessions table pre‑exists from a previous version without
    # the ses_num column, attempt to add it. The ALTER TABLE will fail
    # silently if the column already exists.
    try:
        cur.execute("ALTER TABLE sessions ADD COLUMN ses_num INTEGER")
    except sqlite3.OperationalError:
        pass
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trials (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            trial_index INTEGER NOT NULL,
            well_id TEXT,
            rewarded_well TEXT,
            error_flg TEXT,
            data_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def get_or_create_experiment(conn: sqlite3.Connection, name: str) -> int:
    cur = conn.cursor()
    cur.execute("SELECT id FROM experiments WHERE name = ?", (name,))
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute("INSERT INTO experiments (name) VALUES (?)", (name,))
    conn.commit()
    return cur.lastrowid


def get_or_create_treatment(conn: sqlite3.Connection, experiment_id: int, name: str) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM treatments WHERE experiment_id = ? AND name = ?",
        (experiment_id, name),
    )
    row = cur.fetchone()
    if row:
        return row[0]
    cur.execute(
        "INSERT INTO treatments (experiment_id, name) VALUES (?, ?)",
        (experiment_id, name),
    )
    conn.commit()
    return cur.lastrowid


def get_or_create_animal(
    conn: sqlite3.Connection, animal_id: str, experiment_id: int, treatment_id: int
) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT id FROM animals WHERE animal_id = ? AND experiment_id = ?",
        (animal_id, experiment_id),
    )
    row = cur.fetchone()
    if row:
        # update treatment if changed
        cur.execute(
            "UPDATE animals SET treatment_id = ? WHERE id = ?",
            (treatment_id, row[0]),
        )
        conn.commit()
        return row[0]
    cur.execute(
        "INSERT INTO animals (animal_id, experiment_id, treatment_id) VALUES (?, ?, ?)",
        (animal_id, experiment_id, treatment_id),
    )
    conn.commit()
    return cur.lastrowid


def session_exists(
    conn: sqlite3.Connection, animal_db_id: int, dt: datetime
) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sessions WHERE animal_id = ? AND session_datetime = ?",
        (animal_db_id, dt.isoformat()),
    )
    return cur.fetchone() is not None


def insert_session(
    conn: sqlite3.Connection,
    animal_db_id: int,
    experiment_id: int,
    dt: datetime,
    file_name: str,
) -> int:
    cur = conn.cursor()
    # Determine the next session number for this animal. Sessions are
    # numbered sequentially in chronological order per animal, starting
    # from 1. Use the existing maximum to assign the next number.
    cur.execute(
        "SELECT MAX(ses_num) FROM sessions WHERE animal_id = ?",
        (animal_db_id,),
    )
    row = cur.fetchone()
    next_num = (row[0] or 0) + 1
    cur.execute(
        """
        INSERT INTO sessions (animal_id, experiment_id, session_datetime, file_name, ses_num)
        VALUES (?, ?, ?, ?, ?)
        """,
        (animal_db_id, experiment_id, dt.isoformat(), file_name, next_num),
    )
    conn.commit()
    return cur.lastrowid


def insert_trials(
    conn: sqlite3.Connection,
    session_id: int,
    df: pd.DataFrame,
) -> None:
    """Insert rows from a session DataFrame into the trials table.

    Only a subset of columns is explicitly stored (``well_id``,
    ``rewarded_well`` and ``error_flg``). The full row is stored as a
    JSON blob for later processing.

    Args:
        conn: Open database connection.
        session_id: The parent session identifier.
        df: A pandas DataFrame containing the trials for a single session.
    """
    cur = conn.cursor()
    columns = [c.strip() for c in df.columns]
    df = df.copy()
    df.columns = columns
    for i, row in df.iterrows():
        data_json = row.to_json()  # preserves all original columns
        well_id = row.get('well_id')
        rewarded_well = row.get('rewarded_well')
        error_flg = row.get('error_flg')
        # ensure values are strings or None
        well_id = None if pd.isna(well_id) or well_id == '' else str(well_id)
        rewarded_well = None if pd.isna(rewarded_well) or rewarded_well == '' else str(rewarded_well)
        error_flg_val = None if pd.isna(error_flg) or error_flg == '' else str(error_flg)
        trial_index = i + 1
        cur.execute(
            """
            INSERT INTO trials (session_id, trial_index, well_id, rewarded_well, error_flg, data_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, trial_index, well_id, rewarded_well, error_flg_val, data_json),
        )
    conn.commit()


def ingest_folder(
    folder: str,
    mapping_csv: str,
    db_path: str = 'rat_trials.db',
    skip_unlisted_ids: bool = False,
) -> None:
    """Ingest all CSV files in a directory into the database.

    Args:
        folder: Path to a directory containing CSV files. Only files
            matching the expected filename pattern are processed.
        mapping_csv: Path to the mapping CSV that associates subjects
            with experiments and treatments.
        db_path: Location of the SQLite database file. The file will
            be created if it does not already exist.
        skip_unlisted_ids: If ``True``, silently skip CSV files whose
            subject is not listed in the mapping file. If ``False``, a
            ``ValueError`` is raised on encountering an unlisted subject.
    """
    mapping = load_mapping_file(mapping_csv)
    conn = sqlite3.connect(db_path)
    init_database(conn)

    # discover CSV files in folder
    for fname in sorted(os.listdir(folder)):
        if not fname.lower().endswith('.csv'):
            continue
        try:
            animal_id, dt = parse_filename(fname)
        except ValueError:
            # skip files that do not match naming convention
            continue
        if animal_id not in mapping:
            if skip_unlisted_ids:
                print(f"Skipping unlisted subject {animal_id} in file {fname}")
                continue
            else:
                raise ValueError(
                    f"Animal ID {animal_id} found in file {fname} is not present in mapping file."
                )
        map_entry = mapping[animal_id]
        # ensure experiment exists
        experiment_id = get_or_create_experiment(conn, map_entry.experiment)
        # ensure treatment exists
        treatment_id = get_or_create_treatment(conn, experiment_id, map_entry.treatment)
        # ensure animal exists
        animal_db_id = get_or_create_animal(conn, animal_id, experiment_id, treatment_id)
        # check duplicate session
        if session_exists(conn, animal_db_id, dt):
            print(f"Session for {animal_id} at {dt} already exists – skipping {fname}.")
            continue
        # insert session record
        session_id = insert_session(conn, animal_db_id, experiment_id, dt, fname)
        # read CSV data with robust encoding handling. Default to UTF‑8
        # and fall back to Latin‑1 (ISO‑8859‑1) if decoding fails. If
        # reading still fails for any other reason, report the file and
        # continue without aborting the ingestion process.
        csv_path = os.path.join(folder, fname)
        try:
            try:
                df = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                df = pd.read_csv(csv_path, encoding='ISO-8859-1')
        except Exception as exc:
            print(f"Failed to read {fname}: {exc}. Skipping this file.")
            # Remove the session record we just inserted since we can't
            # populate trials for it
            cur = conn.cursor()
            cur.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
            continue
        insert_trials(conn, session_id, df)
        print(f"Imported {fname} (session_id={session_id})")
    conn.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Ingest rodent trial CSV files into a SQLite database.')
    parser.add_argument('folder', help='Directory containing CSV files to ingest')
    parser.add_argument('mapping_csv', help='CSV file defining experiment/treatment assignments')
    parser.add_argument('--db', default='rat_trials.db', help='Path to SQLite database (default: rat_trials.db)')
    parser.add_argument('--skip-unlisted-ids', action='store_true', help='Silently skip files whose subject is not listed in the mapping file')
    args = parser.parse_args()
    ingest_folder(args.folder, args.mapping_csv, db_path=args.db, skip_unlisted_ids=args.skip_unlisted_ids)