# StudyDataset: Hierarchical Loader for Matching Pennies Data

This module provides a lightweight, **hierarchical data container** for Matching Pennies experiments. 
It supports **multiple experiments** (e.g., `PFC_lesion`) and **multiple paradigms** (e.g., `normal`, `hallway_swap`), 
with dot-notation access and robust, hassle-free CSV loading.

```
StudyDataset
└── Experiment ("PFC_lesion")
    ├── Paradigm ("normal")
    │   ├── Animal ("P4637") → [Session, Session, ...]
    │   └── Animal ("P4638") → [...]
    └── Paradigm ("hallway_swap")
        └── Animal ("P4637") → [...]
```

Key features:
- **Dot-notation** access: `study.PFC_lesion.normal.P4637`
- **Robust CSV reader**: tries multiple engines/encodings, skips and logs unreadable files
- **Flexible treatment map**: merges group labels (e.g., `sham`, `mPFC`, `OFC`) from a CSV with many possible header names
- **One-liner aggregation** with metadata columns via `study.all_trials(...)`
- Simple iteration helpers for filtered loops over sessions

---

## Installation / Project Layout

Keep the module under a package (e.g., `src/database/`) with an `__init__.py` file:
```
repo/
  src/
    analysis.py
    database/
      __init__.py
      DataFrameDataBase.py   # (this module)
```

Run your analysis script from the **repo root**:
```bash
python -m src.analysis
```
This ensures `src` is discoverable on `sys.path`. Alternatively, set `PYTHONPATH` or do an editable install (`pip install -e .`).

---

## Quickstart

```python
from database.DataFrameDataBase import StudyDataset

study = StudyDataset.from_roots(
    roots=[
        ("PFC_lesion", "normal", "/path/to/normal_mp"),
        ("PFC_lesion", "hallway_swap", "/path/to/switching_halls"),
    ],
    animal_treatment_map_csv="/path/to/animal_important_dates_csv.csv",
    filename_regex=r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
    low_memory=False,  # OK for the C-engine; will be ignored for Python engine
)

# Dot-notation access
normal = study.PFC_lesion.normal
p4637_sessions = normal.P4637.sessions

# Aggregate all 'normal' trials with metadata columns
normal_df = study.all_trials(experiment="PFC_lesion", paradigm="normal")
print(normal_df.columns)
# => includes: _experiment, _paradigm, _animal, _treatment, _session_idx, _trial_idx

# Filter by treatment
sham_df = study.all_trials(experiment="PFC_lesion", paradigm="normal", treatment="sham")
```

---

## Treatment Map

Pass a CSV path via `animal_treatment_map_csv=`. The loader tries to find suitable columns by normalizing header names. 
**ID candidates:** `animal_id`, `animal`, `id`, `subject_id`, `subject`, `rat_id`, `rat`  
**Treatment candidates:** `treatment`, `group`, `lesion`, `lesion_group`, `surgery`, `surgery_group`, `condition`, `cohort`

> If no suitable columns are found, a warning is printed and loading proceeds without treatments.

---

## File Name Pattern

You must provide a regex with **named groups**:  
- `(?P<animal>...)` — the animal identifier, e.g., `P4637`  
- `(?P<date>...)`   — optional; used to sort sessions chronologically  

Example:
```python
filename_regex = r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$"
```

---

## Robust CSV Loading

The helper `_read_csv_robust` tries reasonable combinations of **engines** (`"c"`, `"python"`) and **encodings** (`utf-8`, `latin-1`, etc.).  
- If the Python engine is used, unsupported options like `low_memory` are **dropped automatically**.  
- If all attempts fail, the file is **skipped** with a clear log message including its path.  

You can forward any `pandas.read_csv` kwargs via `StudyDataset.from_roots(..., **read_csv_kwargs)`.

Examples:
```python
StudyDataset.from_roots(..., on_bad_lines="skip", dtype={"ColA": "Int64"})
```

---

## API Reference

### `StudyDataset.from_roots(roots, filename_regex, animal_treatment_map_csv=None, **read_csv_kwargs)`
Build a study from multiple roots.  
- **roots**: list of `(experiment_name, paradigm_name, folder_path)` tuples  
- **filename_regex**: must include a named `animal` group; optional `date` group  
- **animal_treatment_map_csv**: optional CSV with animal IDs and treatment groups  
- `**read_csv_kwargs`: passed through to `pandas.read_csv` (robustly)

**Returns**: `StudyDataset`

### `StudyDataset.iter_sessions(experiment=None, paradigm=None, treatment=None)`
Yield `Session` objects, optionally filtered.  
**Yields**: `Session`

### `StudyDataset.all_trials(experiment=None, paradigm=None, treatment=None, sort=False)`
Concatenate all trials (rows) into one `DataFrame`. Adds metadata:  
`_experiment, _paradigm, _animal, _treatment, _session_idx, _trial_idx`.

**Returns**: `pandas.DataFrame`

### `Animal`
- `animal_id`: the ID (e.g., `P4637`)
- `treatment`: merged from treatment map
- `sessions`: list of `Session`
- `concat(sort=False)`: per-animal aggregation helper

### `Session`
- `animal_id`, `path`, `df`, `session_num`, `date`

---

## Common Recipes

**Count trials per animal in a paradigm**
```python
df = study.all_trials(experiment="PFC_lesion", paradigm="normal")
counts = df.groupby("_animal")["_trial_idx"].count().sort_values(ascending=False)
```

**Add a per-session trial index to an existing DataFrame**
```python
df["_trial_idx"] = df.groupby(["_animal", "_session_idx"]).cumcount() + 1
```

**Iterate only mPFC animals in `hallway_swap`**
```python
for sess in study.iter_sessions("PFC_lesion", "hallway_swap", "mPFC"):
    print(sess.animal_id, sess.session_num, sess.path.name, len(sess.df))
```

---

## Troubleshooting

- `ModuleNotFoundError: No module named 'database'`  
  Ensure `src/database/__init__.py` exists and run scripts from the repo root (`python -m src.analysis`) or set `PYTHONPATH`.

- `ValueError: 'low_memory' not supported with 'python' engine`  
  The loader drops `low_memory` automatically when it switches to the Python engine. You can still pass it; it will be respected when the C engine is used.

- `⚠️ Skipping file due to encoding error` / `failed all engine/encoding attempts`  
  The file is malformed or in an unusual encoding. Fix the file or extend the encodings passed to `_read_csv_robust`.

---

## Performance Tips

- Add `dtype=` to avoid type inference overhead on large CSVs.
- Use `on_bad_lines="skip"` for gnarly files.
- Consider **lazy loading** if memory gets tight (store file paths in `Session` and load on demand).
- For extremely large datasets, pre-ingest into a columnar format (e.g., Parquet) and adapt `_read_csv_robust` accordingly.

---

## License / Attribution

This module is a lightweight pattern for research use. No warranty. Feel free to copy/adapt.
