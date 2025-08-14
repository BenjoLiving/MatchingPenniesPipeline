"""
StudyDataset: hierarchical loader/aggregator for Matching Pennies datasets.

This module organizes data as:
    StudyDataset
      └── Experiment (e.g., "PFC_lesion")
          └── Paradigm (e.g., "normal", "hallway_swap")
              └── Animal (e.g., "P4637")
                  └── Session (CSV → DataFrame)

Key features
------------
- Dot-notation access at each level (e.g., study.PFC_lesion.normal.P4637).
- Robust CSV reading with engine/encoding fallbacks; logs & skips unreadable files.
- Optional treatment map integration (accepts many header variants).
- One-liner aggregation to a single DataFrame with metadata columns.

Quickstart
----------
>>> study = StudyDataset.from_roots(
...     roots=[
...         ("PFC_lesion", "normal", "/path/to/normal_mp"),
...         ("PFC_lesion", "hallway_swap", "/path/to/switching_halls"),
...     ],
...     animal_treatment_map_csv="/path/to/animal_important_dates_csv.csv",
...     filename_regex=r"(?P<animal>P\\d{4,})_(?P<date>\\d{4}_\\d{2}_\\d{2}).*\\.csv$",
...     low_memory=False,  # honored for C-engine, auto-dropped for Python engine
... )
>>> df = study.all_trials(experiment="PFC_lesion", paradigm="normal")
>>> df.columns
Index([... '_experiment','_paradigm','_animal','_treatment','_session_idx','_trial_idx', ...])
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import re
import pandas as pd
from mp_metrics import *

# =============================
# Utilities
# =============================

def _safe_attr(name: str, existing: dict) -> str:
    """
    Produce a valid, non-colliding Python attribute name for dot-notation.

    Parameters
    ----------
    name : str
        The desired attribute name (e.g., "P4637" or "hallway-swap").
    existing : dict
        The target object's __dict__ used to avoid collisions.

    Returns
    -------
    str
        A sanitized attribute name that is a valid identifier and does not
        exist in `existing`.

    Notes
    -----
    - Non-word characters are replaced with underscores.
    - If the name starts with a digit, it is prefixed with 'n_'.
    - If there is still a collision, a numeric suffix (_2, _3, ...) is added.
    """
    if name.isidentifier() and name not in existing:
        return name
    alias = re.sub(r"\W", "_", name)
    if not alias or alias[0].isdigit():
        alias = f"n_{alias}"
    base, i = alias, 2
    while base in existing:
        base = f"{alias}_{i}"
        i += 1
    return base


def _norm(s: str) -> str:
    """
    Normalize header names for tolerant column matching.

    Examples
    --------
    >>> _norm("Animal ID")
    'animalid'
    >>> _norm("Lesion Group")
    'lesiongroup'
    """
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _find_col(cols: Iterable[str], candidates_norm: List[str]) -> Optional[str]:
    """
    Find a column in `cols` whose normalized form matches any candidate.

    Parameters
    ----------
    cols : Iterable[str]
        Column names to search.
    candidates_norm : list[str]
        Candidate *normalized* names (e.g., ['animalid','animal','id']).

    Returns
    -------
    Optional[str]
        The original column name if found, else None.
    """
    nmap = {_norm(c): c for c in cols}
    for cand in candidates_norm:
        if cand in nmap:
            return nmap[cand]
    return None

# Normalize a header to compare case/whitespace-insensitively
def _norm_name(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    """Return the actual df column whose normalized form matches `name` (ignoring spaces/case)."""
    nmap = {_norm_name(c): c for c in df.columns}
    return nmap.get(_norm_name(name))


def _read_csv_robust(
    path: str | Path,
    engine_preference: Tuple[str, ...] = ("c", "python"),
    encodings: Tuple[Optional[str], ...] = (None, "utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"),
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Read a CSV with multiple engine/encoding fallbacks.

    Tries combinations of pandas engines and encodings; drops options that are
    not supported by a given engine (e.g., 'low_memory' for the Python engine).
    If all attempts fail, returns None (caller can log & skip).

    Parameters
    ----------
    path : str | Path
        CSV file path.
    engine_preference : tuple[str, ...], optional
        Engines to try in order. Default ("c", "python").
    encodings : tuple[Optional[str], ...], optional
        Encodings to try in order. Default includes UTF-8, latin-1, etc.
    **kwargs
        Extra pandas.read_csv kwargs, forwarded per attempt.

    Returns
    -------
    Optional[pandas.DataFrame]
        The loaded DataFrame, or None if every attempt failed.
    """
    for eng in engine_preference:
        # copy kwargs and drop options not supported by this engine
        k = dict(kwargs)
        if eng == "python":
            # python engine doesn't support low_memory, etc.
            k.pop("low_memory", None)

        for enc in encodings:
            try:
                return pd.read_csv(path, engine=eng, encoding=enc, **k)
            except UnicodeDecodeError:
                # try another encoding/engine
                continue
            except ValueError as e:
                # e.g., "low_memory not supported with the 'python' engine"
                msg = str(e).lower()
                if "low_memory" in msg:
                    k.pop("low_memory", None)
                    try:
                        return pd.read_csv(path, engine=eng, encoding=enc, **k)
                    except Exception:
                        continue
                continue
            except Exception:
                continue
    return None


# =============================
# Core containers
# =============================

@dataclass
class Session:
    """
    A single recording session (one CSV → one DataFrame) for an animal.

    Attributes
    ----------
    animal_id : str
        Animal identifier (e.g., 'P4637').
    path : pathlib.Path
        File path to the loaded CSV.
    df : pandas.DataFrame
        Trials table for this session.
    session_num : int | None
        1-based index *within the animal and paradigm*. Set by loader.
    date : str | None
        Optional date extracted from the filename via `filename_regex`.
    """
    animal_id: str
    path: Path
    df: pd.DataFrame
    session_num: Optional[int] = None   # within animal & paradigm
    date: Optional[str] = None          # optional from filename


@dataclass
class Animal:
    """
    Container for all sessions belonging to a single animal.

    Attributes
    ----------
    animal_id : str
        Identifier (e.g., 'P4637').
    sessions : list[Session]
        All sessions for this animal in the given paradigm.
    treatment : str | None
        Treatment/group label (e.g., 'sham', 'mPFC', 'OFC') if provided via map.
    """
    animal_id: str
    sessions: List[Session] = field(default_factory=list)
    treatment: Optional[str] = None     # set from treatment map CSV

    def __iter__(self) -> Iterator[Session]:
        """Iterate over Session objects."""
        return iter(self.sessions)

    def __len__(self) -> int:
        """Number of sessions for this animal."""
        return len(self.sessions)

    def concat(self, sort: bool = False) -> pd.DataFrame:
        """
        Concatenate trials across this animal's sessions.

        Parameters
        ----------
        sort : bool, optional
            Forwarded to pandas.concat(sort=...), by default False.

        Returns
        -------
        pandas.DataFrame
            Concatenated DataFrame with helper columns:
            `_animal` and `_session_idx` (per-animal).
        """
        if not self.sessions:
            return pd.DataFrame()
        return pd.concat(
            [s.df.assign(_animal=self.animal_id, _session_idx=i + 1)
             for i, s in enumerate(self.sessions)],
            ignore_index=True, sort=sort
        )


@dataclass
class Paradigm:
    """
    A paradigm inside an experiment (e.g., 'normal', 'hallway_swap').

    Attributes
    ----------
    name : str
        Paradigm name.
    animals : dict[str, Animal]
        Mapping of animal_id → Animal.
    """
    name: str
    animals: Dict[str, Animal] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Animal]:
        """Iterate over Animal objects."""
        return iter(self.animals.values())

    def __getitem__(self, animal_id: str) -> Animal:
        """Dict-style access to animals by ID."""
        return self.animals[animal_id]


@dataclass
class Experiment:
    """
    A single experiment grouping multiple paradigms (e.g., 'PFC_lesion').

    Attributes
    ----------
    name : str
        Experiment name.
    paradigms : dict[str, Paradigm]
        Mapping of paradigm name → Paradigm.
    """
    name: str
    paradigms: Dict[str, Paradigm] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Paradigm]:
        """Iterate over Paradigm objects."""
        return iter(self.paradigms.values())

    def __getitem__(self, paradigm_name: str) -> Paradigm:
        """Dict-style access to paradigms by name."""
        return self.paradigms[paradigm_name]


@dataclass
class StudyDataset:
    """
    Root container for many experiments & paradigms with dot-notation access.

    Attributes
    ----------
    experiments : dict[str, Experiment]
        Mapping of experiment name → Experiment.
    _attr_alias : dict[str, str]
        Internal map of dot-notation attribute → true name (for reference).

    Usage
    -----
    See module docstring or the `StudyDataset_Usage.md` guide.
    """
    experiments: Dict[str, Experiment] = field(default_factory=dict)
    _attr_alias: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # ---- creation ----
    @classmethod
    def from_roots(
        cls,
        roots: List[Tuple[str, str, str | Path]],
        filename_regex: str = r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
        animal_treatment_map_csv: Optional[str | Path] = None,
        engine: Optional[str] = None,  # kept for backward-compat; robust reader ignores it
        **read_csv_kwargs,
    ) -> "StudyDataset":
        """
        Build a `StudyDataset` by scanning multiple roots (experiment, paradigm, folder).

        Parameters
        ----------
        roots : list[tuple[str, str, str | Path]]
            Each tuple is (experiment_name, paradigm_name, folder_path).
        filename_regex : str, optional
            Regex that must capture a named 'animal' group, and optionally a 'date' group
            used for chronological session sorting. Defaults to:
            r"(?P<animal>P\\d{4,})_(?P<date>\\d{4}_\\d{2}_\\d{2}).*\\.csv$"
        animal_treatment_map_csv : str | Path | None, optional
            Optional CSV containing animal IDs and treatment/group names. The loader
            tolerates many header variants (e.g., 'Animal ID' ~ 'animalid', 'Group' ~ 'group').
        engine : str | None, optional
            Ignored by the robust reader; present for backward compatibility.
        **read_csv_kwargs :
            Extra keyword arguments forwarded to `pandas.read_csv`. They will be
            applied robustly (unsupported options are dropped for the Python engine).

        Returns
        -------
        StudyDataset
            A populated StudyDataset with dot-notation access to experiments/paradigms/animals.

        Notes
        -----
        - Files that fail all engine/encoding attempts are skipped with a warning.
        - Session numbering (`session_num`) is per-animal **within each paradigm**.
        """
        study = cls()

        # --- robust treatment map ---
        treatment_map: Dict[str, str] = {}
        if animal_treatment_map_csv is not None:
            try:
                tdf = pd.read_csv(animal_treatment_map_csv)
                aid_col = _find_col(
                    tdf.columns,
                    ["animalid", "animal", "id", "subjectid", "subject", "ratid", "rat"]
                )
                trt_col = _find_col(
                    tdf.columns,
                    ["treatment", "group", "lesion", "lesiongroup", "surgery", "surgerygroup", "condition", "cohort"]
                )
                if aid_col and trt_col:
                    for _, row in tdf.iterrows():
                        aid = str(row[aid_col]).strip()
                        trt_val = row[trt_col]
                        trt = None if pd.isna(trt_val) else str(trt_val).strip()
                        if aid:
                            treatment_map[aid] = trt
                else:
                    print("⚠️ Treatment map provided, but could not find ID and treatment/group columns. "
                          f"Seen columns: {list(tdf.columns)}")
            except Exception as e:
                print(f"⚠️ Failed to read treatment map: {e}")

        pat = re.compile(filename_regex)

        # load each (experiment, paradigm, folder)
        for exp_name, par_name, folder in roots:
            folder = Path(folder)

            # add/get experiment with dot access
            exp = study.experiments.get(exp_name)
            if exp is None:
                exp = Experiment(name=exp_name)
                attr = _safe_attr(exp_name, study.__dict__)
                setattr(study, attr, exp)
                study._attr_alias[attr] = exp_name
                study.experiments[exp_name] = exp

            # add/get paradigm with dot access
            paradigm = exp.paradigms.get(par_name)
            if paradigm is None:
                paradigm = Paradigm(name=par_name)
                attr = _safe_attr(par_name, getattr(exp, "__dict__", {}))
                setattr(exp, attr, paradigm)
                exp.paradigms[par_name] = paradigm

            # discover files and group by animal
            found: Dict[str, List[Tuple[Path, Optional[str]]]] = {}
            for p in sorted(folder.rglob("*.csv")):
                m = pat.search(p.name)
                if not m:
                    continue
                animal = m.group("animal")
                date = m.groupdict().get("date")
                found.setdefault(animal, []).append((p, date))

            # build animals & sessions
            for animal_id, lst in found.items():
                lst.sort(key=lambda t: (t[1] or "", t[0].name))

                animal = paradigm.animals.get(animal_id)
                if animal is None:
                    animal = Animal(animal_id=animal_id, treatment=treatment_map.get(animal_id))
                    attr = _safe_attr(animal_id, getattr(paradigm, "__dict__", {}))
                    setattr(paradigm, attr, animal)
                    paradigm.animals[animal_id] = animal

                # append sessions (session_num per animal *within this paradigm*)
                start_idx = len(animal.sessions) + 1
                for offset, (path, date) in enumerate(lst, start=0):
                    df = _read_csv_robust(path, **read_csv_kwargs)
                    if df is None:
                        print(f"⚠️ Skipping file (failed all engine/encoding attempts): {path}")
                        continue
                    # Clean headers: remove BOM, trim spaces
                    df.rename(columns=lambda s: str(s).replace("\ufeff", "").strip(), inplace=True)
                    sess = Session(
                        animal_id=animal_id, path=path, df=df,
                        session_num=start_idx + offset, date=date
                    )
                    animal.sessions.append(sess)

        return study

    # ---- iteration helpers ----
    def iter_sessions(
        self,
        experiment: Optional[str] = None,
        paradigm: Optional[str] = None,
        treatment: Optional[str] = None,
    ) -> Iterator[Session]:
        """
        Iterate over `Session` objects with optional filters.

        Parameters
        ----------
        experiment : str | None, optional
            Restrict to a particular experiment name.
        paradigm : str | None, optional
            Restrict to a particular paradigm name.
        treatment : str | None, optional
            Restrict to animals with a particular treatment label.

        Yields
        ------
        Session
            Matching sessions.
        """
        exps = (
            [self.experiments[experiment]]
            if experiment else self.experiments.values()
        )
        for exp in exps:
            pars = (
                [exp.paradigms[paradigm]]
                if paradigm else exp.paradigms.values()
            )
            for par in pars:
                for animal in par:
                    if treatment and animal.treatment != treatment:
                        continue
                    for sess in animal:
                        yield sess

    def all_trials(
        self,
        experiment: Optional[str] = None,
        paradigm: Optional[str] = None,
        treatment: Optional[str] = None,
        sort: bool = False,
    ) -> pd.DataFrame:
        """
        Concatenate trials across the study (optionally filtered), adding metadata.

        Parameters
        ----------
        experiment : str | None, optional
            Restrict to a particular experiment.
        paradigm : str | None, optional
            Restrict to a particular paradigm.
        treatment : str | None, optional
            Restrict to a particular treatment/group label.
        sort : bool, optional
            Forwarded to pandas.concat(sort=...), default False.

        Returns
        -------
        pandas.DataFrame
            Single table of trials with columns:
            `_experiment`, `_paradigm`, `_animal`, `_treatment`,
            `_session_idx` (per animal & paradigm),
            `_trial_idx` (1-based within each session).

        Notes
        -----
        - If the original DataFrames already contain any of these metadata names,
          they are preserved as `*_orig` before the new columns are added.
        """
        frames = []
        exps = (
            [self.experiments[experiment]]
            if experiment else self.experiments.values()
        )
        for exp in exps:
            pars = (
                [exp.paradigms[paradigm]]
                if paradigm else exp.paradigms.values()
            )
            for par in pars:
                for animal in par:
                    if treatment and animal.treatment != treatment:
                        continue
                    for sess in animal:
                        if sess.df is None or len(sess.df) == 0:
                            continue
                        df = sess.df.copy()
                        # avoid clobbering if user already has columns with these names
                        rename_map = {}
                        for c in ("_experiment", "_paradigm", "_animal", "_treatment", "_session_idx", "_trial_idx"):
                            if c in df.columns:
                                rename_map[c] = c + "_orig"
                        if rename_map:
                            df = df.rename(columns=rename_map)

                        df["_experiment"] = exp.name
                        df["_paradigm"] = par.name
                        df["_animal"] = animal.animal_id
                        df["_treatment"] = animal.treatment
                        df["_session_idx"] = sess.session_num
                        df["_trial_idx"] = range(1, len(df) + 1)
                        frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=sort)
    
    # ======= Trial-level annotators (return a DataFrame) =======

    def annotate_reaction_response_times(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_reaction_response_times(df, **kw)

    def annotate_inter_trial_intervals(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_inter_trial_intervals(df, **kw)

    def annotate_num_rewards(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_num_rewards(df, **kw)

    def annotate_high_value_well_from_bolus(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_high_value_well_from_bolus(df, **kw)

    def annotate_block_id_by_bolus(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_block_id_by_bolus(df, **kw)

    def annotate_block_id_by_prob(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return add_block_id_by_prob(df, **kw)

    def filter_blocks_with_few_trials(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return delete_blocks_with_few_trials(df, **kw)

    # ======= Session-level summaries (return one row per session) =======

    def summarize_session_performance(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return summarize_session_performance(df, **kw)

    def summarize_no_response_rate(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return summarize_no_response_rate(df, **kw)

    def summarize_prob_right(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return summarize_prob_right(df, **kw)

    def summarize_prob_repeat(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return summarize_prob_repeat(df, **kw)

    def summarize_wsls(self, *, experiment=None, paradigm=None, treatment=None, **kw):
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)
        return summarize_wsls(df, **kw)

    def summarize_sessions(
        self,
        *,
        experiment: str | None = None,
        paradigm: str | None = None,
        treatment: str | None = None,
        prefer_wrong_flag: bool = False,
        well_col: str = "Well_on_ts",
        choice_col: str = "Well_id",
        err_col: str = "Error_flg",
    ) -> pd.DataFrame:
        """
        Convenience: merge several common summaries into one table.
        Includes: performance, no-response, prob-right, prob-repeat, WSLS.
        """
        from mp_metrics import (
            DEFAULT_SESSION_KEYS,
            summarize_session_performance,
            summarize_no_response_rate,
            summarize_prob_right,
            summarize_prob_repeat,
            summarize_wsls,
        )

        def _norm_name(s: str) -> str:
            import re
            return re.sub(r"[^a-z0-9]+", "", str(s).lower())

        def _resolve_any(df: pd.DataFrame, candidates: list[str]) -> str | None:
            nmap = {_norm_name(c): c for c in df.columns}
            for cand in candidates:
                hit = nmap.get(_norm_name(cand))
                if hit is not None:
                    return hit
            return None

        keys = DEFAULT_SESSION_KEYS
        df = self.all_trials(experiment=experiment, paradigm=paradigm, treatment=treatment)

        # Resolve the *actual* column names present in your data
        real_well   = _resolve_any(df, [well_col, "well_on_ts"])
        real_choice = _resolve_any(df, [choice_col, "well_id"])
        real_err    = _resolve_any(df, [err_col, "error_flg"])

        # Fall back to the provided names if not found (metrics will warn if still missing)
        real_well   = real_well   or well_col
        real_choice = real_choice or choice_col
        real_err    = real_err    or err_col

        perf = summarize_session_performance(df, keys=keys, prefer_wrong_flag=prefer_wrong_flag)
        nr   = summarize_no_response_rate(df, keys=keys, well_col=real_well)
        pr   = summarize_prob_right(df, keys=keys, choice_col=real_choice)
        rep  = summarize_prob_repeat(df, keys=keys, choice_col=real_choice)
        wsls = summarize_wsls(df, keys=keys, choice_col=real_choice, err_col=real_err)

        out = perf.copy()
        for sub in (nr, pr, rep, wsls):
            out = out.merge(sub.drop(columns=[c for c in sub.columns if c == "num_trials"]),
                            on=list(keys), how="left")
        return out


# =============================
# Example usage (manual test)
# =============================

if __name__ == "__main__":
    study = StudyDataset.from_roots(
        roots=[
            ("PFC_lesion", "normal",
             "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/normal_mp"),
            ("PFC_lesion", "hallway_swap",
             "/Users/ben/Documents/Data/lesion_study/post_sx_testing/reorganized/switching_halls"),
        ],
        # If your map lives elsewhere, point to it here:
        # e.g., "/mnt/data/animal_important_dates_csv.csv"
        animal_treatment_map_csv="/Users/ben/Documents/Data/lesion_study/animal_important_dates_csv.csv",
        filename_regex=r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
        # You may pass low_memory for the C engine; it will be auto-dropped for the Python engine:
        low_memory=False,
    )

    # Iterate only sham animals in hallway_swap
    for sess in study.iter_sessions(experiment="PFC_lesion", paradigm="hallway_swap", treatment="sham"):
        print(sess.animal_id, sess.session_num, sess.path.name, len(sess.df))

    # All trials for the 'normal' paradigm, with metadata columns
    all_normal = study.all_trials(experiment="PFC_lesion", paradigm="normal")
    print(all_normal[["_experiment", "_paradigm", "_animal", "_treatment", "_session_idx", "_trial_idx"]].head())

    # Compare paradigms quickly
    all_trials = study.all_trials(experiment="PFC_lesion")
    summary = (all_trials
               .groupby(["_paradigm", "_treatment"])["_trial_idx"]
               .count()
               .sort_values(ascending=False)
               .head(10))
    print(summary)
