from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import re
import pandas as pd

# =============================
# Utilities
# =============================

def _safe_attr(name: str, existing: dict) -> str:
    """Return a valid, non-colliding attribute name for dot-notation."""
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
    """Normalize header names: 'Animal ID' -> 'animalid' (lower, strip non-alnum)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _find_col(cols: Iterable[str], candidates_norm: List[str]) -> Optional[str]:
    """Return original column name matching any normalized candidate."""
    nmap = {_norm(c): c for c in cols}
    for cand in candidates_norm:
        if cand in nmap:
            return nmap[cand]
    return None

def _read_csv_robust(
    path: str | Path,
    engine_preference: Tuple[str, ...] = ("c", "python"),
    encodings: Tuple[Optional[str], ...] = (None, "utf-8", "utf-8-sig", "latin-1", "iso-8859-1", "cp1252"),
    **kwargs,
) -> Optional[pd.DataFrame]:
    """
    Try multiple engines and encodings. Skips unsupported kwargs per engine.
    Returns a DataFrame or None if all attempts fail.
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
    """One session worth of trials."""
    animal_id: str
    path: Path
    df: pd.DataFrame
    session_num: Optional[int] = None   # within animal & paradigm
    date: Optional[str] = None          # optional from filename

@dataclass
class Animal:
    """All sessions for one animal."""
    animal_id: str
    sessions: List[Session] = field(default_factory=list)
    treatment: Optional[str] = None     # set from treatment map CSV

    def __iter__(self) -> Iterator[Session]:
        return iter(self.sessions)

    def __len__(self) -> int:
        return len(self.sessions)

    def concat(self, sort: bool = False) -> pd.DataFrame:
        """All trials across this animal's sessions (adds animal/session idx)."""
        if not self.sessions:
            return pd.DataFrame()
        return pd.concat(
            [s.df.assign(_animal=self.animal_id, _session_idx=i + 1)
             for i, s in enumerate(self.sessions)],
            ignore_index=True, sort=sort
        )

@dataclass
class Paradigm:
    """A paradigm inside an experiment (e.g., 'normal', 'hallway_swap')."""
    name: str
    animals: Dict[str, Animal] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Animal]:
        return iter(self.animals.values())

    def __getitem__(self, animal_id: str) -> Animal:
        return self.animals[animal_id]

@dataclass
class Experiment:
    """A single experiment (e.g., 'PFC_lesion')."""
    name: str
    paradigms: Dict[str, Paradigm] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Paradigm]:
        return iter(self.paradigms.values())

    def __getitem__(self, paradigm_name: str) -> Paradigm:
        return self.paradigms[paradigm_name]

@dataclass
class StudyDataset:
    """Root container for many experiments & paradigms with dot access."""
    experiments: Dict[str, Experiment] = field(default_factory=dict)
    _attr_alias: Dict[str, str] = field(default_factory=dict, init=False, repr=False)

    # ---- creation ----
    @classmethod
    def from_roots(
        cls,
        roots: List[Tuple[str, str, str | Path]],
        # list of (experiment_name, paradigm_name, folder_path)
        filename_regex: str = r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
        animal_treatment_map_csv: Optional[str | Path] = None,
        # `engine` retained for backwards-compat, but robust reader tries both C and Python engines
        engine: Optional[str] = None,
        **read_csv_kwargs,
    ) -> "StudyDataset":
        """
        Build a StudyDataset from multiple roots.
        Each root is (experiment, paradigm, folder).
        Optionally provide a treatment map CSV with columns like:
          animal_id,treatment   (extra cols are fine)
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
            exp = study.experiments.get(exp_name)
            if exp is None:
                exp = Experiment(name=exp_name)
                # dot access: ds.<experiment>
                setattr(study, _safe_attr(exp_name, study.__dict__), exp)
                study._attr_alias[_safe_attr(exp_name, study.__dict__)] = exp_name
                study.experiments[exp_name] = exp

            paradigm = exp.paradigms.get(par_name)
            if paradigm is None:
                paradigm = Paradigm(name=par_name)
                # dot access: ds.<experiment>.<paradigm>
                setattr(exp, _safe_attr(par_name, getattr(exp, "__dict__", {})), paradigm)
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
                    # dot access: ds.<experiment>.<paradigm>.<animal_id>
                    setattr(paradigm, _safe_attr(animal_id, getattr(paradigm, "__dict__", {})), animal)
                    paradigm.animals[animal_id] = animal

                # append sessions (session_num per animal *within this paradigm*)
                start_idx = len(animal.sessions) + 1
                for offset, (path, date) in enumerate(lst, start=0):
                    df = _read_csv_robust(path, **read_csv_kwargs)
                    if df is None:
                        print(f"⚠️ Skipping file (failed all engine/encoding attempts): {path}")
                        continue
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
        """Yield sessions filtered by experiment, paradigm, and/or treatment."""
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
        Concatenate all trials with metadata columns:
        _experiment, _paradigm, _animal, _treatment,
        _session_idx (per animal & paradigm), _trial_idx (within session).
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

# =============================
# Example usage
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
