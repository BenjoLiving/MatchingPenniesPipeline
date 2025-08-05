"""
Analysis utilities for rodent trial data stored in a SQLite database.

This module exposes the ``DatabaseAnalysis`` class, which provides
high‑level methods for selecting sessions and computing behavioural
metrics. It separates data retrieval, cleaning and metric computation
into distinct operations to keep analyses transparent and reproducible.

The general workflow is:

1. Create an instance of ``DatabaseAnalysis`` pointing at your
   ``rat_trials.db`` database.
2. Select sessions of interest via ``get_sessions``. This method can
   filter by experiment name, animal identifiers and date ranges.
3. For each session ID returned, call ``load_session`` to obtain a
   cleaned ``pandas.DataFrame``. The dataframe includes all fields
   present in the original CSV plus a handful of helper columns.
4. Use the provided metric functions such as ``compute_performance``,
   ``performance_by_bins``, ``count_rewards``, ``response_times``,
   ``probability_right``, ``win_stay_lose_shift`` and
   ``extraneous_feeder_sampling`` to summarise each session.

Example usage::

    from analysis import DatabaseAnalysis
    db = DatabaseAnalysis('rat_trials.db')
    sessions = db.get_sessions(experiment='ExperimentA')
    for session_id in sessions:
        df = db.load_session(session_id)
        df_clean = db.clean_trials(df)
        correct, wrong, perf = db.compute_performance(df_clean)
        print(f'Session {session_id}: {correct} correct, {wrong} wrong, {perf:.2%} performance')

"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class SessionSummary:
    """Container for summarised session metrics."""

    session_id: int
    correct: int
    wrong: int
    performance: float
    num_rewards: int
    prob_right: float
    win_stay_fraction: Optional[float]
    lose_shift_fraction: Optional[float]
    extraneous_sampling_fraction: Optional[float]


class DatabaseAnalysis:
    """
    Provide methods to query and analyse trial data stored in a SQLite database.

    Each analysis method operates on an in‑memory pandas DataFrame
    representing a single session. Data is lazily loaded and parsed
    through the ``load_session`` method, which expands the JSON column
    containing full trial information into individual columns.
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        # ensure rows are returned as dicts when using cursor
        self.conn.row_factory = sqlite3.Row

    def get_sessions(
        self,
        experiment: Optional[str] = None,
        animal_ids: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[int]:
        """Retrieve session IDs filtered by various criteria.

        Args:
            experiment: Name of the experiment. If ``None``, sessions from
                all experiments are returned.
            animal_ids: List of subject identifiers (e.g. ``["P4677"]``).
                If ``None``, sessions for all animals are returned.
            start_date: Only sessions starting on or after this date are
                included.
            end_date: Only sessions starting on or before this date are
                included.

        Returns:
            A list of session IDs matching the filters.
        """
        cur = self.conn.cursor()
        query = "SELECT sessions.id FROM sessions"
        joins: List[str] = []
        wheres: List[str] = []
        params: List[Any] = []

        if experiment is not None:
            joins.append("JOIN experiments ON sessions.experiment_id = experiments.id")
            wheres.append("experiments.name = ?")
            params.append(experiment)
        if animal_ids is not None:
            joins.append("JOIN animals ON sessions.animal_id = animals.id")
            placeholders = ",".join("?" for _ in animal_ids)
            wheres.append(f"animals.animal_id IN ({placeholders})")
            params.extend(animal_ids)
        if start_date is not None:
            wheres.append("sessions.session_datetime >= ?")
            params.append(start_date.isoformat())
        if end_date is not None:
            wheres.append("sessions.session_datetime <= ?")
            params.append(end_date.isoformat())
        if joins:
            query += " " + " ".join(joins)
        if wheres:
            query += " WHERE " + " AND ".join(wheres)
        query += " ORDER BY sessions.session_datetime ASC"
        cur.execute(query, params)
        return [row[0] for row in cur.fetchall()]

    def load_session(self, session_id: int) -> pd.DataFrame:
        """Load all trials for a given session and return a DataFrame.

        Args:
            session_id: Primary key of the session.

        Returns:
            A pandas DataFrame with one row per trial. The original
            JSON payload is expanded into individual columns. Additional
            helper columns (``well_id``, ``rewarded_well``, ``error_flg``
            and ``trial_index``) are also included for convenience.
        """
        df = pd.read_sql(
            "SELECT trial_index, well_id, rewarded_well, error_flg, data_json FROM trials"
            " WHERE session_id = ? ORDER BY trial_index ASC",
            self.conn,
            params=(session_id,),
        )
        # expand JSON column into separate series of dictionaries
        # parse each JSON string; ensure errors don't propagate
        expanded_records: List[Dict[str, Any]] = []
        for data in df["data_json"]:
            try:
                record = json.loads(data)
            except Exception:
                record = {}
            expanded_records.append(record)
        expanded_df = pd.DataFrame(expanded_records)
        # strip whitespace from column names
        expanded_df.columns = [c.strip() for c in expanded_df.columns]
        # join with base df
        # Drop duplicate columns from expanded_df that already exist in the left
        # DataFrame. When concatenating, duplicate names result in a
        # DataFrame being returned when selecting a column (rather than a
        # Series), which breaks downstream string operations. Remove any
        # columns from expanded_df that are present in df (except
        # 'data_json', which will be dropped below).
        left_cols = set(df.columns) - {"data_json"}
        deduped_expanded = expanded_df[[c for c in expanded_df.columns if c not in left_cols]]
        result = pd.concat(
            [df.drop(columns=["data_json"]).reset_index(drop=True), deduped_expanded.reset_index(drop=True)],
            axis=1,
        )
        return result

    @staticmethod
    def clean_trials(df: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid trials from a DataFrame.

        Trials are excluded if any of the following conditions hold:

        * ``error_flg`` is null or missing.
        * ``well_id`` is null or missing.
        * ``rewarded_well`` equals ``'E'`` (extraneous trial).

        Note: the caller is expected to display statistics about how
        many trials are removed for each reason. This function does
        not print anything itself.

        Args:
            df: Raw session DataFrame.

        Returns:
            DataFrame containing only valid trials.
        """
        """Implementation note: build a single boolean mask rather than
        mutating in place with ``&=``. Using the inplace operator
        ``&=`` with pandas can lead to odd behaviour if indices are
        misaligned or if pandas returns ``NotImplemented`` for some
        comparisons. We therefore combine conditions explicitly using
        the ``&`` operator and assign the result to ``mask``.
        """
        # Define individual conditions
        # Strip whitespace from string columns when evaluating conditions
        error_notna = df['error_flg'].notna()
        error_notblank = df['error_flg'].astype(str).str.strip() != ''
        well_notna = df['well_id'].notna()
        well_notblank = df['well_id'].astype(str).str.strip() != ''
        not_extraneous = df['rewarded_well'].astype(str).str.strip().str.upper() != 'E'
        mask = error_notna & error_notblank & well_notna & well_notblank & not_extraneous
        return df.loc[mask].reset_index(drop=True)

    @staticmethod
    def cleaning_statistics(df: pd.DataFrame) -> Dict[str, int]:
        """Compute how many trials would be removed during cleaning.

        This helper mirrors the logic in ``clean_trials`` and reports
        counts for each exclusion criterion. No rows are dropped.

        Args:
            df: Raw session DataFrame.

        Returns:
            Dictionary with keys ``missing_error_flg``, ``missing_well_id``
            and ``extraneous_reward`` counting how many rows fall into
            each category. A row can contribute to more than one count
            if multiple criteria apply.
        """
        missing_error = df['error_flg'].isna() | (df['error_flg'].astype(str).str.strip() == '')
        missing_well = df['well_id'].isna() | (df['well_id'].astype(str).str.strip() == '')
        extraneous = df['rewarded_well'].astype(str).str.strip().str.upper() == 'E'
        return {
            'missing_error_flg': int(missing_error.sum()),
            'missing_well_id': int(missing_well.sum()),
            'extraneous_reward': int(extraneous.sum()),
        }

    @staticmethod
    def compute_performance(df: pd.DataFrame) -> Tuple[int, int, float]:
        """Compute the number of correct and wrong trials and performance.

        A trial is considered correct if ``error_flg`` equals ``'0'`` (as
        a string or numeric zero). All other non‑blank values are treated
        as wrong. Trials should already have been cleaned using
        ``clean_trials``.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Tuple ``(n_correct, n_wrong, performance_fraction)`` where
            ``performance_fraction`` is the fraction of trials that were
            correct (between 0 and 1). If there are no trials in the
            dataframe, ``(0, 0, np.nan)`` is returned.
        """
        if df.empty:
            return 0, 0, float('nan')
        # treat error_flg as string; convert to numeric when possible
        ef = df['error_flg'].astype(str).str.strip()
        correct_mask = ef == '0'
        n_correct = int(correct_mask.sum())
        n_wrong = int((~correct_mask).sum())
        total = n_correct + n_wrong
        perf = n_correct / total if total > 0 else float('nan')
        return n_correct, n_wrong, perf

    @staticmethod
    def performance_by_bins(df: pd.DataFrame, n_bins: int = 10) -> List[float]:
        """Compute performance in equally sized bins across a session.

        Trials should be cleaned prior to passing into this function. The
        session is divided into ``n_bins`` contiguous segments of nearly
        equal length. For each segment, the fraction of correct trials
        (as defined by ``compute_performance``) is returned. If a bin
        contains no trials, ``np.nan`` is returned for that bin.

        Args:
            df: Cleaned trials DataFrame.
            n_bins: Number of bins to partition the trials into.

        Returns:
            List of length ``n_bins`` containing performance fractions for
            each bin.
        """
        if df.empty:
            return [float('nan')] * n_bins
        total_trials = len(df)
        bin_edges = np.linspace(0, total_trials, n_bins + 1, dtype=int)
        perf_by_bin: List[float] = []
        for i in range(n_bins):
            start, stop = bin_edges[i], bin_edges[i + 1]
            subset = df.iloc[start:stop]
            if subset.empty:
                perf_by_bin.append(float('nan'))
            else:
                n_correct, n_wrong, perf = DatabaseAnalysis.compute_performance(subset)
                perf_by_bin.append(perf)
        return perf_by_bin

    @staticmethod
    def count_rewards(df: pd.DataFrame) -> int:
        """Compute the total number of reward deliveries in a session.

        This function counts the number of timestamp entries in the
        ``L_reward_ts`` and ``R_reward_ts`` columns across all trials. These
        columns contain whitespace‑separated timestamps indicating when
        rewards were delivered. Empty strings or missing values are
        ignored.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Total number of reward deliveries across the session.
        """
        total = 0
        for col in ['L_reward_ts', 'R_reward_ts']:
            if col not in df.columns:
                continue
            for val in df[col]:
                if pd.isna(val) or str(val).strip() == '':
                    continue
                tokens = str(val).split()
                total += len(tokens)
        return total

    @staticmethod
    def response_times_and_licks(df: pd.DataFrame) -> Tuple[List[Optional[float]], List[int]]:
        """Compute response times and lick counts for each trial.

        The response time for a trial is defined as the difference between
        the first lick (either left or right) and the trial start
        timestamp. Licks occurring before the trial start are ignored.
        If no lick timestamps are present, the response time is set to
        ``None``. The lick count is the total number of timestamps
        observed across both feeders.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Two lists of equal length: response times (in the same units
            as the timestamps, typically milliseconds) and lick counts.
        """
        response_times: List[Optional[float]] = []
        lick_counts: List[int] = []
        for _, row in df.iterrows():
            start_ts = _safe_float(row.get('trial_start_ts'))
            l_licks = _parse_ts_list(row.get('L_lick_ts'))
            r_licks = _parse_ts_list(row.get('R_lick_ts'))
            # combine and find first lick after trial start
            all_licks = [t for t in l_licks + r_licks if start_ts is not None and t >= start_ts]
            all_licks.sort()
            if all_licks:
                response_times.append(all_licks[0] - start_ts)
            else:
                response_times.append(None)
            lick_counts.append(len(l_licks) + len(r_licks))
        return response_times, lick_counts

    @staticmethod
    def probability_right(df: pd.DataFrame) -> float:
        """Compute the probability of choosing the right feeder.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Fraction of trials in which ``well_id`` equals ``'R'``. If
            the dataframe is empty, returns ``np.nan``.
        """
        if df.empty:
            return float('nan')
        choices = df['well_id'].astype(str).str.upper()
        return (choices == 'R').mean()

    @staticmethod
    def win_stay_lose_shift(df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Compute win‑stay and lose‑shift fractions for a session.

        A trial (other than the first) is classified as:

        * **win‑stay** – the previous trial was correct and the current
          choice is the same as the previous choice.
        * **lose‑shift** – the previous trial was wrong and the current
          choice is different from the previous choice.

        The function returns the fraction of eligible trials that are
        win‑stay and lose‑shift respectively. If there are fewer than two
        trials after cleaning, both values are ``None``.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Tuple ``(win_stay_fraction, lose_shift_fraction)``.
        """
        if len(df) < 2:
            return None, None
        # convert error flag and choices
        ef = df['error_flg'].astype(str).str.strip()
        choices = df['well_id'].astype(str).str.upper()
        win_stay = 0
        win_stay_total = 0
        lose_shift = 0
        lose_shift_total = 0
        for i in range(1, len(df)):
            prev_correct = ef.iloc[i - 1] == '0'
            curr_same = choices.iloc[i] == choices.iloc[i - 1]
            if prev_correct:
                win_stay_total += 1
                if curr_same:
                    win_stay += 1
            else:
                lose_shift_total += 1
                if not curr_same:
                    lose_shift += 1
        win_stay_frac = win_stay / win_stay_total if win_stay_total > 0 else None
        lose_shift_frac = lose_shift / lose_shift_total if lose_shift_total > 0 else None
        return win_stay_frac, lose_shift_frac

    @staticmethod
    def extraneous_feeder_sampling(df: pd.DataFrame) -> Optional[float]:
        """Estimate the rate of extraneous feeder sampling (EFS).

        EFS occurs when, after a trial has ended but before the next
        trial begins, the animal enters the non‑chosen feeder. A simple
        approximation is used here: for each pair of consecutive trials
        (``i`` and ``i+1``), we look at the lick timestamps on the
        opposite side of the chosen feeder during the interval
        ``(well_off_ts[i], trial_start_ts[i+1])``. If any such lick
        occurs, an EFS event is recorded. The rate returned is the
        fraction of inter‑trial intervals containing at least one
        extraneous lick. If fewer than two cleaned trials are present,
        ``None`` is returned.

        Args:
            df: Cleaned trials DataFrame.

        Returns:
            Fraction of inter‑trial gaps containing an extraneous lick, or
            ``None`` if computation is not possible.
        """
        if len(df) < 2:
            return None
        n_intervals = 0
        efs_events = 0
        for i in range(len(df) - 1):
            row = df.iloc[i]
            next_row = df.iloc[i + 1]
            chosen = str(row['well_id']).strip().upper()
            other_side = 'L' if chosen == 'R' else 'R'
            # compute end of trial i
            end_ts_candidates = []
            # well_off_ts marks when the feeder door closes
            end_ts_candidates.append(_safe_float(row.get('well_off_ts')))
            # reward timestamps may extend beyond well_off_ts
            for rcol in [f'{chosen}_reward_ts', f'{other_side}_reward_ts']:
                end_ts_candidates.append(_max_ts(_parse_ts_list(row.get(rcol))))
            end_ts = max([t for t in end_ts_candidates if t is not None], default=None)
            start_next = _safe_float(next_row.get('trial_start_ts'))
            if end_ts is None or start_next is None or start_next <= end_ts:
                # cannot evaluate interval
                continue
            n_intervals += 1
            # lick times on the other side in row i
            lick_col = f'{other_side}_lick_ts'
            lick_times = _parse_ts_list(row.get(lick_col))
            # check for lick after end_ts but before next start
            found = any(end_ts < t < start_next for t in lick_times)
            if found:
                efs_events += 1
        if n_intervals == 0:
            return None
        return efs_events / n_intervals

    def summarise_sessions(
        self,
        session_ids: List[int],
        n_bins: int = 10,
    ) -> List[SessionSummary]:
        """Generate a summary for each session in ``session_ids``.

        The summary includes total correct and wrong trials, overall
        performance, total rewards, probability of choosing the right
        feeder, win‑stay/lose‑shift fractions and extraneous feeder
        sampling rate. Binned performance can be computed separately via
        ``performance_by_bins``.

        Args:
            session_ids: List of session identifiers.
            n_bins: Number of bins to use when computing binned performance
                (unused here; provided for convenience).

        Returns:
            A list of ``SessionSummary`` objects.
        """
        summaries: List[SessionSummary] = []
        for sid in session_ids:
            df = self.load_session(sid)
            df_clean = self.clean_trials(df)
            n_correct, n_wrong, perf = self.compute_performance(df_clean)
            num_rewards = self.count_rewards(df_clean)
            prob_right = self.probability_right(df_clean)
            win_stay_frac, lose_shift_frac = self.win_stay_lose_shift(df_clean)
            efs_frac = self.extraneous_feeder_sampling(df_clean)
            summaries.append(
                SessionSummary(
                    session_id=sid,
                    correct=n_correct,
                    wrong=n_wrong,
                    performance=perf,
                    num_rewards=num_rewards,
                    prob_right=prob_right,
                    win_stay_fraction=win_stay_frac,
                    lose_shift_fraction=lose_shift_frac,
                    extraneous_sampling_fraction=efs_frac,
                )
            )
        return summaries


def _safe_float(value: Optional[Any]) -> Optional[float]:
    """Convert a value to float if possible, otherwise return None."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _parse_ts_list(value: Optional[Any]) -> List[float]:
    """Parse a whitespace separated string of timestamps into floats.

    Args:
        value: String containing timestamps separated by whitespace, or
            any other type. ``None`` or empty strings result in an empty
            list.

    Returns:
        List of floats representing timestamps.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    tokens = s.split()
    result: List[float] = []
    for tok in tokens:
        try:
            result.append(float(tok))
        except Exception:
            continue
    return result


def _max_ts(ts_list: List[float]) -> Optional[float]:
    """Return the maximum timestamp in a list or None if empty."""
    return max(ts_list) if ts_list else None
