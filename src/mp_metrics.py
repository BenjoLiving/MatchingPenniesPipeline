# mp_metrics.py (patched)
# Vectorized metrics for Matching Pennies trials in pandas.
from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple
import warnings
import numpy as np
import pandas as pd


# --------------------------
# helpers
# --------------------------

DEFAULT_SESSION_KEYS = ("_experiment", "_paradigm", "_animal", "_treatment", "_session_idx")

def _ensure_order(df: pd.DataFrame, keys: Sequence[str] = DEFAULT_SESSION_KEYS) -> pd.DataFrame:
    cols = list(keys)
    if "_trial_idx" in df.columns:
        cols = list(keys) + ["_trial_idx"]
    return df.sort_values(cols, kind="mergesort", ignore_index=False)

def _has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing columns {missing}; skipping related metrics.", RuntimeWarning)
        return False
    return True

def _gdf(df: pd.DataFrame, keys: Sequence[str] = DEFAULT_SESSION_KEYS):
    """Group a DataFrame by the session keys (dropna=False, sort=False)."""
    return df.groupby(list(keys), dropna=False, sort=False)

def _gcol(df: pd.DataFrame, col: str, keys: Sequence[str] = DEFAULT_SESSION_KEYS):
    """Group a single column (Series) by session keys using the parent DataFrame."""
    return df.groupby(list(keys), dropna=False, sort=False)[col]

def _norm_name(s: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(s).lower())

def _resolve_col(df: pd.DataFrame, name: str) -> str | None:
    """Return the actual df column whose normalized form matches `name` (ignoring spaces/case)."""
    nmap = {_norm_name(c): c for c in df.columns}
    return nmap.get(_norm_name(name))

def _normalize_key_types(df: pd.DataFrame, keys) -> pd.DataFrame:
    out = df.copy()
    for k in keys:
        if k == "_session_idx":
            # nullable integer to handle any NaNs safely
            out[k] = pd.to_numeric(out[k], errors="coerce").astype("Int64")
        else:
            # use pandas' nullable string dtype to keep NaNs as <NA>
            out[k] = out[k].astype("string")
    return out

def _safe_left_merge(base: pd.DataFrame, summ: pd.DataFrame, keys) -> pd.DataFrame:
    """
    Merge `summ` into `base` on `keys`, but only bring in columns that
    don't already exist in `base` (besides the keys).
    """
    existing = set(base.columns)
    keyset   = set(keys)
    # Only pull in keys plus truly new metric columns
    newcols = [c for c in summ.columns if (c in keyset) or (c not in existing)]
    return base.merge(summ[newcols], on=list(keys), how="left")


# --------------------------
# trial-level annotators
# --------------------------

def add_reaction_response_times(
    df: pd.DataFrame,
    *,
    colmap: Optional[Mapping[str, str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add ReactTime, ResponseTime, InitTime to each trial.

    ReactTime    = Odor_port_off_ts - Tone_on_ts
    ResponseTime = Well_on_ts - Odor_port_off_ts
    InitTime     = Odor_port_on_ts - Trial_start_ts
    """
    cm = {
        "tone_on": "Tone_on_ts",
        "odor_port_off": "Odor_port_off_ts",
        "odor_port_on": "Odor_port_on_ts",
        "trial_start": "Trial_start_ts",
        "well_on": "Well_on_ts",
    }
    if colmap:
        cm.update(colmap)

    # resolve to real columns (case/space-insensitive)
    tone = _resolve_col(df, cm["tone_on"])
    opoff = _resolve_col(df, cm["odor_port_off"])
    opon = _resolve_col(df, cm["odor_port_on"])
    tstart = _resolve_col(df, cm["trial_start"])
    won = _resolve_col(df, cm["well_on"])

    need = [tone, opoff, opon, tstart, won]
    if any(c is None for c in need):
        warnings.warn("Missing one or more timestamp columns; cannot compute reaction/response/init times.", RuntimeWarning)
        return df

    out = df if inplace else df.copy()
    out["ReactTime"] = out[opoff] - out[tone]
    out["ResponseTime"] = out[won] - out[opoff]
    out["InitTime"] = out[opon] - out[tstart]
    return out


def add_inter_trial_intervals(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    colmap: Optional[Mapping[str, str]] = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Compute inter-trial intervals (ITI) within each session.

    ITI_n = Odor_port_on_ts(n) - max(House_light_on_ts, L_reward_ts, R_reward_ts)(n-1)
    Also splits into ITI_afterLose / ITI_afterWin based on previous trial error flag.
    """
    cm = {
        "odor_port_on": "Odor_port_on_ts",
        "house_light_on": "House_light_on_ts",
        "L_reward_ts": "L_reward_ts",
        "R_reward_ts": "R_reward_ts",
        "error": "Error_flg",
    }
    if colmap:
        cm.update(colmap)

    opon = _resolve_col(df, cm["odor_port_on"])
    hlight = _resolve_col(df, cm["house_light_on"])
    Lr = _resolve_col(df, cm["L_reward_ts"])
    Rr = _resolve_col(df, cm["R_reward_ts"])
    err = _resolve_col(df, cm["error"])

    need_prev = [hlight, Lr, Rr, err]
    need_curr = [opon]
    if any(c is None for c in need_prev + need_curr):
        warnings.warn("Missing columns for ITI computation; skipping.", RuntimeWarning)
        return df

    out = df if inplace else df.copy()
    out = _ensure_order(out, keys)

    prev_house = _gcol(out, hlight, keys).shift(1)
    prev_L = _gcol(out, Lr, keys).shift(1)
    prev_R = _gcol(out, Rr, keys).shift(1)
    prev_err = pd.to_numeric(_gcol(out, err, keys).shift(1), errors="coerce")

    prev_reward_end = pd.concat([prev_house, prev_L, prev_R], axis=1).max(axis=1)
    iti = out[opon] - prev_reward_end

    out["InterTrialInterval"] = iti
    out["InterTrialInterval_afterLose"] = np.where(prev_err == 1, iti, np.nan)
    out["InterTrialInterval_afterWin"] = np.where(prev_err == 0, iti, np.nan)
    return out


def add_num_rewards(
    df: pd.DataFrame,
    *,
    prefer_counts: bool = True,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Add per-trial NumRewards.

    By default counts non-NaN timestamps in ['L_reward_ts','R_reward_ts'].
    If columns ['L_reward_num','R_reward_num'] exist and prefer_counts=False,
    NumRewards = L_reward_num + R_reward_num.
    """
    out = df if inplace else df.copy()

    L_num = _resolve_col(out, "L_reward_num")
    R_num = _resolve_col(out, "R_reward_num")
    L_ts  = _resolve_col(out, "L_reward_ts")
    R_ts  = _resolve_col(out, "R_reward_ts")

    has_num = L_num is not None and R_num is not None
    has_ts = L_ts is not None and R_ts is not None

    if not has_num and not has_ts:
        warnings.warn("No reward columns found; skipping NumRewards.", RuntimeWarning)
        return out

    if has_num and not prefer_counts:
        out["NumRewards"] = out[L_num].fillna(0).astype(float) + out[R_num].fillna(0).astype(float)
    elif has_ts:
        out["NumRewards"] = out[[L_ts, R_ts]].notna().sum(axis=1).astype(float)
    else:
        # fallback if only one timestamp exists
        col = L_ts if L_ts is not None else R_ts
        out["NumRewards"] = out[[col]].notna().sum(axis=1).astype(float)
    return out


def add_high_value_well_from_bolus(
    df: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Derive 'HighValWell_id' from bolus counts and flag 'ChoseHighVal_flg'.
    """
    out = df if inplace else df.copy()

    L_num = _resolve_col(out, "L_reward_num")
    R_num = _resolve_col(out, "R_reward_num")
    if L_num is None or R_num is None:
        warnings.warn("Missing L_reward_num/R_reward_num; high-value well not computed.", RuntimeWarning)
        return out

    comp = out[R_num].astype(float) - out[L_num].astype(float)
    hv = np.where(comp > 0, "R", np.where(comp < 0, "L", "N"))
    out["HighValWell_id"] = hv

    well_col = _resolve_col(out, "Well_id")
    if well_col is not None:
        out["ChoseHighVal_flg"] = (out[well_col] == out["HighValWell_id"]).astype("float")
    else:
        out["ChoseHighVal_flg"] = np.nan
    return out


def _rle_blocks(series: pd.Series) -> pd.Series:
    """Run-length block index per identical consecutive value."""
    change = series.ne(series.shift(1)).fillna(True)
    return change.cumsum()


def add_block_id_by_bolus(
    df: pd.DataFrame,
    *,
    min_trials: int = 1,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Assign blocks using which side has more boluses per trial.
    Produces:
      - Block_id: 'Big_R', 'Big_L', or 'N'
      - Block_number: 1.. within session for qualifying blocks (len >= min_trials)
      - Block_trial_indx: 1.. position within block (qualifying blocks only)
    """
    out = df if inplace else df.copy()
    out = _ensure_order(out, keys)

    L_num = _resolve_col(out, "L_reward_num")
    R_num = _resolve_col(out, "R_reward_num")
    if L_num is None or R_num is None:
        warnings.warn("Missing L_reward_num/R_reward_num; block-by-bolus not computed.", RuntimeWarning)
        return out

    comp = out[R_num].astype(float) - out[L_num].astype(float)
    out["Block_id"] = np.where(comp > 0, "Big_R", np.where(comp < 0, "Big_L", "N"))

    # per-session run-length groups
    g = _gdf(out, keys)
    block_grp = g["Block_id"].transform(_rle_blocks)

    # length of each run
    run_sizes = block_grp.map(block_grp.value_counts())

    # mask short runs (non-'N' blocks with size < min_trials) to 'N'
    short_nonN = (out["Block_id"] != "N") & (run_sizes < int(min_trials))
    out.loc[short_nonN, "Block_id"] = "N"

    # recompute runs after masking, then assign numbers & indices
    block_grp2 = _gdf(out, keys)["Block_id"].transform(_rle_blocks)

    # Block_trial_indx: position within each run
    out["Block_trial_indx"] = _gdf(out, keys)["Block_id"].cumcount() + 1

    # Block_number: count only non-'N' blocks, sequential per session
    is_real_block_start = (out["Block_id"] != "N") & block_grp2.ne(block_grp2.shift(1))
    # out["Block_number"] = _gdf(is_real_block_start.astype(int).to_frame("start"), keys)["start"].cumsum()
    out["Block_number"] = _gcol(
        out.assign(__start=is_real_block_start.astype(int)),
        "__start",
        keys
    ).cumsum()
    out.loc[out["Block_id"] == "N", "Block_number"] = np.nan

    return out


def add_block_id_by_prob(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Assign block labels using reward probabilities per trial.
    Produces:
      - Block_id: string "L/R" (e.g., "40/60")
      - Block_agg_id: collapsed "max/min" (e.g., "60/40" for both 60/40 and 40/60)
      - Block_trial_indx: 1.. within block
      - ChoseHighProb_flg: 1 if Well_id matches the higher-prob side
      - ChoseHighProb_lastTrial_flg: previous trial's flag within session
    """
    out = df if inplace else df.copy()
    out = _ensure_order(out, keys)

    Lp_col = _resolve_col(out, "L_reward_prob")
    Rp_col = _resolve_col(out, "R_reward_prob")
    if Lp_col is None or Rp_col is None:
        warnings.warn("Missing L_reward_prob/R_reward_prob; block-by-prob not computed.", RuntimeWarning)
        return out

    Lp = (out[Lp_col] * 100).round().astype("Int64")
    Rp = (out[Rp_col] * 100).round().astype("Int64")
    out["Block_id"] = Lp.astype(str) + "/" + Rp.astype(str)

    # aggregated symmetric
    hi = np.maximum(Lp, Rp).astype("Int64")
    lo = np.minimum(Lp, Rp).astype("Int64")
    out["Block_agg_id"] = hi.astype(str) + "/" + lo.astype(str)

    # trial index within block
    out["Block_trial_indx"] = _gdf(out, keys + ["Block_id"])["Block_id"].cumcount() + 1

    # chose higher-prob side
    high_side = np.where(Rp > Lp, "R", np.where(Lp > Rp, "L", "N"))
    well_col = _resolve_col(out, "Well_id")
    if well_col is not None:
        out["ChoseHighProb_flg"] = (out[well_col] == high_side).astype("float")
        # previous trial flag within session
        out["ChoseHighProb_lastTrial_flg"] = _gcol(out, "ChoseHighProb_flg", keys).shift(1)
    else:
        out["ChoseHighProb_flg"] = np.nan
        out["ChoseHighProb_lastTrial_flg"] = np.nan

    return out


def delete_blocks_with_few_trials(
    df: pd.DataFrame,
    thresh: int,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
) -> pd.DataFrame:
    """
    Return a copy of df with trials removed if their Block_id or Block_agg_id
    occurs fewer than `thresh` times **within each session**.
    """
    out = _ensure_order(df, keys)
    # choose which columns exist
    cols = [c for c in ["Block_id", "Block_agg_id"] if c in out.columns]
    if not cols:
        warnings.warn("No Block_id/Block_agg_id columns; nothing to delete.", RuntimeWarning)
        return out.copy()

    mask = pd.Series(True, index=out.index)
    for c in cols:
        keep = _gcol(out, c, keys).transform(lambda s: s.map(s.value_counts()).ge(thresh))
        mask &= keep

    return out.loc[mask].copy()


# --------------------------
# session-level summaries
# --------------------------

def _agg_template(df: pd.DataFrame, keys: Sequence[str]) -> pd.DataFrame:
    g = _gdf(df, keys)
    return g.size().rename("num_trials").to_frame().reset_index()


def summarize_session_performance(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    prefer_wrong_flag: bool = False,
) -> pd.DataFrame:
    """
    PercentCorrectChoice and MeanRewardsPerTrial per session.

    PercentCorrectChoice:
        - If prefer_wrong_flag and 'Wrong_choice_flg' exists: 1 - mean(Wrong_choice_flg)
        - Else if 'Error_flg' exists: 1 - mean(Error_flg)
        - Else NaN

    MeanRewardsPerTrial:
        - If 'NumRewards' exists, uses its mean
        - Else counts non-NaN reward timestamps per trial and averages
    """
    out = _agg_template(df, keys)

    wrong_col = _resolve_col(df, "Wrong_choice_flg")
    err_col   = _resolve_col(df, "Error_flg")

    if prefer_wrong_flag and (wrong_col is not None):
        s = pd.to_numeric(df[wrong_col], errors="coerce")
        pc = 1 - _gcol(df.assign(__w=s), "__w", keys).mean()
    elif err_col is not None:
        s = pd.to_numeric(df[err_col], errors="coerce")
        pc = 1 - _gcol(df.assign(__e=s), "__e", keys).mean()
    else:
        pc = pd.Series(np.nan, index=out.index)

    numr_col = _resolve_col(df, "NumRewards")
    if numr_col is not None:
        mr = _gcol(df, numr_col, keys).mean()
    else:
        L_ts = _resolve_col(df, "L_reward_ts")
        R_ts = _resolve_col(df, "R_reward_ts")
        if L_ts is not None and R_ts is not None:
            counts = df[[L_ts, R_ts]].notna().sum(axis=1).astype(float)
            mr = _gcol(df.assign(_tmp_counts=counts), "_tmp_counts", keys).mean()
        else:
            mr = pd.Series(np.nan, index=out.index)

    out["PercentCorrectChoice"] = pc.values
    out["MeanRewardsPerTrial"] = mr.values
    return out


def summarize_no_response_rate(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    well_col: str = "Well_on_ts",
) -> pd.DataFrame:
    """
    No-response metrics per session using a well-on timestamp column.

    Returns columns: keys..., no_resp_count, num_trials, no_resp_rate
    """
    out = _agg_template(df, keys)
    real_well = _resolve_col(df, well_col)
    if real_well is None:
        warnings.warn(f"Missing column {well_col}; cannot compute no-response.", RuntimeWarning)
        out["no_resp_count"] = np.nan
        out["no_resp_rate"] = np.nan
        return out

    nr = _gcol(df.assign(__nr=df[real_well].isna().astype(int)), "__nr", keys).sum()
    out["no_resp_count"] = nr.values
    out["no_resp_rate"] = (out["no_resp_count"] / out["num_trials"]).astype(float)
    return out


def summarize_prob_right(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    choice_col: str = "Well_id",
) -> pd.DataFrame:
    """Probability of responding Right per session."""
    out = _agg_template(df, keys)
    real_choice = _resolve_col(df, choice_col)
    if real_choice is None:
        warnings.warn(f"Missing column {choice_col}; cannot compute ProbR.", RuntimeWarning)
        out["ProbR"] = np.nan
        return out

    pr = _gcol(df.assign(__isR=(df[real_choice] == "R").astype(float)), "__isR", keys).mean()
    out["ProbR"] = pr.values
    return out


def summarize_prob_repeat(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    choice_col: str = "Well_id",
) -> pd.DataFrame:
    """Probability of repeating the previous choice per session."""
    dfo = _ensure_order(df, keys)
    real_choice = _resolve_col(dfo, choice_col)
    if real_choice is None:
        out = _agg_template(dfo, keys)
        warnings.warn(f"Missing column {choice_col}; cannot compute ProbSame.", RuntimeWarning)
        out["ProbSame"] = np.nan
        return out

    prev = _gcol(dfo, real_choice, keys).shift(1)
    same = (dfo[real_choice] == prev)

    # Keep keys by attaching helper columns to the original frame:
    denom = _gcol(dfo.assign(__ok=same.notna()), "__ok", keys).sum()
    num   = _gcol(dfo.assign(__same=same.fillna(False).astype(int)), "__same", keys).sum()

    out = _agg_template(dfo, keys)
    out["ProbSame"] = (num / denom.replace(0, np.nan)).values
    return out


def summarize_wsls(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    choice_col: str = "Well_id",
    err_col: str = "Error_flg",
    inplace_trial_flags: bool = False,
) -> pd.DataFrame:
    """
    Win–Stay / Lose–Switch summaries per session.
    """
    dfo = _ensure_order(df, keys)
    real_choice = _resolve_col(dfo, choice_col)
    real_err = _resolve_col(dfo, err_col)
    if real_choice is None or real_err is None:
        out = _agg_template(dfo, keys)
        warnings.warn(f"Missing {choice_col} or {err_col}; cannot compute WSLS.", RuntimeWarning)
        out["ProbWSLS"] = out["ProbLoseSwitch"] = out["ProbLoseStay"] = out["ProbWinStay"] = np.nan
        return out

    prev_choice = _gcol(dfo, real_choice, keys).shift(1)
    prev_err = pd.to_numeric(_gcol(dfo, real_err, keys).shift(1), errors="coerce")

    switched = (dfo[real_choice] != prev_choice)
    valid = prev_err.notna()

    # optional per-trial flags (write into *df* if requested)
    if inplace_trial_flags:
        df["LoseSwitch_all_flg"] = ((prev_err == 1) & switched & valid).astype("float")
        df["LoseStay_all_flg"]   = ((prev_err == 1) & (~switched) & valid).astype("float")
        df["WinStay_all_flg"]    = ((prev_err == 0) & (~switched) & valid).astype("float")

    # denominators (keep keys by using assign on the original frame)
    denom_ls = _gcol(dfo.assign(__ls=(valid & (prev_err == 1))), "__ls", keys).sum()
    denom_ws = _gcol(dfo.assign(__ws=(valid & (prev_err == 0))), "__ws", keys).sum()

    prob_wsls  = _gcol(dfo.assign(__w=(prev_err == switched).astype(float)), "__w", keys).mean()
    prob_ls    = _gcol(dfo.assign(__x=((prev_err == 1) & switched).astype(int)), "__x", keys).sum() / denom_ls.replace(0, np.nan)
    prob_lstay = _gcol(dfo.assign(__y=((prev_err == 1) & (~switched)).astype(int)), "__y", keys).sum() / denom_ls.replace(0, np.nan)
    prob_wstay = _gcol(dfo.assign(__z=((prev_err == 0) & (~switched)).astype(int)), "__z", keys).sum() / denom_ws.replace(0, np.nan)

    out = _agg_template(dfo, keys)
    out["ProbWSLS"]       = prob_wsls.values
    out["ProbLoseSwitch"] = prob_ls.values
    out["ProbLoseStay"]   = prob_lstay.values
    out["ProbWinStay"]    = prob_wstay.values
    return out

# --------------------------
# Convenience functions
# --------------------------

def attach_performance_metrics_to_trials(
    df: pd.DataFrame,
    *,
    keys: Sequence[str] = DEFAULT_SESSION_KEYS,
    prefer_wrong_flag: bool = False
) -> pd.DataFrame:
    """
    Enrich a trial-level DataFrame with both per-trial and per-session performance metrics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame of all trials with at least the columns in DEFAULT_SESSION_KEYS.
    keys : sequence of str, optional
        Column names used to identify a unique session, by default DEFAULT_SESSION_KEYS.
    prefer_wrong_flag : bool, optional
        Passed to summarize_session_performance.

    Returns
    -------
    pandas.DataFrame
        Copy of df with:
        - Trial-level metrics added (NumRewards, reaction times, ITI, high-value well, block IDs)
        - All session-level summaries merged back onto each trial
    """
    out = df.copy()

    # Per-trial enrichments
    out = add_num_rewards(out, prefer_counts=True)
    out = add_reaction_response_times(out)
    out = add_inter_trial_intervals(out, keys=keys)
    out = add_high_value_well_from_bolus(out)
    out = add_block_id_by_bolus(out, keys=keys)  # or add_block_id_by_prob

    # # Session summaries
    # perf   = summarize_session_performance(out, keys=keys, prefer_wrong_flag=prefer_wrong_flag)
    # pright = summarize_prob_right(out, keys=keys)
    # prept  = summarize_prob_repeat(out, keys=keys)
    # wsls   = summarize_wsls(out, keys=keys)
    # nrs    = summarize_no_response_rate(out, keys=keys)

    # # --- normalize dtypes on keys for ALL frames ---
    # out    = _normalize_key_types(out,    keys)
    # perf   = _normalize_key_types(perf,   keys)
    # pright = _normalize_key_types(pright, keys)
    # prept  = _normalize_key_types(prept,  keys)
    # wsls   = _normalize_key_types(wsls,   keys)
    # nrs    = _normalize_key_types(nrs,    keys)

    # # Merge summaries back
    # for summ in (perf, pright, prept, wsls, nrs):
    #     out = out.merge(summ, on=list(keys), how="left")

    # --- session summaries ---
    perf   = summarize_session_performance(out, keys=keys, prefer_wrong_flag=prefer_wrong_flag)
    pright = summarize_prob_right(out, keys=keys)
    prept  = summarize_prob_repeat(out, keys=keys)
    wsls   = summarize_wsls(out, keys=keys)
    nrs    = summarize_no_response_rate(out, keys=keys)

    # (optional but recommended) normalize key dtypes
    out    = _normalize_key_types(out,    keys)
    perf   = _normalize_key_types(perf,   keys)
    pright = _normalize_key_types(pright, keys)
    prept  = _normalize_key_types(prept,  keys)
    wsls   = _normalize_key_types(wsls,   keys)
    nrs    = _normalize_key_types(nrs,    keys)

    # If each summary carries a shared count column like 'NumTrials' or 'num_trials',
    # keep the one from `perf` and drop it from the others to avoid collisions.
    for s in (pright, prept, wsls, nrs):
        for col in ("NumTrials", "num_trials", "n_trials"):
            if col in s.columns:
                s.drop(columns=[col], inplace=True)

    # --- merges: bring in only *new* metrics each time ---
    out = _safe_left_merge(out,    perf,   keys)   # keep perf's count columns
    out = _safe_left_merge(out,    pright, keys)
    out = _safe_left_merge(out,    prept,  keys)
    out = _safe_left_merge(out,    wsls,   keys)
    out = _safe_left_merge(out,    nrs,    keys)


    return out
