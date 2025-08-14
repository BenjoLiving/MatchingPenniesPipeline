from typing import Callable, Optional, Sequence, Union, Iterable, Tuple, List
import pandas as pd
from statistics import NormalDist

MaskLike = Union[str, pd.Series, Callable[[pd.DataFrame], pd.Series], None]

def _to_mask(df: pd.DataFrame, m: MaskLike) -> pd.Series:
    """
    Convert an expression/Series/callable/None into a boolean mask aligned to df.index.
    - str: evaluated with DataFrame.eval (use &, |, ~ for logic)
    - Series: coerced to bool and reindexed
    - callable(df) -> Series: same as Series case
    - None: all True
    """
    if m is None:
        return pd.Series(True, index=df.index)
    if isinstance(m, str):
        s = df.eval(m, engine="python")
        return pd.Series(s, index=df.index).astype(bool)
    if callable(m):
        s = m(df)
        return pd.Series(s, index=df.index).astype(bool)
    # assume Series-like
    return pd.Series(m, index=df.index).astype(bool)

def _wilson_ci(n: int, N: int, alpha: float = 0.05) -> tuple[float, float]:
    """
    Wilson score interval for a binomial proportion.
    Returns (lo, hi). If N==0, returns (nan, nan).
    """
    import math
    if N == 0:
        return (float("nan"), float("nan"))
    z = NormalDist().inv_cdf(1 - alpha/2)
    phat = n / N
    denom = 1 + (z*z)/N
    center = (phat + (z*z)/(2*N)) / denom
    half = (z * math.sqrt((phat*(1-phat)/N) + (z*z)/(4*N*N))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

def probability_of(
    df: pd.DataFrame,
    condition: MaskLike,
    *,
    groupby: Optional[Sequence[str]] = None,
    denom: MaskLike = None,
    subset: MaskLike = None,
    alpha: float = 0.05,
    sort: bool = False,
) -> pd.DataFrame:
    """
    Compute P(condition | denom & subset) with optional grouping and Wilson CIs.

    Parameters
    ----------
    df : DataFrame
        Trials table (e.g., from study.all_trials(...)).
    condition : str | Series | callable(df)->Series
        Event whose probability you want, e.g.:
        - "response == 0"      (no response)
        - "EFS == 1"           (EFS event)
        - "choice == 'R'"      (go right)
        Use &, |, ~ for logic if it's a string expression.
    groupby : sequence[str], optional
        Columns to group by (e.g., ["_animal"], ["_animal","_session_idx"], ["_paradigm","_treatment"], ...).
    denom : str | Series | callable, optional
        What counts in the denominator. Default: all rows (after `subset`).
        Example: denom="response == 1" to exclude no-response trials.
    subset : str | Series | callable, optional
        Pre-filter rows before counting (e.g., "(_trial_idx >= 60) & (_paradigm == 'hallway_swap')").
    alpha : float, default 0.05
        For (1 - alpha) confidence intervals; alpha=0.05 → 95% CI.
    sort : bool, default False
        Forwarded to final sort by groupby keys.

    Returns
    -------
    DataFrame with columns:
        [groupby keys..., 'n', 'N', 'prob', 'ci_low', 'ci_high']
    where:
        n  = count of rows matching condition & denom & subset in group
        N  = count of rows matching denom & subset in group
        prob = n / N
        ci_low, ci_high = Wilson CI bounds
    """
    # global masks
    cond_mask   = _to_mask(df, condition)
    denom_mask  = _to_mask(df, denom)
    subset_mask = _to_mask(df, subset)

    total_mask = denom_mask & subset_mask
    num_mask   = cond_mask & total_mask

    # group or not
    if groupby and len(groupby) > 0:
        groups = df.groupby(list(groupby), dropna=False).indices  # dict: key -> index positions
        rows = []
        for key, idx in groups.items():
            idx = pd.Index(idx)
            N = int(total_mask.loc[idx].sum())
            n = int(num_mask.loc[idx].sum())
            p = (n / N) if N > 0 else float("nan")
            lo, hi = _wilson_ci(n, N, alpha)
            row = {"n": n, "N": N, "prob": p, "ci_low": lo, "ci_high": hi}
            # normalize key to tuple
            if not isinstance(key, tuple):
                key = (key,)
            row.update({k: v for k, v in zip(groupby, key)})
            rows.append(row)
        out = pd.DataFrame(rows)
        cols = list(groupby) + ["n", "N", "prob", "ci_low", "ci_high"]
        out = out[cols]
        if sort:
            out = out.sort_values(list(groupby))
        return out.reset_index(drop=True)
    else:
        N = int(total_mask.sum())
        n = int(num_mask.sum())
        p = (n / N) if N > 0 else float("nan")
        lo, hi = _wilson_ci(n, N, alpha)
        return pd.DataFrame([{"n": n, "N": N, "prob": p, "ci_low": lo, "ci_high": hi}])

def add_trial_bins(df: pd.DataFrame, bin_size: int = 10, *, col: str = "_trial_idx", out_col: str = "_trial_bin") -> pd.DataFrame:
    """
    Add a 1-based integer bin column over a trial index column.
    Example: with bin_size=10, trials 1..10 -> bin 1, 11..20 -> bin 2, etc.
    """
    out = df.copy()
    out[out_col] = ((out[col] - 1) // bin_size) + 1
    return out

def drop_constant_columns(
    df: pd.DataFrame,
    *,
    ignore_na: bool = True,
    protect: Iterable[str] = ("_experiment", "_paradigm", "_animal", "_treatment", "_session_idx", "_trial_idx"),
    inplace: bool = False,
    return_dropped: bool = True,
) -> Tuple[pd.DataFrame, List[str]] | pd.DataFrame:
    """
    Drop columns that are constant.

    Parameters
    ----------
    df : pd.DataFrame
        Input table.
    ignore_na : bool, default True
        If True: drop a column if all *non-NaN* values are identical
        (columns of all-NaN are also dropped).
        If False: drop only if *every* value (treating NaN as a value) is identical.
    protect : Iterable[str], default ()
        Column names never to drop (e.g., metadata like '_animal', '_session_idx').
    inplace : bool, default False
        If True, modify df in place.
    return_dropped : bool, default True
        If True, return (new_df, dropped_columns). If False, return only new_df.

    Returns
    -------
    (DataFrame, List[str]) or DataFrame
        The dataframe without constant columns, and (optionally) the list of dropped columns.

    Notes
    -----
    - With ignore_na=True (the default), columns like:
        [NaN, NaN, 'Mode 2', 'Mode 2']  → dropped (all non-NaN values identical)
        [NaN, NaN, NaN]                  → dropped (all NaN)
    - With ignore_na=False, the same column would *not* be dropped unless
      the entire column is exactly the same value (including NaN placement).
    """
    protect = set(protect)
    check_cols = [c for c in df.columns if c not in protect]

    to_drop: List[str] = []
    for c in check_cols:
        s = df[c]
        if ignore_na:
            # unique among non-NaN values
            u = s.dropna().nunique()
            if u <= 1:
                # drop if all-NaN (u==0) or all non-NaN equal (u==1)
                to_drop.append(c)
        else:
            # treat NaN as a distinct value; all entries must be identical
            if s.nunique(dropna=False) == 1:
                to_drop.append(c)

    if inplace:
        df.drop(columns=to_drop, inplace=True, errors="ignore")
        return (df, to_drop) if return_dropped else df
    else:
        out = df.drop(columns=to_drop, errors="ignore")
        return (out, to_drop) if return_dropped else out

# Analysis Utilities 
