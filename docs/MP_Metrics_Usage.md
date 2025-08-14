
# MP Metrics — Usage Guide

Vectorized metrics and annotators for **Matching Pennies** data stored in pandas DataFrames. Designed to work seamlessly with your `StudyDataset` loader (hierarchy: Study → Experiment → Paradigm → Animal → Session).

This guide covers:
- What each function computes
- How to call them directly or via `StudyDataset` convenience methods
- Column naming & robustness
- Practical examples and troubleshooting

---

## TL;DR (Quickstart)

```python
from database.DataFrameDataBase import StudyDataset

study = StudyDataset.from_roots(
    roots=[
        ("PFC_lesion", "normal", "/path/to/normal_mp"),
        ("PFC_lesion", "hallway_swap", "/path/to/switching_halls"),
    ],
    animal_treatment_map_csv="/path/to/animal_important_dates_csv.csv",
    filename_regex=r"(?P<animal>P\d{4,})_(?P<date>\d{4}_\d{2}_\d{2}).*\.csv$",
    low_memory=False,
    skipinitialspace=True,  # helps with spacey CSVs
)

# One-line session dashboard (performance, no-response, ProbR, ProbSame, WSLS)
dash = study.summarize_sessions(experiment="PFC_lesion", paradigm="normal")
print(dash.head())

# Individual summaries if you want just one metric family:
wsls = study.summarize_wsls(experiment="PFC_lesion", paradigm="normal")
probR = study.summarize_prob_right(experiment="PFC_lesion", paradigm="normal")
```

---

## Column Robustness (case/space-insensitive)

The metrics module is **forgiving** about header styles. It resolves the following logical names against your DataFrame **case- and whitespace-insensitively**:

- `Well_on_ts`, `Well_id`, `Error_flg`
- `Tone_on_ts`, `Odor_port_off_ts`, `Odor_port_on_ts`, `Trial_start_ts`
- `House_light_on_ts`, `L_reward_ts`, `R_reward_ts`
- `L_reward_num`, `R_reward_num`
- `L_reward_prob`, `R_reward_prob`
- `Wrong_choice_flg` (optional)

> Example: headers like `" well_on_ts"` or `"WELL_ID"` will be found.

If a required column genuinely doesn’t exist, the function will:
- emit a clear `RuntimeWarning`, and
- return `NaN` for the affected metrics (instead of erroring).

**Tip:** the loader already strips BOMs and trims spaces:
```python
# in StudyDataset.from_roots(...)
df.rename(columns=lambda s: str(s).replace("\ufeff","").strip(), inplace=True)
```

---

## Session Keys & Trial Order

Most metrics group by the session identity keys:

```
DEFAULT_SESSION_KEYS = ("_experiment", "_paradigm", "_animal", "_treatment", "_session_idx")
```

Trial-order dependent computations (e.g., ProbRepeat, WSLS) also rely on `_trial_idx` **if present** (it is present in `StudyDataset.all_trials`). The library will stable-sort by keys (+ `_trial_idx` if available).

---

## What’s Included

### Trial-level **Annotators** (add columns per trial)

- `add_reaction_response_times(df, colmap=None, inplace=False)`  
  Adds: `ReactTime = Odor_port_off_ts - Tone_on_ts`,  
  `ResponseTime = Well_on_ts - Odor_port_off_ts`,  
  `InitTime = Odor_port_on_ts - Trial_start_ts`

- `add_inter_trial_intervals(df, keys=DEFAULT_SESSION_KEYS, colmap=None, inplace=False)`  
  Adds: `InterTrialInterval`, `InterTrialInterval_afterLose`, `InterTrialInterval_afterWin`  
  ITI(n) = `Odor_port_on_ts(n)` − `max(House_light_on_ts, L_reward_ts, R_reward_ts)(n−1)`

- `add_num_rewards(df, prefer_counts=True, inplace=False)`  
  Adds: `NumRewards`  
  If `prefer_counts=False` and `L_reward_num/R_reward_num` exist: sums those.  
  Else counts non-NaN `L_reward_ts/R_reward_ts` per trial.

- `add_high_value_well_from_bolus(df, inplace=False)`  
  Adds: `HighValWell_id ∈ {L,R,N}`, `ChoseHighVal_flg`  
  Based on `R_reward_num − L_reward_num` sign.

- `add_block_id_by_bolus(df, min_trials=1, keys=DEFAULT_SESSION_KEYS, inplace=False)`  
  Adds: `Block_id ∈ {Big_R,Big_L,N}`, `Block_number`, `Block_trial_indx`  
  Filters runs shorter than `min_trials` to `N`.

- `add_block_id_by_prob(df, keys=DEFAULT_SESSION_KEYS, inplace=False)`  
  Adds: `Block_id` like `"40/60"`, `Block_agg_id` like `"60/40"`,  
  `Block_trial_indx`, `ChoseHighProb_flg`, `ChoseHighProb_lastTrial_flg`  
  Requires `L_reward_prob` and `R_reward_prob`.

- `delete_blocks_with_few_trials(df, thresh, keys=DEFAULT_SESSION_KEYS)`  
  Returns `df` with trials removed if their `Block_id`/`Block_agg_id` occurs `< thresh` times **within session**.

### Session-level **Summaries** (1 row per session)

- `summarize_session_performance(df, keys=DEFAULT_SESSION_KEYS, prefer_wrong_flag=False)`  
  Adds: `num_trials`, `PercentCorrectChoice`, `MeanRewardsPerTrial`  
  `PercentCorrectChoice` uses `1 - mean(Wrong_choice_flg)` if available & requested, else `1 - mean(Error_flg)`.

- `summarize_no_response_rate(df, keys=DEFAULT_SESSION_KEYS, well_col="Well_on_ts")`  
  Adds: `num_trials`, `no_resp_count`, `no_resp_rate`  
  No response := `Well_on_ts` is `NaN`.

- `summarize_prob_right(df, keys=DEFAULT_SESSION_KEYS, choice_col="Well_id")`  
  Adds: `num_trials`, `ProbR` (fraction of `"R"` choices).

- `summarize_prob_repeat(df, keys=DEFAULT_SESSION_KEYS, choice_col="Well_id")`  
  Adds: `num_trials`, `ProbSame` (probability of repeating previous choice).  
  First trial per session is excluded from the denominator.

- `summarize_wsls(df, keys=DEFAULT_SESSION_KEYS, choice_col="Well_id", err_col="Error_flg", inplace_trial_flags=False)`  
  Adds: `num_trials`, `ProbWSLS`, `ProbLoseSwitch`, `ProbLoseStay`, `ProbWinStay`  
  `WSLS` = `prev_err == switched_choice`  
  Optionally writes per-trial flags into `df`: `LoseSwitch_all_flg`, `LoseStay_all_flg`, `WinStay_all_flg`.

---

## Using via `StudyDataset` (Convenience Methods)

All the above are exposed as methods that first call `all_trials(...)` and then run the metric:

**Annotators**
- `study.annotate_reaction_response_times(...)`
- `study.annotate_inter_trial_intervals(...)`
- `study.annotate_num_rewards(...)`
- `study.annotate_high_value_well_from_bolus(...)`
- `study.annotate_block_id_by_bolus(...)`
- `study.annotate_block_id_by_prob(...)`
- `study.filter_blocks_with_few_trials(...)`

**Summaries**
- `study.summarize_session_performance(...)`
- `study.summarize_no_response_rate(...)`
- `study.summarize_prob_right(...)`
- `study.summarize_prob_repeat(...)`
- `study.summarize_wsls(...)`

**Dashboard combo**
- `study.summarize_sessions(experiment=None, paradigm=None, treatment=None, prefer_wrong_flag=False, well_col="Well_on_ts", choice_col="Well_id", err_col="Error_flg")`  
  Merges: performance + no-response + ProbR + ProbSame + WSLS.  
  If any component is missing required inputs, it returns `NaN` for *that* piece and keeps the rest.

### Examples

```python
# 1) One-pass dashboard for an experiment/paradigm subset
dash = study.summarize_sessions(experiment="PFC_lesion", paradigm="normal")
print(dash.head())

# 2) Only WSLS for sham animals across paradigms
wsls = study.summarize_wsls(experiment="PFC_lesion", treatment="sham")

# 3) Add ITIs and then summarize performance on hallway_swap
ann = study.annotate_inter_trial_intervals(experiment="PFC_lesion", paradigm="hallway_swap")
perf = study.summarize_session_performance(experiment="PFC_lesion", paradigm="hallway_swap")
```

---

## Customizing Column Names

- Annotators accept `colmap={logical_name: actual_df_column}` for flexible mapping:
  ```python
  add_reaction_response_times(df, colmap={"tone_on": "toneOn", "well_on": "well_on_ts"})
  ```
- Summaries accept simple strings to override defaults:
  ```python
  summarize_no_response_rate(df, well_col="well_on_ts")
  summarize_prob_right(df, choice_col="well_id")
  summarize_wsls(df, choice_col="well_id", err_col="error_flg")
  ```

> You rarely need these if your headers are close to the defaults; the module already resolves `" well_id"`, `"Well_ID"`, etc.

---

## What Each Metric Requires

| Function                          | Needs at minimum                             | Notes |
|----------------------------------|----------------------------------------------|------|
| `summarize_session_performance`  | `Error_flg` **or** `Wrong_choice_flg`; reward timestamps or `NumRewards` | `prefer_wrong_flag=True` to prioritize `Wrong_choice_flg`. |
| `summarize_no_response_rate`     | `Well_on_ts`                                 | Counts `NaN` as no response. |
| `summarize_prob_right`           | `Well_id`                                    | Treats `"R"` as right. |
| `summarize_prob_repeat`          | `Well_id`                                    | Excludes first trial per session. |
| `summarize_wsls`                 | `Well_id`, `Error_flg`                       | Computes WS, LS, etc. |
| `add_reaction_response_times`    | `Tone_on_ts`, `Odor_port_off_ts`, `Odor_port_on_ts`, `Trial_start_ts`, `Well_on_ts` | Adds `ReactTime`, `ResponseTime`, `InitTime`. |
| `add_inter_trial_intervals`      | `Odor_port_on_ts`, `House_light_on_ts`, `L_reward_ts`, `R_reward_ts`, `Error_flg` | Adds ITI family. |
| `add_num_rewards`                | `L_reward_ts` & `R_reward_ts` **or** counts  | Counts timestamps by default; can use `*_num`. |
| `add_high_value_well_from_bolus` | `L_reward_num`, `R_reward_num` (+ `Well_id` optional) | Adds `HighValWell_id` and `ChoseHighVal_flg`. |
| `add_block_id_by_bolus`          | `L_reward_num`, `R_reward_num`               | Blocks by side with more boli. |
| `add_block_id_by_prob`           | `L_reward_prob`, `R_reward_prob`             | Blocks labeled like `"60/40"` and aggregated `"60/40"`. |

---

## Performance Tips

- **Filter early**: pass `experiment=`, `paradigm=`, `treatment=` to `StudyDataset` methods to reduce data size before metrics.
- **Avoid copies**: Use the default `inplace=False`; only set `inplace=True` when you need to mutate the original frame.
- **Select columns** when plotting or exporting (e.g., `dash[["_animal","_session_idx","ProbR"]]`).

---

## Troubleshooting

- **Warnings like “Missing column Well_on_ts …”**  
  Expected when the needed inputs aren’t present in your CSVs.  
  Double-check `df.columns`:
  ```python
  df = study.all_trials(experiment="PFC_lesion", paradigm="normal")
  print([c for c in df.columns if "well" in c.lower() or "error" in c.lower()][:30])
  ```
  If names differ substantially, pass overrides (e.g., `choice_col="Choice"`).

- **`KeyError: '_experiment'` or similar**  
  Fixed in the latest module by ensuring group-by keys are preserved during helper column creation.  
  Make sure you rebuilt `mp_metrics.py` and reloaded your Python session.

- **All `NaN` metrics**  
  Means required inputs are missing; start by printing the relevant columns and confirm they exist after load.  
  The loader trims spaces from headers; also consider `skipinitialspace=True` in `from_roots(...)`.

---

## Reproducing MATLAB Semantics

The functions mirror your legacy pipeline:
- “No response” := `Well_on_ts` is `NaN`.
- WSLS family uses previous trial’s `Error_flg` and choice change.
- Performance uses `Wrong_choice_flg` if requested (when available), else `Error_flg`.

---

## Changelog (recent)

- **Header robustness**: All metrics/annotators now resolve column names case/whitespace-insensitively.
- **Group-by bug fix**: Helpers now attach temporary columns via `assign` on the original frame to preserve session keys.
- **StudyDataset convenience**: Added `study.summarize_sessions(...)` dashboard and individual `study.summarize_*`/`study.annotate_*` helpers.
