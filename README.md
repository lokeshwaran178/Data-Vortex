# Data Vortex — Round 1 (Phase 1): Social Engine Data Recovery

Cleaning + EDA pipeline for the corrupted `Social_Engine_Posts` and
`Social_Engine_Users` datasets recovered from the Data Vortex recovery
terminal.

## Repo structure
```
data/
  Social_Engine_Posts_Corrupted.csv   # raw input
  Social_Engine_Users.csv             # raw input
  cleaned/
    posts_cleaned.csv / .json
    users_cleaned.csv / .json
notebook/
  DataVortex_Cleaning_EDA.ipynb       # full pipeline + EDA, pre-executed
src/
  clean_data.py                       # standalone, reusable cleaning module
report/
  EDA_Report.html                     # rendered notebook (view without Jupyter)
  figures/                            # exported chart PNGs
build_notebook.py                     # script that assembles the notebook programmatically
```

## Reproducing the pipeline
```bash
pip install pandas numpy matplotlib seaborn nbformat jupyter nbconvert
python src/clean_data.py                       # produces data/cleaned/*
jupyter nbconvert --to notebook --execute --inplace notebook/DataVortex_Cleaning_EDA.ipynb
```

## Corruption found & how it was resolved

| Issue | Rows affected | Resolution | Justification |
|---|---:|---|---|
| Exact duplicate rows | 360 | Dropped | Identical `post_id` + all fields = re-ingestion artifact |
| 3 mixed timestamp formats (ISO 8601 / Unix epoch / DD-MM-YYYY) | 12,360 (all) | Standardized to ISO 8601 UTC | DD-MM-YYYY confirmed (not MM-DD-YYYY) via day values >12 in the first field |
| `likes` negative sign-flip | 525 | `abs()` applied | `abs(negative likes)` distribution (mean 2457, range 11–4987) is statistically indistinguishable from the positive-likes distribution (mean 2495, range 0–5000) — verified before correcting, not assumed |
| `likes` missing | 1,858 | Left `NaN`, flagged `likes_was_missing` | No fabrication — imputing a mean/median would invent data not present in the source |
| `platform` missing | 1,846 | Labelled `"Unknown"`, flagged `platform_was_missing` | Preserves the row for non-platform analyses without guessing a value |
| `text_content` missing (blank + literal string `"NULL"`) | 1,770 | Left `NaN`, flagged `text_content_was_missing` | Text cannot be safely reconstructed |
| HTML entities in text (`&amp;`, etc.) | 341 | Decoded with `html.unescape` | Lossless, mechanical fix — not a judgment call |
| Two different missing-value encodings (blank cell vs. literal `"NULL"` string) | — | Unified to a single `NaN` representation | Ensures every later `.isna()` check is consistent across the whole file |

Users file (`Social_Engine_Users.csv`) was profiled and found already
consistent — single date format, no negative followers, no duplicate
`user_id`s — so only light-touch normalization (splitting `location` into
`city`/`country`, parsing `account_created` to datetime) was applied.

## Assumptions
- Timestamps are assumed UTC (no offset present in the source; consistent with the Unix-epoch rows).
- No missing or corrupted value was ever filled in with a guessed/fabricated number — every unresolved gap stays `NaN` and is flagged in a companion boolean column (`*_was_missing` / `*_was_corrupted_sign`) so downstream analysis can explicitly choose to exclude it.

## Key EDA finding worth flagging in the submission
Engagement metrics (likes/shares/comments) are nearly flat across platforms
and show ~0 correlation with follower count — not consistent with how
engagement behaves on real social platforms. This suggests these fields were
generated close to randomly in the corrupted dataset, and is called out as a
limitation in the notebook rather than over-interpreted as a real insight
(e.g. "Platform X drives more engagement").
