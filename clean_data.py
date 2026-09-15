"""
Data Vortex - Round 1 (Phase 1)
Cleaning pipeline for Social_Engine_Posts_Corrupted.csv and Social_Engine_Users.csv

Design principle followed throughout: NEVER fabricate values. Where a value is
missing/corrupted and cannot be safely recovered, it is left as null (NaN) and
flagged with a boolean *_was_missing / *_was_corrupted column so it can be
excluded from downstream aggregates instead of silently biasing them.

Run:
    python src/clean_data.py
Outputs (into ../data/cleaned/):
    posts_cleaned.csv, posts_cleaned.json
    users_cleaned.csv, users_cleaned.json
"""

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_DIR = RAW_DIR / "cleaned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSTS_IN = RAW_DIR / "Social_Engine_Posts_Corrupted.csv"
USERS_IN = RAW_DIR / "Social_Engine_Users.csv"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def normalize_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """
    The raw file mixes two different 'missing' encodings: real empty cells
    (parsed by pandas as NaN) AND the literal string 'NULL' (any case,
    with/without surrounding whitespace). Collapse both to a single NaN
    representation so every later missing-value check is consistent.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().upper() == "NULL" else v
        )
    return df


def parse_mixed_timestamp(value: str):
    """
    timestamp column arrives in three different formats:
      - ISO 8601, e.g. 2025-02-03T02:09:31
      - Unix epoch (seconds), e.g. 1731041269
      - DD-MM-YYYY, e.g. 28-05-2024   (day-first; verified against dataset's
        own date range - values >12 appear in the first field, e.g. 25-09-2024,
        which rules out MM-DD-YYYY)
    Returns a single tz-aware (UTC) pandas.Timestamp, or NaT if unparseable.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()

    # Unix epoch: 10 digits, all numeric
    if re.fullmatch(r"\d{10}", s):
        return pd.to_datetime(int(s), unit="s", utc=True)

    # DD-MM-YYYY
    if re.fullmatch(r"\d{2}-\d{2}-\d{4}", s):
        return pd.to_datetime(s, format="%d-%m-%Y", utc=True)

    # ISO 8601 (no explicit offset in source -> assume UTC, matches epoch convention above)
    try:
        return pd.to_datetime(s, utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def extract_hashtags(text):
    if pd.isna(text):
        return []
    return re.findall(r"#(\w+)", text)


def extract_mentions(text):
    if pd.isna(text):
        return []
    return re.findall(r"@(\w+)", text)


# --------------------------------------------------------------------------- #
# Posts pipeline
# --------------------------------------------------------------------------- #
def clean_posts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    n_raw = len(df)

    # 1) Unify missing-value tokens ("NULL" string -> NaN)
    df = normalize_null_tokens(df)

    # 2) Drop exact full-row duplicates (same post_id + identical content
    #    in every column => a genuine re-ingestion duplicate, not two
    #    distinct posts that happen to share an id)
    n_before_dedup = len(df)
    df = df.drop_duplicates(keep="first")
    n_dupes_dropped = n_before_dedup - len(df)

    # 3) Standardize timestamp -> single ISO 8601 UTC column
    df["timestamp_clean"] = df["timestamp"].apply(parse_mixed_timestamp)
    n_bad_timestamps = df["timestamp_clean"].isna().sum()
    df = df.drop(columns=["timestamp"]).rename(columns={"timestamp_clean": "timestamp"})

    # 4) Decode HTML entities in free text (&amp; -> &, etc.)
    df["text_content_was_missing"] = df["text_content"].isna()
    df["text_content"] = df["text_content"].apply(
        lambda t: html.unescape(t) if isinstance(t, str) else t
    )

    # 5) likes: negative values are a sign-flip corruption, not a distinct
    #    population. Verified: abs(negative likes) has mean 2457 / range
    #    11-4987, statistically indistinguishable from the positive-likes
    #    distribution (mean 2495 / range 0-5000). Safe to correct with abs().
    #    True missing values (NaN) are left as NaN - not imputed - and flagged.
    df["likes_was_corrupted_sign"] = df["likes"] < 0
    df["likes_was_missing"] = df["likes"].isna()
    df["likes"] = df["likes"].abs()

    # 6) platform: missing values are labelled "Unknown" (an explicit,
    #    honest category) rather than dropped or guessed, so the rows are
    #    still usable for engagement analysis, just excluded from
    #    platform-specific breakdowns.
    df["platform_was_missing"] = df["platform"].isna()
    df["platform"] = df["platform"].fillna("Unknown")

    # 7) Derived EDA-friendly fields (kept separate from the raw text so
    #    nothing about the original content is altered/fabricated)
    df["hashtags"] = df["text_content"].apply(extract_hashtags)
    df["mentions"] = df["text_content"].apply(extract_mentions)
    df["hashtag_count"] = df["hashtags"].apply(len)
    df["mention_count"] = df["mentions"].apply(len)

    # 8) Dtypes
    df["shares"] = df["shares"].astype("Int64")
    df["comments"] = df["comments"].astype("Int64")
    df["likes"] = df["likes"].astype("Float64")

    df = df.reset_index(drop=True)

    print(f"[posts] raw rows: {n_raw}")
    print(f"[posts] exact duplicate rows dropped: {n_dupes_dropped}")
    print(f"[posts] unparseable timestamps -> NaT: {n_bad_timestamps}")
    print(f"[posts] final rows: {len(df)}")
    return df


# --------------------------------------------------------------------------- #
# Users pipeline
# --------------------------------------------------------------------------- #
def clean_users(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = normalize_null_tokens(df)

    # account_created is already a single consistent YYYY-MM-DD format and
    # follower_count has no negatives / no missing values in this file, so
    # only light-touch normalization is needed.
    df["account_created"] = pd.to_datetime(df["account_created"], format="%Y-%m-%d", utc=True)

    # split "City, Country" into two columns for easier grouping in EDA
    split_loc = df["location"].str.split(",", n=1, expand=True)
    df["city"] = split_loc[0].str.strip()
    df["country"] = split_loc[1].str.strip()

    dupes = df.duplicated(subset="user_id").sum()
    print(f"[users] rows: {len(df)}, duplicate user_id rows: {dupes}")
    return df


# --------------------------------------------------------------------------- #
def main():
    posts = clean_posts(POSTS_IN)
    users = clean_users(USERS_IN)

    posts.to_csv(OUT_DIR / "posts_cleaned.csv", index=False)
    posts.to_json(OUT_DIR / "posts_cleaned.json", orient="records", date_format="iso")

    users.to_csv(OUT_DIR / "users_cleaned.csv", index=False)
    users.to_json(OUT_DIR / "users_cleaned.json", orient="records", date_format="iso")

    print(f"\nCleaned files written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
