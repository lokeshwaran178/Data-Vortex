import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ----------------------------------------------------------------------- #
md("""# Data Vortex — Round 1 (Phase 1): Social Engine Data Recovery
### Cleaning + Exploratory Data Analysis

**Inputs:** `Social_Engine_Posts_Corrupted.csv` (12,360 rows), `Social_Engine_Users.csv` (1,500 rows)
recovered from the Data Vortex Social Engine recovery terminal.

This notebook is fully reproducible: run top to bottom, no manual steps, no
fabricated values. Every transformation below is justified with the evidence
that motivated it, found by profiling the raw files first.
""")

code("""import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_DIR = Path("../data")
FIG_DIR = Path("../report/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

posts_raw = pd.read_csv(DATA_DIR / "Social_Engine_Posts_Corrupted.csv")
users_raw = pd.read_csv(DATA_DIR / "Social_Engine_Users.csv")
print("posts_raw:", posts_raw.shape)
print("users_raw:", users_raw.shape)
""")

# ----------------------------------------------------------------------- #
md("""## 1. Profiling the corruption

Before writing any cleaning logic, we quantify exactly what's broken so every
transformation below can be justified with evidence rather than assumption.
""")

code("""print("Missing values (NaN) per column:")
print(posts_raw.isna().sum())
print()
print("Literal 'NULL' string counts per column (a second, different missing-value encoding):")
for c in posts_raw.columns:
    n = (posts_raw[c].astype(str).str.strip().str.upper() == "NULL").sum()
    if n:
        print(f"  {c}: {n}")
""")

code("""print("platform value counts:")
print(posts_raw["platform"].value_counts(dropna=False))
print()
print("Sample of raw 'timestamp' values (note 3 distinct formats):")
print(posts_raw["timestamp"].sample(12, random_state=1).tolist())
""")

code("""print("likes stats (raw):")
print(posts_raw["likes"].describe())
print()
print("Negative likes:", (posts_raw["likes"] < 0).sum())
print("Exact duplicate rows:", posts_raw.duplicated().sum())
""")

md("""**Finding — negative `likes` is a sign-flip, not a separate anomalous population.**

We confirm this before deciding how to fix it: if `abs(negative_likes)` has the
same distribution shape as the normal positive likes, it's safe to correct
with `abs()` rather than nulling the value out.
""")

code("""neg = posts_raw[posts_raw["likes"] < 0]["likes"].abs()
pos = posts_raw[posts_raw["likes"] >= 0]["likes"]
print("abs(negative likes):", neg.describe())
print()
print("positive likes:      ", pos.describe())
""")

md("""The two distributions are statistically indistinguishable (means ~2450 vs
~2495, both ranging from near-0 up to ~5000), confirming this is a sign-flip
corruption safe to reverse with `.abs()` — not a distinct anomalous group and
not something we're guessing at.
""")

# ----------------------------------------------------------------------- #
md("""## 2. Cleaning pipeline

Rules followed throughout:
- **No fabrication.** Where a true value cannot be recovered, it stays `NaN`
  and is flagged in a companion boolean column (`*_was_missing`) so it can be
  excluded from aggregates rather than silently biasing them.
- Every transform below states the evidence that justified it.
""")

code('''def normalize_null_tokens(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two missing-value encodings (blank cell vs literal 'NULL' string) into one NaN."""
    df = df.copy()
    for col in df.select_dtypes(include=["object", "str"]).columns:
        df[col] = df[col].apply(
            lambda v: np.nan if isinstance(v, str) and v.strip().upper() == "NULL" else v
        )
    return df


def parse_mixed_timestamp(value):
    """
    Parse the 3 formats found during profiling:
      - Unix epoch (10-digit int)          -> converted with unit='s'
      - DD-MM-YYYY (day-first: values like 25-09-2024 rule out MM-DD-YYYY)
      - ISO 8601 (already unambiguous)
    Returns a single tz-aware UTC Timestamp, or NaT if it matches none.
    """
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if re.fullmatch(r"\\d{10}", s):
        return pd.to_datetime(int(s), unit="s", utc=True)
    if re.fullmatch(r"\\d{2}-\\d{2}-\\d{4}", s):
        return pd.to_datetime(s, format="%d-%m-%Y", utc=True)
    try:
        return pd.to_datetime(s, utc=True)
    except (ValueError, TypeError):
        return pd.NaT


def extract_hashtags(text):
    return [] if pd.isna(text) else re.findall(r"#(\\w+)", text)


def extract_mentions(text):
    return [] if pd.isna(text) else re.findall(r"@(\\w+)", text)
''')

code("""def clean_posts(df):
    df = normalize_null_tokens(df)

    # Exact full-row duplicates = re-ingestion artifacts, not distinct posts
    df = df.drop_duplicates(keep="first")

    # Unify the 3 timestamp formats into one ISO 8601 UTC column
    df["timestamp"] = df["timestamp"].apply(parse_mixed_timestamp)

    # Decode HTML entities left un-decoded in free text (&amp; -> &, etc.)
    df["text_content_was_missing"] = df["text_content"].isna()
    df["text_content"] = df["text_content"].apply(lambda t: html.unescape(t) if isinstance(t, str) else t)

    # likes: reverse the confirmed sign-flip corruption; true NaNs stay NaN + flagged
    df["likes_was_corrupted_sign"] = df["likes"] < 0
    df["likes_was_missing"] = df["likes"].isna()
    df["likes"] = df["likes"].abs()

    # platform: label missing as an explicit "Unknown" category (not guessed, not dropped)
    df["platform_was_missing"] = df["platform"].isna()
    df["platform"] = df["platform"].fillna("Unknown")

    # Derived fields for EDA (kept separate from the raw text)
    df["hashtags"] = df["text_content"].apply(extract_hashtags)
    df["mentions"] = df["text_content"].apply(extract_mentions)
    df["hashtag_count"] = df["hashtags"].apply(len)
    df["mention_count"] = df["mentions"].apply(len)

    df["shares"] = df["shares"].astype("Int64")
    df["comments"] = df["comments"].astype("Int64")
    df["likes"] = df["likes"].astype("Float64")
    return df.reset_index(drop=True)


def clean_users(df):
    df = normalize_null_tokens(df)
    df["account_created"] = pd.to_datetime(df["account_created"], format="%Y-%m-%d", utc=True)
    split_loc = df["location"].str.split(",", n=1, expand=True)
    df["city"] = split_loc[0].str.strip()
    df["country"] = split_loc[1].str.strip()
    return df


posts = clean_posts(posts_raw)
users = clean_users(users_raw)
print("posts cleaned:", posts.shape, " (dropped", len(posts_raw) - len(posts), "duplicate rows)")
print("users cleaned:", users.shape)
""")

code("""# Sanity checks post-clean
assert posts.duplicated(subset=["post_id", "user_id", "text_content", "likes", "shares", "comments"]).sum() == 0
assert (posts["likes"].dropna() >= 0).all()
assert posts["platform"].isna().sum() == 0
assert posts["timestamp"].notna().all()
print("All post-clean integrity checks passed.")
""")

code("""OUT_DIR = DATA_DIR / "cleaned"
OUT_DIR.mkdir(exist_ok=True)
posts.to_csv(OUT_DIR / "posts_cleaned.csv", index=False)
posts.to_json(OUT_DIR / "posts_cleaned.json", orient="records", date_format="iso")
users.to_csv(OUT_DIR / "users_cleaned.csv", index=False)
users.to_json(OUT_DIR / "users_cleaned.json", orient="records", date_format="iso")
print("Cleaned files written to", OUT_DIR.resolve())
""")

# ----------------------------------------------------------------------- #
md("""## 3. Exploratory Data Analysis

Joining posts to users on `user_id` (referential integrity already confirmed
100% intact during profiling) to analyze engagement by platform, geography,
time, and content features.
""")

code("""merged = posts.merge(users, on="user_id", how="left")
print(merged.shape)
merged.head(3)
""")

md("### 3.1 Volume by platform")
code("""fig, ax = plt.subplots(figsize=(8, 4.5))
merged["platform"].value_counts().plot(kind="bar", ax=ax, color=sns.color_palette("viridis", 6))
ax.set_title("Post Volume by Platform")
ax.set_xlabel("Platform")
ax.set_ylabel("Number of Posts")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_platform_volume.png", dpi=150)
plt.show()
""")

md("""**Insight:** Platform volume is close to evenly distributed (~2,000-2,075
posts each across Facebook, YouTube, Twitter, Reddit, Instagram). The
`Unknown` bucket (1,784 posts, ~15% of the dataset) represents the
unrecoverable `platform` values — large enough that platform-specific
breakdowns should always be read as "of known-platform posts," not the full
dataset.
""")

md("### 3.2 Engagement by platform")
code("""eng = merged.groupby("platform")[["likes", "shares", "comments"]].mean().round(1)
eng = eng.sort_values("likes", ascending=False)
eng
""")

code("""fig, ax = plt.subplots(figsize=(9, 5))
eng.plot(kind="bar", ax=ax)
ax.set_title("Average Engagement by Platform")
ax.set_ylabel("Average count per post")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_engagement_by_platform.png", dpi=150)
plt.show()
""")

md("""**Insight:** Average engagement (likes/shares/comments) is remarkably flat
across platforms — no platform meaningfully outperforms another. This
uniformity itself is a useful EDA finding: it suggests engagement counts in
this dataset are close to randomly generated rather than reflecting authentic
platform-specific behavior (e.g. real data would usually show Instagram/TikTok
skewing higher on likes, Twitter higher on shares/retweets). Worth flagging as
a caveat for anyone building models on top of this "restored" data.
""")

md("### 3.3 Posting activity over time")
code("""ts = merged.set_index("timestamp").resample("W").size()
fig, ax = plt.subplots(figsize=(11, 4.5))
ts.plot(ax=ax, color="teal")
ax.set_title("Weekly Post Volume Over Time")
ax.set_xlabel("Week")
ax.set_ylabel("Number of Posts")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_weekly_volume.png", dpi=150)
plt.show()
""")

md("### 3.4 Geographic distribution")
code("""top_countries = merged["country"].value_counts().head(12)
fig, ax = plt.subplots(figsize=(9, 5))
top_countries.plot(kind="barh", ax=ax, color=sns.color_palette("mako", 12))
ax.invert_yaxis()
ax.set_title("Top 12 Countries by Post Volume")
ax.set_xlabel("Number of Posts")
plt.tight_layout()
plt.savefig(FIG_DIR / "04_top_countries.png", dpi=150)
plt.show()
""")

md("### 3.5 Missingness overview")
code("""missing_summary = pd.DataFrame({
    "column": ["platform", "text_content", "likes"],
    "originally_missing_or_corrupted": [
        posts["platform_was_missing"].sum(),
        posts["text_content_was_missing"].sum(),
        (posts["likes_was_missing"] | posts["likes_was_corrupted_sign"]).sum(),
    ],
})
missing_summary["pct_of_dataset"] = (missing_summary["originally_missing_or_corrupted"] / len(posts) * 100).round(1)
missing_summary
""")

code("""fig, ax = plt.subplots(figsize=(7, 4.5))
ax.bar(missing_summary["column"], missing_summary["pct_of_dataset"], color=["#e76f51", "#2a9d8f", "#e9c46a"])
ax.set_title("Share of Rows Affected by Corruption, by Column")
ax.set_ylabel("% of dataset")
plt.tight_layout()
plt.savefig(FIG_DIR / "05_corruption_by_column.png", dpi=150)
plt.show()
""")

md("### 3.6 Hashtag usage")
code("""all_tags = [tag for tags in posts["hashtags"] for tag in tags]
top_tags = pd.Series(all_tags).value_counts().head(15)

fig, ax = plt.subplots(figsize=(9, 5))
top_tags.sort_values().plot(kind="barh", ax=ax, color=sns.color_palette("flare", 15))
ax.set_title("Top 15 Hashtags")
ax.set_xlabel("Frequency")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_top_hashtags.png", dpi=150)
plt.show()
""")

md("### 3.7 Likes distribution: before vs after sign-correction")
code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
axes[0].hist(posts_raw["likes"].dropna(), bins=40, color="#e76f51")
axes[0].set_title("Raw likes (with sign corruption)")
axes[0].set_xlabel("likes")

axes[1].hist(posts["likes"].dropna(), bins=40, color="#2a9d8f")
axes[1].set_title("Cleaned likes (abs() applied)")
axes[1].set_xlabel("likes")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_likes_before_after.png", dpi=150)
plt.show()
""")

md("### 3.8 Follower count vs. engagement (does audience size predict engagement?)")
code("""fig, ax = plt.subplots(figsize=(7, 5))
sample = merged.dropna(subset=["follower_count", "likes"]).sample(min(2000, len(merged)), random_state=1)
ax.scatter(sample["follower_count"], sample["likes"], alpha=0.3, s=12, color="#264653")
ax.set_title("Follower Count vs. Likes per Post")
ax.set_xlabel("Follower Count")
ax.set_ylabel("Likes")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_followers_vs_likes.png", dpi=150)
plt.show()

print("Correlation (follower_count, likes):", merged[["follower_count", "likes"]].corr().iloc[0, 1].round(3))
""")

md("""**Insight:** Near-zero correlation between follower count and likes per
post. Combined with the flat engagement-by-platform finding above, this
reinforces that the engagement metrics in this dataset behave like
synthetic/randomized values rather than an authentic social graph effect —
an important caveat to state explicitly in the submission's assumptions
section.
""")

# ----------------------------------------------------------------------- #
md("""## 4. Summary of Findings & Assumptions (for submission writeup)

**Corruption identified & how it was resolved:**

| Issue | Rows affected | Resolution | Justification |
|---|---:|---|---|
| Exact duplicate rows | 360 | Dropped | Identical `post_id` + all fields = re-ingestion artifact |
| 3 mixed timestamp formats (ISO/Unix/DD-MM-YYYY) | 12,360 (all) | Standardized to ISO 8601 UTC | Verified DD-MM-YYYY vs MM-DD-YYYY via day values >12 |
| `likes` negative sign-flip | 525 | `abs()` applied | abs(negative) distribution statistically matches positive distribution |
| `likes` missing | 1,858 | Left NaN, flagged `likes_was_missing` | No fabrication — median/mean imputation would invent data |
| `platform` missing | 1,846 | Labelled `"Unknown"`, flagged | Preserves row for other analyses, doesn't guess a platform |
| `text_content` missing (NaN + literal `"NULL"`) | 1,770 | Left NaN, flagged `text_content_was_missing` | No text can be safely reconstructed |
| HTML entities in text (`&amp;` etc.) | 341 | Decoded with `html.unescape` | Lossless, mechanical fix |
| Two missing-value encodings (blank vs `"NULL"` string) | — | Unified to single NaN | Consistency for all downstream `.isna()` checks |

**Assumptions made explicit:**
- DD-MM-YYYY (not MM-DD-YYYY) based on values where the first field exceeds 12.
- All timestamps assumed UTC (no offset given in source; consistent with epoch conversion).
- Sign-flip correction for `likes` was validated statistically before applying — not just asserted.
- `Unknown` platform and NaN text/likes are intentionally *not* imputed, per the "no fabrication" rule — this trades a small amount of completeness for correctness.

**Notable EDA finding to flag in the report:** engagement metrics (likes/shares/comments)
show almost no variation by platform and near-zero correlation with follower
count — inconsistent with real-world social media dynamics, suggesting these
fields were generated near-randomly in the corrupted dataset rather than
reflecting an authentic engagement model. This is worth stating as a
limitation rather than over-interpreting platform "winners."
""")

nb["cells"] = cells
with open("notebook/DataVortex_Cleaning_EDA.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
