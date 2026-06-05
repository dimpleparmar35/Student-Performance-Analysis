"""
========================================================
MEMBER 1 — STEP 3: Data Preprocessing & Cleaning
Student Performance Analysis Project
========================================================
Handles:
  • Duplicate removal
  • Missing value detection & imputation
  • Outlier detection (IQR method)
  • Categorical encoding
  • Train/test split
  • Saves cleaned dataset
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# ── Paths ──────────────────────────────────────────────
RAW_PATH     = "dataset/student_performance.csv"
CLEANED_PATH = "dataset/student_performance_cleaned.csv"
TRAIN_PATH   = "dataset/train.csv"
TEST_PATH    = "dataset/test.csv"

# ── Column groups ──────────────────────────────────────
NUMERIC_COLS     = ["Age", "StudyTimeWeekly", "Absences", "GPA"]
CATEGORICAL_COLS = ["Gender", "Ethnicity", "ParentalEducation",
                    "ParentalSupport", "Tutoring", "Extracurricular",
                    "Sports", "Music", "Volunteering"]
TARGET_COL       = "GradeClass"
ID_COL           = "StudentID"


# ══════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════
def load_raw(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Raw dataset not found at '{path}'.\n"
            "Run step1_problem_analysis.py first."
        )
    df = pd.read_csv(path)
    print(f"[LOAD]  {len(df):,} rows × {df.shape[1]} columns loaded.")
    return df


# ══════════════════════════════════════════════════════
# 2. DUPLICATES
# ══════════════════════════════════════════════════════
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    print(f"[DEDUP] {removed} duplicate rows removed. Remaining: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════
# 3. MISSING VALUES
# ══════════════════════════════════════════════════════
def report_missing(df: pd.DataFrame):
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("[MISSING] No missing values found.")
    else:
        print("[MISSING] Columns with nulls:")
        for col, cnt in missing.items():
            pct = cnt / len(df) * 100
            print(f"  {col:<25} {cnt:>4} ({pct:.1f}%)")


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
      Numeric  → median  (robust to skew / outliers)
      Categorical → mode (most frequent)
    """
    for col in NUMERIC_COLS:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  [IMPUTE] {col}: filled with median ({median_val:.2f})")

    for col in CATEGORICAL_COLS:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            print(f"  [IMPUTE] {col}: filled with mode ({mode_val})")

    return df


# ══════════════════════════════════════════════════════
# 4. OUTLIERS  (IQR method — cap, not drop)
# ══════════════════════════════════════════════════════
def cap_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    print("[OUTLIER] IQR capping applied to numeric columns:")
    for col in cols:
        if col not in df.columns:
            continue
        Q1  = df[col].quantile(0.25)
        Q3  = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 1.5 * IQR
        hi  = Q3 + 1.5 * IQR
        n_lo = (df[col] < lo).sum()
        n_hi = (df[col] > hi).sum()
        df[col] = df[col].clip(lower=lo, upper=hi)
        print(f"  {col:<22} lo={lo:.2f}  hi={hi:.2f}  "
              f"capped {n_lo} low + {n_hi} high values")
    return df


# ══════════════════════════════════════════════════════
# 5. ENCODING
# ══════════════════════════════════════════════════════
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Kaggle dataset already stores categoricals as integers,
    but we verify and document the encoding for reproducibility.
    """
    ENCODINGS = {
        "Gender"           : {0: "Male",    1: "Female"},
        "Tutoring"         : {0: "No",      1: "Yes"},
        "Extracurricular"  : {0: "No",      1: "Yes"},
        "Sports"           : {0: "No",      1: "Yes"},
        "Music"            : {0: "No",      1: "Yes"},
        "Volunteering"     : {0: "No",      1: "Yes"},
        "Ethnicity"        : {0: "Caucasian", 1: "African American",
                              2: "Asian",     3: "Other"},
        "ParentalEducation": {0: "None", 1: "High School", 2: "Some College",
                              3: "Bachelor's", 4: "Higher"},
        "ParentalSupport"  : {0: "None", 1: "Low", 2: "Moderate",
                              3: "High", 4: "Very High"},
        "GradeClass"       : {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"},
    }
    print("[ENCODE] Categorical encodings verified:")
    for col, mapping in ENCODINGS.items():
        if col in df.columns:
            unique_vals = sorted(df[col].dropna().unique())
            print(f"  {col:<22} values: {unique_vals}")

    # All columns are already integer-encoded — no transformation needed.
    # (If raw text labels were present, we would apply LabelEncoder here.)
    return df


# ══════════════════════════════════════════════════════
# 6. DTYPE OPTIMISATION
# ══════════════════════════════════════════════════════
def optimise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS + [TARGET_COL]:
        if col in df.columns:
            df[col] = df[col].astype("int8")
    df["Age"] = df["Age"].astype("int8")
    print(f"[DTYPE] Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    return df


# ══════════════════════════════════════════════════════
# 7. TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════
def split_and_save(df: pd.DataFrame,
                   test_size: float = 0.2,
                   random_state: int = 42):
    train, test = train_test_split(
        df, test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL]
    )
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH,   index=False)
    print(f"[SPLIT] Train: {len(train):,} rows  |  Test: {len(test):,} rows")
    print(f"        Saved → {TRAIN_PATH}")
    print(f"        Saved → {TEST_PATH}")


# ══════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════
def preprocess_pipeline() -> pd.DataFrame:
    print("=" * 60)
    print("  MEMBER 1 — Data Preprocessing Pipeline")
    print("=" * 60)

    df = load_raw(RAW_PATH)

    print("\n── Step 1: Duplicate removal ─────────────────────────")
    df = remove_duplicates(df)

    print("\n── Step 2: Missing value analysis ───────────────────")
    report_missing(df)
    df = impute_missing(df)

    print("\n── Step 3: Outlier capping (IQR) ────────────────────")
    df = cap_outliers(df, NUMERIC_COLS)

    print("\n── Step 4: Categorical encoding ─────────────────────")
    df = encode_categoricals(df)

    print("\n── Step 5: Dtype optimisation ───────────────────────")
    df = optimise_dtypes(df)

    print("\n── Step 6: Save cleaned dataset ─────────────────────")
    df.to_csv(CLEANED_PATH, index=False)
    print(f"[SAVE]  Cleaned dataset → {CLEANED_PATH}")

    print("\n── Step 7: Train/test split ─────────────────────────")
    split_and_save(df)

    print("\n[DONE]  Preprocessing complete.\n")
    return df


if __name__ == "__main__":
    df_clean = preprocess_pipeline()
    print("Preview of cleaned dataset:")
    print(df_clean.head(5).to_string(index=False))