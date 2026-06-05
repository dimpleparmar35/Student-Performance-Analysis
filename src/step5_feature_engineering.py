"""
========================================================
MEMBER 1 — STEP 5: Feature Engineering
Student Performance Analysis Project
========================================================
New features created:
  • AttendanceRate     — % of days attended
  • StudyCategory      — Low / Moderate / High study time
  • ActivityScore      — sum of extracurricular involvement
  • SupportScore       — weighted parental + tutoring support
  • RiskScore          — composite 0–100 at-risk indicator
  • IsAtRisk           — binary flag (RiskScore >= 50)
  • GPAGroup           — GPA binned into 4 bands
"""

import os
import pandas as pd
import numpy as np


# ── Paths ──────────────────────────────────────────────
CLEANED_PATH     = "dataset/student_performance_cleaned.csv"
ENGINEERED_PATH  = "dataset/student_performance_engineered.csv"

TOTAL_SCHOOL_DAYS = 180          # typical academic year


# ══════════════════════════════════════════════════════
# FEATURE FUNCTIONS
# ══════════════════════════════════════════════════════

def add_attendance_rate(df: pd.DataFrame) -> pd.DataFrame:
    """
    AttendanceRate = (TotalDays - Absences) / TotalDays * 100
    Clipped to [0, 100].
    """
    df["AttendanceRate"] = (
        (TOTAL_SCHOOL_DAYS - df["Absences"]) / TOTAL_SCHOOL_DAYS * 100
    ).clip(0, 100).round(2)
    print(f"  [+] AttendanceRate  — range: "
          f"{df['AttendanceRate'].min():.1f}% – {df['AttendanceRate'].max():.1f}%")
    return df


def add_study_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bin weekly study hours into ordinal categories.
      0–5 h  → 0 (Low)
      5–15 h → 1 (Moderate)
      >15 h  → 2 (High)
    """
    df["StudyCategory"] = pd.cut(
        df["StudyTimeWeekly"],
        bins=[-1, 5, 15, 100],
        labels=[0, 1, 2]
    ).astype("int8")
    dist = df["StudyCategory"].value_counts().sort_index()
    print(f"  [+] StudyCategory   — Low:{dist.get(0,0)} | "
          f"Moderate:{dist.get(1,0)} | High:{dist.get(2,0)}")
    return df


def add_activity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum of four binary activity flags: Extracurricular, Sports, Music, Volunteering.
    Range: 0 – 4.
    """
    activity_cols = ["Extracurricular", "Sports", "Music", "Volunteering"]
    df["ActivityScore"] = df[activity_cols].sum(axis=1).astype("int8")
    print(f"  [+] ActivityScore   — mean: {df['ActivityScore'].mean():.2f} / 4")
    return df


def add_support_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weighted combination of parental and tutoring support.
      SupportScore = ParentalSupport * 0.6 + Tutoring * 2
    Scaled to 0–10 range (ParentalSupport max=4, Tutoring max=1).
    """
    df["SupportScore"] = (
        df["ParentalSupport"] * 0.6 + df["Tutoring"] * 2
    ).round(2)
    print(f"  [+] SupportScore    — mean: {df['SupportScore'].mean():.2f} "
          f"(range 0–{df['SupportScore'].max():.2f})")
    return df


def add_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Composite at-risk indicator (0 = no risk, 100 = highest risk).

    Formula (weighted sum of normalised penalty factors):
      Attendance penalty  = (1 - AttendanceRate/100)   weight 35
      Study penalty       = (1 - StudyTimeWeekly/20)   weight 25  [20h = max]
      Support penalty     = (1 - SupportScore/4.4)     weight 20  [4.4 = max]
      Activity penalty    = (1 - ActivityScore/4)      weight 10
      GPA penalty         = (1 - GPA/4)                weight 10

    Each penalty ∈ [0,1]; higher = worse. Weighted sum * 100 = RiskScore.
    """
    att  = (1 - df["AttendanceRate"] / 100).clip(0, 1)
    stu  = (1 - df["StudyTimeWeekly"] / 20).clip(0, 1)
    sup  = (1 - df["SupportScore"] / 4.4).clip(0, 1)
    act  = (1 - df["ActivityScore"] / 4).clip(0, 1)
    gpa  = (1 - df["GPA"] / 4).clip(0, 1)

    df["RiskScore"] = (
        att * 35 + stu * 25 + sup * 20 + act * 10 + gpa * 10
    ).round(2)

    print(f"  [+] RiskScore       — mean: {df['RiskScore'].mean():.1f} | "
          f"min: {df['RiskScore'].min():.1f} | max: {df['RiskScore'].max():.1f}")
    return df


def add_is_at_risk(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """Binary flag: 1 if RiskScore >= threshold, else 0."""
    df["IsAtRisk"] = (df["RiskScore"] >= threshold).astype("int8")
    at_risk_count = df["IsAtRisk"].sum()
    pct = at_risk_count / len(df) * 100
    print(f"  [+] IsAtRisk        — {at_risk_count} students ({pct:.1f}%) "
          f"flagged (threshold={threshold})")
    return df


def add_gpa_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    GPAGroup: 0=Poor (0–1.9), 1=Average (2–2.9), 2=Good (3–3.4), 3=Excellent (3.5–4)
    """
    df["GPAGroup"] = pd.cut(
        df["GPA"],
        bins=[-0.01, 1.99, 2.99, 3.49, 4.0],
        labels=[0, 1, 2, 3]
    ).astype("int8")
    dist = df["GPAGroup"].value_counts().sort_index()
    labels = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
    for g, cnt in dist.items():
        print(f"    GPAGroup {g} ({labels[g]}): {cnt}")
    return df


# ══════════════════════════════════════════════════════
# FEATURE SUMMARY
# ══════════════════════════════════════════════════════
def print_feature_summary(df: pd.DataFrame):
    new_cols = [
        "AttendanceRate", "StudyCategory", "ActivityScore",
        "SupportScore", "RiskScore", "IsAtRisk", "GPAGroup"
    ]
    print("\n── New Feature Statistics ───────────────────────────")
    print(df[new_cols].describe().round(2).to_string())


# ══════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════
def feature_engineering_pipeline() -> pd.DataFrame:
    print("=" * 60)
    print("  MEMBER 1 — Feature Engineering Pipeline")
    print("=" * 60)

    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{CLEANED_PATH}'.\n"
            "Run step3_preprocessing.py first."
        )

    df = pd.read_csv(CLEANED_PATH)
    print(f"[LOAD]  {len(df):,} rows loaded from cleaned dataset.\n")

    print("── Creating new features ────────────────────────────")
    df = add_attendance_rate(df)
    df = add_study_category(df)
    df = add_activity_score(df)
    df = add_support_score(df)
    df = add_risk_score(df)
    df = add_is_at_risk(df, threshold=50.0)
    df = add_gpa_group(df)

    print_feature_summary(df)

    df.to_csv(ENGINEERED_PATH, index=False)
    print(f"\n[SAVE]  Engineered dataset → {ENGINEERED_PATH}")
    print("[DONE]  Feature engineering complete.\n")

    return df


if __name__ == "__main__":
    df_eng = feature_engineering_pipeline()
    print("\nSample rows (select features):")
    sample_cols = ["StudentID", "GPA", "Absences", "AttendanceRate",
                   "StudyTimeWeekly", "RiskScore", "IsAtRisk", "GradeClass"]
    print(df_eng[sample_cols].head(10).to_string(index=False))