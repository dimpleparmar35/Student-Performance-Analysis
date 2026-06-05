"""
========================================================
MEMBER 1 — STEP 1: Problem Analysis & Dataset Collection
Student Performance Analysis Project
========================================================
"""

import os
import pandas as pd
import numpy as np

# ── Project Configuration ──────────────────────────────
PROJECT_NAME = "Student Performance Analysis and Grade Prediction"
DATASET_URL  = "https://www.kaggle.com/datasets/miadul/student-performance-dataset"
DATASET_FILE = "dataset/student_performance.csv"

# Expected columns in the Kaggle dataset
EXPECTED_COLUMNS = [
    "StudentID", "Age", "Gender", "Ethnicity", "ParentalEducation",
    "StudyTimeWeekly", "Absences", "Tutoring", "ParentalSupport",
    "Extracurricular", "Sports", "Music", "Volunteering",
    "GPA", "GradeClass"
]

GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}


def print_banner():
    print("=" * 60)
    print(f"  {PROJECT_NAME}")
    print("  Member 1 — Problem Analysis & Dataset Collection")
    print("=" * 60)


def verify_dataset(filepath: str) -> bool:
    """Check the dataset exists and has expected columns."""
    if not os.path.exists(filepath):
        print(f"[WARNING] Dataset not found at '{filepath}'.")
        print(f"  Please download it from:\n  {DATASET_URL}")
        print("  Save it as: dataset/student_performance.csv\n")
        return False

    df = pd.read_csv(filepath, nrows=5)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARNING] Missing columns: {missing}")
        return False

    print(f"[OK] Dataset found: {filepath}")
    print(f"     Columns verified: {list(df.columns)}\n")
    return True


def load_dataset(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"[OK] Loaded {len(df):,} rows × {df.shape[1]} columns\n")
    return df


def summarise_problem(df: pd.DataFrame):
    """Print a structured problem summary."""
    print("── Problem Summary ──────────────────────────────────")
    print(f"  Total students     : {len(df):,}")
    print(f"  Features available : {df.shape[1]}")
    print(f"  Target variable    : GradeClass  (0=A … 4=F)")
    print()
    grade_dist = df["GradeClass"].value_counts().sort_index()
    print("  Grade distribution:")
    for g, count in grade_dist.items():
        label = GRADE_MAP.get(g, str(g))
        bar   = "█" * (count // 20)
        print(f"    Grade {label} ({g}): {count:>4}  {bar}")
    print()
    avg_gpa = df["GPA"].mean()
    print(f"  Average GPA        : {avg_gpa:.2f}")
    low_gpa = (df["GPA"] < 2.0).sum()
    print(f"  Students GPA < 2.0 : {low_gpa} (at-risk candidates)")
    print("─" * 52)


if __name__ == "__main__":
    print_banner()
    dataset_ok = verify_dataset(DATASET_FILE)

    if dataset_ok:
        df = load_dataset(DATASET_FILE)
        summarise_problem(df)
    else:
        print("[INFO] Running with a SYNTHETIC demo dataset (100 rows).")
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            "StudentID"        : range(1001, 1001 + n),
            "Age"              : np.random.randint(15, 19, n),
            "Gender"           : np.random.choice([0, 1], n),
            "Ethnicity"        : np.random.choice([0, 1, 2, 3], n),
            "ParentalEducation": np.random.choice([0, 1, 2, 3, 4], n),
            "StudyTimeWeekly"  : np.round(np.random.uniform(0, 20, n), 1),
            "Absences"         : np.random.randint(0, 30, n),
            "Tutoring"         : np.random.choice([0, 1], n),
            "ParentalSupport"  : np.random.choice([0, 1, 2, 3, 4], n),
            "Extracurricular"  : np.random.choice([0, 1], n),
            "Sports"           : np.random.choice([0, 1], n),
            "Music"            : np.random.choice([0, 1], n),
            "Volunteering"     : np.random.choice([0, 1], n),
            "GPA"              : np.round(np.random.uniform(0, 4, n), 2),
            "GradeClass"       : np.random.choice([0, 1, 2, 3, 4], n),
        })
        df.to_csv(DATASET_FILE, index=False)
        print("[INFO] Demo dataset saved to dataset/student_performance.csv\n")
        summarise_problem(df)