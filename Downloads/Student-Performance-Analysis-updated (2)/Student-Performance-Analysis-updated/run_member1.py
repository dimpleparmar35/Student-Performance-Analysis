"""
========================================================
MEMBER 1 — MASTER RUNNER
Student Performance Analysis Project
========================================================
Run this single script to execute all Member 1 tasks:

  Step 1  → Problem analysis & dataset verification
  Step 3  → Data preprocessing & cleaning
  Step 5  → Feature engineering (RiskScore, etc.)

Usage:
  cd Student-Performance-Analysis
  python run_member1.py
========================================================
"""

import sys, os

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from step1_problem_analysis  import print_banner, verify_dataset, load_dataset, summarise_problem
from step3_preprocessing     import preprocess_pipeline
from step5_feature_engineering import feature_engineering_pipeline


DATASET_FILE = "dataset/student_performance.csv"


def section(title: str):
    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║  {title:<56}║")
    print("╚" + "═" * 58 + "╝\n")


def main():
    # Ensure all directories exist
    for d in ["dataset", "notebooks", "src", "models", "dashboard",
              "reports", "presentation", "images"]:
        os.makedirs(d, exist_ok=True)

    section("STEP 1 — Problem Analysis & Dataset Collection")
    print_banner()
    ok = verify_dataset(DATASET_FILE)

    if ok:
        df_raw = load_dataset(DATASET_FILE)
    else:
        # Auto-generate synthetic data for demo
        from step1_problem_analysis import np, pd, DATASET_FILE as DF
        import numpy as np
        import pandas as pd
        print("[INFO] Generating synthetic dataset for demonstration...\n")
        np.random.seed(42)
        n = 2000
        df_raw = pd.DataFrame({
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
        df_raw.to_csv(DATASET_FILE, index=False)
        print(f"[OK] Synthetic dataset saved → {DATASET_FILE}")

    summarise_problem(df_raw)

    section("STEP 3 — Data Preprocessing & Cleaning")
    df_clean = preprocess_pipeline()

    section("STEP 5 — Feature Engineering")
    df_eng = feature_engineering_pipeline()

    section("MEMBER 1 — SUMMARY")
    print(f"  ✔  Raw dataset rows       : {len(df_raw):,}")
    print(f"  ✔  Cleaned dataset rows   : {len(df_clean):,}")
    print(f"  ✔  Engineered features    : {df_eng.shape[1]} total columns")
    print(f"  ✔  At-risk students       : {df_eng['IsAtRisk'].sum()} "
          f"({df_eng['IsAtRisk'].mean()*100:.1f}%)")
    print(f"  ✔  Avg RiskScore          : {df_eng['RiskScore'].mean():.1f} / 100")
    print()
    print("  Files produced:")
    for f in [
        "dataset/student_performance.csv",
        "dataset/student_performance_cleaned.csv",
        "dataset/student_performance_engineered.csv",
        "dataset/train.csv",
        "dataset/test.csv",
    ]:
        exists = "✔" if os.path.exists(f) else "✗"
        print(f"    {exists}  {f}")
    print()
    print("  Member 1 tasks complete. Hand off to Member 2 (EDA).")
    print("  Engineered dataset: dataset/student_performance_engineered.csv")
    print()


if __name__ == "__main__":
    main()