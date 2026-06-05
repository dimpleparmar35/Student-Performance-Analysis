"""
Generate a realistic synthetic dataset matching the Kaggle
"Student Performance Dataset" by Arif Miah.

Schema: 5,000 records with 15 columns:
  StudentID, Age, Gender, Ethnicity, ParentalEducation,
  StudyTimeWeekly, Absences, Tutoring, ParentalSupport,
  Extracurricular, Sports, Music, Volunteering, GPA, GradeClass

GradeClass is derived from GPA:
  0 = A (GPA >= 3.5)
  1 = B (3.0 <= GPA < 3.5)
  2 = C (2.5 <= GPA < 3.0)
  3 = D (2.0 <= GPA < 2.5)
  4 = F (GPA < 2.0)
"""

import numpy as np
import pandas as pd
import os

np.random.seed(42)
N = 5000

# --- Demographics ---
student_id = np.arange(1001, 1001 + N)
age = np.random.randint(15, 19, N)                         # 15-18
gender = np.random.choice([0, 1], N)                        # 0=Male, 1=Female
ethnicity = np.random.choice([0, 1, 2, 3], N,
                             p=[0.40, 0.25, 0.20, 0.15])
parental_education = np.random.choice([0, 1, 2, 3, 4], N,
                                      p=[0.05, 0.25, 0.30, 0.25, 0.15])

# --- Study Habits ---
study_time_weekly = np.round(np.random.uniform(0, 20, N), 2)
absences = np.random.randint(0, 31, N)                      # 0-30
tutoring = np.random.choice([0, 1], N, p=[0.65, 0.35])

# --- Parental Involvement ---
parental_support = np.random.choice([0, 1, 2, 3, 4], N,
                                    p=[0.05, 0.15, 0.35, 0.30, 0.15])

# --- Extracurricular Activities ---
extracurricular = np.random.choice([0, 1], N, p=[0.40, 0.60])
sports = np.random.choice([0, 1], N, p=[0.50, 0.50])
music = np.random.choice([0, 1], N, p=[0.60, 0.40])
volunteering = np.random.choice([0, 1], N, p=[0.55, 0.45])

# --- GPA (realistic: influenced by study habits, absences, support) ---
# Use a base centered around ~2.8 GPA with factor-based adjustments
base_gpa = np.random.normal(2.8, 0.5, N)

# Study time effect: more study -> higher GPA (scaled contribution)
study_effect = (study_time_weekly - 10) / 20 * 0.5

# Absence effect: more absences -> lower GPA
absence_effect = -(absences - 15) / 30 * 0.5

# Parental support effect
support_effect = (parental_support - 2) / 4 * 0.3

# Tutoring effect
tutoring_effect = tutoring * 0.15

# Parental education effect
edu_effect = (parental_education - 2) / 4 * 0.2

# Activity effect (small)
activity_count = extracurricular + sports + music + volunteering
activity_effect = (activity_count - 2) / 4 * 0.1

# Compute final GPA
gpa = base_gpa + study_effect + absence_effect + support_effect + tutoring_effect + edu_effect + activity_effect
gpa = np.clip(gpa, 0.0, 4.0)
gpa = np.round(gpa, 2)

# --- GradeClass (derived from GPA) ---
grade_class = np.full(N, 4, dtype=int)  # Default F
grade_class[gpa >= 3.5] = 0   # A
mask_b = (gpa >= 3.0) & (gpa < 3.5)
grade_class[mask_b] = 1   # B
mask_c = (gpa >= 2.5) & (gpa < 3.0)
grade_class[mask_c] = 2   # C
mask_d = (gpa >= 2.0) & (gpa < 2.5)
grade_class[mask_d] = 3   # D
# F remains for gpa < 2.0

# --- Build DataFrame ---
df = pd.DataFrame({
    "StudentID": student_id,
    "Age": age,
    "Gender": gender,
    "Ethnicity": ethnicity,
    "ParentalEducation": parental_education,
    "StudyTimeWeekly": study_time_weekly,
    "Absences": absences,
    "Tutoring": tutoring,
    "ParentalSupport": parental_support,
    "Extracurricular": extracurricular,
    "Sports": sports,
    "Music": music,
    "Volunteering": volunteering,
    "GPA": gpa,
    "GradeClass": grade_class,
})

# --- Save ---
os.makedirs("dataset", exist_ok=True)

# Save as main dataset file (root)
df.to_csv("Student_performance_data _.csv", index=False)
# Also copy into dataset folder
df.to_csv("dataset/student_performance.csv", index=False)

print(f"[OK] Generated {N} student records")
print(f"   Saved: Student_performance_data _.csv")
print(f"   Saved: dataset/student_performance.csv")
print(f"\n   Grade Distribution:")
for g, label in {0: "A", 1: "B", 2: "C", 3: "D", 4: "F"}.items():
    count = (grade_class == g).sum()
    print(f"     Grade {label}: {count} ({count/N*100:.1f}%)")
print(f"\n   Average GPA: {gpa.mean():.2f}")
print(f"   GPA < 2.0 (at-risk): {(gpa < 2.0).sum()}")
