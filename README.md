# Student Performance Analysis and Grade Prediction

## Project Overview
This project uses data analytics and machine learning to predict student `GradeClass`,
identify at-risk students early, and provide actionable recommendations for educators.

## Dataset
- Source: [Kaggle — Students Performance Dataset](https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset)
- File included: `Student_performance_data _.csv` (2,392 students, 15 features)
- The pipeline automatically copies it into the `dataset/` folder on first run.

## Project Structure
```
Student-Performance-Analysis/
├── Student_performance_data _.csv        ← original raw dataset
├── dataset/                              ← processed data files (auto-created)
│   ├── student_performance.csv
│   ├── student_performance_cleaned.csv
│   ├── student_performance_engineered.csv
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── step1_problem_analysis.py         ← Member 1: problem scope & data collection
│   ├── step3_preprocessing.py            ← Member 1: cleaning & missing values
│   └── step5_feature_engineering.py      ← Member 1: RiskScore, AttendanceRate, etc.
├── notebooks/
│   └── Exploratory_and_Statistical_data_analysis.ipynb   ← Member 2: EDA
├── models/
│   └── ML_Grade_Prediction_G_Revanth_Reddy               ← Member 3: ML model
├── dashboard/
│   └── DATA ANALTICS DASHBOARD.twb                       ← Member 2: Tableau dashboard
├── reports/
│   └── Insights (data analysis).pages                    ← Member 2: insights report
├── run_member1.py                        ← ▶ run all Member 1 tasks
└── README.md
```

## How to Run (Member 1)

### Step 1 — Install dependencies
```
pip install pandas numpy scikit-learn
```

### Step 2 — Run the pipeline
```
python run_member1.py
```

That's it. The script handles everything automatically.

### What it produces
| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_problem_analysis.py` | Dataset loaded, grade distribution printed |
| 3 | `step3_preprocessing.py` | Cleaned CSV, train.csv, test.csv |
| 5 | `step5_feature_engineering.py` | Engineered CSV with 7 new features |

## Engineered Features (Member 1)

| Feature | Description | Range |
|---------|-------------|-------|
| `AttendanceRate` | % of school days attended | 0 – 100 |
| `StudyCategory` | Binned study hours (Low / Moderate / High) | 0, 1, 2 |
| `ActivityScore` | Sum of 4 extracurricular activities | 0 – 4 |
| `SupportScore` | Weighted parental + tutoring support | 0 – 4.4 |
| `RiskScore` | Composite at-risk score | 0 – 100 |
| `IsAtRisk` | Binary flag (RiskScore ≥ 50) | 0 or 1 |
| `GPAGroup` | GPA binned (Poor / Average / Good / Excellent) | 0 – 3 |

## Dataset Summary
- Total students: **2,392**
- At-risk students (RiskScore ≥ 50): **420 (17.6%)**
- Average GPA: **1.91**
- Students with GPA < 2.0: **1,274**

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Member 1** | Problem analysis, data collection, cleaning, preprocessing, feature engineering, GitHub |
| Member 2 | EDA, statistics, visualization, Tableau dashboard, documentation |
| Member 3 | ML model development, training/testing, grade prediction, final presentation |

## Technologies
Python · Pandas · NumPy · Scikit-learn · Tableau · Jupyter · GitHub
