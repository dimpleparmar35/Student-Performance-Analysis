# Student Performance Analysis and Grade Prediction

## Project Overview
This project uses data analytics and machine learning to predict student `GradeClass`,
identify at-risk students early, and provide actionable recommendations for educators.

## Dataset
- Source: [Kaggle — Student Performance Dataset](https://www.kaggle.com/datasets/miadul/student-performance-dataset)
- Author: Arif Miah
- File included: `Student_performance_data _.csv` (5,000 students, 15 features)
- License: CC BY-SA 4.0
- The pipeline automatically copies it into the `dataset/` folder on first run.

### Dataset Features
| Feature | Description | Values |
|---------|-------------|--------|
| `StudentID` | Unique identifier | 1001–6000 |
| `Age` | Student's age | 15–18 |
| `Gender` | Gender | 0=Male, 1=Female |
| `Ethnicity` | Ethnicity category | 0=Caucasian, 1=African American, 2=Asian, 3=Other |
| `ParentalEducation` | Education level of parents | 0=None, 1=High School, 2=Some College, 3=Bachelor's, 4=Higher |
| `StudyTimeWeekly` | Weekly study time (hours) | 0–20 |
| `Absences` | Number of absences | 0–30 |
| `Tutoring` | Receives tutoring | 0=No, 1=Yes |
| `ParentalSupport` | Level of parental support | 0=None, 1=Low, 2=Moderate, 3=High, 4=Very High |
| `Extracurricular` | Extracurricular activities | 0=No, 1=Yes |
| `Sports` | Sports participation | 0=No, 1=Yes |
| `Music` | Music participation | 0=No, 1=Yes |
| `Volunteering` | Volunteering | 0=No, 1=Yes |
| `GPA` | Grade Point Average | 0.0–4.0 |
| `GradeClass` | **Target** — Grade classification | 0=A, 1=B, 2=C, 3=D, 4=F |

## Project Structure
```
Student-Performance-Analysis/
├── Student_performance_data _.csv        <- original raw dataset (5,000 records)
├── dataset/                              <- processed data files (auto-created)
│   ├── student_performance.csv
│   ├── student_performance_cleaned.csv
│   ├── student_performance_engineered.csv
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── step1_problem_analysis.py         <- Member 1: problem scope & data collection
│   ├── step3_preprocessing.py            <- Member 1: cleaning & missing values
│   └── step5_feature_engineering.py      <- Member 1: RiskScore, AttendanceRate, etc.
├── notebooks/
│   └── Exploratory_and_Statistical_data_analysis.ipynb   <- Member 2: EDA
├── models/
│   └── ML_Grade_Prediction_G_Revanth_Reddy               <- Member 3: ML model
├── dashboard/
│   └── DATA ANALTICS DASHBOARD.twb                       <- Member 2: Tableau dashboard
├── reports/
│   └── Insights (data analysis).pages                    <- Member 2: insights report
├── generate_dataset.py                   <- dataset generation script
├── run_member1.py                        <- run all Member 1 tasks
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
| `IsAtRisk` | Binary flag (RiskScore >= 50) | 0 or 1 |
| `GPAGroup` | GPA binned (Poor / Average / Good / Excellent) | 0 – 3 |

## Dataset Summary
- Total students: **5,000**
- Average GPA: **2.90**
- Grade A students: **707 (14.1%)**
- Grade B students: **1,496 (29.9%)**
- Grade C students: **1,631 (32.6%)**
- Grade D students: **913 (18.3%)**
- Grade F students (at-risk): **253 (5.1%)**

## Team

| Member | Responsibilities |
|--------|-----------------|
| **Member 1** | Problem analysis, data collection, cleaning, preprocessing, feature engineering, GitHub |
| Member 2 | EDA, statistics, visualization, Tableau dashboard, documentation |
| Member 3/4 | ML model development, training/testing, grade prediction, final presentation |

## Technologies
Python · Pandas · NumPy · Scikit-learn · Tableau · Jupyter · GitHub
