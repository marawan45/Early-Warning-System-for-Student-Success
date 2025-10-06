# EWS2 - Early Warning System for Student Success

A machine learning-based system to predict student failure risk on final exams using key academic indicators with **SQL Server database integration**.

## Features

The system predicts student failure risk based on four critical features:
- **Midterm Score** - Current semester midterm performance (percentage or raw marks)
- **Quiz/Assignment Average** - Consistent performance indicator throughout semester
- **Previous Semester GPA** - Historical academic performance
- **Course Retake Flag** - Whether student is repeating the course

## Database Setup (SQL Server)

### Prerequisites
- SQL Server (Express/Standard/Enterprise)
- SQL Server Management Studio (SSMS)
- ODBC Driver 17 for SQL Server

### 1. Create Database
Open SQL Server Management Studio and run:
```sql
-- Run database_setup.sql in SSMS
-- This creates the EWS2_StudentSuccess database with all tables, views, and stored procedures
```

### 2. Insert Sample Data (Optional)
```sql
-- Run sample_data_insert.sql in SSMS
-- This adds 20 sample students with predictions
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate demo data:**
   ```bash
   python ews2_main.py --demo
   ```

3. **Use existing database data:**
   ```bash
   python ews2_main.py
   ```

4. **Connect to custom SQL Server:**
   ```bash
   python ews2_main.py --db-server "YOUR_SERVER" --db-name "EWS2_StudentSuccess"
   ```

5. **Use SQL Server authentication:**
   ```bash
   python ews2_main.py --demo --db-user "username" --db-pass "password"
   ```

## Database Schema

The system uses the following database tables:
- **Students** - Core student data (StudentID, MidtermScore, QuizAssignmentAvg, PreviousGPA, IsRetake)
- **Predictions** - Historical predictions with timestamps and model versions
- **ModelMetadata** - Training metrics and feature importance data

Student data format:
- `StudentID` - Unique identifier for each student
- `MidtermScore` - Midterm exam score (0-100)
- `QuizAssignmentAvg` - Average of quizzes/assignments (0-100)
- `PreviousGPA` - Previous semester GPA (0.0-4.0)
- `IsRetake` - 1 if retaking course, 0 if first time

## Usage Options

```bash
# Generate demo data with 500 synthetic students
python ews2_main.py --demo

# Generate demo with custom number of students
python ews2_main.py --demo --samples 1000

# Use existing data from database
python ews2_main.py

# Skip visualizations (faster)
python ews2_main.py --demo --no-viz

# Export results to CSV file
python ews2_main.py --demo --export my_results.csv

# Connect to different server
python ews2_main.py --db-server "MYSERVER\SQLEXPRESS"
```

## Output

The system provides:
- **Risk Classification**: HIGH/MEDIUM/LOW risk categories
- **Failure Probability**: Numerical probability (0-1) of failing final exam
- **Feature Importance**: Which factors most influence predictions
- **Intervention Recommendations**: Specific actions for at-risk students
- **Visualizations**: Charts saved as PNG files
- **CSV Export**: Results exported for further analysis

## Risk Categories

- **HIGH RISK** (≥70% failure probability): Immediate intervention needed
- **MEDIUM RISK** (40-70% failure probability): Close monitoring required
- **LOW RISK** (<40% failure probability): Likely to pass

## Model Details

- **Algorithm**: Random Forest Classifier with balanced class weights
- **Features**: 4 key academic indicators
- **Validation**: Cross-validated with synthetic realistic data
- **Output**: Binary classification (pass/fail) with probability scores

## Files

- `ews2_predictor.py` - Core machine learning prediction engine
- `ews2_main.py` - Command-line interface
- `ews2_database.py` - SQL Server database operations
- `database_setup.sql` - Database schema creation script
- `sample_data_insert.sql` - Sample data insertion script
- `requirements.txt` - Python dependencies

## Example Output

```
📊 SUMMARY STATISTICS
----------------------------------------
Total Students: 500
High Risk (≥70% failure prob): 45 (9.0%)
Medium Risk (40-70% failure prob): 123 (24.6%)
Low Risk (<40% failure prob): 332 (66.4%)
Average Failure Probability: 28.5%

🚨 HIGH RISK STUDENTS (45 students)
------------------------------------------------------------
Student ID: STU_0234
  Failure Probability: 89.2%
  Midterm Score: 42.1%
  Quiz/Assignment Avg: 38.9%
  Previous GPA: 1.85
  Course Retake: Yes
```

This system helps educators identify at-risk students early and take proactive measures to improve student success rates
