"""
EWS2 Database Module for SQL Server
Handles all database operations for the Early Warning System
"""

import pyodbc
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EWS2Database:
    def __init__(self, server: str = 'localhost', database: str = 'EWS2_StudentSuccess', 
                 trusted_connection: bool = True, username: str = None, password: str = None):
        """
        Initialize database connection
        
        Args:
            server: SQL Server instance name
            database: Database name
            trusted_connection: Use Windows Authentication if True
            username: SQL Server username (if not using Windows Auth)
            password: SQL Server password (if not using Windows Auth)
        """
        self.server = server
        self.database = database
        self.trusted_connection = trusted_connection
        self.username = username
        self.password = password
        self.connection_string = self._build_connection_string()
        
    def _build_connection_string(self) -> str:
        """Build SQL Server connection string"""
        if self.trusted_connection:
            return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};Trusted_Connection=yes;"
        else:
            return f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={self.server};DATABASE={self.database};UID={self.username};PWD={self.password};"
    
    def test_connection(self) -> bool:
        """Test database connection"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                return True
        except Exception as e:
            print(f"❌ Database connection failed: {e}")
            return False
    
    def get_all_students(self) -> pd.DataFrame:
        """Retrieve all students from database"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                query = """
                    SELECT StudentID as student_id, 
                           MidtermScore as midterm_score,
                           QuizAssignmentAvg as quiz_assignment_avg,
                           PreviousGPA as previous_gpa,
                           IsRetake as is_retake
                    FROM Students
                    ORDER BY StudentID
                """
                df = pd.read_sql(query, conn)
                print(f"✅ Retrieved {len(df)} students from database")
                return df
        except Exception as e:
            print(f"❌ Error retrieving students: {e}")
            return pd.DataFrame()
    
    def get_student(self, student_id: str) -> Optional[Dict]:
        """Get a specific student by ID"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT StudentID, MidtermScore, QuizAssignmentAvg, PreviousGPA, IsRetake
                    FROM Students WHERE StudentID = ?
                """, student_id)
                
                row = cursor.fetchone()
                if row:
                    return {
                        'student_id': row[0],
                        'midterm_score': row[1],
                        'quiz_assignment_avg': row[2],
                        'previous_gpa': row[3],
                        'is_retake': row[4]
                    }
                return None
        except Exception as e:
            print(f"❌ Error retrieving student {student_id}: {e}")
            return None
    
    def add_student(self, student_id: str, midterm_score: float = None, 
                   quiz_assignment_avg: float = None, previous_gpa: float = None, 
                   is_retake: int = 0) -> bool:
        """Add a single student to the database"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    EXEC sp_AddStudent ?, ?, ?, ?, ?
                """, student_id, midterm_score, quiz_assignment_avg, previous_gpa, is_retake)
                
                result = cursor.fetchone()
                if result and result[0] == 'SUCCESS':
                    print(f"✅ Added student {student_id}")
                    return True
                else:
                    print(f"❌ Failed to add student {student_id}: {result[1] if result else 'Unknown error'}")
                    return False
        except Exception as e:
            print(f"❌ Error adding student {student_id}: {e}")
            return False
    
    def add_students_batch(self, students_df: pd.DataFrame) -> bool:
        """Add multiple students from DataFrame"""
        try:
            success_count = 0
            for _, student in students_df.iterrows():
                if self.add_student(
                    student['student_id'],
                    student.get('midterm_score'),
                    student.get('quiz_assignment_avg'),
                    student.get('previous_gpa'),
                    student.get('is_retake', 0)
                ):
                    success_count += 1
            
            print(f"✅ Successfully added {success_count}/{len(students_df)} students")
            return success_count == len(students_df)
        except Exception as e:
            print(f"❌ Error in batch insert: {e}")
            return False
    
    def update_student(self, student_id: str, **kwargs) -> bool:
        """Update student data"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    EXEC sp_UpdateStudent ?, ?, ?, ?, ?
                """, 
                student_id,
                kwargs.get('midterm_score'),
                kwargs.get('quiz_assignment_avg'),
                kwargs.get('previous_gpa'),
                kwargs.get('is_retake')
                )
                
                result = cursor.fetchone()
                if result and result[0] == 'SUCCESS':
                    print(f"✅ Updated student {student_id}")
                    return True
                else:
                    print(f"❌ Failed to update student {student_id}: {result[1] if result else 'Unknown error'}")
                    return False
        except Exception as e:
            print(f"❌ Error updating student {student_id}: {e}")
            return False
    
    def save_predictions(self, predictions_df: pd.DataFrame, model_version: str = None) -> bool:
        """Save predictions to database"""
        try:
            success_count = 0
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                
                for _, pred in predictions_df.iterrows():
                    cursor.execute("""
                        EXEC sp_SavePredictions ?, ?, ?, ?
                    """, 
                    pred['student_id'],
                    float(pred['failure_probability']),
                    pred['risk_category'],
                    model_version
                    )
                    
                    result = cursor.fetchone()
                    if result and result[0] == 'SUCCESS':
                        success_count += 1
            
            print(f"✅ Saved {success_count}/{len(predictions_df)} predictions")
            return success_count == len(predictions_df)
        except Exception as e:
            print(f"❌ Error saving predictions: {e}")
            return False
    
    def get_latest_predictions(self) -> pd.DataFrame:
        """Get latest predictions for all students"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                query = """
                    SELECT s.StudentID as student_id,
                           s.MidtermScore as midterm_score,
                           s.QuizAssignmentAvg as quiz_assignment_avg,
                           s.PreviousGPA as previous_gpa,
                           s.IsRetake as is_retake,
                           p.FailureProbability as failure_probability,
                           p.RiskCategory as risk_category,
                           p.PredictionDate as prediction_date
                    FROM Students s
                    LEFT JOIN (
                        SELECT StudentID, FailureProbability, RiskCategory, PredictionDate,
                               ROW_NUMBER() OVER (PARTITION BY StudentID ORDER BY PredictionDate DESC) as rn
                        FROM Predictions
                    ) p ON s.StudentID = p.StudentID AND p.rn = 1
                    ORDER BY s.StudentID
                """
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            print(f"❌ Error retrieving predictions: {e}")
            return pd.DataFrame()
    
    def get_high_risk_students(self) -> pd.DataFrame:
        """Get all high-risk students"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                query = "SELECT * FROM vw_HighRiskStudents ORDER BY FailureProbability DESC"
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            print(f"❌ Error retrieving high-risk students: {e}")
            return pd.DataFrame()
    
    def get_risk_summary(self) -> pd.DataFrame:
        """Get risk category summary statistics"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                query = "SELECT * FROM vw_StudentRiskSummary"
                df = pd.read_sql(query, conn)
                return df
        except Exception as e:
            print(f"❌ Error retrieving risk summary: {e}")
            return pd.DataFrame()
    
    def save_model_metadata(self, model_version: str, train_accuracy: float,
                           test_accuracy: float, cv_score: float, 
                           feature_importance: Dict, model_path: str = None) -> bool:
        """Save model training metadata"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Convert feature importance to JSON string
                feature_json = json.dumps(feature_importance)
                
                cursor.execute("""
                    INSERT INTO ModelMetadata 
                    (ModelVersion, TrainAccuracy, TestAccuracy, CVScore, FeatureImportance, ModelPath)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, model_version, train_accuracy, test_accuracy, cv_score, feature_json, model_path)
                
                conn.commit()
                print(f"✅ Saved model metadata for version {model_version}")
                return True
        except Exception as e:
            print(f"❌ Error saving model metadata: {e}")
            return False
    
    def delete_student(self, student_id: str) -> bool:
        """Delete a student and all related predictions"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                
                # Delete predictions first (foreign key constraint)
                cursor.execute("DELETE FROM Predictions WHERE StudentID = ?", student_id)
                
                # Delete student
                cursor.execute("DELETE FROM Students WHERE StudentID = ?", student_id)
                
                if cursor.rowcount > 0:
                    conn.commit()
                    print(f"✅ Deleted student {student_id}")
                    return True
                else:
                    print(f"❌ Student {student_id} not found")
                    return False
        except Exception as e:
            print(f"❌ Error deleting student {student_id}: {e}")
            return False
    
    def get_student_count(self) -> int:
        """Get total number of students"""
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM Students")
                return cursor.fetchone()[0]
        except Exception as e:
            print(f"❌ Error getting student count: {e}")
            return 0
