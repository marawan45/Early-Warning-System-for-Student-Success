"""
EWS2 - Early Warning System for Student Final Exam Failure Prediction
Command-line interface for predicting student failure risk using SQL Server database

Usage:
    python ews2_main.py --demo                    # Generate demo data and save to database
    python ews2_main.py                           # Use existing data from database
    python ews2_main.py --db-server "MyServer"    # Connect to custom SQL Server
    python ews2_main.py --help                    # Show help
"""

import argparse
import pandas as pd
import numpy as np
from ews2_predictor import EWS2Predictor
from ews2_database import EWS2Database
import matplotlib.pyplot as plt
import seaborn as sns

def print_banner():
    """Print EWS2 banner"""
    print("="*60)
    print("  EWS2 - Early Warning System for Student Success")
    print("  Predicting Final Exam Failure Risk")
    print("="*60)

def print_summary_stats(predictions_df):
    """Print summary statistics"""
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 40)
    
    total = len(predictions_df)
    high_risk = len(predictions_df[predictions_df['risk_category'] == 'HIGH'])
    medium_risk = len(predictions_df[predictions_df['risk_category'] == 'MEDIUM'])
    low_risk = len(predictions_df[predictions_df['risk_category'] == 'LOW'])
    
    print(f"Total Students: {total}")
    print(f"High Risk (≥70% failure prob): {high_risk} ({high_risk/total*100:.1f}%)")
    print(f"Medium Risk (40-70% failure prob): {medium_risk} ({medium_risk/total*100:.1f}%)")
    print(f"Low Risk (<40% failure prob): {low_risk} ({low_risk/total*100:.1f}%)")
    
    avg_prob = predictions_df['failure_probability'].mean()
    print(f"Average Failure Probability: {avg_prob:.1%}")

def print_high_risk_students(predictions_df):
    """Print details of high-risk students"""
    high_risk = predictions_df[predictions_df['risk_category'] == 'HIGH'].copy()
    
    if len(high_risk) == 0:
        print("\n✅ No high-risk students found!")
        return
    
    print(f"\n🚨 HIGH RISK STUDENTS ({len(high_risk)} students)")
    print("-" * 60)
    
    # Sort by failure probability (highest first)
    high_risk = high_risk.sort_values('failure_probability', ascending=False)
    
    for _, student in high_risk.iterrows():
        print(f"Student ID: {student['student_id']}")
        print(f"  Failure Probability: {student['failure_probability']:.1%}")
        print(f"  Midterm Score: {student['midterm_score']:.1f}%")
        print(f"  Quiz/Assignment Avg: {student['quiz_assignment_avg']:.1f}%")
        print(f"  Previous GPA: {student['previous_gpa']:.2f}")
        print(f"  Course Retake: {'Yes' if student['is_retake'] else 'No'}")
        print()

def print_medium_risk_students(predictions_df):
    """Print details of medium-risk students"""
    medium_risk = predictions_df[predictions_df['risk_category'] == 'MEDIUM'].copy()
    
    if len(medium_risk) == 0:
        print("\n✅ No medium-risk students found!")
        return
    
    print(f"\n⚠️  MEDIUM RISK STUDENTS ({len(medium_risk)} students)")
    print("-" * 60)
    
    # Sort by failure probability (highest first)
    medium_risk = medium_risk.sort_values('failure_probability', ascending=False)
    
    for _, student in medium_risk.iterrows():
        print(f"Student ID: {student['student_id']}")
        print(f"  Failure Probability: {student['failure_probability']:.1%}")
        print(f"  Midterm Score: {student['midterm_score']:.1f}%")
        print(f"  Quiz/Assignment Avg: {student['quiz_assignment_avg']:.1f}%")
        print(f"  Previous GPA: {student['previous_gpa']:.2f}")
        print(f"  Course Retake: {'Yes' if student['is_retake'] else 'No'}")
        print()

def print_feature_importance(predictor):
    """Print feature importance analysis"""
    if not predictor.is_trained:
        return
    
    importance = dict(zip(predictor.feature_names, predictor.model.feature_importances_))
    
    print("\n🔍 FEATURE IMPORTANCE ANALYSIS")
    print("-" * 40)
    
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    for feature, imp in sorted_features:
        feature_name = {
            'midterm_score': 'Midterm Score',
            'quiz_assignment_avg': 'Quiz/Assignment Average',
            'previous_gpa': 'Previous Semester GPA',
            'is_retake': 'Course Retake Flag'
        }.get(feature, feature)
        
        bar = "█" * int(imp * 50)  # Visual bar
        print(f"{feature_name:<25}: {imp:.3f} {bar}")

def print_recommendations(predictions_df):
    """Print intervention recommendations"""
    high_risk = predictions_df[predictions_df['risk_category'] == 'HIGH']
    medium_risk = predictions_df[predictions_df['risk_category'] == 'MEDIUM']
    
    print("\n💡 INTERVENTION RECOMMENDATIONS")
    print("-" * 40)
    
    if len(high_risk) > 0:
        print("🚨 IMMEDIATE ACTIONS NEEDED:")
        print(f"  • Schedule urgent meetings with {len(high_risk)} high-risk students")
        print("  • Provide intensive tutoring sessions before final exam")
        print("  • Consider alternative assessment options")
        print("  • Implement daily check-ins")
        print()
    
    if len(medium_risk) > 0:
        print("⚠️  MONITORING ACTIONS:")
        print(f"  • Monitor {len(medium_risk)} medium-risk students weekly")
        print("  • Provide additional practice materials")
        print("  • Offer optional review sessions")
        print("  • Send regular progress updates")
        print()
    
    # Analyze common patterns
    all_at_risk = pd.concat([high_risk, medium_risk]) if len(high_risk) > 0 or len(medium_risk) > 0 else pd.DataFrame()
    
    if len(all_at_risk) > 0:
        print("📈 GENERAL RECOMMENDATIONS:")
        
        avg_midterm = all_at_risk['midterm_score'].mean()
        avg_quiz = all_at_risk['quiz_assignment_avg'].mean()
        retake_rate = all_at_risk['is_retake'].mean()
        
        if avg_midterm < 60:
            print("  • Review midterm material - many at-risk students struggled")
        
        if avg_quiz < 65:
            print("  • Provide more frequent formative assessments")
        
        if retake_rate > 0.3:
            print("  • Focus extra attention on students retaking the course")
        
        print(f"  • At-risk students average midterm: {avg_midterm:.1f}%")
        print(f"  • At-risk students average quiz/assignment: {avg_quiz:.1f}%")

def create_visualizations(predictions_df):
    """Create and save visualization plots"""
    print("\n📊 Creating visualizations...")
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('EWS2 - Student Risk Analysis', fontsize=16, fontweight='bold')
    
    # 1. Risk distribution
    risk_counts = predictions_df['risk_category'].value_counts()
    colors = {'HIGH': '#FF4444', 'MEDIUM': '#FFA500', 'LOW': '#44AA44'}
    ax1 = axes[0, 0]
    bars = ax1.bar(risk_counts.index, risk_counts.values, 
                   color=[colors.get(x, 'blue') for x in risk_counts.index])
    ax1.set_title('Risk Level Distribution')
    ax1.set_xlabel('Risk Level')
    ax1.set_ylabel('Number of Students')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # 2. Failure probability histogram
    ax2 = axes[0, 1]
    ax2.hist(predictions_df['failure_probability'], bins=20, alpha=0.7, color='skyblue')
    ax2.set_title('Failure Probability Distribution')
    ax2.set_xlabel('Failure Probability')
    ax2.set_ylabel('Number of Students')
    
    # 3. Midterm vs Quiz/Assignment scatter
    ax3 = axes[1, 0]
    scatter = ax3.scatter(predictions_df['midterm_score'], 
                         predictions_df['quiz_assignment_avg'],
                         c=predictions_df['failure_probability'], 
                         cmap='RdYlGn_r', alpha=0.6)
    ax3.set_title('Midterm vs Quiz/Assignment Performance')
    ax3.set_xlabel('Midterm Score (%)')
    ax3.set_ylabel('Quiz/Assignment Average (%)')
    plt.colorbar(scatter, ax=ax3, label='Failure Probability')
    
    # 4. GPA vs Risk
    ax4 = axes[1, 1]
    risk_order = ['LOW', 'MEDIUM', 'HIGH']
    sns.boxplot(data=predictions_df, x='risk_category', y='previous_gpa', 
                order=risk_order, ax=ax4)
    ax4.set_title('Previous GPA by Risk Level')
    ax4.set_xlabel('Risk Level')
    ax4.set_ylabel('Previous GPA')
    
    plt.tight_layout()
    plt.savefig('ews2_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✅ Saved visualization to 'ews2_analysis.png'")
    
    # Show the plot
    plt.show()

def export_results(predictions_df, filename='ews2_results.csv'):
    """Export results to CSV"""
    # Select relevant columns for export
    export_df = predictions_df[[
        'student_id', 'midterm_score', 'quiz_assignment_avg', 
        'previous_gpa', 'is_retake', 'failure_probability', 'risk_category'
    ]].copy()
    
    # Round numerical values
    export_df['failure_probability'] = export_df['failure_probability'].round(3)
    export_df['midterm_score'] = export_df['midterm_score'].round(1)
    export_df['quiz_assignment_avg'] = export_df['quiz_assignment_avg'].round(1)
    export_df['previous_gpa'] = export_df['previous_gpa'].round(2)
    
    export_df.to_csv(filename, index=False)
    print(f"  ✅ Results exported to '{filename}'")


def main():
    parser = argparse.ArgumentParser(description='EWS2 - Early Warning System for Student Success')
    parser.add_argument('--demo', action='store_true', help='Generate synthetic demo data and save to database')
    parser.add_argument('--samples', type=int, default=500, help='Number of synthetic samples for demo (default: 500)')
    parser.add_argument('--no-viz', action='store_true', help='Skip visualization generation')
    parser.add_argument('--export', type=str, help='Export results to CSV file')
    
    # Database connection arguments
    parser.add_argument('--db-server', type=str, default='localhost', help='SQL Server instance (default: localhost)')
    parser.add_argument('--db-name', type=str, default='EWS2_StudentSuccess', help='Database name (default: EWS2_StudentSuccess)')
    parser.add_argument('--db-user', type=str, help='Database username (leave empty for Windows Auth)')
    parser.add_argument('--db-pass', type=str, help='Database password (leave empty for Windows Auth)')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Initialize database configuration (always required)
    db_config = {
        'server': args.db_server,
        'database': args.db_name,
        'trusted_connection': args.db_user is None,
        'username': args.db_user,
        'password': args.db_pass
    }
    
    # Initialize predictor with database
    predictor = EWS2Predictor(db_config=db_config)
    
    # Verify database connection
    if not predictor.db or not predictor.db.test_connection():
        print("❌ Database connection failed. Please check your connection settings.")
        print("   Make sure SQL Server is running and the database exists.")
        return
    
    # Handle demo data generation
    if args.demo:
        print(f"\n🔄 Generating {args.samples} synthetic students for demonstration...")
        data = predictor.generate_synthetic_data(args.samples)
        print("✅ Synthetic data generated successfully")
        
        print("💾 Saving synthetic data to database...")
        predictor.db.add_students_batch(data)
    else:
        print("\n🔄 Using existing data from database...")
        student_count = predictor.db.get_student_count()
        if student_count == 0:
            print("⚠️  No students found in database. Use --demo to generate sample data.")
            return
        print(f"✅ Found {student_count} students in existing database")
        print("📋 Using real student data for predictions")
    
    # Train model
    print("\n🤖 Training prediction model...")
    training_results = predictor.train_model()
    
    if not training_results['success']:
        print(f"❌ Training failed: {training_results['message']}")
        return
    
    print("✅ Model trained successfully")
    print(f"  Training Accuracy: {training_results['train_accuracy']:.3f}")
    print(f"  Test Accuracy: {training_results['test_accuracy']:.3f}")
    print(f"  Cross-validation Score: {training_results['cv_mean']:.3f} ± {training_results['cv_std']:.3f}")
    
    # Make predictions
    print("\n🔮 Making failure risk predictions...")
    predictions = predictor.predict_failure_risk()
    print("✅ Predictions completed")
    
    # Display results
    print_summary_stats(predictions)
    print_feature_importance(predictor)
    print_high_risk_students(predictions)
    print_medium_risk_students(predictions)
    print_recommendations(predictions)
    
    # Create visualizations
    if not args.no_viz:
        try:
            create_visualizations(predictions)
        except Exception as e:
            print(f"⚠️  Could not create visualizations: {e}")
    
    # Export results
    if args.export:
        print(f"\n💾 Exporting results...")
        export_results(predictions, args.export)
    else:
        print(f"\n💾 Exporting results...")
        export_results(predictions)
    
    print("\n" + "="*60)
    print("  EWS2 Analysis Complete!")
    print("  Use the recommendations above to help at-risk students")
    print("="*60)

if __name__ == "__main__":
    main()
