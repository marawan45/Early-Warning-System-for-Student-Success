import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import warnings
from ews2_database import EWS2Database

warnings.filterwarnings('ignore')


class EWS2PredictorMultiModel:
    def __init__(self, db_config: dict):
        self.models = {
            'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10,
                                                   min_samples_split=5, min_samples_leaf=2,
                                                   random_state=42, class_weight='balanced'),
            'GradientBoosting': GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=500, class_weight='balanced'),
            'SVM': SVC(probability=True, class_weight='balanced')
        }
        self.scaler = StandardScaler()
        self.feature_names = ['midterm_score', 'quiz_assignment_avg', 'previous_gpa', 'is_retake']
        self.db = EWS2Database(**db_config)
        self.trained_models = {}
        self.model_version = "v1.0"

        if not self.db.test_connection():
            raise ConnectionError("Database connection failed")

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feature_names:
            if feature not in data.columns:
                raise ValueError(f"Missing feature: {feature}")
        features = data[self.feature_names].copy()
        features['midterm_score'].fillna(features['midterm_score'].mean(), inplace=True)
        features['quiz_assignment_avg'].fillna(features['quiz_assignment_avg'].mean(), inplace=True)
        features['previous_gpa'].fillna(2.0, inplace=True)
        features['is_retake'].fillna(0, inplace=True)
        features['is_retake'] = features['is_retake'].astype(int)
        return features

    def _add_synthetic_target(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        risk_scores = []
        for _, row in data.iterrows():
            score = ((100 - row['midterm_score']) * 0.4 +
                     (100 - row['quiz_assignment_avg']) * 0.3 +
                     (4.0 - row['previous_gpa']) * 15 +
                     row['is_retake'] * 20 +
                     np.random.normal(0, 10))
            risk_scores.append(score)
        fail_prob = [1 / (1 + np.exp(-0.1 * (s - 50))) for s in risk_scores]
        data['will_fail_final'] = [1 if np.random.random() < p else 0 for p in fail_prob]
        return data

    def train_models(self) -> pd.DataFrame:
        data = self.db.get_all_students()
        if data.empty:
            raise ValueError("No student data in DB")
        if 'will_fail_final' not in data.columns:
            data = self._add_synthetic_target(data)

        X = self.prepare_features(data)
        y = data['will_fail_final']
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        results_summary = []

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            cv_score = cross_val_score(model, X_scaled, y, cv=5)

            # Store model and metrics
            self.trained_models[name] = model

            # Feature importance only available for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance = dict(zip(self.feature_names, model.feature_importances_))
            else:
                importance = 'N/A'

            summary = {
                'Model': name,
                'Train Accuracy': model.score(X_train, y_train),
                'Test Accuracy': model.score(X_test, y_test),
                'CV Mean': cv_score.mean(),
                'CV Std': cv_score.std(),
                'Feature Importance': importance,
                'Classification Report': classification_report(y_test, y_pred, output_dict=True),
                'Confusion Matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            results_summary.append(summary)

            print(f"--- {name} Results ---")
            print(f"Train Accuracy: {summary['Train Accuracy']:.4f}")
            print(f"Test Accuracy: {summary['Test Accuracy']:.4f}")
            print(f"CV Mean: {summary['CV Mean']:.4f} | CV Std: {summary['CV Std']:.4f}")
            print(f"Confusion Matrix:\n{summary['Confusion Matrix']}")
            if importance != 'N/A':
                print(f"Feature Importance: {importance}")

        return pd.DataFrame(results_summary)

    def predict_failure_risk(self, model_name='RandomForest') -> pd.DataFrame:
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} not trained")
        model = self.trained_models[model_name]
        student_data = self.db.get_all_students()
        X = self.prepare_features(student_data)
        X_scaled = self.scaler.transform(X)
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]
        results = student_data.copy()
        results['failure_risk'] = preds
        results['failure_probability'] = probs
        results['risk_category'] = results['failure_probability'].apply(
            lambda x: 'HIGH' if x >= 0.7 else 'MEDIUM' if x >= 0.4 else 'LOW'
        )
        print(f"\nPrediction summary using {model_name}:")
        print(results[['student_id', 'failure_probability', 'risk_category']].head(10))
        self.db.save_predictions(results, self.model_version)
        return results
