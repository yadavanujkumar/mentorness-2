import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load data
file_path = 'healthcare-dataset-stroke-data.csv'
stroke_data = pd.read_csv(file_path)

# Data Preprocessing
imputer = SimpleImputer(strategy='mean')
stroke_data['bmi'] = imputer.fit_transform(stroke_data[['bmi']])  # Impute missing BMI values

label_encoder = LabelEncoder()
stroke_data['ever_married'] = label_encoder.fit_transform(stroke_data['ever_married'])
stroke_data['Residence_type'] = label_encoder.fit_transform(stroke_data['Residence_type'])
stroke_data['gender'] = stroke_data['gender'].replace({'Other': 'Male'})  # Handle 'Other' gender category
stroke_data['gender'] = label_encoder.fit_transform(stroke_data['gender'])

# One-hot encoding for work_type and smoking_status
stroke_data_encoded = pd.get_dummies(stroke_data, columns=['work_type', 'smoking_status'], drop_first=True)

# Drop 'id' column
stroke_data_encoded = stroke_data_encoded.drop(columns=['id'])

# Split features and target
X = stroke_data_encoded.drop('stroke', axis=1)
y = stroke_data_encoded['stroke']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Selection and Hyperparameter Tuning with Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

# Print best parameters and F1 score from GridSearchCV
print("Best parameters:", grid_search.best_params_)
print("Best F1 score:", grid_search.best_score_)

# Train best model on full training data
best_rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
best_rf.fit(X_train_scaled, y_train)

# Model Evaluation on test set
y_pred = best_rf.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# Save the best model
joblib.dump(best_rf, 'best_stroke_model.pkl')
