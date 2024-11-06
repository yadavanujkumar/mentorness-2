import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the data
file_path = 'healthcare-dataset-stroke-data.csv'
stroke_data = pd.read_csv(file_path)

# Impute missing values in 'bmi' column with mean
imputer = SimpleImputer(strategy='mean')
stroke_data['bmi'] = imputer.fit_transform(stroke_data[['bmi']])

# Encode categorical variables
label_encoder = LabelEncoder()

# Binary encodings
stroke_data['ever_married'] = label_encoder.fit_transform(stroke_data['ever_married'])
stroke_data['Residence_type'] = label_encoder.fit_transform(stroke_data['Residence_type'])
stroke_data['gender'] = stroke_data['gender'].replace({'Other': 'Male'})  # Handling 'Other' by assuming 'Male'
stroke_data['gender'] = label_encoder.fit_transform(stroke_data['gender'])

# One-hot encode multi-class columns (work_type and smoking_status)
stroke_data_encoded = pd.get_dummies(stroke_data, columns=['work_type', 'smoking_status'], drop_first=True)

# Drop 'id' column
stroke_data_encoded = stroke_data_encoded.drop(columns=['id'])

# Split data into features and target
X = stroke_data_encoded.drop('stroke', axis=1)
y = stroke_data_encoded['stroke']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Scale the continuous features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
log_reg = LogisticRegression(random_state=42)
decision_tree = DecisionTreeClassifier(random_state=42)
random_forest = RandomForestClassifier(random_state=42)
svm = SVC(random_state=42)

# Dictionary to store results
results = {}

# Train and evaluate models
models = {
    'Logistic Regression': log_reg,
    'Decision Tree': decision_tree,
    'Random Forest': random_forest,
    'SVM': svm
}

for model_name, model in models.items():
    # Train model
    model.fit(X_train_scaled, y_train)
    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    # Evaluate performance
    results[model_name] = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred)
    }

# Display the results
results_df = pd.DataFrame(results).T
print(results_df)



print("done")