

# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('customer_purchase_data.csv')  # Replace with your file path

# Separate features and target variable
X = data.drop(columns='PurchaseStatus')
y = data['PurchaseStatus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# Train and evaluate each model using cross-validation
model_scores = {}
for model_name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    model_scores[model_name] = scores.mean()
    print(f"{model_name} Cross-Validation Accuracy: {scores.mean():.4f}")

# Select the best model based on cross-validation accuracy
best_model_name = max(model_scores, key=model_scores.get)
best_model = models[best_model_name]
print(f"\nBest Initial Model: {best_model_name} with Accuracy: {model_scores[best_model_name]:.4f}")

# Hyperparameter tuning for the best model (Gradient Boosting in this case)
if best_model_name == 'Gradient Boosting':
    # Define the parameter grid for Gradient Boosting
    param_grid = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    # Set up GridSearchCV
    grid_search = GridSearchCV(estimator=best_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_  # Update best model with tuned parameters
    best_params = grid_search.best_params_
    print(f"\nBest Hyperparameters for Gradient Boosting: {best_params}")

# Evaluate the tuned best model on the test set
y_pred = best_model.predict(X_test_scaled)
print("\nClassification Report on Test Set:\n", classification_report(y_test, y_pred))

# Prediction example for new sample data
sample_data = [[25, 1, 50000, 5, 1, 45, 1, 2]]  # Example input
sample_scaled = scaler.transform(sample_data)
purchase_prediction = best_model.predict(sample_scaled)

# Output the prediction
print("Purchase Prediction for Sample Data:", "Yes" if purchase_prediction[0] == 1 else "No")
