import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # Added for saving/loading models

# Load data
df = pd.read_csv("german_credit_data_with_labels.csv")
X = df.drop("Credit_Risk", axis=1)
y = df["Credit_Risk"]

# Categorical and numerical columns
categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop="first")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numerical_cols),
    ("cat", categorical_transformer, categorical_cols)
])

# Train-test split
X_processed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, stratify=y, random_state=42)

# Models to evaluate
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
    plt.close()

# Extract feature names after encoding
encoded_cols = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(encoded_cols)

# Function to plot feature importances
def plot_feature_importance(model, model_name):
    importances = model.feature_importances_
    indices = importances.argsort()[::-1]
    plt.figure(figsize=(12, 6))
    plt.title(f"{model_name} - Feature Importances")
    sns.barplot(x=importances[indices], y=[all_feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Plot feature importances for tree-based models only
for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    plot_feature_importance(model, name)

# Save the best performing model (choose one, e.g., XGBoost)
best_model_name = "XGBoost"  # Choose your best model here
best_model = models[best_model_name]

# Save the model
joblib.dump(best_model, f'{best_model_name}_model.joblib')
print(f"{best_model_name} model saved as {best_model_name}_model.joblib")

# Save the preprocessing steps (optional, if you want to include them with the model)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', best_model)
])

# Save the entire pipeline
joblib.dump(pipeline, 'credit_risk_pipeline.joblib')
print("Pipeline with preprocessing and model saved as 'credit_risk_pipeline.joblib'")

# Now, you can load and use the saved model or pipeline

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the saved model or pipeline
pipeline = joblib.load('credit_risk_pipeline.joblib')
print("Model and preprocessing pipeline loaded.")

# Function to make a prediction for new input data
def predict_credit_risk(input_data):
    # The input data needs to be in the same format as the training data
    # Assuming input_data is a dictionary with feature names and values:
    
    # Convert input data into a DataFrame (if it is not already a DataFrame)
    input_df = pd.DataFrame([input_data])

    # Make predictions
    prediction = pipeline.predict(input_df)
    
    # Output the prediction result
    if prediction[0] == 1:
        return "Good Credit Risk"
    else:
        return "Bad Credit Risk"

# Example of user input (replace with actual user input)
input_data = {
    'Age': 25,
    'Sex': 'male',
    'Job': 2,
    'Housing': 'own',
    'Saving accounts': 'moderate',
    'Checking account': 'rich',
    'Credit amount': 10000,
    'Duration': 24,
    'Purpose': 'radio/TV'
}

# Get prediction for the input data
result = predict_credit_risk(input_data)
print(f"Prediction: {result}")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Function to plot feature importances as percentage
def plot_feature_importance_percentage(model, model_name):
    # Get the feature importances
    importances = model.feature_importances_
    
    # Normalize the importances to sum up to 1 (percentage format)
    importances_percentage = (importances / np.sum(importances)) * 100
    
    # Sort the importances in descending order
    indices = importances_percentage.argsort()[::-1]
    
    # Plotting the graph
    plt.figure(figsize=(12, 6))
    plt.title(f"{model_name} - Feature Importances (Percentage)")
    sns.barplot(x=importances_percentage[indices], y=[all_feature_names[i] for i in indices])
    plt.xlabel("Importance (%)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

# Call the function for Random Forest and XGBoost models (or whichever models you want)
for name in ["Random Forest", "XGBoost"]:
    model = models[name]
    plot_feature_importance_percentage(model, name)


