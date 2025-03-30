import pandas as pd
import numpy as np
import joblib
from google.colab import files
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from google.colab import files

# Upload the dataset
uploaded = files.upload()
file_path = list(uploaded.keys())[0] 
print(f"Using file: {file_path}")

df = pd.read_csv(file_path)
print("Columns in Dataset:", df.columns)

# Detect target column
target_col = df.columns[-1]
print(f"Detected Target Column: {target_col}")

# Convert timestamp columns to numeric format
non_numeric_cols = df.select_dtypes(include=['object']).columns
for col in non_numeric_cols:
    try:
        df[col] = pd.to_datetime(df[col], errors='coerce').astype('int64') // 10**9
    except:
        pass 

# Label Encoding categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  
    label_encoders[col] = le

print("Preprocessing complete!")

# Remove highly correlated features
correlation_matrix = df.corr()
high_corr_features = correlation_matrix.columns[
    (correlation_matrix[target_col].abs() > 0.95) & (correlation_matrix[target_col].abs() < 1)
]
df.drop(columns=high_corr_features, inplace=True)
print("Dropped highly correlated features:", list(high_corr_features))

# Define features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# Save feature names
feature_columns = X.columns
joblib.dump(feature_columns, 'feature_columns.pkl')

print("Feature selection complete!")

# Define models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500)
}

# Train models and evaluate
results = {}
for name, model in models.items():
    model.fit(X, y)
    y_pred = model.predict(X)
    results[name] = {
        "Accuracy": accuracy_score(y, y_pred),
        "Precision": precision_score(y, y_pred),
        "Recall": recall_score(y, y_pred),
        "F1-score": f1_score(y, y_pred),
        "ROC-AUC": roc_auc_score(y, y_pred)
    }

results_df = pd.DataFrame(results).T
print("
Model Performance:
", results_df)

# Save the best model
best_model = models["Random Forest"]
model_filename = 'fraud_detection_model.pkl'
joblib.dump(best_model, model_filename)
print(f"Model saved successfully as '{model_filename}'.")

# Download model file
files.download(model_filename)
