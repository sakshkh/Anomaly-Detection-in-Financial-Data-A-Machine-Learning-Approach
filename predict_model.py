import pandas as pd
import joblib
from google.colab import files
from sklearn.preprocessing import LabelEncoder

# Load trained model
loaded_model = joblib.load('fraud_detection_model.pkl')
print("Model loaded successfully.")

# Define feature columns
feature_columns = [
    'transaction_id', 'timestamp', 'customer_id', 'merchant_id',
    'transaction_amount', 'transaction_type', 'payment_method', 'location',
    'device_used', 'previous_transaction_gap', 'day_of_week', 'hour_of_day',
    'is_weekend', 'transaction_velocity', 'average_transaction_amount'
]

# Initialize encoders for categorical features
encoders = {col: LabelEncoder() for col in ['transaction_type', 'payment_method', 'location', 'device_used']}

# User input for prediction mode
option = input("Enter 'csv' to upload a CSV file or 'manual' to enter transaction details manually: ").strip().lower()

if option == 'csv':
    # Upload CSV file
    new_uploaded = files.upload()
    new_file_path = list(new_uploaded.keys())[0]
    print(f"Using file: {new_file_path}")

    # Load CSV data
    new_df = pd.read_csv(new_file_path)

    # Convert timestamps to numeric format
    if 'timestamp' in new_df.columns:
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp']).apply(lambda x: int(x.timestamp()))

    # Encode categorical features
    for col in encoders:
        if col in new_df.columns:
            new_df[col] = encoders[col].fit_transform(new_df[col])

    # Select required features
    new_df = new_df[[col for col in feature_columns if col in new_df.columns]]

    # Make predictions
    predictions_new = loaded_model.predict(new_df)
    new_df['predicted_class'] = predictions_new
    new_df['predicted_label'] = new_df['predicted_class'].apply(lambda x: "Fraud" if x == 1 else "Legitimate")

    # Print results
    print("\nPredictions for CSV File:")
    print(new_df.head())

    fraudulent_transactions = new_df[new_df['predicted_class'] == 1]
    print("\nFraudulent Transactions:")
    print(fraudulent_transactions)

elif option == 'manual':
    print("Enter transaction details:")
    features = {}

    # Manual input for transaction details
    features['transaction_id'] = int(input("Transaction ID: "))
    timestamp_str = input("Timestamp (YYYY-MM-DD HH:MM:SS): ")
    features['timestamp'] = int(pd.to_datetime(timestamp_str).timestamp())  
    features['customer_id'] = int(input("Customer ID: "))
    features['merchant_id'] = int(input("Merchant ID: "))
    features['transaction_amount'] = float(input("Transaction Amount: "))
    features['transaction_type'] = input("Transaction Type: ")
    features['payment_method'] = input("Payment Method: ")
    features['location'] = input("Location: ")
    features['device_used'] = input("Device Used: ")
    features['previous_transaction_gap'] = float(input("Previous Transaction Gap: "))
    features['day_of_week'] = int(input("Day of Week: "))
    features['hour_of_day'] = int(input("Hour of Day: "))
    features['is_weekend'] = int(input("Is Weekend (0 or 1): "))
    features['transaction_velocity'] = int(input("Transaction Velocity: "))
    features['average_transaction_amount'] = float(input("Average Transaction Amount: "))

    # Convert to DataFrame
    manual_data = pd.DataFrame([features])

    # Encode categorical features
    for col in encoders:
        manual_data[col] = encoders[col].fit_transform(manual_data[col])

    # Select required features
    manual_data = manual_data[feature_columns]

    # Make prediction
    predictions_manual = loaded_model.predict(manual_data)
    manual_data['predicted_class'] = predictions_manual
    manual_data['predicted_label'] = manual_data['predicted_class'].apply(lambda x: "Fraud" if x == 1 else "Legitimate")

    # Print results
    print("\nPrediction for Manually Entered Transaction:")
    print(manual_data)

else:
    print("Invalid option! Please restart the script.")
