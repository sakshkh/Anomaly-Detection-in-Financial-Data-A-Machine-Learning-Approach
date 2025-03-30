import pandas as pd
import numpy as np

# Define number of samples
num_samples = 1000

# Generate synthetic data
np.random.seed(42)

synthetic_data = pd.DataFrame({
    'transaction_id': np.arange(1, num_samples + 1),
    'timestamp': np.random.randint(1609459200, 1640995200, num_samples),
    'customer_id': np.random.randint(1000, 9999, num_samples),
    'merchant_id': np.random.randint(5000, 15000, num_samples),
    'transaction_amount': np.round(np.random.uniform(5, 5000, num_samples), 2),
    'transaction_type': np.random.choice(['Online', 'In-Store', 'ATM'], num_samples),
    'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'UPI', 'Net Banking'], num_samples),
    'location': np.random.choice(['Delhi', 'Mumbai', 'Bangalore', 'Chennai', 'Hyderabad'], num_samples),
    'device_used': np.random.choice(['Mobile', 'Desktop', 'POS Terminal', 'ATM Machine'], num_samples),
    'previous_transaction_gap': np.round(np.random.uniform(0, 72, num_samples), 2),
    'day_of_week': np.random.randint(0, 7, num_samples),
    'hour_of_day': np.random.randint(0, 24, num_samples),
    'is_weekend': np.random.choice([0, 1], num_samples),
    'transaction_velocity': np.random.randint(1, 50, num_samples),
    'average_transaction_amount': np.round(np.random.uniform(100, 2000, num_samples), 2),
    'fraudulent': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])
})

# Save dataset locally
file_name = "synthetic_fraud_data.csv"
synthetic_data.to_csv(file_name, index=False)

print("Synthetic dataset generated and saved!")
