Anomaly-Detection-in-Financial-Data-A-Machine-Learning-Approach

📌 Project Overview

This project is a Machine Learning-based Fraud Detection System designed to identify fraudulent transactions in highly imbalanced datasets. It compares multiple classification models, implements data preprocessing, and provides real-time fraud predictions.

🚀 Features

Multi-Model Comparison – Random Forest, Decision Tree, Naïve Bayes, and Logistic Regression.

Data Preprocessing – Handles categorical encoding, feature selection, and data balancing.

Fraud Prediction – Allows predictions via CSV file upload or manual transaction input.

Performance Metrics – Evaluates accuracy, precision, recall, F1-score, and ROC-AUC.


🛠️ Technologies Used

Python

Scikit-Learn

Pandas & NumPy

Joblib (Model Persistence)

Google Colab (Model Training & Execution)


📂 Project Structure

📁Anomaly-Detection-in-Financial-Data-A-Machine-Learning-Approach
│── 📜 train_model.py             
│── 📜 predict_model.py           
│── 📜 synthetic_fraud_data.csv   
│── 📜 training_data_code.py     
│── 📜 fraud_detection_model.pkl  
└── 📜 README.md                  

📌 How to Use

🔹 1. Train the Model :

      Run train_model.py to train and evaluate models. The best model (Random Forest) is saved as fraud_detection_model.pkl.

🔹 2. Predict Fraudulent Transactions :

      Run predict_model.py and choose between:
      CSV Mode: Upload a dataset for batch predictions.
      Manual Mode: Enter transaction details manually for real-time fraud detection.


📊 Model Performance

![image](https://github.com/user-attachments/assets/0e04b77d-a148-4001-8d7c-3db6957ef229)

📌 Random Forest achieved the best performance and is used as the final model.

🔮 Future Enhancements

🔹 Deep Learning – Integrate neural networks for fraud detection.

🔹 Real-Time Streaming – Enable continuous fraud monitoring.

🔹 Explainable AI (XAI) – Improve model transparency for better trust and interpretation.

👨‍💻 Author

Developed by Sakshham Khanijau.

