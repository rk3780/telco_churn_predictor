import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Ensure model directory exists
os.makedirs("model", exist_ok=True)

# Load dataset
df = pd.read_csv('telco.csv')

# Clean blank values in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(" ", pd.NA)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Convert 'Churn' to 0/1
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Encode 'gender'
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

# Select features and target
features = ['gender', 'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'model/churn_model.pkl')

print("✅ Model trained and saved to model/churn_model.pkl")
