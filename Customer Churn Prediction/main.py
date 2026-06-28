
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Generate Customer Dataset
# -----------------------------

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Age": np.random.randint(18,70,n),
    "MonthlyCharges": np.random.randint(200,5000,n),
    "Tenure": np.random.randint(1,72,n),
    "SupportCalls": np.random.randint(0,10,n),
    "ContractType": np.random.randint(0,2,n)
})

# Generate Target

data["Churn"] = (
    (data["MonthlyCharges"] > 3000) &
    (data["SupportCalls"] > 5) &
    (data["Tenure"] < 12)
).astype(int)

print("Dataset Preview")
print(data.head())

print("\nDataset Shape:", data.shape)

print("\nTarget Distribution")
print(data["Churn"].value_counts())

# -----------------------------
# Features & Target
# -----------------------------

X = data.drop("Churn", axis=1)
y = data["Churn"]

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Train Model
# -----------------------------

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

print("\nModel Training Completed Successfully!")

# -----------------------------
# Prediction
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------

print("\nAccuracy Score")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "customer_churn_model.pkl")

print("\nModel Saved Successfully!")

# -----------------------------
# User Prediction
# -----------------------------

print("\n===== Customer Churn Prediction =====")

age = int(input("Age: "))
monthly = float(input("Monthly Charges: "))
tenure = int(input("Tenure (Months): "))
support = int(input("Support Calls: "))
contract = int(input("Contract Type (0=Monthly, 1=Yearly): "))

sample = [[age, monthly, tenure, support, contract]]

prediction = model.predict(sample)

if prediction[0] == 1:
    print("\n⚠️ Customer is likely to Churn.")
else:
    print("\n✅ Customer is likely to Stay.")

print("\nProject Completed Successfully!")
