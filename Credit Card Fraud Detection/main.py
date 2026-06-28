import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Generate Dataset
np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "Amount": np.random.randint(10,5000,n),
    "Transaction_Time": np.random.randint(0,24,n),
    "Location_Change": np.random.randint(0,2,n),
    "International": np.random.randint(0,2,n),
    "Card_Present": np.random.randint(0,2,n)
})

# Fraud Label
data["Fraud"] = (
    (data["Amount"]>3000) &
    (data["International"]==1) &
    (data["Card_Present"]==0)
).astype(int)

print("Dataset Preview")
print(data.head())

print("\nDataset Shape:",data.shape)

print("\nFraud Distribution")
print(data["Fraud"].value_counts())

# Features & Target
X = data.drop("Fraud",axis=1)
y = data["Fraud"]

# Split Data
X_train,X_test,y_train,y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)

model.fit(X_train,y_train)

print("\nModel Training Completed")

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy")
print(accuracy_score(y_test,y_pred))

print("\nClassification Report")
print(classification_report(y_test,y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test,y_pred))

# Save Model
joblib.dump(model,"fraud_detection_model.pkl")

print("\nModel Saved Successfully")

# User Prediction
print("\nCredit Card Fraud Prediction")

amount = float(input("Transaction Amount: "))
time = int(input("Transaction Time (0-23): "))
location = int(input("Location Changed (1=Yes,0=No): "))
international = int(input("International Transaction (1=Yes,0=No): "))
card = int(input("Card Present (1=Yes,0=No): "))

sample = [[amount,time,location,international,card]]

prediction = model.predict(sample)

if prediction[0]==1:
    print("\n⚠️ Fraudulent Transaction")
else:
    print("\n✅ Legitimate Transaction")
