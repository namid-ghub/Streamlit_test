import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load data
df = pd.read_csv("loan_data.csv")

# Example: select two features for simplicity
X = df[['income_annum', 'loan_amount']]
y = df['loan_approval']  # Replace this with your actual target column

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model
with open('Random_Forest.sav', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model saved as Random_Forest.sav")
