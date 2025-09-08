# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pickle
import os

# Load dataset
df = pd.read_csv("data/Housing.csv")

# Rename columns (if needed)
df.columns = ['area', 'bedrooms', 'bathrooms', 'price']

# Drop missing values
df = df.dropna()

# Remove outliers
df = df[(df['area'] > 200) & (df['area'] < 10000)]
df = df[(df['bedrooms'] >= 1) & (df['bedrooms'] <= 10)]
df = df[(df['bathrooms'] >= 1) & (df['bathrooms'] <= 10)]
df = df[df['price'] > 0]

# Split into features and target
X = df[['area', 'bedrooms', 'bathrooms']]
y = df['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"✅ R² Score after training: {r2:.2f}")

# Save model
os.makedirs("model", exist_ok=True)
with open("model/house_model.pkl", "wb") as f:
    pickle.dump((model, r2), f)

# Save accuracy separately
with open("model/accuracy.txt", "w") as f:
    f.write(str(r2))

print("✅ Model and accuracy saved successfully.")
