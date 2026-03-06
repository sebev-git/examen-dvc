import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv").values.ravel()

best_params = joblib.load("models/best_params.pkl")

model = RandomForestRegressor(**best_params, random_state=42)

model.fit(X_train, y_train)

joblib.dump(model, "models/model_optimal.pkl")

print("Modèle entraîné et sauvegardé.")