import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json

X_test = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv").values.ravel()

model = joblib.load("models/model_optimal.pkl")

predictions = model.predict(X_test)

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

metrics = {
    "mse": mse,
    "r2": r2
}

prediction_df = pd.DataFrame({
    "y_true": y_test,
    "y_pred": predictions
})

prediction_df.to_csv("data/predictions.csv", index=False)

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

print(metrics)