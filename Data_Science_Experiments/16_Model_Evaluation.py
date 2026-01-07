import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

data = {
    'Hours_Studied': [1, 2, 3, 4, 5],
    'Score':         [10, 20, 30, 40, 50]
}
df = pd.DataFrame(data)
X = df[['Hours_Studied']]
y = df['Score']

model = LinearRegression()
model.fit(X,y)
predictions = model.predict(X)

mse_score = mean_squared_error(y,predictions)
r2_val = r2_score(y,predictions)

print(f"R-Squared (Accuracy): {r2_val}")
print(f"MSE (Error Penalty):  {mse_score}")

y_messy = [12, 18, 33, 37, 55]

model.fit(X, y_messy)
predictions_messy = model.predict(X)

mse_messy = mean_squared_error(y_messy, predictions_messy)
r2_messy = r2_score(y_messy, predictions_messy)

print("\n--- Messy Model Results ---")
print(f"R-Squared (Accuracy): {r2_messy:.4f} (Lower is worse)")
print(f"MSE (Error Penalty):  {mse_messy:.4f} (Higher is worse)")