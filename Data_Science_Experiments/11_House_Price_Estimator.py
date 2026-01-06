import pandas as pd
from sklearn.linear_model import LinearRegression

data = {
    'Size_SqFt': [1500, 2500, 1800, 4000, 2000, 1200],
    'Bedrooms':  [3,    4,    3,    5,    3,    2],
    'Age_Years': [10,   5,    20,   2,    15,   30],
    'Price':     [300000, 500000, 280000, 750000, 320000, 200000]
}

df = pd.DataFrame(data)

print("--- Real Estate Data ---")
print(df)

X = df[["Size_SqFt","Bedrooms","Age_Years"]]
Y = df["Price"]

model = LinearRegression()

model.fit(X,Y)

print(f"Base Price (Intercept): ${model.intercept_:.2f}")
print("Price Adjustments (Coefficients):")
print(f" - Per SqFt:    ${model.coef_[0]:.2f}")
print(f" - Per Bedroom: ${model.coef_[1]:.2f}")
print(f" - Per Year Old:${model.coef_[2]:.2f}")

new_home = pd.DataFrame({
    "Size_SqFt":[3000],
    "Bedrooms":[4],
    "Age_Years":[10]
})

predicted_price = model.predict(new_home)

print(f"\nPrediction for New House: ${predicted_price[0]:,.2f}")
