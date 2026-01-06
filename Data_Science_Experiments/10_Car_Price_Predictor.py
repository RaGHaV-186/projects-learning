import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = {
    'Horsepower': [130, 165, 150, 150, 140, 198, 220, 215, 225, 190],
    'Price': [16500, 19000, 18500, 20000, 17000, 35000, 40000, 38000, 42000, 32000]
}

df = pd.DataFrame(data)
print(df)

X = df[['Horsepower']]

Y = df['Price']

model = LinearRegression()

model.fit(X,Y)

print(f"Intercept (Starting Price):${model.intercept_:.2f}")
print(f"Slope (price per HP):${model.coef_[0]:.2f}")

new_car_hp = pd.DataFrame({'Horsepower': [250]})
predicted_price = model.predict(new_car_hp)

print(f"A car with 250HP should roughly cost ${predicted_price[0]:.2f}")

plt.scatter(df["Horsepower"], df["Price"], color='blue', label='Actual Cars')

plt.plot(df["Horsepower"], model.predict(X), color='red', label='Prediction Line')

plt.xlabel('Horsepower')
plt.ylabel('Price ($)')
plt.title('Car Price Prediction Model')
plt.legend()
plt.show()

