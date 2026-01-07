import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = {
    'Size': [1500, 2500, 1800, 4000, 2000, 1200, 3000, 1600],
    'Rooms': [3, 4, 3, 5, 3, 2, 4, 3],
    'Age':   [10, 5, 20, 2, 15, 30, 8, 18],
    'Price': [300000, 500000, 280000, 750000, 320000, 200000, 600000, 290000]
}
df = pd.DataFrame(data)

X = df[["Size","Rooms","Age"]]
Y = df["Price"]

model = LinearRegression()
model.fit(X,Y)
Yhat = model.predict(X)

plt.figure(figsize=(12,6))

sns.kdeplot(Y, color="blue", label="Actual Values", fill=True, alpha=0.3)

sns.kdeplot(Yhat, color="red", label="Predicted Values", fill=True, alpha=0.3)

plt.title('Distribution Plot: Actual vs Predicted Prices')
plt.xlabel('Price ($)')
plt.ylabel('Density (Frequency)')
plt.legend()
plt.show()

