from cProfile import label

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = {
    'Position': ['Business Analyst', 'Junior Consultant', 'Senior Consultant', 'Manager',
                 'Country Manager', 'Region Manager', 'Partner', 'Senior Partner', 'C-Level', 'CEO'],
    'Level':  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Salary': [45000, 50000, 60000, 80000, 110000, 150000, 200000, 300000, 500000, 1000000]
}
df = pd.DataFrame(data)

X = df[["Level"]]
Y = df["Salary"]

lin_reg = LinearRegression()
lin_reg.fit(X, Y)

poly_reg = PolynomialFeatures(degree=4)
X_poly =poly_reg.fit_transform(X)


lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,Y)

plt.figure(figsize=(10,6))

plt.scatter(X,Y,color='red',label="Actual Salary Data")

plt.plot(X,lin_reg.predict(X),color="blue",label = "Linear (Straight)")

plt.plot(X, lin_reg_2.predict(X_poly), color='green', label='Polynomial (Degree 4)')

plt.title('Truth or Bluff: Salary Detector')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.legend()
plt.show()

pred_linear = lin_reg.predict([[6.5]])
print(f"Linear Model Guesses: ${pred_linear[0]:,.2f}")

pred_poly = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(f"Polynomial Model Guesses: ${pred_poly[0]:,.2f}")


predicted_value = pred_poly[0]
claimed_value = 160000

difference = abs(predicted_value - claimed_value)

print(f"Difference: ${difference:,.2f}")

if difference < 5000:
    print("Verdict: TELLING THE TRUTH (It's close enough!)")
else:
    print("Verdict: BLUFFING (The gap is too big!)")