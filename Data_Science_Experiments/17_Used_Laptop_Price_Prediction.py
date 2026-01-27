import pandas as pd
from scipy.stats import alpha

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

filepath = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(filepath, header= 0)

#print(df.head(10))

X = df[["CPU_frequency"]]
y = df["Price"]

model = LinearRegression()
model.fit(X, y)

Yhat = model.predict(X)

plt.figure(figsize=(10,6))

sns.kdeplot(y,color="blue",label="Actual Values",fill=True,alpha=0.3)

sns.kdeplot(Yhat,color="red",label="Predicted Values",fill=True,alpha=0.3)

# plt.title('Distribution Plot: Actual vs Predicted Prices')
# plt.xlabel('Price')
# plt.ylabel('Proportion of laptops')
# plt.legend()
#plt.show()

mse_slr = mean_squared_error(df['Price'],Yhat)
r2_val = model.score(X,y)
print(f"The mean squared error for slr is : {mse_slr :.2f}")
print(f"The r2 score for the slr is : {r2_val :.2f}")

A = df[["CPU_frequency","RAM_GB","Storage_GB_SSD","CPU_core","OS","GPU","Category"]]
B = df["Price"]

model2 = LinearRegression()
model2.fit(A,B)

Y_hat = model2.predict(A)

plt.figure(figsize=(10,6))

sns.kdeplot(B,color="blue",label="Actual Values",fill=True,alpha=0.3)

sns.kdeplot(Y_hat,color="red",label="Predicted Values",fill=True,alpha=0.3)
#
# plt.title('Actual vs Fitted Values for Price')
# plt.xlabel('Price')
# plt.ylabel('Proportion of laptops')
#
# plt.legend()
# plt.show()

mse_mlr = mean_squared_error(df["Price"],Y_hat)
r2_val_mlr = model2.score(A,B)

print(f"The mse for mlr is: {mse_mlr}")
print(f"The r2 socre for mlr is: {r2_val_mlr}")

X = X.to_numpy().flatten()
f1 = np.polyfit(X, y, 1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X, y, 3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X, y, 5)
p5 = np.poly1d(f5)

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')
    plt.legend()
    plt.show()

PlotPolly(p1, X, y, 'CPU_frequency')

PlotPolly(p3, X, y, 'CPU_frequency')

PlotPolly(p5, X, y, 'CPU_frequency')


r_squared_1 = r2_score(y, p1(X))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(y,p1(X)))
r_squared_3 = r2_score(y, p3(X))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(y,p3(X)))
r_squared_5 = r2_score(y, p5(X))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(y,p5(X)))

Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model',LinearRegression())]
pipe=Pipeline(Input)
A = A.astype(float)
pipe.fit(A,y)
ypipe=pipe.predict(A)

print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(y, ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(y, ypipe))