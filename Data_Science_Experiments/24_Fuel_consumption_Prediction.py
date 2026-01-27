import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.model_selection import  train_test_split



url= "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/FuelConsumptionCo2.csv"

df = pd.read_csv(url)

print(df.head())
print(df.columns)
df = df.drop(['MODELYEAR', 'MAKE', 'MODEL', 'VEHICLECLASS', 'TRANSMISSION', 'FUELTYPE',],axis=1)

print(df.corr())

df = df.drop(['CYLINDERS', 'FUELCONSUMPTION_CITY', 'FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB',],axis=1)

print(df.head(9))

axes = pd.plotting.scatter_matrix(df, alpha=0.2)

# for ax in axes.flatten():
#     ax.xaxis.label.set_rotation(90)
#     ax.yaxis.label.set_rotation(0)
#     ax.yaxis.label.set_ha('right')
#
# plt.tight_layout()
# plt.gcf().subplots_adjust(wspace=0, hspace=0)
# plt.show()

X = df.iloc[:,[0,1]].to_numpy()
y = df.iloc[:,[2]].to_numpy()

std_scaler = preprocessing.StandardScaler()

X_std = std_scaler.fit_transform(X)

pd.DataFrame(X_std).describe().round(2)

X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.2,random_state=42)

regressor = LinearRegression()

regressor.fit(X_train,y_train)

coef_ = regressor.coef_
intercept_ = regressor.intercept_

print ('Coefficients: ',coef_)
print ('Intercept: ',intercept_)

means_ = std_scaler.mean_
std_devs_ = np.sqrt(std_scaler.var_)

coef_original = coef_ / std_devs_
intercept_original = intercept_ - np.sum((means_ * coef_) / std_devs_)

print ('Coefficients: ', coef_original)
print ('Intercept: ', intercept_original)


X_train_1 = X_train[:,0]
regressor_1 = LinearRegression()
regressor_1.fit(X_train_1.reshape(-1,1),y_train)
coef_1 =  regressor_1.coef_
intercept_1 = regressor_1.intercept_
print ('Coefficients: ',coef_1)
print ('Intercept: ',intercept_1)

# plt.scatter(X_train_1, y_train,  color='blue')
# plt.plot(X_train_1, coef_1[0] * X_train_1 + intercept_1, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.show()

X_test_1 = X_test[:,0]
plt.scatter(X_test_1, y_test,  color='blue')
plt.plot(X_test_1, coef_1[0] * X_test_1 + intercept_1, '-r')
plt.xlabel("Engine size")
plt.ylabel("CO2 Emission")

plt.show()
