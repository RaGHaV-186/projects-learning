import numpy as np
import  pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV
from sklearn.linear_model import  LinearRegression ,Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from tqdm import tqdm

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv'
df = pd.read_csv(filepath, header=0)
df = df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"], axis=1)
print(df.head())

x_data = df.drop('Price',axis=1)
y_data = df['Price']

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.1,random_state=42)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre = LinearRegression()

lre.fit(x_train[['CPU_frequency']],y_train)
print(lre.score(x_test[['CPU_frequency']],y_test))
print(lre.score(x_train[['CPU_frequency']], y_train))

Rcross = cross_val_score(lre, x_train[['CPU_frequency']], y_train, cv=4)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.5,random_state=42)
print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])

lre = LinearRegression()
Rsqu_test = []
order = [1, 2, 3, 4, 5]

for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lre.fit(x_train_pr, y_train)
    Rsqu_test.append(lre.score(x_test_pr, y_test))

# plt.plot(order, Rsqu_test)
# plt.xlabel('order')
# plt.ylabel('R^2')
# plt.title('R^2 Using Test Data')
# plt.show()

pr = PolynomialFeatures(degree=2)

x_train_pr=pr.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])
x_test_pr=pr.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']])


Rsqu_test = []
Rsqu_train = []
Alpha = np.arange(0.001,1,0.001)
pbar = tqdm(Alpha)

for alpha in pbar:
    RidgeModel = Ridge(alpha=alpha)
    RidgeModel.fit(x_train_pr, y_train)
    test_score, train_score = RidgeModel.score(x_test_pr, y_test), RidgeModel.score(x_train_pr, y_train)
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

# plt.figure(figsize=(10, 6))
# plt.plot(Alpha, Rsqu_test, label='validation data')
# plt.plot(Alpha, Rsqu_train, 'r', label='training Data')
# plt.xlabel('alpha')
# plt.ylabel('R^2')
# plt.ylim(0, 1)
# plt.legend()
# plt.show()

parameters1 = [{'alpha':[0.0001,0.001,0.01, 0.1, 1, 10]}]

RR = Ridge()
Grid1 = GridSearchCV(RR,parameters1,cv=4)
Grid1.fit(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_train)
BestRR=Grid1.best_estimator_
print(BestRR.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']], y_test))