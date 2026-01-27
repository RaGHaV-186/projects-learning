import pandas as pd
from pygments.lexer import include
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

filepath='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-SkillsNetwork/labs/FinalModule_Coursera/data/kc_house_data_NaN.csv'
df = pd.read_csv(filepath, header=0)
df.drop(columns=['Unnamed: 0','id'],axis=1,inplace=True)
#print(df.head())

#print(df.describe())
# print(df['bedrooms'].isnull().sum())
# print(df['bathrooms'].isnull().sum())

mean_bedrooms = df['bedrooms'].mean()
mean_bathrooms = df['bathrooms'].mean()
df['bedrooms'].replace(np.nan,mean_bedrooms,inplace=True)
df['bathrooms'].replace(np.nan,mean_bathrooms,inplace=True)

floor_counts = df['floors'].value_counts().to_frame()

# print(floor_counts)
# print(df.columns)

#plt.figure(figsize=(10,6))
#
# sns.boxplot(x='waterfront',y='price',data=df)
#
# plt.show()
#
# sns.regplot(x='sqft_above',y='price',data=df,line_kws={'color':'red'})
#
# plt.show()
X = df[['long']]
Y = df['price']
lre = LinearRegression()

lre.fit(X,Y)
Yhat = lre.predict(X)
print(r2_score(Y,Yhat))

features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X_multi = df[features]
lre2 = LinearRegression()
lre2.fit(X_multi,Y)

Y_hat = lre2.predict(X_multi)

print(r2_score(Y,Y_hat))

pipe = Pipeline([
    ('scale',StandardScaler()),
    ('polynomial',PolynomialFeatures(include_bias=False)),
    ('model',LinearRegression())
])

pipe.fit(X_multi,Y)

Y_pipe = pipe.predict(X_multi)

print(r2_score(Y,Y_pipe))

x_test,x_train,y_test,y_train = train_test_split(X_multi,Y,test_size=0.15,random_state=42)


RidgeModel = Ridge(alpha=0.1)

RidgeModel.fit(x_train,y_train)

Y_train_pred = RidgeModel.predict(x_train)
Y_test_pred = RidgeModel.predict(x_test)

print(r2_score(y_train,Y_train_pred))
print(r2_score(y_test, Y_test_pred))

pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

Ridge_Model = Ridge(alpha=10000)

Ridge_Model.fit(x_train_pr,y_train)

Y_hat_train_pr = Ridge_Model.predict(x_train_pr)
Y_hat_test_pr = Ridge_Model.predict(x_test_pr)

print(r2_score(y_train,Y_hat_train_pr))
print(r2_score(y_test,Y_hat_test_pr))








