import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split

filepath = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/medical_insurance_dataset.csv'
df = pd.read_csv(filepath, header=None)

# print(df.head(10))

headers = ['age','gender','bmi','no_of_children','smoker','region','charges']

df.columns = headers

df.replace('?',np.nan,inplace=True)

#print(df.head(10))

#print(df.info())

is_smoker = df['smoker'].value_counts().idxmax()

df['smoker'] = df['smoker'].replace(np.nan,is_smoker)

mean_age = df['age'].astype('float').mean(axis = 0)

df['age'] = df['age'].replace(np.nan,mean_age)

df['charges'] = np.round(df['charges'])

# print(df.head(10))

plt.figure(figsize=(10,6))

# sns.regplot(x='bmi',y='charges',data=df,line_kws={'color':'red'})
#
# plt.show()
#
# sns.boxplot(x='smoker',y='charges',data=df)
#
# plt.show()

#print(df.corr())

x = df[['smoker']]

y = df['charges']

lre = LinearRegression()

lre.fit(x,y)

print(lre.score(x,y))

z = df[['age','gender','bmi','no_of_children','smoker','region']]

lre2 = LinearRegression()

lre2.fit(z,y)

print(lre2.score(z,y))

pipe = Pipeline([
    ('scale',StandardScaler()),
    ('polynomial',PolynomialFeatures()),
    ('model',LinearRegression())
])

z = z.astype('float')

pipe.fit(z,y)
ypipe = pipe.predict(z)

print(r2_score(y,ypipe))

x_train,x_test,y_train,y_test = train_test_split(z,y,test_size=0.2,random_state=42)

RidgeModel = Ridge(alpha=0.1)

RidgeModel.fit(x_train,y_train)

yhat = RidgeModel.predict(x_test)

print(r2_score(y_test,yhat))

pr = PolynomialFeatures(degree=2)

x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)

RidgeModel.fit(x_train_pr,y_train)

y_hat = RidgeModel.predict(x_test_pr)

print(r2_score(y_test,y_hat))