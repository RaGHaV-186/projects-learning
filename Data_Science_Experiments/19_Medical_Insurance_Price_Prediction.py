from statistics import LinearRegression

import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

df = pd.read_csv(url)

# print(df.head())
#
# print(df.dtypes)

df_clean = pd.get_dummies(df,columns=["smoker","sex"],drop_first=True,dtype=int)

df_clean = df_clean.drop("region",axis=1)
print(df_clean.head())

# plt.figure(figsize=(10,6))
#
# sns.scatterplot(x='age',y='charges',hue='smoker',data=df)
# plt.title('Medical Charges by Age and Smoker Status')
# plt.show()

x = df_clean[['age','bmi','children','smoker_yes','sex_male']]

y = df_clean['charges']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

lr = LinearRegression()

lr.fit(x_train,y_train)

train_score = lr.score(x_train,y_train)
test_score = lr.score(x_test,y_test)

print(f"the training score is : {train_score}")
print(f"the test score is : {test_score}")


ridge = Pipeline([
    ('scaler',StandardScaler()),
    ('poly',PolynomialFeatures(degree=2)),
    ('ridge',Ridge())
])

parameters = {"ridge__alpha":[0.1, 1, 10, 100]}

grid = GridSearchCV(ridge,parameters,cv=5)
grid.fit(x_train,y_train)

best_model = grid.best_estimator_
best_alpha = grid.best_params_['ridge__alpha']

print(f"Best Alpha Found: {best_alpha}")
print(f"Training R²:      {best_model.score(x_train, y_train):.4f}")
print(f"Testing R²:       {best_model.score(x_test, y_test):.4f}")