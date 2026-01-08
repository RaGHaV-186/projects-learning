import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

url = "https://raw.githubusercontent.com/Ankit152/Fish-Market/main/Fish.csv"

df = pd.read_csv(url)

#print(df.head())

x = df[['Length1','Length2','Length3','Height','Width']]
y = df['Weight']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print(f"Training Samples:{x_train.shape[0]}")
print(f"Testing Samples:{x_test.shape[0]}")

# print(x_train.head())

bad_model = Pipeline([
    ('scaler',StandardScaler()),
    ('poly',PolynomialFeatures(degree=5)),
    ('linear',LinearRegression())
])

bad_model.fit(x_train,y_train)

train_score = bad_model.score(x_train,y_train)
test_score = bad_model.score(x_test,y_test)

print(f"Training Score: {train_score:.4f} (Perfect!)")
print(f"Testing Score:  {test_score:.4f} (Terrible!)")

#Ridge regression

ridge_pipe = Pipeline([
    ('scaler',StandardScaler()),
    ('poly',PolynomialFeatures(degree=5)),
    ('ridge',Ridge())
])

parameters = {'ridge__alpha':[0.01, 0.1, 1, 5, 10, 50, 100]}

grid = GridSearchCV(ridge_pipe,parameters,cv=5)
grid.fit(x_train,y_train)

best_model = grid.best_estimator_
best_alpha = grid.best_params_['ridge__alpha']

print(f"Best Alpha Found: {best_alpha}")
print(f"Training R²:      {best_model.score(x_train, y_train):.4f}")
print(f"Testing R²:       {best_model.score(x_test, y_test):.4f}")
