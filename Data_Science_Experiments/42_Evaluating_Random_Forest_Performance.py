import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error,root_mean_squared_error,mean_absolute_error,r2_score


data = fetch_california_housing()
X = data.data
y = data.target

# print(data.DESCR)

X_train ,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

eda = pd.DataFrame(data=X_train)
eda.columns = data.feature_names
eda['MedHouseVal'] = y_train

print(eda.describe())

rf_regressor = RandomForestRegressor(n_estimators=100,random_state=42)

rf_regressor.fit(X_train,y_train)

y_pred_test = rf_regressor.predict(X_test)

mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = root_mean_squared_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

