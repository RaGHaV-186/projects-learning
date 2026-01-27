import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split
from  sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

import warnings
warnings.filterwarnings('ignore')

url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/pu9kbeSaAtRZ7RxdJKX9_A/yellow-tripdata.csv'
raw_data = pd.read_csv(url)

print(raw_data.head())

#correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
# correlation_values.plot(kind='barh', figsize=(10, 6))
#
# plt.show()

y = raw_data[['tip_amount']].values.astype('float32')

proc_data = raw_data.drop(['tip_amount'], axis=1)

X = proc_data.values

X = normalize(X, axis=1, norm='l1', copy=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_reg = DecisionTreeRegressor(criterion="squared_error",max_depth=8,random_state=42)

dt_reg.fit(X_train,y_train)

y_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)

r2_score = dt_reg.score(X_test,y_test)

print('MSE score : {0:.3f}'.format(mse_score))

print('R^2 score : {0:.3f}'.format(r2_score))

correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
print(abs(correlation_values).sort_values(ascending=False)[:3])

raw_data = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)

y = raw_data[['tip_amount']].values.astype('float32')
proc_data = raw_data.drop(['tip_amount'], axis=1)
X = proc_data.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_reg = DecisionTreeRegressor(criterion="squared_error", max_depth=8, random_state=42)
dt_reg.fit(X_train, y_train)

y_pred = dt_reg.predict(X_test)

mse_score = mean_squared_error(y_test, y_pred)
r2_score = dt_reg.score(X_test, y_test)

print('MSE score : {0:.3f}'.format(mse_score))
print('R^2 score : {0:.3f}'.format(r2_score))