import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import make_regression

def regression_results(y_true,y_pred,regr_type):
    ev = explained_variance_score(y_true,y_pred)
    mae = mean_absolute_error(y_true,y_pred)
    mse = mean_squared_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)

    print('Evaluation metrics for ' + regr_type + ' Linear Regression')
    print('explained_variance: ', round(ev, 4))
    print('r2: ', round(r2, 4))
    print('MAE: ', round(mae, 4))
    print('MSE: ', round(mse, 4))
    print('RMSE: ', round(np.sqrt(mse), 4))
    print()

noise = 1
np.random.seed(42)
X = 2 * np.random.rand(1000,1)

y = 4 + 3 * X + noise * np.random.randn(1000,1)

y_ideal = 4 + 3 * X

y_outlier  = pd.Series(y.reshape(-1).copy())

threshold = 1.5
outlier_indices = np.where(X.flatten()>threshold)[0]

num_outliers = 5
selected_indices = np.random.choice(outlier_indices,num_outliers,replace=False)

y_outlier[selected_indices] += np.random.uniform(50, 100, num_outliers)

# plt.figure(figsize=(12, 6))
#
#
# plt.scatter(X, y_outlier, alpha=0.4,ec='k', label='Original Data with Outliers')
# plt.plot(X, y_ideal,  linewidth=3, color='g',label='Ideal, noise free data')
#
# plt.xlabel('Feature (X)')
# plt.ylabel('Target (y)')
# plt.title('')
# plt.legend()
# plt.show()
#
# plt.figure(figsize=(12, 6))
#
# plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data without Outliers')
# plt.plot(X, y_ideal,  linewidth=4, color='g',label='Ideal, noise free data')
#
# plt.xlabel('Feature (X)')
# plt.ylabel('Target (y)')
# plt.title('')
# plt.legend()
# plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X,y_outlier)
y_outlier_pred_lin = lin_reg.predict(X)

ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X,y_outlier)
y_outlier_pred_ridge = ridge_reg.predict(X)

lasso_reg = Lasso(alpha=.2)
lasso_reg.fit(X, y_outlier)
y_outlier_pred_lasso = lasso_reg.predict(X)

regression_results(y, y_outlier_pred_lin, 'Ordinary')
regression_results(y, y_outlier_pred_ridge, 'Ridge')
regression_results(y, y_outlier_pred_lasso, 'Lasso')

plt.figure(figsize=(12, 6))

# Scatter plot of the original data with outliers
plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data')

# Plot the ideal regression line (noise free data)
plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')

# Plot predictions from the simple linear regression model
plt.plot(X, y_outlier_pred_lin,  linewidth=5, label='Linear Regression')

# Plot predictions from the ridge regression model
plt.plot(X, y_outlier_pred_ridge, linestyle='--', linewidth=2, label='Ridge Regression')

# Plot predictions from the lasso regression model
plt.plot(X, y_outlier_pred_lasso,  linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of Predictions with Outliers')
plt.legend()
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)


ridge_reg = Ridge(alpha=1)
ridge_reg.fit(X, y)
y_pred_ridge = ridge_reg.predict(X)


lasso_reg = Lasso(alpha=0.2)
lasso_reg.fit(X, y)
y_pred_lasso = lasso_reg.predict(X)

regression_results(y, y_pred_lin, 'Ordinary')
regression_results(y, y_pred_ridge, 'Ridge')
regression_results(y, y_pred_lasso, 'Lasso')

plt.figure(figsize=(12, 8))

plt.scatter(X, y, alpha=0.4,ec='k', label='Original Data')


plt.plot(X, y_ideal,  linewidth=2, color='k',label='Ideal, noise free data')

plt.plot(X, y_pred_lin,  linewidth=5, label='Linear Regression')

plt.plot(X, y_pred_ridge, linestyle='--',linewidth=2, label='Ridge Regression')

plt.plot(X, y_pred_lasso,  linewidth=2, label='Lasso Regression')

plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Comparison of predictions with no outliers')
plt.legend()
plt.show()

X, y, ideal_coef = make_regression(n_samples=100, n_features=100, n_informative=10, noise=10, random_state=42, coef=True)

ideal_predictions = X @ ideal_coef

X_train, X_test, y_train, y_test, ideal_train, ideal_test = train_test_split(X, y, ideal_predictions, test_size=0.3, random_state=42)

lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=1.0)
linear = LinearRegression()

lasso.fit(X_train, y_train)
ridge.fit(X_train, y_train)
linear.fit(X_train, y_train)

y_pred_linear = linear.predict(X_test)
y_pred_ridge = ridge.predict(X_test)
y_pred_lasso = lasso.predict(X_test)

regression_results(y_test, y_pred_linear, 'Ordinary')
regression_results(y_test, y_pred_ridge, 'Ridge')
regression_results(y_test, y_pred_lasso, 'Lasso')