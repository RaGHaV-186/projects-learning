import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/srinivasav22/Graduate-Admission-Prediction/master/Admission_Predict_Ver1.1.csv"
df = pd.read_csv(url)

print(df.head())

df.columns = df.columns.str.strip()

df = df.drop('Serial No.',axis=1)

print(df.head())

X = df.drop('Chance of Admit',axis=1)
y = df['Chance of Admit']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

rf_model = RandomForestRegressor(n_estimators=100,random_state=42)
xgb_model = XGBRegressor(n_estimators=100,random_state=42)

rf_model.fit(X_train,y_train)
xgb_model.fit(X_train,y_train)

rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

r2_score_rf = r2_score(y_test,rf_pred)
r2_score_xgb = r2_score(y_test,xgb_pred)

print(r2_score_rf)
print(r2_score_xgb)

if r2_score_xgb > r2_score_rf:
    print("XGB winner")
else:
    print("RF winner")

plt.figure(figsize=(10, 6))

importances = rf_model.feature_importances_
features = X.columns

df_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
df_imp = df_imp.sort_values(by='Importance', ascending=False)

print(df_imp.head())

sns.barplot(x='Importance', y='Feature', data=df_imp)
plt.title('What Matters Most for Grad Admission?')
plt.xlabel('Importance Score')
plt.show()







