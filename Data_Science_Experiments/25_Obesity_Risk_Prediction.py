import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score

import warnings
warnings.filterwarnings('ignore')

file_path = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/GkDzb7bWrtvGXdPOfk6CIg/Obesity-level-prediction-dataset.csv"
data = pd.read_csv(file_path)
# print(data.head())

# sns.countplot(y='NObeyesdad',data=data)
# plt.title('Distribution of Obesity levels')
# plt.show()

print(data.info())

print(data.describe())

continuous_columns = data.select_dtypes(include=['float64']).columns.to_list()

scaler = StandardScaler()

scaled_features = scaler.fit_transform((data[continuous_columns]))

scaled_df = pd.DataFrame(scaled_features, columns=scaler.get_feature_names_out(continuous_columns))

scaled_data = pd.concat([data.drop(columns=continuous_columns),scaled_df],axis=1)

print(scaled_data.head())

categorical_columns = scaled_data.select_dtypes(include=['object']).columns.to_list()

categorical_columns.remove('NObeyesdad')

print(categorical_columns)

encoder = OneHotEncoder(sparse_output=False,drop='first')

encoded_features = encoder.fit_transform(scaled_data[categorical_columns])

encoded_df = pd.DataFrame(encoded_features,columns=encoder.get_feature_names_out(categorical_columns))

prepped_data = pd.concat([scaled_data.drop(columns=categorical_columns),encoded_df],axis=1)

print(prepped_data.head())

prepped_data['NObeyesdad'] = prepped_data['NObeyesdad'].astype('category').cat.codes

print(prepped_data.head())

X = prepped_data.drop('NObeyesdad',axis=1)
y = prepped_data['NObeyesdad']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model_ova = OneVsRestClassifier(LogisticRegression(max_iter=1000))

model_ova.fit(X_train,y_train)

y_pred_ova = model_ova.predict(X_test)

print("One-vs-All (OvA) Strategy")

print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ova),2)}%")

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))

model_ovo.fit(X_train,y_train)

y_pred_ovo = model_ovo.predict(X_test)

print("One-vs-One (OvO) Strategy")
print(f"Accuracy: {np.round(100*accuracy_score(y_test, y_pred_ovo),2)}%")

for test_size in [0.1,0.3]:
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    model_ova.fit(X_train,y_train)
    y_pred_ova = model_ova.predict(X_test)
    print("One-vs-All (OvA) Strategy")
    print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ova), 2)}%")
    model_ovo.fit(X_train,y_train)
    y_pred_ovo = model_ovo.predict(X_test)
    print("One-vs-One (OvO) Strategy")
    print(f"Accuracy: {np.round(100 * accuracy_score(y_test, y_pred_ovo), 2)}%")