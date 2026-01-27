import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/teleCust1000t.csv')
print(df.head())

print(df['custcat'].value_counts())

correlation_matrix = df.corr()

# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
# plt.show()

correlation_values = abs(df.corr()['custcat'].drop('custcat')).sort_values(ascending=False)

print(correlation_values)

X = df.drop('custcat',axis=1)
y = df['custcat']

X_norm = StandardScaler().fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2,random_state=42)

k = 3

knn_classifier = KNeighborsClassifier(n_neighbors=k)
knn_model = knn_classifier.fit(X_train,y_train)

yhat = knn_model.predict(X_test)

print("Test set Accuracy: ", accuracy_score(y_test, yhat))

