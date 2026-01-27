import pandas as pd
from sklearn.metrics import accuracy_score
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

url = "https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv"

df = pd.read_csv(url)

print(df.head())

print(df['Purchased'].value_counts())

# plt.figure(figsize=(10,6))
#
# sns.scatterplot(x='Age',y='EstimatedSalary',data=df,hue='Purchased')
#
# plt.show()

X = df.drop(['Purchased','User ID'],axis=1)
y = df['Purchased']

le = LabelEncoder()

X['Gender'] = le.fit_transform(X['Gender'])

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

# k = 5
#
# knn_classifier = KNeighborsClassifier(n_neighbors=k)
#
# knn_model = knn_classifier.fit(X_train,y_train)
#
# knn_model_pred = knn_model.predict(X_test)
#
# print("KNN Model accuracy")
#
# print(f"Accuracy:{accuracy_score(knn_model_pred,y_test)}")

accuracy_list= []

for k in range(1,40):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    knn_model = knn_classifier.fit(X_train,y_train)

    knn_model_pred = knn_model.predict(X_test)

    acc = accuracy_score(y_test, knn_model_pred)
    accuracy_list.append(acc)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), accuracy_list, color='green', linestyle='dashed',
         marker='o', markerfacecolor='orange', markersize=10)
plt.title('Accuracy Score vs. K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()
