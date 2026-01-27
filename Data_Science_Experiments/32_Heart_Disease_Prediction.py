import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report

url = "https://raw.githubusercontent.com/amankharwal/Website-data/master/heart.csv"

df = pd.read_csv(url)

print(df.head())

print(df.info())

print((df == '?').sum())

print(df['target'].value_counts())

# plt.figure(figsize=(10,6))
#
# sns.scatterplot(x='age',y='chol',data=df,hue='target')
#
# plt.show()
#
# sns.scatterplot(x='thalach',y='chol',data=df,hue='target')
#
# plt.show()

X = df.drop('target',axis=1)
y = df['target']

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y,test_size=0.2,random_state=42)

svc_model = SVC()

tree_model = DecisionTreeClassifier(criterion='entropy',max_depth=5,random_state=42)

svc_model.fit(X_train,y_train)

tree_model.fit(X_train,y_train)

svc_model_pred = svc_model.predict(X_test)

tree_model_pred = tree_model.predict(X_test)

print("SVC MODEL")

print(f"accuracy score {accuracy_score(svc_model_pred,y_test)}")

print("TREE MODEL")

print(f"accuracy score {accuracy_score(tree_model_pred,y_test)}")

svc_cm = confusion_matrix(y_test, svc_model_pred)
tree_cm = confusion_matrix(y_test, tree_model_pred)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

sns.heatmap(svc_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('SVC Confusion Matrix')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

sns.heatmap(tree_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title('Decision Tree Confusion Matrix')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.show()

print("--- SVC Detailed Report ---")
print(classification_report(y_test, svc_model_pred))

print("\n--- Decision Tree Detailed Report ---")
print(classification_report(y_test, tree_model_pred))




