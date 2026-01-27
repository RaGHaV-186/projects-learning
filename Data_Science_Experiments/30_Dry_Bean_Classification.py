import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import  StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsOneClassifier,OneVsRestClassifier


filepath = "Dry_Bean_Dataset.xlsx"

df = pd.read_excel(filepath)

print(df.head())

print(df['Class'].value_counts())

print(df.info())

print((df == '?').sum())

X = df.drop('Class',axis=1)

y = df['Class']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train,X_test,y_train,y_test = train_test_split(X_scaled,y_encoded,test_size=0.3,random_state=42)


model_ovr = OneVsRestClassifier(LogisticRegression(max_iter=1000))

model_ovr.fit(X_train,y_train)

model_ovr_pred = model_ovr.predict(X_test)

print("One VS Rest Strategy")

print(f"Accuracy:{accuracy_score(model_ovr_pred,y_test)}")

model_ovo = OneVsOneClassifier(LogisticRegression(max_iter=1000))

model_ovo.fit(X_train,y_train)

model_ovo_pred = model_ovo.predict(X_test)

print("One VS One Strategy")

print(f"Accuracy:{accuracy_score(model_ovo_pred,y_test)}")

cm = confusion_matrix(y_test,model_ovo_pred)