import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier , plot_tree
from sklearn.metrics import  accuracy_score ,confusion_matrix,ConfusionMatrixDisplay,classification_report



url = "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"

columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring",
    "stalk-surface-below-ring", "stalk-color-above-ring",
    "stalk-color-below-ring", "veil-type", "veil-color", "ring-number",
    "ring-type", "spore-print-color", "population", "habitat"
]

df = pd.read_csv(url, names=columns)

print(df.head())

print(df['class'].value_counts())

print((df == '?').sum())

df_encoded = df.copy()

le = LabelEncoder()

for col in df_encoded.columns:
    df_encoded[col] = le.fit_transform(df_encoded[col])

print(df_encoded.head())


X = df_encoded.drop('class',axis = 1)
y = df_encoded['class']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

mush_tree = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=42)

mush_tree.fit(X_train,y_train)

mush_tree_pred = mush_tree.predict(X_test)

print("Decision Trees's Accuracy: ", accuracy_score(y_test, mush_tree_pred))


plt.figure(figsize=(20, 10))
plot_tree(mush_tree,
          feature_names=X.columns,
          class_names=['Edible', 'Poisonous'],
          filled=True,
          rounded=True,
          fontsize=12)
plt.title("Mushroom Survival Decision Tree")
plt.show()

importances = pd.Series(mush_tree.feature_importances_, index=X.columns)

importances.sort_values().plot(kind='barh', color='teal', figsize=(10, 8))
plt.title("Which Mushroom Traits Matter Most?")
plt.xlabel("Importance Score")
plt.show()

cm = confusion_matrix(y_test,mush_tree_pred)

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Edible', 'Poisonous'])
disp.plot(cmap='Reds')
plt.title("Mushroom Safety Confusion Matrix")
plt.show()

print(classification_report(y_test, mush_tree_pred, target_names=['Edible', 'Poisonous']))