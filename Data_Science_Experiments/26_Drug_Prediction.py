import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
my_data = pd.read_csv(path)
print(my_data.head())
print(my_data.info())

label_encoder = LabelEncoder()
my_data['Sex'] = label_encoder.fit_transform(my_data['Sex'])
my_data['BP'] = label_encoder.fit_transform(my_data['BP'])
my_data['Cholesterol'] = label_encoder.fit_transform(my_data['Cholesterol'])

print(my_data.head())

print(my_data.isnull().sum())

custom_map = {'drugA':0,'drugB':1,'drugC':2,'drugX':3,'drugY':4}

my_data['Drug_num'] = my_data['Drug'].map(custom_map)

print(my_data.head())

print(my_data.drop('Drug',axis=1).corr()['Drug_num'])

X = my_data.drop(['Drug','Drug_num'],axis=1)

y = my_data['Drug']

X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size=0.3,random_state=42)

drugTree = DecisionTreeClassifier(criterion='entropy',max_depth=4)

drugTree.fit(X_train,y_train)

tree_predictions = drugTree.predict(X_test)

print("Decision Trees's Accuracy: ", metrics.accuracy_score(y_test, tree_predictions))

plot_tree(drugTree)
plt.show()
