import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
from  sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')

file = "Credit Card Data.csv"

raw_data = pd.read_csv(file)

print(raw_data.head())
#
# print(raw_data.info())

labels = raw_data['Class'].unique()

print(labels)

sizes = raw_data['Class'].value_counts().values

print(sizes)

# fig, ax = plt.subplots()
# ax.pie(sizes, labels=labels, autopct='%1.3f%%')
# ax.set_title('Target Variable Value Counts')
# plt.show()
#
# correlation_values = raw_data.corr()['Class'].drop('Class')
# correlation_values.plot(kind='barh', figsize=(10, 6))
#
# plt.show()

print(raw_data.iloc[:,1:30])

raw_data.iloc[:,1:30] = StandardScaler().fit_transform(raw_data.iloc[:,1:30])

print(raw_data.head())

data_matrix = raw_data.values

X = data_matrix[:,1:30]

y = data_matrix[:,30]

X = normalize(X,norm='l1')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

w_train = compute_sample_weight('balanced', y_train)

dt = DecisionTreeClassifier(max_depth=4, random_state=35)

dt.fit(X_train, y_train, sample_weight=w_train)

svm = LinearSVC(class_weight='balanced', random_state=31, loss="hinge", fit_intercept=False)

svm.fit(X_train, y_train)

y_pred_dt = dt.predict_proba(X_test)[:,1]

roc_auc_dt = roc_auc_score(y_test, y_pred_dt)
print('Decision Tree ROC-AUC score : {0:.3f}'.format(roc_auc_dt))

y_pred_svm = svm.decision_function(X_test)

roc_auc_svm = roc_auc_score(y_test, y_pred_svm)
print("SVM ROC-AUC score: {0:.3f}".format(roc_auc_svm))

