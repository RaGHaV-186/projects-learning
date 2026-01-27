import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import plot_tree


url = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

df = pd.read_csv(url)

print(df.head())

plt.figure(figsize=(10,6))
sns.histplot(df['charges'],kde=True)
plt.show()

sns.boxplot(x='smoker',y='charges',data=df)
plt.show()

sns.scatterplot(x='bmi',y='charges',data=df)
plt.show()

sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.show()


le = LabelEncoder()

df['sex'] = le.fit_transform(df['sex'])

df['smoker'] = le.fit_transform(df['smoker'])

print(df[['sex', 'smoker']].head())

encoder = OneHotEncoder(sparse_output=False,drop='first')

region_encoded = encoder.fit_transform(df[['region']])

region_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['region']))

df = pd.concat([df.drop('region', axis=1), region_df], axis=1)

print(df.head())

X = df.drop('charges',axis=1)
y = df['charges']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

cost_tree = DecisionTreeRegressor(criterion='squared_error',max_depth=4,random_state=42)

cost_tree.fit(X_train,y_train)

cost_tree_pred = cost_tree.predict(X_test)

mae = mean_absolute_error(y_test, cost_tree_pred)
print(f"Mean Absolute Error: ${mae:.2f}")

r2 = r2_score(y_test, cost_tree_pred)
print(f"R2 Score: {r2:.4f}")

plt.figure(figsize=(20,10))

plot_tree(cost_tree,
          feature_names=X.columns,
          filled=True,
          rounded=True,
          fontsize=10)

plt.title("Insurance Cost Regression Tree (Depth 4)")
plt.show()
