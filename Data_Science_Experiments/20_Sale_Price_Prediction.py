import pandas as pd
from sklearn.metrics import mean_absolute_error

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import  matplotlib.pyplot as plt
import seaborn as sns

url = "http://jse.amstat.org/v19n3/decock/AmesHousing.txt"
df = pd.read_csv(url, sep='\t')

print(df.head())

features = ['Overall Qual', 'Year Built', 'Year Remod/Add', 'Total Bsmt SF',
            '1st Flr SF', 'Gr Liv Area', 'Full Bath', 'TotRms AbvGrd',
            'Garage Cars', 'Garage Area']

target = 'SalePrice'

df_subset = df[features + [target]]
print("Missing values before cleaning:\n", df_subset.isnull().sum())

df_clean = df_subset.dropna()

print(f"\nOriginal Size: {df.shape}")
print(f"Clean Size:    {df_clean.shape}")

x = df_clean[features]
y = df_clean[target]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', RidgeCV(alphas=[0.1, 1.0, 10.0]))
])

pipeline.fit(x_train,y_train)

best_alpha = pipeline.named_steps['ridge'].alpha_
test_score = pipeline.score(x_test,y_test)
y_pred = pipeline.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)

print(f"Best Alpha Chosen: {best_alpha}")
print(f"Test R² Score:     {test_score:.4f} (Accuracy)")
print(f"Mean Error (MAE):  ${mae:,.0f} (Average miss in dollars)")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, edgecolor=None)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2) # The Perfect Line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Truth Plot: How close are we?")
plt.show()