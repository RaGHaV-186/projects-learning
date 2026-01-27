import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

data = {
    'Size_SqFt': [1500, 2500, 1800, 1200, 3000],
    'Type':      ['Apartment', 'House', 'Apartment', 'Studio', 'House'], # <--- TEXT!
    'Price':     [300000, 500000, 350000, 200000, 600000]
}
df = pd.DataFrame(data)

X = df[['Size_SqFt', 'Type']]
y = df['Price']

print(X.head())

column_trans = make_column_transformer((OneHotEncoder(),['Type']),remainder='passthrough')

pipe = make_pipeline(column_trans,LinearRegression())

pipe.fit(X,y)


new_home = pd.DataFrame({
    "Size_SqFt":[2000],
    'Type':["Apartment"]
})

prediction = pipe.predict(new_home)

print(f"Predicted Price for 2,000 sqft Apartment: ${prediction[0]:,.2f}")