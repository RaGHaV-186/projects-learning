import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

smartphone_data = {
    'Smartphone': ['iPhone 13', 'Samsung S21', 'OnePlus 9', 'Xiaomi Mi 11', 'Pixel 6', 'iPhone SE', 'Samsung A52'],
    'Brand': ['Apple', 'Samsung', 'OnePlus', 'Xiaomi', 'Google', 'Apple', 'Samsung'],
    'Price_USD': ['$799', '$799', '$729', '$749', '$599', '$399', '$349'],
    'RAM': ['4GB', '8GB', '12GB', '8GB', '8GB', '4GB', '6GB'],
    'Battery': [3240, 4000, 4500, 4600, 4614, 2018, 4500]
}

df = pd.DataFrame(smartphone_data)

print(df.head())

df["Price_USD"] = df["Price_USD"].str.replace("$","").astype(float)
df["RAM"] = df["RAM"].str.replace("GB","").astype(int)
print("-------------")
print(df.head())

max_price = df["Price_USD"].max()
min_price = df["Price_USD"].min()

df["Normalized_Price"] = (df['Price_USD']-min_price)/(max_price-min_price)
print("-------------")
print(df[['Smartphone', 'Price_USD', 'Normalized_Price']].head())


bins = np.linspace(min(df["Battery"]),max(df["Battery"]),4)

group_names = ["Small","Medium","Large"]

df["Battery_Binned"] = pd.cut(df["Battery"],bins,labels=group_names,include_lowest=True)

print(df[['Smartphone', 'Battery', 'Battery_Binned']].head())

dummy_variable_1 = pd.get_dummies(df["Brand"])

dummy_variable_1.rename(columns=lambda x: f"Brand_{x}", inplace=True)

df = pd.concat([df, dummy_variable_1], axis=1)

df.drop("Brand", axis=1, inplace=True)

print(df)
