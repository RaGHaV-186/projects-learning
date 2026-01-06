import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
real_estate_data = {
    'Property_ID': ['101', '102', '103', '104', '105', '106', '107'],
    'Neighborhood': ['Downtown', 'Suburb', 'Downtown', 'Rural', 'Suburb', 'Downtown', 'Rural'],
    'Price': ['$850,000', '$450,000', '$920,000', '$250,000', '$520,000', '$780,000', '$300,000'],
    'Size': ['1200 sqft', '2500 sqft', '1100 sqft', '3500 sqft', '2100 sqft', '1300 sqft', '3000 sqft'],
    'Year_Built': [1920, 2005, 1998, 2015, 1985, 1940, 2021]
}

df = pd.DataFrame(real_estate_data)

print(df.head())
#data formatting handling
df["Price"] = df["Price"].str.replace("$","").str.replace(",","").astype(float)
df["Size"] = df["Size"].str.replace(" sqft","").astype(int)

print(df.head())

#Normalization
max_price = df["Price"].max()
min_price = df["Price"].min()

df["Price_Normalization"] = (df["Price"]-min_price)/(max_price-min_price)

print(df[["Property_ID","Price","Price_Normalization"]].head())

#Binning

bins = [1900,1980,2010,2025]

group_names = ["Historic","Standard","Modern"]

df["House_Category"] = pd.cut(df["Year_Built"],bins,labels=group_names)

print(df.head())

#Convert 'Neighborhood' (Text) into math (0s and 1s)

dummy_variable = pd.get_dummies(df["Neighborhood"])

#print(dummy_variable)

dummy_variable.rename(columns=lambda x:f"Neighbourhood_{x}",inplace=True)

df = pd.concat([df,dummy_variable],axis=1)
df.drop("Neighborhood",axis=1,inplace=True)

print(df)