import pandas as pd

pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

data = {
    'Product': ['Apples', 'Bread', 'Chicken', 'Bananas', 'Milk', 'Beef', 'Eggs', 'Spinach'],
    'Department': ['Produce', 'Bakery', 'Meat', 'Produce', 'Dairy', 'Meat', 'Dairy', 'Produce'],
    'Price': [1.20, 2.50, 8.00, 0.80, 3.00, 12.00, 4.00, 1.50],
    'Quantity_Sold': [100, 50, 30, 120, 60, 20, 40, 80]
}

df = pd.DataFrame(data)

df["Total_Revenue"] = df['Price'] * df["Quantity_Sold"]

print(df.head())

dept_sales = df.groupby("Department")["Total_Revenue"].sum()

print(dept_sales)

dept_sales_sorted = dept_sales.sort_values(ascending=False)

print(dept_sales_sorted)

