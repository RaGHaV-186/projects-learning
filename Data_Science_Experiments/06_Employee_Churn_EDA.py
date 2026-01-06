import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns",None)
pd.set_option("display.width",1000)

data = {
    'Employee_ID': [101, 102, 103, 104, 105, 106, 107, 108],
    'Department': ['Sales', 'IT', 'HR', 'Sales', 'IT', 'Sales', 'HR', 'IT'],
    'Years_At_Company': [2, 8, 3, 1, 5, 2, 4, 7],
    'Salary': [45000, 85000, 50000, 42000, 78000, 46000, 52000, 82000],
    'Left_Company': ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No']
}

df = pd.DataFrame(data)
#print(df.head())

print(df.groupby('Left_Company')[['Years_At_Company','Salary']].mean())

plt.figure(figsize=(8,5))
sns.boxplot(x="Left_Company",y="Salary",data = df)
#plt.show()


dept_churn = df[df["Left_Company"]=="Yes"]["Department"].value_counts()
print(dept_churn)