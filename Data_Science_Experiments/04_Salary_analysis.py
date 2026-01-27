import pandas as pd

data = {
    'Employee': ['Michael', 'Dwight', 'Jim', 'Pam', 'Stanley'],
    'Salary': [80000, 65000, 68000, 55000, 72000]
}
df = pd.DataFrame(data)

print(df)

#average_salary

avg_salary =  df['Salary'].mean()

print(avg_salary)

df['Difference_From_Mean'] = df["Salary"].map(lambda p:p-avg_salary)

print(df)


#creating a readable string using apply

def generate_report(row):

    name = row['Employee']
    diff = row['Difference_From_Mean']

    if diff > 0:
        status = "above"
    else:
        status = "below"

    return f"{name} is paid ${abs(diff)} {status} average"

df['Report'] = df.apply(generate_report,axis=1)

print(df['Report'])