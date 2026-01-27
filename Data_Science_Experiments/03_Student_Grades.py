import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.width',1000)

data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Score': [65, 50, 75, 40, 90]
}
df = pd.DataFrame(data)

print("--- Original Scores ---")
print(df)

#average score

avg_score = df.Score.mean()
print(avg_score)

#curved grades
curved_amount = 80 - avg_score

df['Curved_Score'] = df.Score.map(lambda p:p+curved_amount)

print(df)

#assigning grades

def get_grades(score):
    if score >= 90: return 'A'
    elif score >= 80: return 'B'
    elif score >= 70: return 'C'
    else : return 'F'

df['Grade'] = df.Curved_Score.apply(get_grades)

print(df)