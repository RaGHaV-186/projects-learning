import pandas as pd

data = {
    'Student': ['Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie', 'Alice', 'Bob', 'Charlie'],
    'Subject': ['Math', 'Math', 'Math', 'Science', 'Science', 'Science', 'History', 'History', 'History'],
    'Score': [85, 70, 95, 90, 60, 80, 75, 85, 90]
}

df = pd.DataFrame(data)

print("--- Student Scores ---")
print(df)

#subject difficulty

subject_difficulty = df.groupby("Subject")["Score"].agg(["mean","min","max"])

print(subject_difficulty)