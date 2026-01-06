import pandas as pd

data = {
    'Customer_ID': [1, 2, 3, 4, 5],
    'Review': [
        "Great coffee but bitter taste",
        "Amazing service and great price",
        "Too bitter for me",
        "It was okay, not great",
        "Bitter coffee, bad service"
    ]
}
df = pd.DataFrame(data)
print(df)

#word finding with maps

df['Mentioned_Bitter'] = df['Review'].map(lambda desc: "bitter" in desc.lower())

print(df[['Review', 'Mentioned_Bitter']])

#counting

complaint_counts = df["Mentioned_Bitter"].value_counts()

print(complaint_counts)