import pandas as pd

data = {
    'Date': ['2023-10-01', '2023-10-01', '2023-10-01', '2023-10-02', '2023-10-02', '2023-10-02'],
    'Page': ['Home', 'About', 'Home', 'Product', 'Home', 'Product'],
    'Visitor_Region': ['US', 'UK', 'US', 'US', 'CA', 'UK'],
    'Load_Time_ms': [120, 300, 110, 500, 130, 480]
}

df = pd.DataFrame(data)

print(df)

#Page Popularity

page_popularity = df['Page'].value_counts()

print(page_popularity)

#Reginal Analysis

regional_analysis = df.groupby("Visitor_Region")["Load_Time_ms"].mean()

print(regional_analysis)

#Multi sorting

sorted_traffic = df.sort_values(by=["Page","Load_Time_ms"],ascending=[True,False])

print(sorted_traffic)