import sqlite3
import pandas as pd

conn = sqlite3.connect("Hotel_Lost_Found.db")
cursor = conn.cursor()

guests_data = {
    'Guest_ID': [1, 2, 3, 4, 5],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Room_Number': [101, 102, 103, 104, 105]
}
df_guests = pd.DataFrame(guests_data)
df_guests.to_sql('GUESTS', conn, if_exists='replace', index=False)

items_data = {
    'Item_ID': [10, 11, 12, 13],
    'Item_Name': ['Wallet', 'Phone', 'Watch', 'Umbrella'],
    'Owner_ID': [1, 2, 1, 99]
}
df_items = pd.DataFrame(items_data)
df_items.to_sql('LOST_ITEMS', conn, if_exists='replace', index=False)

print("Hotel Database Built: Guests & Lost Items created.")

query1 = """SELECT g.Name,li.Item_Name FROM GUESTS AS g INNER JOIN  LOST_ITEMS AS li ON g.Guest_ID = li.Owner_ID"""

df_query1 = pd.read_sql_query(query1,conn)

print(df_query1)

query_left = """SELECT g.Name,li.Item_Name FROM GUESTS AS g LEFT JOIN LOST_ITEMS AS li ON g.Guest_ID = li.Owner_ID"""

df_query_left = pd.read_sql_query(query_left,conn)

print(df_query_left)


#swapped tables since sqlite does not support right join
query_right = """
SELECT g.Name, li.Item_Name 
FROM LOST_ITEMS AS li 
LEFT JOIN GUESTS AS g 
ON li.Owner_ID = g.Guest_ID
"""
df_query_right = pd.read_sql_query(query_right,conn)

print(df_query_right)


query_full = """
SELECT g.Name, li.Item_Name 
FROM GUESTS AS g 
LEFT JOIN LOST_ITEMS AS li 
ON g.Guest_ID = li.Owner_ID

UNION

SELECT g.Name, li.Item_Name 
FROM LOST_ITEMS AS li 
LEFT JOIN GUESTS AS g 
ON li.Owner_ID = g.Guest_ID
"""

df_query_full = pd.read_sql_query(query_full, conn)
print(df_query_full)