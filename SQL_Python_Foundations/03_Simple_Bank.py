import sqlite3
import pandas as pd

conn = sqlite3.connect("Simple_Bank.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS ACCOUNTS")
cursor.execute("DROP TABLE IF EXISTS TRANSACTION_LOG")
cursor.execute("DROP VIEW IF EXISTS VIP_CUSTOMERS")

cursor.execute("""
CREATE TABLE ACCOUNTS (
    Account_ID INTEGER PRIMARY KEY,
    Name TEXT,
    Balance INTEGER
)
""")

cursor.execute("""
CREATE TABLE TRANSACTION_LOG (
    Log_ID INTEGER PRIMARY KEY AUTOINCREMENT,
    Message TEXT
)
""")

accounts_data = [
    (1, 'Alice', 1000),
    (2, 'Bob', 1000),
    (3, 'Charlie', 1000)
]
cursor.executemany("INSERT INTO ACCOUNTS VALUES (?,?,?)", accounts_data)
conn.commit()

print("Bank Reset and Open!")

query_view = """
CREATE VIEW VIP_CUSTOMERS AS 
SELECT Name, Balance FROM ACCOUNTS 
WHERE Balance >= 1000
"""
cursor.execute(query_view)

# --- CHECK RESULTS ---
print(pd.read_sql_query("SELECT * FROM VIP_CUSTOMERS", conn))

print("\n--- 💸 STARTING TRANSACTION: Alice -> Bob ($100) ---")

try:
    cursor.execute("UPDATE ACCOUNTS SET Balance = Balance - 100 WHERE Name = 'Alice'")

    cursor.execute("UPDATE ACCOUNTS SET Balance = Balance + 100 WHERE Name = 'Bob'")

    conn.commit()
    print("Transaction Successful: Committed to DB.")

except Exception as e:
    conn.rollback()
    print(f" Error: {e}")
    print("Changes Rolled Back.")

print(pd.read_sql_query("SELECT * FROM ACCOUNTS", conn))

print("\n--- STARTING BROKEN TRANSACTION: Alice -> Bob ($100) ---")

try:
    cursor.execute("UPDATE ACCOUNTS SET Balance = Balance - 100 WHERE Name = 'Alice'")
    print("Step 1 Complete: Taken $100 from Alice...")
    raise Exception("POWER FAILURE!")

    cursor.execute("UPDATE ACCOUNTS SET Balance = Balance + 100 WHERE Name = 'Bob'")



except Exception as e:

    print(f"Error Occurred: {e}")
    conn.rollback()  # <--- THE MAGIC BUTTON
    print("Rolling back changes... No money lost.")

print(pd.read_sql_query("SELECT * FROM ACCOUNTS", conn))