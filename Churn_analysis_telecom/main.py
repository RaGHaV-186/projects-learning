import random
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns

conn = sqlite3.connect("Churn_Analysis.db")

num_customers = 500

customer_ids = list(range(1001, 1001 + num_customers))

data_customer = {
    "customer_id": customer_ids,
    "gender":[random.choice(["Male","Female"]) for _ in range(num_customers)],
    "age":[random.randint(18,70) for _ in range(num_customers)],
    'city': [random.choice(["Hyderabad","Bengaluru","Mumbai","Delhi"]) for _ in range(num_customers)]
}

data_subscriptions = {
    'customer_id': customer_ids,
    'contract_type': [random.choice(['Month-to-Month', 'One Year', 'Two Year']) for _ in range(num_customers)],
    'monthly_charges': [round(random.uniform(30, 120), 2) for _ in range(num_customers)],
    'churn_status': [random.choices([0, 1], weights=[0.7, 0.3])[0] for _ in range(num_customers)]
}

df_cust = pd.DataFrame(data_customer)
df_sub = pd.DataFrame(data_subscriptions)

df_cust.to_sql('customers',conn,if_exists='replace',index=False)
df_sub.to_sql('subscriptions',conn,if_exists='replace',index=False)

query = """
SELECT 
    churn_status,
    COUNT(customer_id) as count,
    ROUND(AVG(monthly_charges), 2) as avg_price
FROM subscriptions
GROUP BY churn_status
"""

df_result = pd.read_sql_query(query,conn)
print("--- Churn Status and Average Price ---")
print(df_result)

query_city = """
SELECT 
    c.city,
    COUNT(s.customer_id) as total_customers,
    SUM(s.churn_status) as churned_customers,
    ROUND(AVG(s.churn_status) * 100, 2) as churn_rate_percent
FROM customers c
JOIN subscriptions s ON c.customer_id = s.customer_id
GROUP BY c.city
ORDER BY churn_rate_percent DESC
"""

df_city = pd.read_sql_query(query_city, conn)

print("\n--- Churn Rate by City ---")
print(df_city)

query_contract = """
SELECT 
    contract_type,
    COUNT(customer_id) as total_customers,
    SUM(churn_status) as churned_customers,
    ROUND(AVG(churn_status) * 100, 2) as churn_rate_percent
FROM subscriptions
GROUP BY contract_type
ORDER BY churn_rate_percent DESC
"""

df_contract = pd.read_sql_query(query_contract, conn)

print("\n--- Churn Rate by Contract ---")
print(df_contract)

plt.figure(figsize=(8, 5))

sns.barplot(data=df_contract, x='contract_type', y='churn_rate_percent', palette='Reds_d')

plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate (%)')
plt.xlabel('Contract Type')
plt.show()

conn.close()


