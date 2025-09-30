import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

n_customers = 1000
n_transactions = 5000

# Create customers
customer_ids = [f'C{1000+i}' for i in range(n_customers)]
ages = np.random.choice(range(18, 70), size=n_customers)
regions = np.random.choice(['Riyadh','Jeddah','Dammam','Khobar','Other'], size=n_customers, p=[0.35,0.25,0.15,0.1,0.15])
loyalty = np.random.choice(['Bronze','Silver','Gold','Platinum'], size=n_customers, p=[0.5,0.3,0.15,0.05])

customers = pd.DataFrame({
    'customer_id': customer_ids,
    'age': ages,
    'region': regions,
    'loyalty': loyalty
})

# Create sales transactions
start_date = datetime(2024,1,1)
transactions = []
for _ in range(n_transactions):
    cust = random.choice(customer_ids)
    date = start_date + timedelta(days=np.random.poisson(60)) + timedelta(days=random.randint(0,365))
    product_cat = np.random.choice(['Electronics','Clothing','Home','Grocery','Beauty'], p=[0.25,0.2,0.2,0.25,0.1])
    qty = np.random.poisson(2) + 1
    price = {
        'Electronics': np.random.normal(300,80),
        'Clothing': np.random.normal(50,20),
        'Home': np.random.normal(120,40),
        'Grocery': np.random.normal(20,8),
        'Beauty': np.random.normal(35,12)
    }[product_cat]
    
    # Apply loyalty discount
    cust_loyalty = customers.loc[customers.customer_id==cust,'loyalty'].values[0]
    if cust_loyalty=='Gold': price *= 0.95
    if cust_loyalty=='Platinum': price *= 0.9
    total = max(1, round(price * qty,2))
    transactions.append([cust, date.strftime('%Y-%m-%d'), product_cat, qty, total])

sales = pd.DataFrame(transactions, columns=['customer_id','date','category','quantity','total_amount'])

# Merge customer data
sales = sales.merge(customers, on='customer_id', how='left')

# Save to CSV
import os
os.makedirs('data', exist_ok=True)
sales.to_csv('data/sample_sales.csv', index=False)
customers.to_csv('data/customers.csv', index=False)
print('Saved data/sample_sales.csv and data/customers.csv')
