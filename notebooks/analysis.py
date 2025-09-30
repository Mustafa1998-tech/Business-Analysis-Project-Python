# analysis.py
# Run with: python notebooks/analysis.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('assets/plots', exist_ok=True)

# 1. Load data
sales = pd.read_csv('data/sample_sales.csv', parse_dates=['date'])
customers = pd.read_csv('data/customers.csv')

# 2. Overview
print('Dataset shape:', sales.shape)
print('\nFirst few rows:')
print(sales.head())

# 3. Convert date to month period
sales['year_month'] = sales['date'].dt.to_period('M')

# 4. Monthly sales analysis
monthly = sales.groupby('year_month').total_amount.sum().reset_index()
monthly['year_month'] = monthly['year_month'].astype(str)

plt.figure(figsize=(10,4))
plt.plot(monthly['year_month'], monthly['total_amount'])
plt.xticks(rotation=45)
plt.title('Total Sales by Month')
plt.tight_layout()
plt.savefig('assets/plots/total_sales_by_month.png')
plt.close()

# 5. Product category analysis
cat = sales.groupby('category').total_amount.sum().sort_values(ascending=False)
print('\nSales by Category:')
print(cat)

plt.figure(figsize=(6,6))
cat.plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.title('Sales Distribution by Category')
plt.savefig('assets/plots/sales_by_category.png')
plt.close()

# 6. RFM Analysis
rfm = sales.groupby('customer_id').agg({
    'date': lambda x: (sales['date'].max() - x.max()).days,
    'customer_id': 'count',
    'total_amount': 'sum'
}).rename(columns={'date':'recency','customer_id':'frequency','total_amount':'monetary'})

# Clean data
rfm['recency'] = rfm['recency'].astype(int)

# 7. Customer Segmentation using KMeans
scaler = StandardScaler()
X = scaler.fit_transform(rfm[['recency','frequency','monetary']].fillna(0))
km = KMeans(n_clusters=4, random_state=42)
rfm['cluster'] = km.fit_predict(X)

print('\nCustomer segments (cluster counts):')
print(rfm['cluster'].value_counts())

# Save results
rfm.to_csv('data/rfm_segments.csv')

# 8. Generate insights report
insights = {
    'top_category': cat.index[0],
    'top_month': monthly.loc[monthly.total_amount.idxmax(),'year_month'],
    'num_customers': customers.shape[0]
}

with open('reports/insights.md','w', encoding='utf-8') as f:
    f.write('# Insights\n\n')
    f.write('## Sales Overview\n')
    f.write(f"- Top selling category: {insights['top_category']}\n")
    f.write(f"- Best performing month: {insights['top_month']}\n")
    f.write(f"- Total customers: {insights['num_customers']}\n\n")
    
    f.write('## Business Recommendations\n')
    f.write('1. Focus marketing efforts on the top performing category\n')
    f.write('2. Analyze what made the top month successful and replicate\n')
    f.write('3. Review customer segments for targeted campaigns\n')
    f.write('4. Consider loyalty program adjustments based on RFM analysis\n')

print('\nAnalysis complete. Results saved to:')
print('- assets/plots/: Visualizations')
print('- data/rfm_segments.csv: Customer segments')
print('- reports/insights.md: Key findings and recommendations')
