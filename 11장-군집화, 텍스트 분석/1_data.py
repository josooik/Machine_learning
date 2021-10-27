import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt

print("데이터 전처리 \n")

df = pd.read_excel('./data/retail.xlsx')
print(df.head(3))
print(df.info())

print("\n")

df = df[df['Quantity'] > 0]
df = df[df['UnitPrice'] > 0]
df = df[df['CustomerID'].notnull()]
print(df.shape)
print("\n")

print(df.isnull().sum())

print("\n")

print(df['Country'].value_counts()[:5])

print("\n")

df = df[df['Country'] == 'United Kingdom']
print(df.shape)

print("\n")

print(df['Country'].value_counts()[:5])

print("\n")

df['sale_amount'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)

print(df['CustomerID'].value_counts().head(5))

print("\n")

print(df.groupby('CustomerID')['sale_amount'].sum().sort_values(ascending=False)[:5])

aggs = {
    'InvoiceDate' : 'max',
    'InvoiceNo' : 'count',
    'sale_amount' : 'sum'
}

print("\n")

cust_df = df.groupby('CustomerID').agg(aggs)
cust_df = cust_df.rename(columns={'InvoiceDate' : 'Recency', 'InvoiceNo' : 'Frequency', 'sale_amount' : 'Monetary'})
cust_df = cust_df.reset_index()
print(cust_df.head(3))

print("\n")

cust_df['Recency'] = dt.datetime(2011, 12, 10) - cust_df['Recency']
cust_df['Recency'] = cust_df['Recency'].apply(lambda x: x.days + 1)
print(cust_df.shape)

print("\n")

print(cust_df.head(3))

print("\n")

print("데이터 분석 \n")

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(12, 4), nrows=1, ncols=3)
ax1.set_title('Recency')
ax1.hist(cust_df['Recency'])
ax2.set_title('Frequency')
ax2.hist(cust_df['Frequency'])
ax3.set_title('Monetary')
ax3.hist(cust_df['Monetary'])
plt.show()