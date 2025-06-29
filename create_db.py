import sqlite3
import pandas as pd

df = pd.read_csv('telco.csv')
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

conn = sqlite3.connect('data/customers.db')
df.to_sql('customers', conn, if_exists='replace', index=False)
conn.close()
