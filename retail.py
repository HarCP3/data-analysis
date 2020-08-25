import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import warnings
import seaborn as sns
import plotly as py
import plotly.graph_objs as go
import plotly.offline as py
py.offline.init_notebook_mode()
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns',None)
file_path = './data.csv'
df = pd.read_csv(file_path)
start=time.clock()
#数据清洗


df.apply(lambda x:sum(x.isnull())/len(x),axis=0)
df1 = df.dropna(how='any').copy()
#print(df1.head())
df1['InvoiceDate'] = pd.to_datetime(df1['InvoiceDate'], errors='coerce')
df1['InvoiceDate'] = df1['InvoiceDate'].dt.date
#print(df1.head())
#print(df1.info())
df1['Price'] = df1.apply(lambda x: x[3]*x[5],axis=1)
print(df1.head())


#数据分析和可视化

#购买商品前十国家
df2 = df1[df1['Quantity']>0].groupby('Country').sum()['Quantity']
#print(type(df2))
#print(df2)
quantity_first_10 = df2.sort_values(ascending=False).head(10)
#print(quantity_first_10)
# trace_basic = [go.Bar(x = quantity_first_10.index.tolist(), y =quantity_first_10.values.tolist(),marker = dict(color = 'orange'),opacity=0.50)]
# layout = go.Layout(title = '购买商品前十国家',xaxis=dict(title='国家'))
# figure_basic = go.Figure(data=trace_basic,layout=layout)
# py.plot(figure_basic)


#交易额前十国家
# df3 = df1[df1['Quantity']>0].groupby('Country').sum()['Price']
# Price_first_10 = df3.sort_values(ascending=False).head(10)
# trace_basic = [go.Bar(x = Price_first_10.index.tolist(), y =Price_first_10.values.tolist(),marker = dict(color = 'orange'),opacity=0.50)]
# layout = go.Layout(title = '交易额前十国家',xaxis=dict(title='国家'))
# figure_basic = go.Figure(data=trace_basic,layout=layout)
# py.plot(figure_basic)

df1['month'] = pd.to_datetime(df1['InvoiceDate'], errors='coerce').dt.month
#print(df1.head())
df4 = df1[df1['Quantity']>0].groupby('month').sum()['Quantity']
quantity_month = df4.sort_values(ascending=False)
print(quantity_month)
sns.set(style='darkgrid',context='notebook',font_scale=1.2)
plt.figure(figsize=(20,8),dpi=80)
df4.sort_values(ascending=False).plot(kind='bar')
plt.xticks(rotation=45)
plt.show()
sumPrice = df1[df1['Quantity']>0]['Price'].sum().astype(float)
count_id = df1[df1['Quantity']>0]['InvoiceNo'].drop_duplicates().count()

avgPrice = sumPrice/count_id
print(avgPrice)
end=time.clock()
print("final is in:\n",end-start)

customer = df1[df1['Quantity']>0].groupby('CustomerID').agg({'InvoiceNo':'nunique','Quantity':np.sum,'Price':np.sum})









