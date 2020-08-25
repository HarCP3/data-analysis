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
#print(df.head())
#print(df.info())
#print(df[df['InvoiceNo'].str[0] == 'C'])


#数据清洗
print(df.apply(lambda x:sum(x.isnull())/len(x),axis=0))
df.drop(['Description'],axis=1,inplace=True)
#print(df)
df['CustomerID'] = df['CustomerID'].fillna('U')
df['amount'] = df['Quantity']*df['UnitPrice']
#print(df.info())
df['date']= [i.split(' ')[0] for i in df['InvoiceDate']]
# df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
# df['date'] = [i.strftime('%d-%m-%Y') for i in df['InvoiceDate']]

# print(df.info())
# print(df.head())
df['time'] = [i.split(' ')[1] for i in df['InvoiceDate']]
#print(df[['time','date']])
df.drop(['InvoiceDate'],axis=1,inplace=True)
#print(df['date'].head())
df['year'] = [i.split('/')[2] for i in df['date']]
df['month'] = [i.split('/')[0] for i in df['date']]
df['day'] = [i.split('/')[1] for i in df['date']]
#print(df[['date','year','month','day']].head())
df['date'] = pd.to_datetime(df['date'])
df = df.drop_duplicates()
# print(df.describe())
df2 = df.loc[df['UnitPrice']<=0]
# print(df2.shape[0]/df.shape[0])
# print(df2['UnitPrice'].groupby(by=df2['UnitPrice']).count())

#数据分析
df1 = df.loc[(df['Quantity']<=0)]
tt = pd.pivot_table(df1, index='year',columns = 'month', values = 'amount', aggfunc= np.sum)
# print(tt)
df2 = df[(df['Quantity']>0) & (df['UnitPrice']>0)]
pp = pd.pivot_table(df2, index='year',columns = 'month', values = 'amount', aggfunc= np.sum)
# print(pp)
# print(np.abs(tt/pp))
np.abs(tt/pp).loc['2011'].mean()

#画图(已解决)

R_value = df.groupby('CustomerID')['date'].max()
df2['date'].max()
R_value = (df2['date'].max()-R_value).dt.days
F_value = df2.groupby('CustomerID')['InvoiceNo'].nunique()
M_value = df2.groupby('CustomerID')['amount'].sum()
sns.set(style = 'darkgrid')
# plt.hist(R_value)
# plt.show()
R_bins = [0,30,90,180,360,720]
F_bins = [1,2,5,10,20,5000]
M_bins = [0,55,2000,5000,10000,200000]
R_score = pd.cut(R_value,R_bins,labels=[5,4,3,2,1],right=False)
#print(R_score)
F_score = pd.cut(F_value,F_bins,labels=[1,2,3,4,5],right=False)
M_score = pd.cut(M_value,M_bins,labels=[1,2,3,4,5],right=False)
rfm = pd.concat([R_score,F_score,M_score],axis=1)
#print(F_score.shape,M_score.shape,R_score.shape)
# print(rfm)
rfm.rename(columns={'date':'R_score','InvoiceNo':'F_score','amount':'M_score'},inplace=True)
for i in ['R_score','F_score','M_score']:
    rfm[i] = rfm[i].astype(float)
rfm['R'] = np.where(rfm['R_score']>3.82,'高','低')
rfm['F'] = np.where(rfm['F_score']>2.03,'高','低')
rfm['M'] = np.where(rfm['M_score']>1.8,'高','低')
rfm['value'] = rfm['R'].str[:] +rfm['F'].str[:] + rfm['M'].str[:]
#print(rfm.info())
def trans_value(x):
    if x == '高高高':
        return '重要价值客户'
    elif x=='高低高':
        return '重要发展客户'
    elif x== '低高高':
        return '重要保持客户'
    elif x== '低低高':
        return '重要挽留客户'
    elif x=='高高低':
        return '一般价值客户'
    elif x== '高低低':
        return '一般发展客户'
    elif x=='低高低':
        return '一般保持客户'
    else:
        return '一般挽留客户'

rfm['用户等级'] = rfm['value'].apply(trans_value)
rfm['用户等级'].value_counts()
trade_basic = [go.Bar(x = rfm['用户等级'].value_counts().index, y=rfm['用户等级'].value_counts().values,marker = dict(color='orange'),opacity=0.50)]
layout = go.Layout(title='用户等级情况',xaxis = dict(title='用户重要度'))
figure_basic = go.Figure(data= trade_basic,layout=layout)
py.plot(figure_basic)
# trace = [go.Pie(labels= rfm['用户等级'].value_counts().index, values=rfm['用户等级'].value_counts().values,textfont=dict(size=12,color='white'))]
# layout2 = go.Layout(title='用户等级比例')
# figure_basic2 = go.Figure(data= trace,layout=layout2)
# py.plot(figure_basic2)

#结论和建议














