import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid',font_scale=1.5)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns',None)
# import os
# os.chdir()
data_user = pd.read_csv('tianchi_mobile_recommend_train_user.csv',dtype=str)
#data_user = data_user.sample(frac=0.2,replace=False)
data_user.info()
data_user.head()

#3.数据预处理（数据清洗）
#3.1统计缺失值
data_user.apply(lambda x:sum(x.isnull())/len(x))
data_user['date'] = data_user['time'].str[0:10]
data_user['hour'] = data_user['time'].str[11:]
data_user.dtypes
data_user['date'] = pd.to_datetime(data_user['date'])
data_user['time'] = pd.to_datetime(data_user['time'])
data_user['hour'] = data_user['hour'].astype(int)
data_user.sort_values(by='time',ascending=True,inplace=True)
data_user.reset_index(drop=True,inplace=True)
data_user.describe(include=['object'])

#4.构建模型

pv_daily = data_user.groupby('date').count()['user_id']
uv_daily = data_user.groupby('date')['user_id'].apply(lambda x: x.drop_duplicates().count())
pv_uv_daily = pd.concat([pv_daily,uv_daily],axis=1)
pv_uv_daily.rename(columns={'0':'pv','user_id':'uv'},inplace=True)
pv_uv_daily.corr(method='pearson')
plt.figure(figsize=(16,9))
plt.subplot(211)
plt.plot(pv_daily,color='red')
plt.xticks(fontsize=10)
plt.title('每天访问量')
plt.subplot(212)
plt.plot(uv_daily,color='green')
plt.xticks(fontsize=10)
plt.title('每天访问用户数')
plt.suptitle('UV和PV变化趋势',fontsize=20)
plt.show()
pv_daily = data_user.groupby('hour').count()['user_id']
uv_daily = data_user.groupby('hour')['user_id'].apply(lambda x: x.drop_duplicates().count())
pv_uv_daily = pd.concat([pv_daily,uv_daily],axis=1)
pv_uv_daily.columns=['pv','uv']
plt.figure(figsize=(16,9))
pv_uv_daily['pv'].plot(color='steelblue',label='每个小时访问量')
plt.ylabel('访问量')
plt.legend(loc='upper left')
pv_uv_daily['uv'].plot(color='red',label='每个小时不同用户访问量',secondary_y=True)
plt.ylabel('访问用户数')
plt.xticks(range(0,24),pv_uv_daily.index)
plt.legend(loc='upper center')
plt.grid(True)
plt.show()

pv_detail = pd.pivot_table(columns='behavior_type',index='hour',data=data_user,values='user_id',aggfunc=np.size)
plt.figure(figsize=(16,9))
sns.lineplot(data=pv_detail.iloc[:,1:])
plt.show()
data_user_buy = data_user[data_user.behavior_type=='4'].groupby('user_id').size()
plt.hist(x=data_user_buy,bins=30)
data_user_buy1 = data_user[data_user.behavior_type=='4'].groupby(['date','user_id'])
data_user_buy1.count()['behavior_type'].reset_index().rename(columns = {'bahavior_type':'total'})
data_user_buy2 = data_user_buy1.gropuby('date').sum()['total'] / data_user_buy1.groupby('date').count()['total']

data_user_buy2.plot()
plt.show()

#日ARPU
data_user['operation'] = 1
data_user_buy2 = data_user.groupby(['date','user','behavior'])['opration'].count().reset_index.rename(columns = {'operation':'total'})
data_user_buy2.groupby('date').apply(lambda x: x[x['behavior_type'] == '4'].total.sum()/len(x.user_id.unique()))


