import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
# pd.set_option('display.unicode.ambiguous_as_wide', True)
# pd.set_option('display.unicode.east_asian_width', True)
# pd.set_option('display.width', 160) # 设置打印宽度(**重要**)
# pd.set_option('expand_frame_repr', False)
#理解数据

pd.set_option('display.max_columns',None)
data = pd.read_csv('./LCIS.csv',dtype=({'ListingId':str}))
# print(data.head(10))
# print(data.info())
# print(data.describe())

#数据预处理(数据清洗)
columns = {'ListingId':'列表序号','recorddate':'记录日期'}
data.rename(columns=columns,inplace=True)
data.apply(lambda x: sum(x.isnull())/len(x),axis=0)
miss_rate = pd.DataFrame(data.apply(lambda x: sum(x.isnull())/len(x),axis=0))
miss_rate.columns = ['缺失率']
miss_rate[miss_rate['缺失率']>0]['缺失率'].apply(lambda x: format(x,' .3%'))
#data[data['下次计划还款利息'].isnull()]
data1=data[data['下次计划还款利息'].isnull()]['标当前状态'].value_counts()
#print(data1.value_counts)
data[data['上次还款利息'].isnull()].iloc[:,-9:-1]
data[data['上次还款利息'].isnull()]['标当前状态'].value_counts()
data[data['历史成功借款金额'].isnull()]
data[data['记录日期'].isnull()]
data[data['记录日期'].isnull()][['手机认证','户口认证']]
data.dropna(subset=['记录日期'],how='any',inplace=True)


#处理重复值

data[data.duplicated()]
data.drop_duplicates(inplace=True)
data['手机认证'].value_counts().plot(kind='bar',figsize=(20,8))
data['户口认证'].value_counts().plot(kind='bar',figsize=(20,8))
data=data[(data['手机认证']== '成功认证') | (data['手机认证']== '未成功认证')]

#构建模型
#1.不同性别的放贷比例与逾期关系
df_gender = pd.pivot_table(data=data,columns=['标当前状态'],index='性别',values='列表序号',aggfunc=np.size)
df_gender['借款笔数占比'] = df_gender.sum(axis=1)/df_gender.sum().sum()
df_gender['逾期笔数占比'] = df_gender['逾期中']/df_gender.sum(axis=1)
plt.figure(figsize=(16,9))
plt.subplot(121)
plt.title('男女借款比例')
plt.bar(x = df_gender.index,height=df_gender['借款笔数占比'],color=['c','g'])
plt.subplot(122)
plt.bar(x = df_gender.index,height=df_gender['逾期笔数占比'],color=['c','g'])
plt.title('男女逾期情况')
plt.suptitle('不同性别的客户画像')
plt.show()
df_age = data.groupby(['年龄'])['借款金额'].sum()
df_age = pd.DataFrame(df_age)
df_age['借款金额累计'] = df_age['借款金额'].cumsum()
df_age['借款金额累计占比'] = df_age['借款金额累计']/df_age['借款金额'].sum()
index_num = df_age[df_age['借款金额累计占比']>0.8].index[0]   #百分之八十的贷款给了36岁以下的酷虎
cum_percent = df_age.loc[index_num,'借款金额累计占比']
plt.figure(figsize=(16,9))
plt.xlabel('年龄',fontsize=20)
plt.bar(x = df_age.index,height=df_age['借款金额'],color = 'steelblue',alpha = 0.5, width = 0.7)
plt.axvline(x=index_num,color='orange',linestyle='--',alpha=0.8)
df_age['借款金额累计占比'].plot(style='--ob',secondary_y = True)       #虚线，o点，蓝色
plt.text(index_num+0.4,cum_percent,'累计占比为: %.3f%%' % (cum_percent*100),color='indianred')
plt.show()
data['年龄分段情况'] = pd.cut(data['年龄'],[17,24,30,36,42,48,54,65],right=True)
df_age = pd.pivot_table(data=data,columns='标当前状态',index=data['年龄分段情况'],values='列表序号',aggfunc=np.size)
df_age['借款笔数'] = df_age.sum(axis=1)
df_age['借款笔数分布'] = df_age['借款笔数']/df_age['借款笔数'].sum()
df_age['逾期占比'] = df_age['逾期中']/df_age['借款笔数']
df_age['借款笔数分布%'] = df_age['借款笔数分布'].apply(lambda x: format(x,' .3%'))
df_age['逾期占比%'] = df_age['逾期占比'].apply(lambda x: format(x,' .3%'))
plt.figure(figsize=(12,9))
# plt.show()
df_age['借款笔数分布'].plot(kind='bar',rot=45,color='steelblue',alpha=0.5)
#plt.hist(x= df_age['借款笔数分布'],bins= df_age.index,alpha=0.5, color='steelblue')
plt.ylabel('借款笔数分布')
df_age['逾期占比'].plot(kind='line',rot=45,color='steelblue',alpha=0.5,secondary_y=True)
plt.xlabel('年龄分段情况',fontsize=20)
plt.ylabel('逾期占比情况')
plt.grid(True)
plt.show()
df_edu = pd.pivot_table(data=data,columns='标当前状态',index='学历认证',values='列表序号',aggfunc=np.size)
df_edu['借款笔数']  = df_edu.sum(axis=1)
df_edu['借款笔数占比'] = df_edu['借款笔数']/df_edu['借款笔数'].sum()
df_edu['逾期占比'] = df_edu['逾期中']/df_edu['借款笔数']
# plt.figure(figsize=(16,9))
# plt.subplot(121)
# plt.pie(x = df_edu['借款笔数占比'],labels=['成功认证','未成功认证'],colors = ['red','yellow'],autopct='%.1f%%',pctdistance=0.5,labeldistance=1.1)
# plt.title('学历认证比例')
# plt.subplot(122)
# plt.bar(x = df_edu.index,height = df_edu['逾期占比'],color = ['orange','c'])
# plt.title('不同学历的人逾期情况')
# plt.suptitle('不同学历的人客户画像')
# plt.show()
def trans(data,col,ind):
    df = pd.pivot_table(data=data, columns=col, index=ind, values='列表序号', aggfunc=np.size)
    df['借款笔数'] = df.apply(np.sum,axis=1)
    df['借款笔数占比'] = df['借款笔数']/df['借款笔数'].sum()
    df['逾期占比'] = df['逾期中']/df['借款笔数']
    plt.figure(figsize=(16, 12))
    plt.subplot(121)
    plt.pie(x=df['借款笔数占比'], labels=['成功认证', '未成功认证'], colors=['red', 'yellow'], autopct='%.1f%%', pctdistance=0.5,
            labeldistance=1.1)
    plt.title('%s比例' % ind)
    plt.subplot(122)
    plt.bar(x=df.index, height=df['逾期占比'], color=['orange', 'c'])
    plt.title('不同%s的人逾期情况' % ind)
    plt.suptitle('%s客户画像' % ind)
    plt.show()

trans(data,'标当前状态','淘宝认证')










