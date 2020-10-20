import pandas as pd
import numpy as np

# T01
# pd.show_versions()

# data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
#         'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
#         'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
#         'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}
# labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# df = pd.DataFrame(data,index = labels)
# print(df)
# T02 详细信息
# print(df.info())
# T03 索引
# print(df.iloc[:3])
# T04 指定选择数据范围
# print(df[df['visits'] > 2])
# T05 查看缺失值
# print(df[df['age'].isnull()])
# T06 通过给定范围查找某一属性
# print(df[(df['animal'] =='cat') & (df['age'] < 3)])
# T07 改变数值
# df.loc['f','age'] = 1.5
# print(df[(df['animal'] =='cat') & (df['age'] < 3)])
# T08 groupby求均值
# print(df.groupby('animal')['age'].mean())
# T09 计算相同属性值的个数
# print(df['animal'].value_counts())
# T10 属性值进行映射
# df['priority'] = df['priority'].map({'yes':True,'no':False})
# print(df.head())
# T11 属性值进行替换
# df['animal'] = df['animal'].replace('snake','dragon')
# print(df.head())
# T12 数据透视表
# print(df.pivot_table(index='animal',columns='visits',values='age',aggfunc='mean'))

# T13 提取均值组成新的数据
# df = pd.DataFrame(np.random.randint(0,10,[5,3]))
# print(df.head())
# print(df.mean(axis=1))
# print(df.sub(df.mean(axis=1),axis = 0))
# T14 统计不同属性值的个数
# print(len(df) - df.duplicated(keep=False).sum())
# print(len(df.drop_duplicates(keep=False)))
# print(df)

# T15 给定数据，分别求滑动窗口的均值（加入补0操作）
# df = pd.DataFrame({'group': list('aabbabbbabab'),
#                    'value': [1, 2, 3, np.nan, 2, 3, np.nan, 1, 7, 3, np.nan, 8]})
# print(df.head(12))
# g1 = df.groupby(['group'])['value']
# g2 = df.fillna(0).groupby(['group'])['value']
# print(g2.describe())
# print(g2.rolling(3,min_periods=1).sum())
# s = g2.rolling(3,min_periods=1).sum()/g2.rolling(3,min_periods=1).count()
# print(s)
# print(s.reset_index(level=0,drop=True).sort_index())

# T16 指定时间序列进行计算
# dt = pd.date_range(start='2017-01-01', end='2017-12-31',freq='D')
# s = pd.Series(np.random.rand(len(dt)), index=dt)
# print(s)
# print(s[s.index.weekday == 2].sum())
# print(s.resample('M').mean())

# T17 对缺失值数据自动计算
# df = pd.DataFrame({'From_To': ['LoNDon_paris', 'MAdrid_miLAN', 'londON_StockhOlm',
#                                'Budapest_PaRis', 'Brussels_londOn'],
#               'FlightNumber': [10045, np.nan, 10065, np.nan, 10085],
#               'RecentDelays': [[23, 47], [], [24, 43, 87], [13], [67, 32]],
#                    'Airline': ['KLM(!)', '<Air France> (12)', '(British Airways. )',
#                                '12. Air France', '"Swiss Air"']})
# print(df.head())
# 插值
# df['FlightNumber'] = df['FlightNumber'].interpolate().astype(int)
# print(df.head())
# 拆分
# temp = df.From_To.str.split('_',expand = True)
# temp.columns = ['From','To']
# temp['From'] = temp['From'].str.capitalize()
# temp['To'] = temp['To'].str.capitalize()
# df = df.join(temp)
# print(df.head())
# 去掉airline中多余的字符
# df['Airline'] = df['Airline'].str.extract('([a-zA-Z\s]+)',expand = False).str.strip()
# print(df.head())
# 将RecentDelays中的数据分开写
# delays = df['RecentDelays'].apply(pd.Series)
# delays.columns = ['delay_{}'.format(n) for n in range(1,len(delays.columns)+1)]
# print(delays)

# 多重索引
# letters = ['A', 'B', 'C']
# numbers = list(range(10))
# mi = pd.MultiIndex.from_product([letters, numbers])
# s = pd.Series(np.random.randint(0,10,30), index=mi)
# print(s)
# 定位
# print(s.loc[pd.IndexSlice[:'B', 5:]])
# 按索引计算
# print(s.sum(level=1))
# 变换索引
# new = s.swaplevel(0,1)
# print(new)
