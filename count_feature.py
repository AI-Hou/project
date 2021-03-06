# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
import warnings

warnings.filterwarnings("ignore")

"""
dataset split:
                      (date_received)                              
           dateset3: 20160701~20160731 (113640),features3 from 20160315~20160630  (off_test)
           dateset2: 20160515~20160615 (258446),features2 from 20160201~20160514  
           dateset1: 20160414~20160514 (138303),features1 from 20160101~20160413        
1.merchant related: 
      sales_use_coupon. total_coupon
      transfer_rate = sales_use_coupon/total_coupon.
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon 
      total_sales.  coupon_rate = sales_use_coupon/total_sales.  

2.coupon related: 
      discount_rate. discount_man. discount_jian. is_man_jian
      day_of_week,day_of_month. (date_received)

3.user related: 
      distance. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      avg_diff_date_datereceived. min_diff_date_datereceived. max_diff_date_datereceived.  
      count_merchant.  
4.user_merchant:
      times_user_buy_merchant_before.

5. other feature:
      this_month_user_receive_all_coupon_count
      this_month_user_receive_same_coupon_count
      this_month_user_receive_same_coupon_lastone
      this_month_user_receive_same_coupon_firstone
      this_day_user_receive_all_coupon_count
      this_day_user_receive_same_coupon_count
      day_gap_before, day_gap_after  (receive the same coupon)
"""

# ,header=0，表明第0行代表列名
# off_train = pd.read_csv('data/ccf_offline_stage1_train.csv',header=0)#如果都这样写下面省不少事

# 1754884 record,1053282 with coupon_id,9738 coupon. date_received:20160101~20160615,date:20160101~20160630, 539438 users, 8415 merchants
off_train = pd.read_csv('data/ccf_offline_stage1_train.csv', header=None)
off_train.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']
# 2050 coupon_id. date_received:20160701~20160731, 76309 users(76307 in trainset, 35965 in online_trainset), 1559 merchants(1558 in trainset)
off_test = pd.read_csv('data/ccf_offline_stage1_test_revised.csv', header=None)

off_test.columns = ['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received']
# 11429826 record(872357 with coupon_id),762858 user(267448 in off_train)
on_train = pd.read_csv('data/ccf_online_stage1_train.csv', header=None)
on_train.columns = ['user_id', 'merchant_id', 'action', 'coupon_id', 'discount_rate', 'date_received', 'date']

dataset3 = off_test
feature3 = off_train[((off_train.date >= 20160315) & (off_train.date <= 20160630)) | (
			(off_train.date == np.nan) & (off_train.date_received >= 20160315) & (
				off_train.date_received <= 20160630))]
dataset2 = off_train[(off_train.date_received >= 20160515) & (off_train.date_received <= 20160615)]
feature2 = off_train[(off_train.date >= 20160201) & (off_train.date <= 20160514) | (
			(off_train.date == np.nan) & (off_train.date_received >= 20160201) & (
				off_train.date_received <= 20160514))]
dataset1 = off_train[(off_train.date_received >= 20160414) & (off_train.date_received <= 20160514)]
feature1 = off_train[(off_train.date >= 20160101) & (off_train.date <= 20160413) | (
			(off_train.date == np.nan) & (off_train.date_received >= 20160101) & (
				off_train.date_received <= 20160413))]

"""
1.merchant related: 
      total_sales. sales_use_coupon.  total_coupon
      coupon_rate = sales_use_coupon/total_sales.  
      transfer_rate = sales_use_coupon/total_coupon. 
      merchant_avg_distance,merchant_min_distance,merchant_max_distance of those use coupon
"""

#for dataset3
merchant3 = feature3[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant3[['merchant_id']]
#删除重复行数据深拷贝
t.drop_duplicates(inplace=True)
#去除null只要id列
t1 = merchant3[merchant3.date!= np.nan][['merchant_id']]
#显示卖出的商品
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()
#显示使用了优惠券消费的商品，正样本
t2 = merchant3[(merchant3.date!= np.nan)&(merchant3.coupon_id!= np.nan)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()
#显示了商品的优惠券的总数量
t3 = merchant3[merchant3.coupon_id!= np.nan][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()
#显示商品销量和距离的关系
t4 = merchant3[(merchant3.date!= np.nan)&(merchant3.coupon_id!= np.nan)][['merchant_id','distance']]
#-1替换null
t4.replace(np.nan ,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
#返回用户离商品的距离最小值
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)
#返回用户离商品的距离最大值
t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)
#返回用户离商品的距离平均值
t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)
#返回用户离商品的距离中值
t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)
#逐个揉进各个特征left代表把t放到左边
merchant3_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t2,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t3,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t5,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t6,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t7,on='merchant_id',how='left')
merchant3_feature = pd.merge(merchant3_feature,t8,on='merchant_id',how='left')
 #fill NAN  with 0
merchant3_feature.sales_use_coupon = merchant3_feature.sales_use_coupon.replace(np.nan,0)
#领取的券中使用券的比率
merchant3_feature['merchant_coupon_transfer_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_coupon
#即卖出商品中使用优惠券的占比
merchant3_feature['coupon_rate'] = merchant3_feature.sales_use_coupon.astype('float') / merchant3_feature.total_sales
#fillna with 0
merchant3_feature.total_coupon = merchant3_feature.total_coupon.replace(np.nan,0)
merchant3_feature.to_csv('data/merchant3_feature.csv',index=None)


#for dataset2与上面同理
merchant2 = feature2[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant2[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant2[merchant2.date!=np.nan][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant2[(merchant2.date!=np.nan)&(merchant2.coupon_id!=np.nan)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant2[merchant2.coupon_id!=np.nan][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant2[(merchant2.date!=np.nan)&(merchant2.coupon_id!=np.nan)][['merchant_id','distance']]
t4.replace('null',-1,inplace=True)
t4.distance = t4.distance.astype('float')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)

merchant2_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t2,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t3,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t5,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t6,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t7,on='merchant_id',how='left')
merchant2_feature = pd.merge(merchant2_feature,t8,on='merchant_id',how='left')
merchant2_feature.sales_use_coupon = merchant2_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant2_feature['merchant_coupon_transfer_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_coupon
merchant2_feature['coupon_rate'] = merchant2_feature.sales_use_coupon.astype('float') / merchant2_feature.total_sales
merchant2_feature.total_coupon = merchant2_feature.total_coupon.replace(np.nan,0) #fillna with 0
merchant2_feature.to_csv('data/merchant2_feature.csv',index=None)

#for dataset1与上面同理
merchant1 = feature1[['merchant_id','coupon_id','distance','date_received','date']]

t = merchant1[['merchant_id']]
t.drop_duplicates(inplace=True)

t1 = merchant1[merchant1.date!=np.nan][['merchant_id']]
t1['total_sales'] = 1
t1 = t1.groupby('merchant_id').agg('sum').reset_index()

t2 = merchant1[(merchant1.date!=np.nan)&(merchant1.coupon_id!=np.nan)][['merchant_id']]
t2['sales_use_coupon'] = 1
t2 = t2.groupby('merchant_id').agg('sum').reset_index()

t3 = merchant1[merchant1.coupon_id!=np.nan][['merchant_id']]
t3['total_coupon'] = 1
t3 = t3.groupby('merchant_id').agg('sum').reset_index()

t4 = merchant1[(merchant1.date!=np.nan)&(merchant1.coupon_id!=np.nan)][['merchant_id','distance']]
t4.replace(np.nan,-1,inplace=True)
t4.distance = t4.distance.astype('int')
t4.replace(-1,np.nan,inplace=True)
t5 = t4.groupby('merchant_id').agg('min').reset_index()
t5.rename(columns={'distance':'merchant_min_distance'},inplace=True)

t6 = t4.groupby('merchant_id').agg('max').reset_index()
t6.rename(columns={'distance':'merchant_max_distance'},inplace=True)

t7 = t4.groupby('merchant_id').agg('mean').reset_index()
t7.rename(columns={'distance':'merchant_mean_distance'},inplace=True)

t8 = t4.groupby('merchant_id').agg('median').reset_index()
t8.rename(columns={'distance':'merchant_median_distance'},inplace=True)


merchant1_feature = pd.merge(t,t1,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t2,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t3,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t5,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t6,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t7,on='merchant_id',how='left')
merchant1_feature = pd.merge(merchant1_feature,t8,on='merchant_id',how='left')
merchant1_feature.sales_use_coupon = merchant1_feature.sales_use_coupon.replace(np.nan,0) #fillna with 0
merchant1_feature['merchant_coupon_transfer_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_coupon
merchant1_feature['coupon_rate'] = merchant1_feature.sales_use_coupon.astype('float') / merchant1_feature.total_sales
merchant1_feature.total_coupon = merchant1_feature.total_coupon.replace(np.nan,0) #fillna with 0
merchant1_feature.to_csv('data/merchant1_feature.csv',index=None)

"""
3.user related: 
      count_merchant. 
      user_avg_distance, user_min_distance,user_max_distance. 
      buy_use_coupon. buy_total. coupon_received.
      buy_use_coupon/coupon_received. 
      buy_use_coupon/buy_total
      user_date_datereceived_gap

"""


def get_user_date_datereceived_gap(s):
	s = s.split(':')
	return (date(int(s[0][0:4]), int(s[0][4:6]), int(s[0][6:8])) - date(int(s[1][0:4]), int(s[1][4:6]),
																		int(s[1][6:8]))).days


# for dataset3
user3 = feature3[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user3[['user_id']]
t.drop_duplicates(inplace=True)  # 删除重复数据

t1 = user3[user3.date != np.nan][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
# 每个有购买记录用户的购买商品的总数量
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)
# 每个领券购买用户的距离
t2 = user3[(user3.date != np.nan) & (user3.coupon_id != np.nan)][['user_id', 'distance']]
t2.replace(np.nan, -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)
# 有购买用户使用券数量统计
t7 = user3[(user3.date != np.nan) & (user3.coupon_id != np.nan)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()
# 有购物的用户买商品总数
t8 = user3[user3.date != np.nan][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()
# 有券用户的券数统计
t9 = user3[user3.coupon_id != np.nan][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()
# 用户从接受券到消费券之间时间间隔很好的构造条件，然后又利用条件在函数里处理
t10 = user3[(user3.date_received.notnull()) & (user3.date.notnull())][['user_id', 'date_received', 'date']]  # 先筛选初始条件
t10['user_date_datereceived_gap'] = t10.date.astype('str') + ':' + t10.date_received.astype('str')
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user3_feature = pd.merge(t, t1, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t3, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t4, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t5, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t6, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t7, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t8, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t9, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t11, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t12, on='user_id', how='left')
user3_feature = pd.merge(user3_feature, t13, on='user_id', how='left')
# 查看一些数据，发现有特殊的nan,所以替换一下，应该也是看一步写一步
user3_feature.count_merchant = user3_feature.count_merchant.replace(np.nan, 0)
user3_feature.buy_use_coupon = user3_feature.buy_use_coupon.replace(np.nan, 0)
# 用户买东西用券的比率
user3_feature['buy_use_coupon_rate'] = user3_feature.buy_use_coupon.astype('float') / user3_feature.buy_total.astype(
	'float')
# 收到总券中消费券的比例
user3_feature['user_coupon_transfer_rate'] = user3_feature.buy_use_coupon.astype(
	'float') / user3_feature.coupon_received.astype('float')
user3_feature.buy_total = user3_feature.buy_total.replace(np.nan, 0)
user3_feature.coupon_received = user3_feature.coupon_received.replace(np.nan, 0)
user3_feature.to_csv('data/user3_feature.csv', index=None)

# for dataset2
user2 = feature2[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user2[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user2[user2.date != np.nan][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user2[(user2.date != np.nan) & (user2.coupon_id != np.nan)][['user_id', 'distance']]
t2.replace(np.nan, -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user2[(user2.date != np.nan) & (user2.coupon_id != np.nan)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user2[user2.date != np.nan][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user2[user2.coupon_id != np.nan][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user2[(user2.date_received.notnull()) & (user2.date.notnull())][['user_id', 'date_received', 'date']]
t10['user_date_datereceived_gap'] = t10.date.astype('str') + ':' + t10.date_received.astype('str')
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user2_feature = pd.merge(t, t1, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t3, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t4, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t5, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t6, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t7, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t8, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t9, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t11, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t12, on='user_id', how='left')
user2_feature = pd.merge(user2_feature, t13, on='user_id', how='left')
user2_feature.count_merchant = user2_feature.count_merchant.replace(np.nan, 0)
user2_feature.buy_use_coupon = user2_feature.buy_use_coupon.replace(np.nan, 0)
user2_feature['buy_use_coupon_rate'] = user2_feature.buy_use_coupon.astype('float') / user2_feature.buy_total.astype(
	'float')
user2_feature['user_coupon_transfer_rate'] = user2_feature.buy_use_coupon.astype(
	'float') / user2_feature.coupon_received.astype('float')
user2_feature.buy_total = user2_feature.buy_total.replace(np.nan, 0)
user2_feature.coupon_received = user2_feature.coupon_received.replace(np.nan, 0)
user2_feature.to_csv('data/user2_feature.csv', index=None)

# for dataset1
user1 = feature1[['user_id', 'merchant_id', 'coupon_id', 'discount_rate', 'distance', 'date_received', 'date']]

t = user1[['user_id']]
t.drop_duplicates(inplace=True)

t1 = user1[user1.date != np.nan][['user_id', 'merchant_id']]
t1.drop_duplicates(inplace=True)
t1.merchant_id = 1
t1 = t1.groupby('user_id').agg('sum').reset_index()
t1.rename(columns={'merchant_id': 'count_merchant'}, inplace=True)

t2 = user1[(user1.date != np.nan) & (user1.coupon_id != np.nan)][['user_id', 'distance']]
t2.replace(np.nan, -1, inplace=True)
t2.distance = t2.distance.astype('int')
t2.replace(-1, np.nan, inplace=True)
t3 = t2.groupby('user_id').agg('min').reset_index()
t3.rename(columns={'distance': 'user_min_distance'}, inplace=True)

t4 = t2.groupby('user_id').agg('max').reset_index()
t4.rename(columns={'distance': 'user_max_distance'}, inplace=True)

t5 = t2.groupby('user_id').agg('mean').reset_index()
t5.rename(columns={'distance': 'user_mean_distance'}, inplace=True)

t6 = t2.groupby('user_id').agg('median').reset_index()
t6.rename(columns={'distance': 'user_median_distance'}, inplace=True)

t7 = user1[(user1.date != np.nan) & (user1.coupon_id != np.nan)][['user_id']]
t7['buy_use_coupon'] = 1
t7 = t7.groupby('user_id').agg('sum').reset_index()

t8 = user1[user1.date != np.nan][['user_id']]
t8['buy_total'] = 1
t8 = t8.groupby('user_id').agg('sum').reset_index()

t9 = user1[user1.coupon_id != np.nan][['user_id']]
t9['coupon_received'] = 1
t9 = t9.groupby('user_id').agg('sum').reset_index()

t10 = user1[(user1.date_received.notnull()) & (user1.date.notnull())][['user_id', 'date_received', 'date']]
t10['user_date_datereceived_gap'] = t10.date.astype('str') + ':' + t10.date_received.astype('str')
t10.user_date_datereceived_gap = t10.user_date_datereceived_gap.apply(get_user_date_datereceived_gap)
t10 = t10[['user_id', 'user_date_datereceived_gap']]

t11 = t10.groupby('user_id').agg('mean').reset_index()
t11.rename(columns={'user_date_datereceived_gap': 'avg_user_date_datereceived_gap'}, inplace=True)
t12 = t10.groupby('user_id').agg('min').reset_index()
t12.rename(columns={'user_date_datereceived_gap': 'min_user_date_datereceived_gap'}, inplace=True)
t13 = t10.groupby('user_id').agg('max').reset_index()
t13.rename(columns={'user_date_datereceived_gap': 'max_user_date_datereceived_gap'}, inplace=True)

user1_feature = pd.merge(t, t1, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t3, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t4, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t5, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t6, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t7, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t8, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t9, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t11, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t12, on='user_id', how='left')
user1_feature = pd.merge(user1_feature, t13, on='user_id', how='left')
user1_feature.count_merchant = user1_feature.count_merchant.replace(np.nan, 0)
user1_feature.buy_use_coupon = user1_feature.buy_use_coupon.replace(np.nan, 0)
user1_feature['buy_use_coupon_rate'] = user1_feature.buy_use_coupon.astype('float') / user1_feature.buy_total.astype(
	'float')
user1_feature['user_coupon_transfer_rate'] = user1_feature.buy_use_coupon.astype(
	'float') / user1_feature.coupon_received.astype('float')
user1_feature.buy_total = user1_feature.buy_total.replace(np.nan, 0)
user1_feature.coupon_received = user1_feature.coupon_received.replace(np.nan, 0)
user1_feature.to_csv('data/user1_feature.csv', index=None)

"""
4.user_merchant:
      times_user_buy_merchant_before.
"""
#for dataset3
all_user_merchant = feature3[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)#去除重复值

t = feature3[['user_id','merchant_id','date']]
t = t[t.date!=np.nan][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature3[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!=np.nan][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature3[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!=np.nan)&(t2.date_received!=np.nan)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature3[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature3[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!=np.nan)&(t4.coupon_id==np.nan)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant3 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t1,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t2,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t3,on=['user_id','merchant_id'],how='left')
user_merchant3 = pd.merge(user_merchant3,t4,on=['user_id','merchant_id'],how='left')
user_merchant3.user_merchant_buy_use_coupon = user_merchant3.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant3.user_merchant_buy_common = user_merchant3.user_merchant_buy_common.replace(np.nan,0)
user_merchant3['user_merchant_coupon_transfer_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_received.astype('float')
user_merchant3['user_merchant_coupon_buy_rate'] = user_merchant3.user_merchant_buy_use_coupon.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3['user_merchant_rate'] = user_merchant3.user_merchant_buy_total.astype('float') / user_merchant3.user_merchant_any.astype('float')
user_merchant3['user_merchant_common_buy_rate'] = user_merchant3.user_merchant_buy_common.astype('float') / user_merchant3.user_merchant_buy_total.astype('float')
user_merchant3.to_csv('data/user_merchant3.csv',index=None)

#for dataset2
all_user_merchant = feature2[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature2[['user_id','merchant_id','date']]
t = t[t.date!=np.nan][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature2[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!=np.nan][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature2[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!=np.nan)&(t2.date_received!=np.nan)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature2[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature2[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!=np.nan)&(t4.coupon_id==np.nan)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant2 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t1,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t2,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t3,on=['user_id','merchant_id'],how='left')
user_merchant2 = pd.merge(user_merchant2,t4,on=['user_id','merchant_id'],how='left')
user_merchant2.user_merchant_buy_use_coupon = user_merchant2.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant2.user_merchant_buy_common = user_merchant2.user_merchant_buy_common.replace(np.nan,0)
user_merchant2['user_merchant_coupon_transfer_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_received.astype('float')
user_merchant2['user_merchant_coupon_buy_rate'] = user_merchant2.user_merchant_buy_use_coupon.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2['user_merchant_rate'] = user_merchant2.user_merchant_buy_total.astype('float') / user_merchant2.user_merchant_any.astype('float')
user_merchant2['user_merchant_common_buy_rate'] = user_merchant2.user_merchant_buy_common.astype('float') / user_merchant2.user_merchant_buy_total.astype('float')
user_merchant2.to_csv('data/user_merchant2.csv',index=None)

#for dataset1
all_user_merchant = feature1[['user_id','merchant_id']]
all_user_merchant.drop_duplicates(inplace=True)

t = feature1[['user_id','merchant_id','date']]
t = t[t.date!=np.nan][['user_id','merchant_id']]
t['user_merchant_buy_total'] = 1
t = t.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t.drop_duplicates(inplace=True)

t1 = feature1[['user_id','merchant_id','coupon_id']]
t1 = t1[t1.coupon_id!=np.nan][['user_id','merchant_id']]
t1['user_merchant_received'] = 1
t1 = t1.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t1.drop_duplicates(inplace=True)

t2 = feature1[['user_id','merchant_id','date','date_received']]
t2 = t2[(t2.date!=np.nan)&(t2.date_received!=np.nan)][['user_id','merchant_id']]
t2['user_merchant_buy_use_coupon'] = 1
t2 = t2.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t2.drop_duplicates(inplace=True)

t3 = feature1[['user_id','merchant_id']]
t3['user_merchant_any'] = 1
t3 = t3.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t3.drop_duplicates(inplace=True)

t4 = feature1[['user_id','merchant_id','date','coupon_id']]
t4 = t4[(t4.date!=np.nan)&(t4.coupon_id==np.nan)][['user_id','merchant_id']]
t4['user_merchant_buy_common'] = 1
t4 = t4.groupby(['user_id','merchant_id']).agg('sum').reset_index()
t4.drop_duplicates(inplace=True)

user_merchant1 = pd.merge(all_user_merchant,t,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t1,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t2,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t3,on=['user_id','merchant_id'],how='left')
user_merchant1 = pd.merge(user_merchant1,t4,on=['user_id','merchant_id'],how='left')
user_merchant1.user_merchant_buy_use_coupon = user_merchant1.user_merchant_buy_use_coupon.replace(np.nan,0)
user_merchant1.user_merchant_buy_common = user_merchant1.user_merchant_buy_common.replace(np.nan,0)
user_merchant1['user_merchant_coupon_transfer_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_received.astype('float')
user_merchant1['user_merchant_coupon_buy_rate'] = user_merchant1.user_merchant_buy_use_coupon.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1['user_merchant_rate'] = user_merchant1.user_merchant_buy_total.astype('float') / user_merchant1.user_merchant_any.astype('float')
user_merchant1['user_merchant_common_buy_rate'] = user_merchant1.user_merchant_buy_common.astype('float') / user_merchant1.user_merchant_buy_total.astype('float')
user_merchant1.to_csv('data/user_merchant1.csv',index=None)

##################  generate training and testing set ################
def get_label(s):
    s = s.split(':')
    if s[0]== 'nan':
        return 0
    elif (date(int(s[0][0:4]),int(s[0][4:6]),int(s[0][6:8]))-date(int(s[1][0:4]),int(s[1][4:6]),int(s[1][6:8]))).days<=15:
        return 1
    else:
        return -1


coupon3 = pd.read_csv('data/coupon3_feature.csv')
merchant3 = pd.read_csv('data/merchant3_feature.csv')
user3 = pd.read_csv('data/user3_feature.csv')
user_merchant3 = pd.read_csv('data/user_merchant3.csv')
other_feature3 = pd.read_csv('data/other_feature3.csv')
#观察各表链接的键值
dataset3 = pd.merge(coupon3,merchant3,on='merchant_id',how='left')
dataset3 = pd.merge(dataset3,user3,on='user_id',how='left')
dataset3 = pd.merge(dataset3,user_merchant3,on=['user_id','merchant_id'],how='left')
dataset3 = pd.merge(dataset3,other_feature3,on=['user_id','coupon_id','date_received'],how='left')
dataset3.drop_duplicates(inplace=True)
#print(dataset3.shape)

dataset3.user_merchant_buy_total = dataset3.user_merchant_buy_total.replace(np.nan,0)
dataset3.user_merchant_any = dataset3.user_merchant_any.replace(np.nan,0)
dataset3.user_merchant_received = dataset3.user_merchant_received.replace(np.nan,0)
#是否为周末
dataset3['is_weekend'] = dataset3.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset3.day_of_week)
#shape[0]和shape[1]分别代表矩阵行和列的长度
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset3 = pd.concat([dataset3,weekday_dummies],axis=1)
dataset3.drop(['merchant_id','day_of_week','coupon_count'],axis=1,inplace=True)
dataset3 = dataset3.replace('null',np.nan)
dataset3.to_csv('data/dataset3.csv',index=None)

#一下重复上面的思想
coupon2 = pd.read_csv('data/coupon2_feature.csv')
merchant2 = pd.read_csv('data/merchant2_feature.csv')
user2 = pd.read_csv('data/user2_feature.csv')
user_merchant2 = pd.read_csv('data/user_merchant2.csv')
other_feature2 = pd.read_csv('data/other_feature2.csv')
dataset2 = pd.merge(coupon2,merchant2,on='merchant_id',how='left')
dataset2 = pd.merge(dataset2,user2,on='user_id',how='left')
dataset2 = pd.merge(dataset2,user_merchant2,on=['user_id','merchant_id'],how='left')
dataset2 = pd.merge(dataset2,other_feature2,on=['user_id','coupon_id','date_received'],how='left')
dataset2.drop_duplicates(inplace=True)
#print(dataset2.shape)

dataset2.user_merchant_buy_total = dataset2.user_merchant_buy_total.replace(np.nan,0)
dataset2.user_merchant_any = dataset2.user_merchant_any.replace(np.nan,0)
dataset2.user_merchant_received = dataset2.user_merchant_received.replace(np.nan,0)
dataset2['is_weekend'] = dataset2.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset2.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset2 = pd.concat([dataset2,weekday_dummies],axis=1)
dataset2['label'] = dataset2.date.astype('str') + ':' +  dataset2.date_received.astype('str')
print(dataset2['label'])
dataset2.label = dataset2.label.apply(get_label)

dataset2.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset2 = dataset2.replace('null',np.nan)
dataset2.to_csv('data/dataset2.csv',index=None)


coupon1 = pd.read_csv('data/coupon1_feature.csv')
merchant1 = pd.read_csv('data/merchant1_feature.csv')
user1 = pd.read_csv('data/user1_feature.csv')
user_merchant1 = pd.read_csv('data/user_merchant1.csv')
other_feature1 = pd.read_csv('data/other_feature1.csv')
dataset1 = pd.merge(coupon1,merchant1,on='merchant_id',how='left')
dataset1 = pd.merge(dataset1,user1,on='user_id',how='left')
dataset1 = pd.merge(dataset1,user_merchant1,on=['user_id','merchant_id'],how='left')
dataset1 = pd.merge(dataset1,other_feature1,on=['user_id','coupon_id','date_received'],how='left')
dataset1.drop_duplicates(inplace=True)
#print( dataset1.shape)

dataset1.user_merchant_buy_total = dataset1.user_merchant_buy_total.replace(np.nan,0)
dataset1.user_merchant_any = dataset1.user_merchant_any.replace(np.nan,0)
dataset1.user_merchant_received = dataset1.user_merchant_received.replace(np.nan,0)
dataset1['is_weekend'] = dataset1.day_of_week.apply(lambda x:1 if x in (6,7) else 0)
weekday_dummies = pd.get_dummies(dataset1.day_of_week)
weekday_dummies.columns = ['weekday'+str(i+1) for i in range(weekday_dummies.shape[1])]
dataset1 = pd.concat([dataset1,weekday_dummies],axis=1)
dataset1['label'] = dataset1.date.astype('str') + ':' +  dataset1.date_received.astype('str')
dataset1.label = dataset1.label.apply(get_label)
dataset1.drop(['merchant_id','day_of_week','date','date_received','coupon_id','coupon_count'],axis=1,inplace=True)
dataset1 = dataset1.replace('null',np.nan)
dataset1.to_csv('data/dataset1.csv',index=None)


