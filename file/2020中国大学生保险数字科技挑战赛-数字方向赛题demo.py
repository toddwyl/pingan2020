################################2020中国大学生保险数字科技挑战赛-数字方向赛题##########################################
from datetime import datetime
import gc
import random

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import numpy as np

df_columns = ['region','city','eventbody','eventname','eventtime','lat','lgt','nettype','title','uadevice','uaos','userid','date'
]
############################################分批读取训练集数据##########################################################
s = datetime.now()
df_0 = pd.read_csv(r'/home/kesci/input/train1541/tps_training_dataset_2_text/000000_0', sep='\001',header=None)
df_0.columns = df_columns
df_1 = pd.read_csv(r'/home/kesci/input/train1541/tps_training_dataset_2_text/000001_0', sep='\001',header=None)
df_1.columns = df_columns
df_2 = pd.read_csv(r'/home/kesci/input/train1541/tps_training_dataset_2_text/000002_0', sep='\001',header=None)
df_2.columns = df_columns
df_3 = pd.read_csv(r'/home/kesci/input/train1541/tps_training_dataset_2_text/000003_0', sep='\001',header=None)
df_3.columns = df_columns
print ('time:{}'.format(datetime.now()-s))

############################################通过变换dataframe中连续型数据格式，减小内存消耗################################
def reduce_mean_usage(df, verbose = True):
    numberics = ['int16', 'int32', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum()/1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numberics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum()/1024**2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb {:.1f}% reduction'.format(end_mem, 100*(start_mem - end_mem)/start_mem))

    return df

df_0 = reduce_mean_usage(df_0, verbose = True)
df_1 = reduce_mean_usage(df_1, verbose = True)
df_2 = reduce_mean_usage(df_2, verbose = True)
df_3 = reduce_mean_usage(df_3, verbose = True)

df = pd.concat([df_0,df_1], axis = 0)
del df_0
del df_1
gc.collect()

df = pd.concat([df,df_2], axis = 0)
del df_2
gc.collect()

df = pd.concat([df,df_3], axis = 0)
del df_3
gc.collect()

###########################################################################################################################
#########################################################训练集中y值提取####################################################
###########################################################################################################################
df_get_chunk_trainy = df[df['date'] >= 20200421]  
df_get_chunk_trainy['hzcModule'] = df_get_chunk_trainy['eventbody'].apply(lambda x: x.split('\x02')[3].split('\x03')[1] if len(x.split('\x02'))>=4 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
df_get_chunk_trainy['productId'] = df_get_chunk_trainy['eventbody'].apply(lambda x: x.split('\x02')[6].split('\x03')[1] if len(x.split('\x02'))>=7 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
df_get_chunk_trainy = df_get_chunk_trainy[df_get_chunk_trainy.eventname == 'h320057']
df_get_chunk_trainy = df_get_chunk_trainy[df_get_chunk_trainy.hzcModule.isin (['hczoi_good_best','hczoi_good_accident','hczoi_good_fitness','hczoi_good_trip','hczoi_good_estate','hczoi_good_company','hczoi_good_special'])]

df_pro_train_y = df_get_chunk_trainy[['userid','productId', 'eventtime']].groupby(['userid','productId']).count()
df_pro_train_y = df_pro_train_y.reset_index()
df_pro_train_y.rename(columns={'eventtime': 'productId_isin'})

del df_get_chunk_trainy
gc.collect()
##########################################################################################################################
######################################################训练集中x信息提取####################################################
##########################################################################################################################
df_get_chunk_trainx = df[df['date'] < 20200421]
df_get_chunk_trainx['hzcModule'] = df_get_chunk_trainx['eventbody'].apply(lambda x: x.split('\x02')[3].split('\x03')[1] if len(x.split('\x02'))>=4 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
df_get_chunk_trainx['productId'] = df_get_chunk_trainx['eventbody'].apply(lambda x: x.split('\x02')[6].split('\x03')[1] if len(x.split('\x02'))>=7 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
df_get_chunk_trainx = df_get_chunk_trainx[df_get_chunk_trainx.eventname == 'h320057']

####################################################训练集中userid提取########################################################
df_userid_data = (df_get_chunk_trainx[['userid','eventtime']].groupby(['userid']).count()).reset_index()

####################################################训练集中userid对应uaos众数提取##############################################
df_uaos_traindata = (df_get_chunk_trainx[['userid','uaos','eventtime']].groupby(['userid','uaos']).count()).reset_index()
df_uaos_traindata = (df_uaos_traindata.sort_values(by = ['userid','uaos'] ,ascending =['asc','asc']))
df_uaos_traindata.drop_duplicates('userid', keep='last', inplace=True)

#####################################################训练集中userid对应nettype众数提取#########################################
df_net_traindata = (df_get_chunk_trainx[['userid','nettype','eventtime']].groupby(['userid','nettype']).count()).reset_index()
df_net_traindata = (df_net_traindata.sort_values(by = ['userid','nettype'] ,ascending =['asc','asc']))
df_net_traindata.drop_duplicates('userid', keep='last', inplace=True)

######################################################训练集中userid对应hzcModule、productId次数提取###########################
df_pro_traindata = (df_get_chunk_trainx[['userid', 'hzcModule', 'productId', 'eventtime']].groupby(['userid', 'hzcModule' , 'productId']).count()).reset_index()
df_userid_traindata = df_userid_data

#############################################################################################################################
######################################################得到整体userid与productId关联###########################################
#############################################################################################################################
productId_ = list(np.unique(df_pro_traindata['productId']))
df_userid_traindata_userid = pd.DataFrame(list(df_userid_traindata['userid'])*len(productId_),columns = ['userid']).sort_values(by=['userid']).reset_index(drop =True)
df_product_traindata = (productId_ * len(df_userid_traindata['userid']))
df_userid_traindata_ = pd.concat((df_userid_traindata_userid, pd.DataFrame(df_product_traindata)), axis = 1)
df_userid_traindata_.columns = ['userid','productId']

del df_userid_traindata
del df_userid_data
del df_userid_traindata_userid
del df_product_traindata
gc.collect()

##############################################################################################################################
###################################################整体userid与productId关联表与训练集相应特征以及y值关联########################
#############################################################################################################################
df_userid_traindata_ = pd.merge(df_userid_traindata_, df_pro_traindata ,on = ['userid','productId'], how = 'left')
df_userid_traindata_ = df_userid_traindata_[['userid','productId','eventtime','hzcModule']]
df_userid_traindata_.columns = ['userid','productId','productId_cnt','hzcModule']
del df_pro_traindata
gc.collect()

df_userid_traindata_ = pd.merge(df_userid_traindata_, df_uaos_traindata ,on = 'userid', how = 'left' )
df_userid_traindata_ = df_userid_traindata_[['userid','productId','productId_cnt','uaos','hzcModule']]
del df_uaos_traindata
gc.collect()

df_userid_traindata_ = pd.merge(df_userid_traindata_, df_net_traindata ,on = 'userid', how = 'left' )
df_userid_traindata_ = df_userid_traindata_[['userid','productId','productId_cnt','uaos','nettype','hzcModule']]
del df_net_traindata
gc.collect()

df_userid_traindata_all = pd.merge(df_userid_traindata_, df_pro_train_y ,on = ['userid','productId'], how = 'left' )
df_userid_traindata_all = df_userid_traindata_all[['userid','productId','productId_cnt','uaos','nettype','hzcModule','eventtime']]
df_userid_traindata_all.columns = ['userid','productId','productId_cnt','uaos','nettype','hzcModule','productId_isin']
del df_pro_train_y
del df_userid_traindata_
gc.collect()

##############################################################################################################################
########################################################训练集数据缺失值处理###################################################
#############################################################################################################################
df_userid_traindata_all['productId_isin'] = df_userid_traindata_all['productId_isin'].fillna(0)
df_userid_traindata_all['productId_cnt'] = df_userid_traindata_all['productId_cnt'].fillna(0)
df_userid_traindata_all['hzcModule'] = df_userid_traindata_all['hzcModule'].fillna('hzcModule_null')
df_userid_traindata_all['uaos'] = df_userid_traindata_all['uaos'].fillna('uaos_null')
df_userid_traindata_all['nettype'] = df_userid_traindata_all['nettype'].fillna('nettype_null')
df_userid_traindata_all['lable'] = df_userid_traindata_all['productId_isin'].apply(lambda x: 1 if x>0 else 0)
df_userid_traindata_all_y = df_userid_traindata_all[df_userid_traindata_all['lable'] == 1]
len_y = len(df_userid_traindata_all_y)
df_userid_traindata_all_n = df_userid_traindata_all[df_userid_traindata_all['lable'] == 0].sample(n = len_y)
df_userid_traindata_all = pd.concat([df_userid_traindata_all_y, df_userid_traindata_all_n])

del df_userid_traindata_all_y
del df_userid_traindata_all_n 
gc.collect()

##############################################################################################################################
########################################################训练集数据离散型进行编码################################################
#############################################################################################################################
categorical_features = ['productId','uaos','nettype','hzcModule']
categorical_entities = np.unique(df_userid_traindata_all[categorical_features].values
        .reshape([-1]))    
categorical_entities = categorical_entities.tolist()

encoder = {}
decoder = {}
for i, e in enumerate(categorical_entities):
    encoder[e] = i+1
    decoder[i] = e 

data_ = df_userid_traindata_all[categorical_features].values
for entity in categorical_entities:
    data_ = np.where(data_ == entity, encoder[entity], data_)
df_userid_traindata_all[categorical_features] = data_

##############################################################################################################################
###########################################################训练集数据划分######################################################
#############################################################################################################################
user_unique = np.unique(df_userid_traindata_all['userid'])
random.shuffle(user_unique)
len_user = len(user_unique)
test_ratio = 0.2 
val_ratio = 0.2
train_test_user = user_unique[:int(len_user*(1-test_ratio))]
test_user = user_unique[int(len_user*(1-test_ratio)):]
len_train_user = len(train_test_user)
train_user = train_test_user[:int(len_train_user*(1-val_ratio))]
val_user = train_test_user[int(len_train_user*(1-val_ratio)):]

X_train = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in train_user])][['productId','productId_cnt','uaos','nettype','hzcModule']]
y_train = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in train_user])]['lable']

X_val = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in val_user])][['productId','productId_cnt','uaos','nettype','hzcModule']]
y_val = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in val_user])]['lable']

X_test = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in test_user])][['productId','productId_cnt','uaos','nettype','hzcModule']]
y_test = df_userid_traindata_all[df_userid_traindata_all.userid.isin([i for i in test_user])]['lable']

lgb_train = lgb.Dataset(X_train, y_train)
lgb_val = lgb.Dataset(X_val, y_val, reference = lgb_train)

##############################################################################################################################
###########################################################训练模型###########################################################
#############################################################################################################################
def train_data(train_data, valid_data):
    params={
    "objective": "binary",
    "metric": "auc",
    "is_unbalance": True,
    "learning_rate": 0.075,
    "bagging_fraction": 0.6,
    "bagging_freq": 5,
    "lambda_2": 0.1,
    "nthread": 8,
    "verbosity": 1,
    "num_iterations": 600,
    "num_leaves": 64,
    "min_data_in_leaf": 20
    }

    m_lgb = lgb.train(
        params,                    #参数字典
        train_data,                #训练集
        valid_sets = [train_data, valid_data], #验证集
        verbose_eval = 10          #迭代多少次打印
        )
    return m_lgb

m_lgb = train_data(lgb_train, lgb_val)
print ('time:{}'.format(datetime.now()-s))

##############################################################################################################################
#######################################################测试集数据读取###########################################################
#############################################################################################################################
df_test = pd.read_csv(r'/home/kesci/input/test5469/000000_0', sep='\001',header=None)
df_test.columns = df_columns

df_get_chunk_trainx = df_test[df_test['date'] < 20200421]
df_get_chunk_trainx['hzcModule'] = df_get_chunk_trainx['eventbody'].apply(lambda x: x.split('\x02')[3].split('\x03')[1] if len(x.split('\x02'))>=4 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
df_get_chunk_trainx['productId'] = df_get_chunk_trainx['eventbody'].apply(lambda x: x.split('\x02')[6].split('\x03')[1] if len(x.split('\x02'))>=7 and len(x.split('\x02')[3].split('\x03'))>=2 else 0)
#df_get_chunk_trainx = df_get_chunk_trainx[df_get_chunk_trainx.eventname == 'h320057']

##############################################################################################################################
###############################################测试集数据userid,uaos,nettype,hzcModule,productId次数等特征提取##################
#############################################################################################################################
df_userid_data = (df_get_chunk_trainx[['userid','eventtime']].groupby(['userid']).count()).reset_index()

df_uaos_traindata = (df_get_chunk_trainx[['userid','uaos','eventtime']].groupby(['userid','uaos']).count()).reset_index()
df_uaos_traindata = (df_uaos_traindata.sort_values(by = ['userid','uaos'] ,ascending =['asc','asc']))
df_uaos_traindata.drop_duplicates('userid', keep='last', inplace=True)
del df_test
gc.collect()

df_net_traindata = (df_get_chunk_trainx[['userid','nettype','eventtime']].groupby(['userid','nettype']).count()).reset_index()
df_net_traindata = (df_net_traindata.sort_values(by = ['userid','nettype'] ,ascending =['asc','asc']))
df_net_traindata.drop_duplicates('userid', keep='last', inplace=True)

df_get_chunk_trainx = df_get_chunk_trainx[df_get_chunk_trainx.eventname == 'h320057']
df_pro_traindata = (df_get_chunk_trainx[['userid', 'hzcModule', 'productId', 'eventtime']].groupby(['userid', 'hzcModule' , 'productId']).count()).reset_index()
df_userid_traindata = df_userid_data

##############################################################################################################################
#######################################################测试集整体userid与productId关联#########################################
#############################################################################################################################
productId_ = list(set(productId_).union(set(np.unique(df_pro_traindata['productId']))))
df_userid_traindata_userid = pd.DataFrame(list(df_userid_traindata['userid'])*len(productId_),columns = ['userid']).sort_values(by=['userid']).reset_index(drop =True)
df_product_traindata = (productId_ * len(df_userid_traindata['userid']))
df_userid_traindata_ = pd.concat((df_userid_traindata_userid, pd.DataFrame(df_product_traindata)), axis = 1)
df_userid_traindata_.columns = ['userid','productId']

del df_userid_traindata
del df_userid_data
del df_userid_traindata_userid
del df_product_traindata
gc.collect()

##############################################################################################################################
###################################################测试集整体userid与productId关联表与测试集相应特征以及y值关联##################
#############################################################################################################################
df_userid_traindata_ = pd.merge(df_userid_traindata_, df_pro_traindata ,on = ['userid','productId'], how = 'left')
df_userid_traindata_ = df_userid_traindata_[['userid','productId','eventtime','hzcModule']]
df_userid_traindata_.columns = ['userid','productId','productId_cnt','hzcModule']
del df_pro_traindata
gc.collect()

df_userid_traindata_ = pd.merge(df_userid_traindata_, df_uaos_traindata ,on = 'userid', how = 'left' )
df_userid_traindata_ = df_userid_traindata_[['userid','productId','productId_cnt','uaos','hzcModule']]
del df_uaos_traindata
gc.collect()

df_userid_traindata_ = pd.merge(df_userid_traindata_, df_net_traindata ,on = 'userid', how = 'left' )
df_userid_traindata_ = df_userid_traindata_[['userid','productId','productId_cnt','uaos','nettype','hzcModule']]
del df_net_traindata
gc.collect()

##############################################################################################################################
########################################################测试集数据缺失值处理###################################################
#############################################################################################################################
df_userid_traindata_['productId_cnt'] = df_userid_traindata_['productId_cnt'].fillna(0)
df_userid_traindata_['hzcModule'] = df_userid_traindata_['hzcModule'].fillna('hzcModule_null')
df_userid_traindata_['uaos'] = df_userid_traindata_['uaos'].fillna('uaos_null')
df_userid_traindata_['nettype'] = df_userid_traindata_['nettype'].fillna('nettype_null')

##############################################################################################################################
########################################################根据训练集编码字典，测试集数据离散型进行编码#############################
#############################################################################################################################
categorical_entities_test = np.unique(df_userid_traindata_[categorical_features].values)
data_ = df_userid_traindata_[categorical_features].values
for entity in categorical_entities:
    data_ = np.where(data_ == entity, encoder[entity], data_)
df_userid_traindata_[categorical_features] = data_

categorical_entities_extra = set(categorical_entities_test)-set(categorical_entities)
num = len(encoder)
if len(categorical_entities_extra) >0:
    data_ = df_userid_traindata_[categorical_features].values
    for entity in list(categorical_entities_extra):
        data_ = np.where(data_ == entity, num, data_)        
        encoder[entity] = num +1
        decoder[num] = entity
        categorical_entities.append(entity)
        num = num +1
    df_userid_traindata_[categorical_features] = data_

##############################################################################################################################
########################################################测试集数据预测,每个userid选取前3保留####################################
#############################################################################################################################
userid_dataset = df_userid_traindata_[['userid']]
test_dataset = df_userid_traindata_[['productId','productId_cnt','uaos','nettype','hzcModule']]

pred = pd.Series(m_lgb.predict(test_dataset, num_iteration = m_lgb.best_iteration))
pred_product = pd.concat((test_dataset, pd.DataFrame(pred, columns = ['pred'])), axis = 1)[['productId','pred']]
pred_product_userid = pd.concat((pred_product, userid_dataset), axis = 1)[['userid', 'productId', 'pred']]

pred_product_userid.sort_values(['userid', 'pred'], ascending = [True, False], inplace = True)
pred_product_final = pred_product_userid.groupby(['userid']).head(3)[['userid', 'productId']]

##############################################################################################################################
########################################################测试集数据预测集，productId反编码######################################
#############################################################################################################################
categorical_pred = ['productId']
data_ = pred_product_final[categorical_pred].values
print (data_)
pred_product_final = pred_product_final.astype(str)
pred_decoder = {str(key):value for key, value in decoder.items() if value in productId_}
print ((pred_decoder))
categorical_pred_entities = pred_decoder.keys()
for entity in categorical_pred_entities:
    data_ = np.where(data_ == entity, pred_decoder[entity], data_)
pred_product_final[categorical_pred] = data_

##############################################################################################################################
########################################################测试集数据预测集，按要求格式############################################
#############################################################################################################################
pred_product_final_csv = pred_product_final.groupby('userid')['productId'].apply(list).reset_index()[['userid']]
pred_product_final_csv[['pred1', 'pred2', 'pred3']] = pd.DataFrame(list(pred_product_final.groupby('userid')['productId'].apply(list).values))

##############################################################################################################################
######################################################预测结果保存以及提交#####################################################
#############################################################################################################################
pred_product_final_csv.to_csv(r'/home/kesci/pred/pred_product_final.csv',sep=',',index=False)
!wget -nv -O kesci_submit https://cdn.kesci.com/submit_tool/v4/kesci_submit&&chmod +x kesci_submit
!./kesci_submit -token cce2d681351465f6 -file /home/kesci/pred/pred_product_final.csv


import os
os.makedirs()
