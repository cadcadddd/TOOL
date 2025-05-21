import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from sklearn.metrics import make_scorer
import numpy as np


'''对数据进行统计分析，查看数据的分布情况'''
data_PL = pd.read_csv(r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')
data_BT=pd.read_csv(r'D:\TemporaryDirectory\testBT.csv',index_col=0,encoding='utf-8')
index_PL=data_PL.index
index_BT=data_BT.index
'''查看各变量间的相关系数'''
correlation_matrix_PL = data_PL.corr()
correlation_matrix_BT = data_BT.corr()

'''划分训练集和验证集'''
data_train, data_test= train_test_split(data_PL,test_size=0.2, random_state=999)
#获取训练集和验证集
X_background=data_PL.iloc[:,0:-1]
X_train=data_train.iloc[:,0:-1]
X_test =data_test.iloc[:,0:-1]
feature_train=data_train.iloc[:,0:-1].columns
feature_test=data_test.iloc[:,0:-1].columns
feature_BT =data_BT.iloc[:,0:-1].columns
y_train=data_train.iloc[:,-1]
y_test=data_test.iloc[:,-1]
#读取BT数据
X_BT=data_BT.iloc[:,0:-1]
Y_BT=data_BT.iloc[:,-1]


# 使用最佳参数训练最终模型
final_model = xgb.XGBRegressor(
    max_depth=5,
    colsample_bytree=0.875,
    colsample_bylevel=0.4,
    colsample_bynode=0.152,
    n_estimators=180,
    learning_rate=0.400,
    subsample=0.5,
    random_state=999
)
# 拟合最终模型
final_model.fit(X_train, y_train)

# 使用最佳模型进行预测
y_test_pred = final_model.predict(X_test)
y_train_pred = final_model.predict(X_train)
Y_BT_pred    =final_model.predict(X_BT)


print ("XGBoost模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))

print ("XGBoost模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))

