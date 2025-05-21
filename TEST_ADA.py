from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
import random
from sklearn.model_selection import train_test_split, cross_val_score  # 导入 cross_val_score
from sklearn.ensemble import AdaBoostRegressor
import joblib

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = make_scorer(rmse,greater_is_better=False)  # 创建均方根误差的评分器

# 限制参数范围的函数
def constrain(value, min_val, max_val):
    return max(min_val, min(max_val, value))

'''对数据进行统计分析，查看数据的分布情况'''
data_PL=pd.read_csv('testPL.csv',index_col=0,encoding='utf-8')
data_BT=pd.read_csv('testBT.csv',index_col=0,encoding='utf-8')
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


AdaBoost_model = AdaBoostRegressor(
    base_estimator=DecisionTreeRegressor(max_depth=11, max_features=2),
    n_estimators=30,  # 使用 n_estimators
    learning_rate=0.252,  # 使用 learning_rate
    random_state=999
)

# 拟合最终模型
AdaBoost_model.fit(X_train, y_train)

# 使用最佳模型进行预测
y_test_pred = AdaBoost_model.predict(X_test)
y_train_pred = AdaBoost_model.predict(X_train)
Y_BT_pred    =AdaBoost_model.predict(X_BT)



# 计算训练集和测试集的RRMSE
print ("AdaBoost模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", rmse(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("AdaBoost模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:",rmse(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("AdaBoost模型评估--BT方法：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:",rmse(Y_BT,Y_BT_pred))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))

joblib.dump(AdaBoost_model, "AdaBoost_model.pkl")  # 推荐（压缩格式）

# 重新加载验证
loaded_model = joblib.load("AdaBoost_model.pkl")
print("模型是否已训练:", hasattr(loaded_model, 'estimators_'))  # 应输出 True
print("特征重要性:", loaded_model.feature_importances_)  # 应输出非空数组