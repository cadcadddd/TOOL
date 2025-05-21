import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from pyswarm import pso
from sklearn.model_selection import cross_val_score


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = make_scorer(rmse,greater_is_better=False)

'''对数据进行统计分析，查看数据的分布情况'''
data_PL=pd.read_csv('testPL.csv',index_col=0,encoding='utf-8')
data_BT=pd.read_csv('testBT.csv',index_col=0,encoding='utf-8')
index_PL=data_PL.index
index_BT=data_BT.index
'''查看各变量间的相关系数'''
correlation_matrix_PL = data_PL.corr()
correlation_matrix_BT = data_BT.corr()

'''划分训练集和验证集'''
data_train, data_test= train_test_split(data_PL,test_size=0.2, random_state=42)
#获取训练集和验证集
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



# max_depth 的可能值
max_depth_options = [None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# 目标函数
def objective_function(params):
    n_estimators = int(params[0])
    max_features = params[1]
    max_depth_index = int(params[2])
    min_samples_split = int(params[3])
    min_samples_leaf = int(params[4])
    # 获取 max_depth
    max_depth = max_depth_options[max_depth_index]
    # 将 bootstrap 转换为布尔值
    bootstrap = bool(int(params[5] >= 0.5))  # 大于等于 0.5 为 True，否则为 False
    print("bootstrap:",bootstrap)
    # 构建模型
    clf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=max_features,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        bootstrap=bootstrap
    )

    # 使用交叉验证计算均方根误差
    scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=10,verbose=2)
    rmse = np.sqrt(-scores.mean())  # 转换为 RMSE
    return rmse

# PSO 参数设置
lb = [500, 0.1, 2, 2, 1, 0]  # 参数下界
ub = [2000, 1.0,  len(max_depth_options) - 1, 4, 4,1]  # 参数上界


# 设置惯性权重、全局吸引因子、粒子数量
w = 0.5
c1 = 2
c2 = 2
# 执行 PSO
best_params, best_rmse = pso(objective_function, lb, ub, omega=w, phip=c1, phig=c2,swarmsize=30, maxiter=100)


# 最终输出最佳 RMSE 和最佳参数位置
print ("最佳参数：",best_params)
# 使用最佳参数构建模型
best_model = RandomForestRegressor(
    n_estimators=int(best_params[0]),
    max_features=best_params[1],
    max_depth=int(best_params[2]),
    min_samples_split=int(best_params[3]),
    min_samples_leaf=int(best_params[4]),
    bootstrap= bool(int(best_params[5] >= 0.5))
)

# 训练模型
best_model.fit(X_train, y_train)
# 使用最佳模型进行预测
y_test_pred = best_model.predict(X_test)
y_train_pred = best_model.predict(X_train)
Y_BT_pred    =best_model.predict(X_BT)


#定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

# 计算训练集和测试集的RRMSE
print ("RF模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("RF模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("RF模型评估--BT方法：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_BT,Y_BT_pred)))
print("RRMSE:", relative_root_mean_squared_error(Y_BT,Y_BT_pred))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))

