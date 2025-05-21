import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import  mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

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
X_test=data_test.iloc[:,0:-1]
feature=data_train.iloc[:,0:-1].columns
y_train=data_train.iloc[:,-1]
y_test=data_test.iloc[:,-1]
#读取BT数据
X_BT=data_BT.iloc[:,0:-1]
Y_BT=data_BT.iloc[:,-1]

# 对输入特征进行归一化
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
# 只在训练集上拟合
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # 训练集拟合
#对测试集和其他数据进行转换
X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.values.reshape(-1, 1)).ravel()
X_BT_scaled=scaler_X.transform(X_BT)

# 创建MLPRegressor对象
mlp = MLPRegressor( random_state=42)

# 创建GridSearchCV对象，并指定参数
param_grid = {
    'hidden_layer_sizes': [(12,12),(11,11),(10,10),(9,9),(8,8),(7,7)],  # 隐藏层神经元数量的不同组合
    'activation': ['tanh'],  # 激活函数的不同选择
    'solver': ['sgd'],  # 优化算法的不同选择
    'max_iter':np.arange(200,2500,10),  # 迭代次数的不同选择
    'learning_rate_init': np.arange(0.01,0.3,0.01)
}
scoring = make_scorer(rmse,greater_is_better=False)  # 创建均方根误差的评分器
grid_search = GridSearchCV(mlp, param_grid=param_grid, cv=10, scoring=scoring, verbose=2)

# 在训练数据上执行网格搜索
grid_search.fit(X_train_scaled, y_train_scaled)

# 获取最佳参数组合和对应模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_
print("最佳参数组合：", best_params)


# 使用最佳模型进行预测
y_test_pred_scaled = best_model.predict(X_test_scaled).ravel()
y_train_pred_scaled = best_model.predict(X_train_scaled).ravel()
Y_BT_pred_scaled    =best_model.predict(X_BT_scaled).ravel()

# 将 y_test_pred_scaled 转换为二维数组
y_test_pred_scaled_reshaped = y_test_pred_scaled.reshape(-1, 1)
y_train_pred_scaled_reshaped = y_train_pred_scaled.reshape(-1, 1)
Y_BT_pred_scaled_reshaped=Y_BT_pred_scaled.reshape(-1, 1)
# 逆转化
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled_reshaped).ravel()
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled_reshaped).ravel()
Y_BT_pred=scaler_y.inverse_transform(Y_BT_pred_scaled_reshaped).ravel()

#定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


# 计算训练集和测试集的RRMSE
print ("MLP模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("MLP模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("MLP模型评估--BT方法：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_BT,Y_BT_pred)))
print("RRMSE:", relative_root_mean_squared_error(Y_BT,Y_BT_pred))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))

