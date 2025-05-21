from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
import joblib


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
print(data_PL.columns)


#'''划分训练集和验证集'''
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
# 输出为 Excel 文件

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

# 使用最佳参数训练最终模型
KNN_model = KNeighborsRegressor(
    n_neighbors=2,
    leaf_size=4*2,
    weights='distance'
)

KNN_model.fit(X_train_scaled, y_train_scaled)
# 使用最佳模型进行预测

# 使用最佳模型进行预测
y_test_pred_scaled = KNN_model.predict(X_test_scaled)  # 确保使用带有特征名称的 DataFrame
y_train_pred_scaled = KNN_model.predict(X_train_scaled)
Y_BT_pred_scaled = KNN_model.predict(X_BT_scaled)

# 将 y_test_pred_scaled 转换为二维数组
y_test_pred_scaled_reshaped = y_test_pred_scaled.reshape(-1, 1)
y_train_pred_scaled_reshaped = y_train_pred_scaled.reshape(-1, 1)
Y_BT_pred_scaled_reshaped = Y_BT_pred_scaled.reshape(-1, 1)
# 逆转化
y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled_reshaped).ravel()
y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled_reshaped).ravel()
Y_BT_pred = scaler_y.inverse_transform(Y_BT_pred_scaled_reshaped).ravel()

#定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

# 计算训练集和测试集的RRMSE
print ("KNN模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("KNN模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("KNN模型评估--BT方法：")
print ('R^2:',r2_score(Y_BT,Y_BT_pred))
print ('MSE',mean_squared_error(Y_BT,Y_BT_pred))
print("RMSE:", np.sqrt(mean_squared_error(Y_BT,Y_BT_pred)))
print("RRMSE:", relative_root_mean_squared_error(Y_BT,Y_BT_pred))
print ('MAE',mean_absolute_error(Y_BT,Y_BT_pred))

joblib.dump(KNN_model, "KNN_model.pkl")  # 推荐（压缩格式）