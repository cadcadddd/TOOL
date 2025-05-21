from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer

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

# 保存原始数据的副本





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


#参数空间
param_grid = {
    'n_neighbors': [2,3, 4, 5,6,7,8,9,10],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'leaf_size': [5,10,15,20,40,30, 25, 35]
}

# 创建KNN回归模型
knn = KNeighborsRegressor()
scoring = make_scorer(rmse,greater_is_better =False)  # 创建均方根误差的评分器
# 创建GridSearchCV对象
grid_search = GridSearchCV(knn, param_grid, cv=10, scoring=scoring)

# 执行网格搜索
grid_search.fit(X_train_scaled, y_train_scaled)

# 获取最佳超参数和模型
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 在训练集和验证集上进行预测
print("最佳参数： ", grid_search.best_params_)

# 使用最佳模型进行预测
y_test_pred_scaled = best_model.predict(X_test_scaled).ravel()
y_train_pred_scaled = best_model.predict(X_train_scaled).ravel()
Y_BT_pred_scaled    =best_model.predict(X_BT_scaled).ravel()







data_PL_original = data_PL.copy()
# 初始化一个 DataFrame 来存储噪声
noise_df = pd.DataFrame(index=data_PL.index, columns=data_PL.columns)
# 设置每个特征的均值和标准差
noise_params = {
    'lf/df': {'mean': 0, 'std_dev': 0.85},  # 第一个特征
    'lf(mm)': {'mean': 0, 'std_dev': 0.583333333333},  # 第二个特征
    'Vf(%)': {'mean': 0, 'std_dev': 0.011333333333},  # 第三个特征
    'fc(MPa)': {'mean': 0, 'std_dev': 0.73},  # 第四个特征
    'd(mm)': {'mean': 0, 'std_dev': 0.233333333333},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.075},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.085} , # 第七个特征
    'τu,exp(MPa)': {'mean': 0, 'std_dev': 0.2588333333}  # 输出
}

# 为每个特征生成噪声并添加到特征中
for i, feature in enumerate(data_PL.columns):
    mean = noise_params[feature]['mean']
    std_dev = noise_params[feature]['std_dev']
    # 生成高斯噪声
    noise = np.random.normal(mean, std_dev, data_PL.iloc[:, i].shape)
    # 仅对非零值添加噪声
    mask = data_PL.iloc[:, i] != 0
    # 添加噪声到特征并记录噪声到 noise_df
    data_PL.iloc[:, i] += noise * mask  # 仅在 mask 为 True 的位置添加噪声
    noise_df[feature] = noise * mask  # 仅记录添加到特征中的噪声

# 如果需要，可以将加噪声的数据转换回DataFrame格式
data_PL_noisy = data_PL.copy()  # 使用 copy 方法确保数据完整
with pd.ExcelWriter(r'D:\TemporaryDirectory\NOISY\NOISY.1.xlsx') as writer:
    data_PL_original.to_excel(writer, sheet_name='Original Data')
    data_PL.to_excel(writer, sheet_name='Noisy Data')
    noise_df.to_excel(writer, sheet_name='Noise Added')
print("数据已成功输出到文件中。")






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

mse_train=mean_squared_error(y_train,y_train_pred)
mse_test=mean_squared_error(y_test,y_test_pred)
rmse_train =np.sqrt(mse_train)
rrmse_train = relative_root_mean_squared_error(y_train,y_train_pred)
rmse_test = np.sqrt(mse_test)
rrmse_test = relative_root_mean_squared_error(y_test,y_test_pred)

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


# 将预测数据添加到验证集中
data_test_with_predictions = data_test.copy()
data_test_with_predictions['Predicted Values'] = y_test_pred

data_train_with_predictions = data_train.copy()
data_train_with_predictions['Predicted Values'] = y_train_pred

data_BT_with_predictions = data_BT.copy()
data_BT_with_predictions['Predicted Values'] = Y_BT_pred

# 导出到 Excel 文件
with pd.ExcelWriter(r'D:\TemporaryDirectory\NOISY\Predictions_KNN.xlsx') as writer:
    data_test_with_predictions.to_excel(writer, sheet_name='Test Predictions', index=False)
    data_train_with_predictions.to_excel(writer, sheet_name='Train Predictions', index=False)
    data_BT_with_predictions.to_excel(writer, sheet_name='BT Predictions', index=False)


print("预测数据已成功输出到 Excel 文件。")