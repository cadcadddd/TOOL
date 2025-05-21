import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score


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



'''模型调参'''
##参数选择
criterion=['absolute_error']
n_estimators =  np.arange(100,2000,5)
max_features = np.arange(0.01,1,0.01)
max_depth =  [None]+list(np.arange(1,15,1))
min_samples_split = np.arange(2,15,1)
min_samples_leaf = np.arange(1,15,1)
bootstrap= [False]

random_grid = {'criterion':criterion,
                'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
                'bootstrap':bootstrap
               }
#构建模型
scoring = make_scorer(rmse,greater_is_better=False)  # 创建均方根误差的评分器
clf= RandomForestRegressor(random_state=42)
clf_random = RandomizedSearchCV(estimator=clf, param_distributions=random_grid,
                              n_iter = 50 ,
                              cv = 10, verbose=2,  n_jobs=1,scoring=scoring)




#回归
clf_random.fit(X_train, y_train)
best_params = clf_random.best_params_
best_model = clf_random.best_estimator_
print ("最佳参数：",best_params)

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

# 创建一个新的数据框，包含预测数据
test_PL_predictions = pd.DataFrame({
    'Dataset': 'Test',
    'True Values': y_test,
    'Predicted Values': y_test_pred
})

train_PL_predictions = pd.DataFrame({
    'Dataset': 'Train',
    'True Values': y_train,
    'Predicted Values': y_train_pred
})
BT_predictions = pd.DataFrame({
    'Dataset': 'BT',
    'True Values': Y_BT,
    'Predicted Values': Y_BT_pred
})
# 将预测数据添加到验证集中
data_test_with_predictions = data_test.copy()
data_test_with_predictions['Predicted Values'] = y_test_pred

data_train_with_predictions = data_train.copy()
data_train_with_predictions['Predicted Values'] = y_train_pred

data_BT_with_predictions = data_BT.copy()
data_BT_with_predictions['Predicted Values'] = Y_BT_pred
# 导出包含预测数据的验证集到CSV文件
data_test_with_predictions.to_csv('RF0.8-test-predicted_data.csv', encoding='utf-8', index=False)
data_train_with_predictions.to_csv('RF0.8-train-predicted_data.csv', encoding='utf-8', index=False)
data_BT_with_predictions.to_csv('RF0.8-BT-predicted_data.csv', encoding='utf-8', index=False)