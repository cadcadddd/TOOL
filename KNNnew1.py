from sklearn.metrics import r2_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import make_scorer
import random
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
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


# 遗传算法参数
n = 2  # 参数个数
p = 500  # 种群数量
m = 100  # 随机参数的最大值
mr = 0.7  # 变异率
epochs = 200  # 代数
k_folds = 5  # 交叉验证折数

def randomGeneration(NumberOfRows, m):
    generation_list = []

    for _ in range(NumberOfRows):
        gene = [
            random.randint(1, m),  # n_neighbors
            random.randint(1, m),  # leaf_size
        ]
        generation_list.append(gene)

    return generation_list


def cross_over(generation_list, p,n):
    if p % 2 != 0:
        p -= 1  # 如果是奇数，减少一个个体

    for i in range(0, p, 2):
        child1 = generation_list[i][:n // 2] + generation_list[i + 1][n // 2:]
        child2 = generation_list[i + 1][:n // 2] + generation_list[i][n // 2:]
        generation_list.append(child1)
        generation_list.append(child2)

    return generation_list


def mutation(generation_list, p, n, m, mr):
    chosen_ones = list(range(p, len(generation_list)))
    random.shuffle(chosen_ones)
    chosen_ones = chosen_ones[:int(p * mr)]

    for i in chosen_ones:
        cell = random.randint(0, n - 1)
        val = random.uniform(1, m)  # 保持与原范围一致
        generation_list[i][cell] = val
    return generation_list


def fitness(population_list, X, y):
    fitness_results = []
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=999)

    for individual in population_list:
        try:
            rmse_values = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                model = KNeighborsRegressor(
                    n_neighbors=int(individual[0]),
                    leaf_size=int(individual[1]) * 2,
                    weights='distance'
                )
                model.fit(X_train, y_train)
                y_pred_scaled = model.predict(X_test)
                y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
                #y——test逆归一化
                y_test=scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
                # 计算当前折的 RMSE
                rmse_value = np.sqrt(mean_squared_error(y_test, y_pred))
                rmse_values.append(rmse_value)  # 添加 RMSE 到列表

            # 计算平均 RMSE
            mean_rmse = np.mean(rmse_values)

            fitness_results.append({
                'params': individual,
                'rmse': mean_rmse  # 使用平均 RMSE
            })
        except Exception as e:
            print(f'Error evaluating parameters {individual}: {e}')
            fitness_results.append({'params': individual, 'rmse': None})

    return fitness_results


# 主循环
population = randomGeneration(p,  m)
best_params = None
no_improvement_count = 0  # 连续无改进计数

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}: Population size = {len(population)}')
    population = cross_over(population, p, n)
    population = mutation(population, p, n, m, mr)

    fitness_results = fitness(population, X_train_scaled, y_train_scaled)
    valid_results = [result for result in fitness_results if result['rmse'] is not None]

    if valid_results:
        valid_results.sort(key=lambda x: x['rmse'])

        # 更新种群
        population = [result['params'] for result in valid_results[:p]]

        # 保存最佳超参数
        if best_params is None or valid_results[0]['rmse'] < best_params['rmse']:
            best_params = {
                'params': valid_results[0]['params'],
                'rmse': valid_results[0]['rmse']
            }
            # 打印当前最佳参数
            current_best = best_params['params']
            print(
                f"Best parameters after epoch {epoch + 1}: n_neighbors={current_best[0]}, leaf_size={current_best[1]}")
            print(f"Best RMSE: {best_params['rmse']}")
            no_improvement_count = 0  # 重置无改进计数
        else:
            no_improvement_count += 1
            print("No improvement found in this epoch.")
            print("最大无改进次数：",no_improvement_count)
    else:
        print("No valid results found, continuing to next epoch.")

# 检查 valid_results 是否为空
if best_params:
    print('Best hyperparameters found:')
    print(best_params['params'])
    print(f'Best RMSE: {best_params["rmse"]}')

    # 使用最佳参数训练最终模型
    final_model = KNeighborsRegressor(
        n_neighbors=best_params['params'][0] ,
        leaf_size=best_params['params'][1]*2 ,
        weights='distance'
    )

    # 拟合最终模型
    final_model.fit(X_train_scaled, y_train_scaled)

    # 使用最佳模型进行预测
    y_test_pred_scaled = final_model.predict(X_test_scaled)  # 确保使用带有特征名称的 DataFrame
    y_train_pred_scaled = final_model.predict(X_train_scaled)
    Y_BT_pred_scaled = final_model.predict(X_BT_scaled)

    # 将 y_test_pred_scaled 转换为二维数组
    y_test_pred_scaled_reshaped = y_test_pred_scaled.reshape(-1, 1)
    y_train_pred_scaled_reshaped = y_train_pred_scaled.reshape(-1, 1)
    Y_BT_pred_scaled_reshaped = Y_BT_pred_scaled.reshape(-1, 1)
    # 逆转化
    y_test_pred = scaler_y.inverse_transform(y_test_pred_scaled_reshaped).ravel()
    y_train_pred = scaler_y.inverse_transform(y_train_pred_scaled_reshaped).ravel()
    Y_BT_pred = scaler_y.inverse_transform(Y_BT_pred_scaled_reshaped).ravel()


#定义rmse、rrmse
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


# 将预测数据添加到验证集中
data_test_with_predictions = data_test.copy()
data_test_with_predictions['Predicted Values'] = y_test_pred

data_train_with_predictions = data_train.copy()
data_train_with_predictions['Predicted Values'] = y_train_pred

data_BT_with_predictions = data_BT.copy()
data_BT_with_predictions['Predicted Values'] = Y_BT_pred

# 导出到 Excel 文件
with pd.ExcelWriter(r'D:\TemporaryDirectory\KNN-DH.xlsx') as writer:
    data_test_with_predictions.to_excel(writer, sheet_name='Test Predictions', index=False)
    data_train_with_predictions.to_excel(writer, sheet_name='Train Predictions', index=False)
    data_BT_with_predictions.to_excel(writer, sheet_name='BT Predictions', index=False)


print("预测数据已成功输出到 Excel 文件。")