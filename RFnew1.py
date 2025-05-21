import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
import random

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
scoring = make_scorer(rmse,greater_is_better=False)  # 创建均方根误差的评分器

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


n = 5  # Number of parameters (like hyperparameters)
p =300  # Number of Population
m = 10  # Maximum value for random parameters
mr = 0.7  # Mutation rate
epochs =150  # Number of generations

def randomGeneration(NumberOfRows, NumberOfQueens, m):
    generation_list = []
    for i in range(NumberOfRows):
        gene = [
            random.randint(1, m),    # n_estimators
            random.uniform(1, m),  # max_features
            random.randint(1, m),    # max_depth
            random.randint(1, m),    #min_samples_split
            random.randint(1, m),    #min_samples_leaf
        ]
        generation_list.append(gene)
    return generation_list

def cross_over(generation_list, p, n):
    if p % 2 != 0:
        p -= 1  # 如果是奇数，减少一个个体

    for i in range(0, p, 2):
        child1 = generation_list[i][:n // 2] + generation_list[i + 1][n // 2:n]
        child2 = generation_list[i + 1][:n // 2] + generation_list[i][n // 2:n]
        generation_list.append(child1)
        generation_list.append(child2)

    return generation_list

def mutation(generation_list, p, n, m, mr):
    chosen_ones = list(range(p, len(generation_list)))
    random.shuffle(chosen_ones)
    chosen_ones = chosen_ones[:int(p * mr)]

    for i in chosen_ones:
        cell = random.randint(0, n - 1)
        val = random.randint(1, m)
        generation_list[i][cell] = val
    return generation_list

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def fitness(population_list):
    fitness_results = []
    for individual in population_list:
        try:
            model = RandomForestRegressor(
                n_estimators=individual[0] * 100,               # 设置树的数量
                max_features=individual[1] / 10,              # 学习率
                max_depth=individual[2] + 1,                    # 最大深度
                min_samples_split=individual[3]+1 ,                   # 子样本比例
                min_samples_leaf=individual[4] ,           # 每棵树的特征比例
                random_state=999                                # 随机状态
            )

            model.fit(X_train, y_train)                       # 拟合模型
            y_pred = model.predict(X_test)                    # 进行预测

            rmse_value = rmse(y_test, y_pred)                 # 计算 RMSE
            fitness_results.append({
                'params': individual,                           # 保存参数
                'rmse': rmse_value                              # 保存 RMSE
            })
        except Exception as e:
            print(f'Error evaluating parameters {individual}: {e}')
            fitness_results.append({'params': individual, 'rmse': None})

    return fitness_results

# 主循环
population = randomGeneration(p, n, m)
best_params = None
no_improvement_count = 0  # 连续无改进计数

for epoch in range(epochs):
    print(f'Epoch {epoch + 1}: Population size = {len(population)}')
    population = cross_over(population, p, n)
    population = mutation(population, p, n, m, mr)

    fitness_results = fitness(population)
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
                f"Best parameters after epoch {epoch + 1}: n_estimators={current_best[0]}, max_features={current_best[1]}, max_depth={current_best[2]},min_samples_split={current_best[3]},min_samples_leaf={current_best[4]}")
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
    final_model = RandomForestRegressor(
        n_estimators=best_params['params'][0] * 100,  # 设置树的数量
        max_features=best_params['params'][1] / 10,  # 学习率
        max_depth=best_params['params'][2] + 1,  # 最大深度
        min_samples_split=best_params['params'][3] + 1,  # 子样本比例
        min_samples_leaf=best_params['params'][4],  # 每棵树的特征比例
        random_state=999
    )

    # 拟合最终模型
    final_model.fit(X_train, y_train)

else:
    print("No valid parameters found after optimization.")

# 使用最佳模型进行预测
y_test_pred = final_model.predict(X_test)
y_train_pred = final_model.predict(X_train)
Y_BT_pred    =final_model.predict(X_BT)



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

# 将预测数据添加到验证集中
data_test_with_predictions = data_test.copy()
data_test_with_predictions['Predicted Values'] = y_test_pred

data_train_with_predictions = data_train.copy()
data_train_with_predictions['Predicted Values'] = y_train_pred

data_BT_with_predictions = data_BT.copy()
data_BT_with_predictions['Predicted Values'] = Y_BT_pred

# 导出到 Excel 文件
with pd.ExcelWriter(r'D:\TemporaryDirectory\RF-DH.xlsx') as writer:
    data_test_with_predictions.to_excel(writer, sheet_name='Test Predictions', index=False)
    data_train_with_predictions.to_excel(writer, sheet_name='Train Predictions', index=False)
    data_BT_with_predictions.to_excel(writer, sheet_name='BT Predictions', index=False)


print("预测数据已成功输出到 Excel 文件。")