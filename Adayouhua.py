from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor
from deap import base, creator, tools, algorithms
import random
from sklearn.model_selection import train_test_split, cross_val_score  # 导入 cross_val_score
import pygad

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


# 定义适应度函数
def fitness_func(solution, solution_idx):
    n_estimators = int(solution[0])  # 第一个基因
    max_depth = int(solution[1])  # 第二个基因
    learning_rate = solution[2]  # 第三个基因

    base_estimator = DecisionTreeClassifier(max_depth=max_depth)
    model = AdaBoostClassifier(base_estimator=base_estimator,
                               n_estimators=n_estimators,
                               learning_rate=learning_rate)

    # 使用交叉验证评估模型性能
    scores = cross_val_score(model, X_train, y_train, cv=10)
    return np.mean(scores)  # 返回平均得分作为适应度


# 设置遗传算法参数
num_generations = 50
num_parents_mating = 10

# 创建遗传算法对象
ga = pygad.GA(num_generations=num_generations,
              num_parents_mating=num_parents_mating,
              fitness_func=fitness_func,
              sol_per_pop=20,
              num_genes=3,  # 三个基因：n_estimators, max_depth, learning_rate
              gene_type=[int, int, float],  # 类型分别为整数和浮点数
              gene_range=[(50, 200), (1, 10), (0.01, 1.0)],  # 基因范围
              parent_selection_type="tournament",
              keep_parents=1,
              crossover_probability=0.5,
              mutation_probability=0.7)

# 运行遗传算法
ga.run()

# 获取最佳解
solution, solution_fitness, solution_idx = ga.best_solution()
print("最佳解:", solution)
print("适应度:", solution_fitness)

# 使用最佳参数训练最终模型
best_n_estimators = int(solution[0])
best_max_depth = int(solution[1])
best_learning_rate = solution[2]

final_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=best_max_depth),
    n_estimators=best_n_estimators,
    learning_rate=best_learning_rate
)

# 在测试集上评估最终模型
final_model.fit(X_train, y_train)

# 使用最佳模型进行预测
y_test_pred =final_model.predict(X_test)
y_train_pred =final_model.predict(X_train)
Y_BT_pred    =final_model.predict(X_BT)


#定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

# 计算训练集和测试集的RRMSE
print ("AdaBoost模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("AdaBoost模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("AdaBoost模型评估--BT方法：")
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
with pd.ExcelWriter(r'D:\TemporaryDirectory\Ada-DH.xlsx') as writer:
    data_test_with_predictions.to_excel(writer, sheet_name='Test Predictions', index=False)
    data_train_with_predictions.to_excel(writer, sheet_name='Train Predictions', index=False)
    data_BT_with_predictions.to_excel(writer, sheet_name='BT Predictions', index=False)


print("预测数据已成功输出到 Excel 文件。")