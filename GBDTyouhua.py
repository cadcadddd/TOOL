from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import shap
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score
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


# 限制参数范围的函数
def constrain(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def evaluate(params):
    learning_rate = constrain(params[0], 0.01, 1)
    n_estimators = int(constrain(params[1], 300, 1500))# 确保是整数
    max_depth = int(constrain(params[2], 1, 15))# 确保是整数
    subsample = constrain(params[3], 0.01, 1.0)
    min_samples_split = int(constrain(params[4], 2, 15))  # 确保是整数
    min_samples_leaf = int(constrain(params[5], 1, 15))# 确保是整数
    max_features = constrain(params[6], 0.01, 1.0)

    model = GradientBoostingRegressor(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        max_depth=max_depth,
        subsample=subsample,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=999
    )
    scores = cross_val_score(model, X_train, y_train, cv=10, scoring=scoring)
    return -scores.mean(),

# 自定义变异操作
def custom_mutate(individual):
    for i in range(len(individual)):
        if random.random() < 0.7:
            if i == 0:  # learning_rate
                individual[i] = random.uniform(0.01, 1)
            elif i == 1:  # n_estimators
                individual[i] = random.randint(300, 1500)# 整数
            elif i == 2:  # max_depth
                individual[i] = random.randint(1, 15)# 整数
            elif i == 3:  # subsample
                individual[i] = random.uniform(0.01, 1.0)
            elif i == 4:  # min_samples_split
                individual[i] = random.randint(2, 15)  # 整数
            elif i == 5:  # min_samples_leaf
                individual[i] = random.randint(1, 15)# 整数
            elif i == 6:  # max_features
                individual[i] = random.uniform(0.01, 1.0)
    return individual,

# 定义遗传算法的基础设置
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("learning_rate", random.uniform, 0.01, 1)
toolbox.register("n_estimators", random.randint, 300, 1500)
toolbox.register("max_depth", random.randint, 1, 15)
toolbox.register("subsample", random.uniform, 0.01, 1.0)
toolbox.register("min_samples_split", random.randint, 2, 15)
toolbox.register("min_samples_leaf", random.randint, 1, 15)
toolbox.register("max_features", random.uniform, 0.01, 1.0)

# 创建个体和种群
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.learning_rate, toolbox.n_estimators, toolbox.max_depth,
                  toolbox.subsample, toolbox.min_samples_split, toolbox.min_samples_leaf,
                  toolbox.max_features), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# 注册评估函数和变异操作
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", custom_mutate)
toolbox.register("select", tools.selTournament, tournsize=3)


# 遗传算法主过程
def main():
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)

    for gen in range(10):  # 迭代10代
        # 评估所有个体的适应度
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # 选择
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # 交叉和变异
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:  # 50%的交叉概率
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            toolbox.mutate(mutant)
            del mutant.fitness.values
        # 评估所有个体的适应度
        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # 替换种群
        pop[:] = offspring

        # 更新最优个体
        hof.update(pop)

    return hof[0]


# 运行遗传算法
best_params = main()
print("最佳参数: ", best_params)

# 使用最佳参数训练最终模型
best_model = GradientBoostingRegressor(
    learning_rate=constrain(best_params[0], 0.01, 0.3),
    n_estimators=int(best_params[1]),
    max_depth=int(best_params[2]),
    min_samples_split=int(best_params[4]),
    min_samples_leaf=int(best_params[5]),
    subsample=constrain(best_params[3], 0.1, 1.0),
    max_features=constrain(best_params[6], 0.1, 1.0),  # 使用最佳的 max_features
    random_state=999
)

# 训练最终模型
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
print ("gbdt模型评估--PL训练集：")
print ('R^2:',r2_score(y_train,y_train_pred))
print ('MSE',mean_squared_error(y_train,y_train_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_train,y_train_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_train,y_train_pred))
print ('MAE',mean_absolute_error(y_train,y_train_pred))

print ("gbdt模型评估--PL验证集：")
print ('R^2:',r2_score(y_test,y_test_pred))
print ('MSE',mean_squared_error(y_test,y_test_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test,y_test_pred)))
print("RRMSE:", relative_root_mean_squared_error(y_test,y_test_pred))
print ('MAE',mean_absolute_error(y_test,y_test_pred))

print ("gbdt模型评估--BT方法：")
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
with pd.ExcelWriter(r'D:\TemporaryDirectory\GBDT-DH.xlsx') as writer:
    data_test_with_predictions.to_excel(writer, sheet_name='Test Predictions', index=False)
    data_train_with_predictions.to_excel(writer, sheet_name='Train Predictions', index=False)
    data_BT_with_predictions.to_excel(writer, sheet_name='BT Predictions', index=False)


print("预测数据已成功输出到 Excel 文件。")
