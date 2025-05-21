from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import shap
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import RandomizedSearchCV
from deap import base, creator, tools
from sklearn.model_selection import cross_val_score
import random
from sklearn.preprocessing import MinMaxScaler


# 定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

'''对数据进行统计分析，查看数据的分布情况'''
data_PL = pd.read_csv(r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')
index_PL = data_PL.index

n_iterations = 20
train_rmse_list = []
test_rmse_list = []
train_r2_list = []
test_r2_list = []
data_train, data_test = train_test_split(data_PL, test_size=0.2, random_state=999)
# 提取特征数据和目标变量
X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
X_test = data_test.iloc[:, :-1].copy()  # 使用 copy() 创建副本以防止修改原始数据
y_test = data_test.iloc[:, -1]

# 噪声参数
noise_params = {
    'lf/df': {'mean': 0, 'std_dev': 0.85},  # 第一个特征
    'lf(mm)': {'mean': 0, 'std_dev': 0.583333333333},  # 第二个特征
    'Vf(%)': {'mean': 0, 'std_dev': 0.011333333333},  # 第三个特征
    'fc(MPa)': {'mean': 0, 'std_dev': 0.73},  # 第四个特征
    'd(mm)': {'mean': 0, 'std_dev': 0.233333333333},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.075},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.085},  # 第七个特征
    }


# 仅对输入特征添加噪声
for feature in X_test.columns:
    mean = noise_params[feature]['mean']
    std_dev = noise_params[feature]['std_dev']
    noise = np.random.normal(mean, std_dev, X_test[feature].shape)
    X_test[feature] += noise  # 仅修改 X_test


# 模型训练
final_model = GradientBoostingRegressor(
    learning_rate=0.475,
    n_estimators=300,
    max_depth=3,
    subsample=0.918,
    min_samples_split=9,
    min_samples_leaf=1,
    max_features=0.367,
    random_state=999
)
# 拟合最终模型
final_model.fit(X_train, y_train)

# 预测
y_train_pred = final_model.predict(X_train)
y_test_pred = final_model.predict(X_test)

# 计算训练集和测试集的指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# 存储指标
train_rmse_list.append(train_rmse)
test_rmse_list.append(test_rmse)
train_r2_list.append(train_r2)
test_r2_list.append(test_r2)

# 输出结果
print("gbdt模型评估--PL训练集：")
print('R^2:', train_r2)
print('RMSE:', train_rmse)

print("gbdt模型评估--PL验证集：")
print('R^2:', test_r2)
print('RMSE:', test_rmse)

# 设置字体全局属性
plt.rcParams['font.family'] = 'serif'  # 字体类型
plt.rcParams['font.serif'] = ['Times New Roman']  # 指定具体字体
plt.rcParams['font.size'] = 14  # 字体大小

# 可视化预测结果
plt.figure(figsize=(8, 8))  # 设置图形大小
plt.scatter(y_test, y_test_pred, alpha=1, s=60)
# 设置坐标轴标题
plt.xlabel('Actual Values', fontsize=16, fontweight='normal')  # x 轴标题
plt.ylabel('Predicted Values', fontsize=16, fontstyle='normal')  # y 轴标题
plt.title('', fontsize=18)  # 图形标题

# 设置坐标轴范围
min_value = min(y_test.min(), y_test_pred.min())
max_value = max(y_test.max(), y_test_pred.max())
plt.xlim(min_value, max_value)
plt.ylim(min_value, max_value)


# 设置刻度
plt.xticks(ticks=np.arange(0, 30, step=5))  # x 轴刻度
plt.yticks(ticks=np.arange(0, 30, step=5))  # y 轴刻度

# 添加 y = x 的对角线
plt.plot([0, 30], [0, 30], 'k-', lw=1.6, label='y = x')  # 对角线
plt.plot([0, 25], [0, 30],color='k', linestyle='--', lw=1.2, alpha=0.8,)
plt.plot([0, 30], [0, 24], color='k', linestyle='--', lw=1.2, alpha=0.8, label='') #20%误差线


plt.legend()  # 添加图例
plt.show()