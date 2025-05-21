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
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.ticker as ticker

# 定义rmse、rrmse
def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


scoring = make_scorer(rmse, greater_is_better=False)  # 创建均方根误差的评分器

'''对数据进行统计分析，查看数据的分布情况'''
data_PL = pd.read_csv(''r'D:\TemporaryDirectory\testPL.csv', index_col=0, encoding='utf-8')
index_PL = data_PL.index

n_iterations = 200
train_rmse_list = []
test_rmse_list = []
train_r2_list = []
test_r2_list = []

# 拆分数据
data_train, data_test = train_test_split(data_PL, test_size=0.2, random_state=999)

# 提取特征数据和目标变量
X_train = data_train.iloc[:, :-1]
y_train = data_train.iloc[:, -1]
y_test = data_test.iloc[:, -1]

# 噪声参数（根据数据特征设定合适的均值和标准差）
noise_params5 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.425},  # 第一个特征
    'l_f': {'mean': 0, 'std_dev': 0.2912},  # 第二个特征
    'V_f': {'mean': 0, 'std_dev': 0.0056},  # 第三个特征
    "f_c'": {'mean': 0, 'std_dev': 0.365},  # 第四个特征
    'd': {'mean': 0, 'std_dev': 0.1165},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.0375},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.0425}}  # 输出

noise_params1 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.085},  # 第一个特征
    'l_f': {'mean': 0, 'std_dev': 0.05824},  # 第二个特征
    'V_f': {'mean': 0, 'std_dev': 0.00112},  # 第三个特征
    "f_c'": {'mean': 0, 'std_dev': 0.073},  # 第四个特征
    'd': {'mean': 0, 'std_dev': 0.0233},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.0075},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.0085}}  # 输出

noise_params10 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.425*2},  # 第一个特征
    'l_f': {'mean': 0, 'std_dev': 0.05824*2},  # 第二个特征
    'V_f': {'mean': 0, 'std_dev': 0.0056*2},  # 第三个特征
    "f_c'": {'mean': 0, 'std_dev': 0.365*2},  # 第四个特征
    'd': {'mean': 0, 'std_dev': 0.1165*2},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.0375*2},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.0425*2}}  # 输出

noise_params20 = {
    'l_f/d_f': {'mean': 0, 'std_dev': 0.425*4},  # 第一个特征
    'l_f': {'mean': 0, 'std_dev': 0.05824*4},  # 第二个特征
    'V_f': {'mean': 0, 'std_dev': 0.0056*4},  # 第三个特征
    "f_c'": {'mean': 0, 'std_dev': 0.365*4},  # 第四个特征
    'd': {'mean': 0, 'std_dev': 0.1165*4},  # 第五个特征
    'c/d': {'mean': 0, 'std_dev': 0.0375*4},  # 第六个特征
    'l/d': {'mean': 0, 'std_dev': 0.0425*4}}  # 输出

# 迭代运行
for i in range(n_iterations):
    # 每次迭代重新复制 X_test
    X_test = data_test.iloc[:, :-1].copy()
    # 仅对输入特征添加噪声
    for feature in X_test.columns:
        mean = noise_params20[feature]['mean']
        std_dev = noise_params20[feature]['std_dev']
        noise = np.random.normal(mean, std_dev, X_test[feature].shape)

        # 仅对不为0的元素添加噪声
        X_test.loc[X_test[feature] != 0, feature] += noise[X_test[feature] != 0]

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
for i in range(n_iterations):
    print(f"Iteration {i + 1}:")
    print(f"R^2 (Train): {train_r2_list[i]}")
    print(f"RMSE (Train): {train_rmse_list[i]}")
    print(f"R^2 (Test): {test_r2_list[i]}")
    print(f"RMSE (Test): {test_rmse_list[i]}\n")

# 计算并打印平均测试集 RMSE
average_test_rmse = np.mean(test_rmse_list)
max_test_rmse = np.max(test_rmse_list)
min_test_rmse = np.min(test_rmse_list)

print(f"平均 RMSE (Test): {average_test_rmse:.4f}")
print(f"最大 RMSE (Test): {max_test_rmse:.4f}")
print(f"最小 RMSE (Test): {min_test_rmse:.4f}\n")
# 设置图形大小
plt.figure(figsize=(10, 6))  # 调整图形大小以减少空白
# 设置字体全局属性
plt.rcParams['font.family'] = 'serif'  # 字体类型
plt.rcParams['font.serif'] = ['Times New Roman']  # 指定具体字体
plt.rcParams['font.size'] = 14  # 字体大小

# 设置坐标轴标题
plt.xlabel('Actual Values', fontsize=16, fontweight='normal')  # x 轴标题
plt.ylabel('Predicted Values', fontsize=16, fontstyle='normal')  # y 轴标题
plt.title('', fontsize=18)  # 图形标题

# 绘制 RMSE
plt.subplot(1, 1, 1)
plt.plot(range(1, n_iterations + 1), test_rmse_list, label='test RMSE', marker='o')
plt.xlabel('Iteration')
plt.ylabel('RMSE')

# 设置 x 轴范围
plt.xlim(0, n_iterations)  # 设置 x 轴起始和结束值

# 设置 x 轴大刻度
plt.xticks(range(1, n_iterations + 1, 5))  # 每隔 5 个显示一个大刻度

# 设置小刻度
ax = plt.gca()  # 获取当前轴
ax.xaxis.set_major_locator(ticker.MultipleLocator(5))  # 设置主刻度
ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))  # 设置次刻度

# 隐藏小刻度的标签
ax.tick_params(axis='x', which='minor', labelbottom=False)

# 设置图例位置为左上角
plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

plt.tight_layout()
plt.show()
