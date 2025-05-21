import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv('testPL.csv',index_col=0,encoding='utf-8')
print (data.head())
print (data.shape)
index=data.index
col=data.columns


'''查看各变量间的相关系数'''
correlation_matrix = data.corr()
print("相关性矩阵：",correlation_matrix)


# 绘制热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()
