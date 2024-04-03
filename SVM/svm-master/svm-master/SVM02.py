# -*- coding:utf-8 -*-
"""
@author:Lisa
@file:svm_Iris.py
@func:Use SVM to achieve Iris flower classification
@time:2018/5/30 0030上午 9:58
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sklearn
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


# define converts(字典)
def Iris_label(s):
    it = {b'0': 0, b'1': 1, b'2': 2,b'3': 3 ,b'4': 4 ,b'5': 5 ,b'6': 6 ,b'7': 7,b'8':8,b'9':9,b'10':10,b'11':11,b'12':12,b'13':13}
    return it[s]


# 1.读取数据集
path = 'D:\POI2vec\SVM\svm-master\svm-master\group\zonevec_593.csv'
data = np.loadtxt(path, dtype=float, delimiter=',', converters={200: Iris_label})
# converters={4:Iris_label}中“4”指的是第5列：将第5列的str转化为label(number)
# print(data.shape)

# 2.划分数据与标签
x, y = np.split(data, indices_or_sections=(200,), axis=1)  # x为数据，y为标签
x = x[:, 0:200]
train_data, test_data, train_label, test_label = train_test_split(x, y, random_state=1, train_size=0.7,
                                                                  test_size=0.3)  # sklearn.model_selection.
# print(train_data.shape)

# 3.训练svm分类器
classifier = svm.SVC(C=18, kernel='rbf', gamma=0.8, decision_function_shape='ovo')  # ovr:一对多策略
classifier.fit(train_data, train_label.ravel())  # ravel函数在降维时 默认是行序优先
#kernel='linear'时，为线性核，C越大分类效果越好，但有可能会过拟合（defaul C=1）。
#kernel='rbf'时（default），为高斯核，gamma值越小，分类界面越连续；gamma值越大，分类界面越“散”，分类效果越好，但有可能会过拟合。
#decision_function_shape='ovr'时，为one v rest（一对多），即一个类别与其他类别进行划分，
#decision_function_shape='ovo'时，为one v one（一对一），即将类别两两之间进行划分，用二分类的方法模拟多分类的结果。

# 4.计算svc分类器的准确率
print(len(train_data))
print("训练集：", classifier.score(train_data, train_label))
print("测试集：", classifier.score(test_data, test_label))

# 也可直接调用accuracy_score方法计算准确率
from sklearn.metrics import accuracy_score

tra_label = classifier.predict(train_data)  # 训练集的预测标签
tes_label = classifier.predict(test_data)  # 测试集的预测标签
print("训练集：", accuracy_score(train_label, tra_label))
print("测试集：", accuracy_score(test_label, tes_label))

# 查看决策函数
np.set_printoptions(threshold=1e6)

# print('train_decision_function:\n', classifier.decision_function(train_data))
# print('predict_result:\n', classifier.predict(train_data))
# print('test_decision_function:\n', classifier.decision_function(test_data))
# print('predict_result:\n', classifier.predict(test_data))



tra_flabel = open(r"D:\POI2vec\SVM\svm-master\svm-master\group\zonecls_train.txt","w",encoding = "utf-8") #预测标签
tra_rule=classifier.predict(train_data)
for x in tra_rule:
    m=int(x)
    tra_flabel.write(str(m))
    tra_flabel.write('\n')
tra_flabel.close()

tes_flabel = open(r"D:\POI2vec\SVM\svm-master\svm-master\group\zonecls_test.txt","w",encoding = "utf-8") #预测标签
tes_rule=classifier.predict(test_data)
for j in tes_rule:
    n=int(j)
    tes_flabel.write(str(n))
    tes_flabel.write('\n')
tes_flabel.close()

# tralabel = open(r"D:\POI2vec\SVM\svm-master\svm-master\group\train_label.txt","w",encoding = "utf-8") #预测标签
# tralabel_rule=train_label
# for m in tralabel_rule:
#     n=int(m)
#     tralabel.write(str(n))
#     tralabel.write('\n')
# tralabel.close()
#
# teslabel = open(r"D:\POI2vec\SVM\svm-master\svm-master\group\test_label.txt","w",encoding = "utf-8") #预测标签
# rule0=test_label
# for m in rule0:
#     n=int(m)
#     teslabel.write(str(n))
#     teslabel.write('\n')
# teslabel.close()

print("over")


# # 5.绘制图形
# # 确定坐标轴范围
# x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0维特征的范围
# x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1维特征的范围
# x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网络采样点
# grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点
# # 指定默认字体
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# # 设置颜色
# cm_light = matplotlib.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = matplotlib.colors.ListedColormap(['g', 'r', 'b'])
#
# grid_hat = classifier.predict(grid_test)  # 预测分类值
# grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
#
# plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
# plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=30, cmap=cm_dark)  # 样本
# plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label[:, 0], s=30, edgecolors='k', zorder=2,
#             cmap=cm_dark)  # 圈中测试集样本点
# plt.xlabel('花萼长度', fontsize=13)
# plt.ylabel('花萼宽度', fontsize=13)
# plt.xlim(x1_min, x1_max)
# plt.ylim(x2_min, x2_max)
# plt.title('鸢尾花SVM二特征分类')
# plt.show()