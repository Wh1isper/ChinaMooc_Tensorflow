# coding:utf-8
import numpy as np
import  matplotlib.pyplot as plt
seed = 2
def generateds():
    # 基于seed产生随机数
    rdm = np.random.RandomState(seed)
    # 随机数返回300行2列的矩阵，表示300组坐标点(x0,x1)，
    X = rdm.randn(300, 2)
    # 从这300行2列的而矩阵中取出一行，判断：
    # 如果两个坐标平方和小于1，则给Y赋值1，否则赋值0
    # 作为输入数据集的标签（模拟标注过程）
    Y_ = [int(x0 * x0 + x1 * x1 < 2) for (x0, x1) in X]
    # 遍历Y中元素，将1标记为red，其余标记为blue
    Y_c = [['red' if y else 'blue'] for y in Y_]
    # 对数据集X和标签Y进行shape整理，n行用-1表示，把X整理为n行2列，把Y整理为n行1列
    X = np.vstack(X).reshape(-1, 2)
    Y_ = np.vstack(Y_).reshape(-1, 1)
    return X, Y_, Y_c

X, Y_, Y_c = generateds()
print(X, Y_, Y_c)
# 用plt.scatter画出数据集
# 横坐标为X第一列元素，Y坐标为X第二列元素
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.show()