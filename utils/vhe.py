# -*- coding: utf-8 -*-
# @Time    : 2019/11/12 17:13
# @Author  : QixianZhou
# @FileName: vhe.py
# @Abstract:


import numpy as np
from numpy import *
# 设置打印结果不按科学计数法输出
np.set_printoptions(suppress=True, threshold=np.inf)
# 大整数
w = 10 ** 30
# 比特化参数
l = 200
# 随机矩阵中随机值范围
aBound, tBound, eBound = 1000, 1000, 1000
import random


def KeySwitch(M, c):
    cstar = getBitVector(c)
    # result=M.dot(cstar).astype(np.int64)
    # print(M,M.dot(cstar))
    # print("M",type( M[0][0]))
    # print( "cstar", type( cstar[0][0]))
    # print( np.dot(M, cstar))
    return M.dot(cstar)


def getRandomMatrix(row, col, bound):
    # A = np.array([x for x in random(0,bound,size=row*col)],dtype=object)
    A = np.zeros(row * col, dtype=object)
    for i in range(row * col):
        A[i] = random.randint(0, bound)

    A = A.reshape(row, col)
    # print("A", type(A[0][0]))
    return A


def getBitMatrix(S):
    rows, cols = S.shape[0], S.shape[1]
    powers = np.array([2 ** x for x in range(l)], dtype=object)
    result = np.zeros([rows, l * cols], dtype=object)
    for i in range(rows):
        for j in range(cols):
            for k in range(l):
                result[i][j * l + k] = S[i][j] * powers[k]
    return result


def getBitVector(c):
    length = c.shape[0]
    result = np.zeros(length * l, dtype=object)
    for i in range(length):
        sign = (c[i] < 0) and -1 or 1
        value = c[i] * sign
        length1 = len(bin(value)) - 2
        for j in range(0, length1):
            result[i * l + j] = sign * int(bin(value)[length1 - j + 1])
        for j in range(length1, l):
            result[i * l + j] = 0
    # result.resize( length*l,1)
    return result


def hCat(A, B):
    assert A.shape[0] == B.shape[0]
    return np.hstack((A, B))


def vCat(A, B):
    assert A.shape[1] == B.shape[1]
    return np.vstack((A, B))


def getSecretKey(T):
    I = np.eye(T.shape[0], dtype=object)
    return hCat(I, T)


def nearestInteger(x, w):
    return int((x + (w + 1) / 2) / w)


def decrypt(s, c):
    Sc = np.dot(s, c)
    output = np.zeros(Sc.shape[0], dtype=object)
    for i in range(0, Sc.shape[0]):
        output[i] = nearestInteger(Sc[i], w)
    return output


def KeySwitchMatrix(S, T):
    Sstar = getBitMatrix(S)
    A = getRandomMatrix(T.shape[1], Sstar.shape[1], aBound)
    E = getRandomMatrix(Sstar.shape[0], Sstar.shape[1], eBound)
    return vCat(Sstar + E - T.dot(A), A)


def encrypt(T, x):
    I = np.eye(x.shape[0], dtype=object)
    M = KeySwitchMatrix(I, T)
    return KeySwitch(M, np.dot(w, x))


def addVectors(c1, c2):
    return c1 + c2


def linearTransform(M, c):
    return M.dot(getBitVector(c))


def linearTransformClient(G, S, T):
    return KeySwitchMatrix(np.dot(G, S), T)


def innerProd(c1, c2, M):
    cc1 = np.zeros((c1.shape[0], 1), dtype=object)
    cc2 = np.zeros((1, c2.shape[0]), dtype=object)
    for i in range(0, c1.shape[0]):
        cc1[i][0] = c1[i]
    for i in range(0, c2.shape[0]):
        cc2[0][i] = c2[i]
    cc = vectorize(np.dot(cc1, cc2))
    output = np.zeros(cc.shape[0], dtype=object)
    for i in range(0, cc.shape[0]):
        output[i] = nearestInteger(cc[i][0], w)
    return M.dot(getBitVector(output))


def innerProdClient(T):
    S = getSecretKey(T)
    tvsts = (vectorize(S.T.dot(S))).T
    mvsts = copyRows(tvsts, T.shape[0])
    return KeySwitchMatrix(mvsts, T)


def copyRows(row, numrows):
    ans = np.zeros((numrows, row.shape[1]), dtype=object)
    for i in range(0, ans.shape[0]):
        for j in range(0, ans.shape[1]):
            ans[i][j] = row[0][j]
    return ans


def vectorize(M):
    ans = np.zeros((M.shape[0] * M.shape[1], 1), dtype=object)
    for i in range(0, M.shape[0]):
        for j in range(0, M.shape[1]):
            ans[i * M.shape[1] + j][0] = M[i][j]
    return ans


if __name__ == '__main__':
    # l=150
    array1=np.array([[[[1, 2, 3],[1,2,3]]],[[[1, 2, 3],[1,2,3]]]])
    array1=array1.flatten()
    print(array1)
    # 1.基本的加解密功能
    print("---------------1.基本的加解密功能------------------------")
    # 明文
    # x = np.array([1, 2, 3], dtype=object)
    x = np.ones((150,), dtype=object)

    '''
        # 获取相关加密参数 待加密数据维度  安全参数，一般取1 随机数范围
    '''
    T = getRandomMatrix(len(x), 1, tBound)
    # 加密
    c = encrypt(T, x)
    # 私钥
    S = getSecretKey(T)
    # 解密
    dc = decrypt(S, c)

    print("明文:", x)
    print("密文:", c)
    print("解密结果：", dc)

    print("-------------2.密文加法-----------------------------------------")
    x1 = np.array([1, 1, 1], dtype=object)
    x2 = np.array([2, 5, 2], dtype=object)
    T = getRandomMatrix(3, 1, tBound)
    S = getSecretKey(T)
    c1 = encrypt(T, x1)
    c2 = encrypt(T, x2)
    c = c1 + c2
    dc = decrypt(S, c)
    print("明文加法结果：", x1 + x2)
    print("密文加密解密：", dc)

    print("---------------3.线性变换------------------------")
    # 2.线性变换 线性变换矩阵 G 明文向量 x  目标是在密文下计算 Gx
    x = np.array([4, 5, 6], dtype=object)
    G = np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 0, 1]
        ], dtype=object
    )

    # 先加密得到 x对应的密文
    T = getRandomMatrix(len(x), 1, tBound)
    # 私钥
    S = getSecretKey(T)
    # 加密
    c = encrypt(T, x)

    # 根据线性变换矩阵 G 得到密钥转换矩阵
    M = KeySwitchMatrix(G.dot(S), T)

    # 对密文c进行一次密钥转换
    cc = KeySwitch(M, c)

    # 解密cc
    dcc = decrypt(S, cc)

    print("明文计算Gx:", G.dot(x))
    print("密文计算Gx,再解密:", dcc)

    print("---------------4.密文内积------------------------")

    x1 = np.array([2, 2, 2], dtype=object)
    x2 = np.array([2, 5, 2], dtype=object)
    T = getRandomMatrix(3, 1, tBound)
    S = getSecretKey(T)
    c1 = encrypt(T, x1)
    c2 = encrypt(T, x2)
    # 密钥转换矩阵
    M = innerProdClient(T)
    # 密文内积计算
    cc = innerProd(c1, c2, M)
    # cc[1] = 0
    # cc[2] = 0
    # 解密结果是一个向量，取第一个元素
    dcc = decrypt(S, cc)[0]

    print("明文内积结果:", x1.dot(x2))
    print("密文内积解密：", dcc)




