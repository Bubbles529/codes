#-*- coding:utf8 -*-
'''
K最近邻算法
'''
import numpy as np
import collections

class KNearestNeighbor():

    def __init__(self, trans_fun=None, dis_fun=None, y_fun=None):
        self.dis_fun = dis_fun
        self.trans_fun = trans_fun
        self.y_fun = y_fun
        self.x_fun = None
    

    def predict(self, x, k):
        '''预测分类或者回归'''
        if self.x_fun:
            x = self.x_fun(x)

        #算距离,求k最近
        if self.dis_fun:
            sample_n, _ = self.X.shape
            distance = np.zeros(sample_n)
            for i in range(sample_n):
                distance[i] = self.dis_fun(X[i], x)
        else:
            X_ = np.sum(np.power(self.X - x, 2), axis=1)
            distance = np.sqrt(X_) 
        choosed = distance.argsort()[:k]

        #求y并输出
        if self.y_fun is None:
            return self._yfun_vote_most(self.Y[choosed])
        else:
            return self.y_fun(X[choosed], Y[choosed], distance[choosed])            


    def fit(self, X, Y):
        '''训练，进行转换'''
        if self.trans_fun:
            X, self.x_fun = self.trans_fun(X)
            
        self.X = X
        self.Y = Y

        return self


    @staticmethod
    def _transfun_std(X):
        '''将X归一化，以std为1'''
        std = np.sum(np.std(X, axis=0))
        mean = X.mean(axis=0)
        X = (X -mean) / std
        def x_fun(x):
            return (x-mean)/std
        return X, x_fun
        
    
    @staticmethod
    def _yfun_vote_most(Y):
        '''Y值通过投票产生'''
        counter = collections.Counter(Y)
        return counter.most_common(1)[0][0]

    STD_TRANS = KNearestNeighbor._transfun_std
    VOTE_MOST = KNearestNeighbor._yfun_vote_most
