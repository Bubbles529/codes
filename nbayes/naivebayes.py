#-*- coding:utf8 -*-
'''
朴素贝叶斯
'''
import operator
import sys
import math
import numpy as np
import logging

import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class Bayes(object):

    DEFAULT_PROBE = 0.01

    def __init__(self, lambda_=None, default_probe=None, with_log=False):
        '''参数lambda作为Lidstone参数，该值为None时计算未出现值时则默认为default_probe'''
        if default_probe is None:
            default_probe = Bayes.DEFAULT_PROBE
        self.lambda_ = lambda_
        self.default_probe = default_probe
        self.with_log = with_log


    def fit(self, X, Y):
        '''训练，X为二维array，Y为一维array'''
        n_samples, n_features = X.shape

        class_counter = collections.Counter(Y)
        self.y_list = list(class_counter.keys())
        n_class = len(self.y_list)

        self.probs_xi_base_y = {}
        for feature_index in range(n_features):
            self.probs_xi_base_y[feature_index] = self._calc_x_base_y_probs(X[:,feature_index], Y, class_counter)

        if self.lambda_:
            self.probs_y = {y: (class_counter[y] + self.lambda_) / (n_samples + self.lambda_*n_class) for y in self.y_list}
        else:
            self.probs_y = {y:  class_counter[y]  /  n_samples  for y in self.y_list}

        if self.with_log:
            self.probs_y = {y: math.log(v) for y,v in self.probs_y.items()}

        return self


    def _calc_x_base_y_probs(self, X, Y, class_counter, n_class, with_log):
        '''计算概率,x_i | y'''
        probs = collections.defaultdict(lambda: collections.defaultdict(float))
                
        for x, y in zip(X, Y):
            probs[x][y] += 1.0

        for x, y_dict in probs.items():
            for y in self.y_list:
                if self.lambda_:
                    y_dict[y] = (y_dict[y] + self.lambda_) / (class_counter[y] + self.lambda_*n_class)
                else:
                    y_dict[y] = y_dict[y]/class_counter[y] if y_dict[y]>0 else self.default_probe

                if with_log:
                    y_dict[y] = math.log(y_dict[y])
                
        return probs
        
            
    def predict_prob(self, x):
        '''预测数据，x为一维array'''
        probes = [self.probs_y[y] for y in self.y_list]
        
        for index, value in enumerate(x):
            probs_dict = self.probs_xi_base_y[index].get(value, None)
            if probs_dict:
                for i, y in enumerate(self.y_list):
                    if self.with_log:
                        probes[i] += probs_dict[y]
                    else:
                        probes[i] *= probs_dict[y]
            #else:如果该值之前没有出现过，则对于每类是一样的

        return probes


    def predict(self, x):
        '''预测数据，x为一维array'''
        probes = self.predict_prob(x)
        return self.y_list[np.argmax(probes)]
