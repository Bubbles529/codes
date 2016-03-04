#-*- coding: utf8 -*-
"""
ID3决策树实现
1. 使用信息增益作为属性选择标准
2. 多分支树
3. 属性消耗
"""

import operator
import sys
import math
import numpy as np
import logging
import collections

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ID3(object):

    class ClasfyNode(object):
        def __init__(self, index, subtrees):
            self.index = index
            self.subtrees = subtrees
            logger.debug('ClasfyNode:'+'index:'+str(index)+';subtree:'+str(subtrees))


        def predict(self, test_data):
            test_value = test_data[self.index]
            sub_node = self.subtrees[test_value]
            return sub_node.predict(test_data)


        def __eq__(self, other):
            return isinstance(other, ID3.ClasfyNode) and self.index == other.index and self.subtrees == other.subtrees


    class LeafNode(object):
        def __init__(self, label):
            self.label = label
            logger.debug('LeafNode:'+'label:'+str(label))


        def predict(self, test_data):
            return self.label


        def __eq__(self, other):
            return isinstance(other, ID3.LeafNode) and self.label == other.label
            

    def fit(self, X, Y):
        '''训练函数，X为二维array，Y为一维array'''
        sample_n, feature_n = X.shape
        unused_features = np.ones(feature_n) == 1
        unclass_samples = np.ones(sample_n) == 1 
        self.node = self._train_tree(X, Y, unused_features, unclass_samples)
        return self


    def _train_tree(self, train_data, train_labels, unused_features,  unclass_samples):
        '''递归调用用以创建树'''
        logger.debug('train node ->  X :  < {} >   Y :   < {} >'.format( str(train_data).replace('\n',''), str(train_labels).replace('\n','')))
        logger.debug('train node ->  avalid features :  < {} >   avalid samples :   < {} >'.format( str(unused_features).replace('\n',''), str(unclass_samples).replace('\n','')))

        #只有一类，直接创建叶子节点
        Y = train_labels[unclass_samples]
        if len(set(Y)) == 1:
            logger.debug('train node ->  to make LeafNode:'+str(Y[0]))
            return ID3.LeafNode(Y[0])
        
        min_feature, min_values, min_entropy = self._find_split_feature(train_data[unclass_samples], Y, unused_features,  unclass_samples)
        if min_feature is None:
            lable = get_most_common_label(Y)
            logger.debug('train node ->  '+ ' to make LeafNode:'+ str(lable))
            return ID3.LeafNode(lable)

        #创建子树 
        else:
            logger.debug('train node ->  '+' make subtree->'+'min_col:'+str(min_feature)+ ';min values:'+str(min_values)+ ';min_entropy:'+str( min_entropy))
            subtrees = {value:None for value in min_values}
            unused_features = unused_features.copy()
            unused_features[min_feature] = 0
            for value in min_values:
                value_unclass_samples = unclass_samples.copy()
                value_unclass_samples[train_data[:, min_feature] != value] = 0
                subtrees[value]  = self._train_tree(train_data, train_labels,  unused_features, value_unclass_samples)
                
            return ID3.ClasfyNode(min_feature, subtrees)


    def _find_split_feature(self, X, Y, unused_features,  unclass_samples):
        ''''寻找最优的拆分feature'''
        min_entropy, min_feature, min_values = np.inf, None, None
        
        for feature_index in range(len(unused_features)):
            if unused_features[feature_index] == 0:
                continue
        
            values = set(X[:, feature_index])
            entropy = 0.0
            for value in values:
                X_index = (X[:,feature_index] == value)
                Y_value =  Y[X_index]
                entropy += float(X_index.sum()) / unclass_samples.sum() * entropy_labels(Y_value)
            logger.debug('train node ->: split feature : '+str(feature_index) +'    calc entropy:'+str(entropy))
            
            if entropy <= min_entropy:
                min_entropy = entropy
                min_feature = feature_index
                min_values = values

        return min_feature, min_values, min_entropy
        

    def predict(self, x):
        '''预测'''
        if self.node is None:
            raise Exception("predict not train")

        return self.node.predict(x)
    

def get_most_common_label(Y):
    '''获取出现次数最大的标签，相同的不确定'''
    if len(Y) == 0: return None
    
    count = collections.Counter(Y)
    return count.most_common(1)[0][0]


def entropy_labels(Y):
    '''计算熵'''
    counter = collections.Counter(Y)
    sum_ = sum(counter.values())

    entropy = 0.0
    for num in counter.values():
        p = num / sum_
        entropy += - p * math.log(p, 2)

    #logger.debug('entropy: labels:' + str(list(Y))+':'+ str(entropy))
    return entropy




