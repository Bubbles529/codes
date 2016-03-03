#-*- coding:utf8 -*-
'''
CART,分类回归树
'''

import numpy as np
import logging

class CartTree():

    class Node():
        
        def __init__(self, feature=None, value=None, left=None, right=None):
            self.feature = feature
            self.value = value
            self.left = left
            self.right = right


        def _is_tree(self):
            return self.feature is not None


        def __repr__(self):
            if self.feature is not None :
                return '(F:{})[L::<= {}->{}][R::>{}->{}]'.format(self.feature,  self.value, self.left if not self.left is None else '', self.value, self.right if not self.right is None else '')
            else:
                return str(self.value)


    def fit(self, x, y, model=None, min_sample_num = 1, min_delta_error = 0.0001):
        '''训练数据'''
        if model is None:
            self.model = self.LINE_MODEL
        else:
            self.model = model
        leaf_fun, error_fun, predict_fun = self.model
    
        self.tree = self._rescu_create_tree(x, y, leaf_fun, error_fun, min_sample_num, min_delta_error)

        return self
        
    def predict(self, x, predict_fun=None):
        '''预测'''
        if predict_fun is None:
            predict_fun = self.model[2]

        node = self.tree
        while node._is_tree():
            v = x[node.feature]
            if v <= node.value:
                node = node.left
            else:
                node = node.right
                
        return predict_fun(node.value, x)


    def post_prun(self, testdata, testy):
        '''采用后剪枝的方法对回归树进行剪枝'''
        self.tree = self._resc_post_prun(self.tree, testdata, testy)
    

    def _resc_post_prun(self, tree, testdata, testy):
        if testdata.shape[0] == 0: return mean(tree)

        left_index, right_index = testdata[:,tree.feature]<= tree.value, testdata[:,tree.feature]>tree.value
        left_set, right_set = testdata[left_index,:], testdata[right_index,:]
        left_y, right_y = testy[left_index,:], testy[right_index,:]

        if tree.left._is_tree():
            tree.left = self._resc_post_prun(tree.left, left_set, left_y)
        if tree.right._is_tree():
            tree.right = self._resc_post_prun(tree.right, right_set, right_y)

        if (not tree.left._is_tree() ) and (not tree.right._is_tree() ):
            tree_mean = (tree.left.value + tree.right.value)/2
            spilt_error = np.sum(np.power((left_y - tree.left.value), 2)) + np.sum(np.power((right_y - tree.right.value), 2))
            merge_error = np.sum(np.power((test_y - tree_mean), 2))
            if merge_error < spilt_error:
                return self.Node(value = tree_mean)

        return tree


    def _rescu_create_tree(self, x, y, leaf_fun, error_fun, min_sample_num, min_delta_error):
        '''创建树或子树，用于递归调用'''
        feature, value = self._choose_best_split(x, y, error_fun, min_sample_num, min_delta_error)
        if feature is None:
            return leaf_fun(x, y)

        left_index, right_index = x[:,feature]<=value,  x[:,feature]>value
        left = self._rescu_create_tree(x[left_index,:], y[left_index],leaf_fun, error_fun, min_sample_num, min_delta_error)
        right = self._rescu_create_tree(x[right_index], y[right_index],leaf_fun, error_fun, min_sample_num, min_delta_error)
        node = self.Node(feature, value, left, right)

        return node


    def _choose_best_split(self, x, y, error_fun, min_sample_num, min_delta_error):
        '''选择最佳的属性以及值进行拆分数据'''
        if len(set(y)) == 1: return None,None
        
        min_error, feature_best, value_best = np.inf, None, None
        _, features_n = x.shape
        error = error_fun(x, y)
        
        for feature in range(features_n):
            for value in set(x[:,feature]):
                left_index, right_index = x[:,feature] <= value,x[:,feature] > value
                data_left, data_right = x[left_index,:], x[right_index,:]
                y_left, y_right = y[left_index], y[right_index]
                if data_left.shape[0] < min_sample_num or data_right.shape[0] < min_sample_num:
                    continue
                
                new_error = error_fun(data_left, y_left) + error_fun(data_right, y_right)
                if new_error < min_error and error - new_error > min_delta_error:
                    min_error, feature_best, value_best = new_error, feature, value

        return feature_best, value_best


    @staticmethod
    def _line_error(x, y):
        '''总方差'''
        return np.var(y)*len(y)

    
    @staticmethod
    def _line_leaf(x, y):
        '''平均值模型'''
        value = np.mean(y)
        return CartTree.Node(value = value)


    @staticmethod
    def _line_predict(value, x):
        '''平均值模型的预测'''
        return value


    @staticmethod
    def _line_solve(x, y):
        '''线性拟合'''
        X = np.matrix(np.ones((x.shape[0], x.shape[1]+1)))
        y = np.matrix(y).T
        X[:,1:] = x
        import numpy.linalg as linalg
        XTX = X.T*X
        if linalg.det(XTX) == 0:
            raise "Cannot line model"
        w = XTX.I*(X.T*y)
        return w,X,y
    

    @staticmethod
    def _piece_line_error(x, y):
        '''分段线性模型的误差'''
        w,X,y= CartTree._line_solve(x, y)
        return np.sum(np.power(y - X*w ,2))


    @staticmethod
    def _piece_line_leaf(x, y):
        '''分段线性模型的叶子生成'''
        w,X,y= CartTree._line_solve(x, y)
        return CartTree.Node(value=w)


    @staticmethod
    def _piece_line_predict(value, x):
        '''分段线性模型的预测'''
        new_x = np.ones(x.shape[0]+1)
        new_x[1:] = x
        return np.sum(new_x*value)

    PIECE_LINE_MODEL = [CartTree._piece_line_leaf,  CartTree._piece_line_error,  CartTree._piece_line_predict]
    LINE_MODEL = [CartTree._line_leaf,  CartTree._line_error,  CartTree._line_predict]


    
