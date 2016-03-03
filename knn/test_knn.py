#-*- coding: utf8 -*-

import unittest
import importlib
from numpy import *
import knn

importlib.reload(knn)

KNN = knn.KNearestNeighbor

class TestKNN(unittest.TestCase):


    def test_basic_predict(self):
        '''全部默认状态使用KNN'''
        X = array([[1,2],[1.5,3],[1.3,2.5],[4,7],[6,6],[5,8]])
        Y = array(['a','c','a','b','b','a'])
        knn = KNN().fit(X,Y)
        self.assertEqual(knn.predict(array([2,2]), 3),'a')
        self.assertEqual(knn.predict(array([5,5]), 3),'b')
        self.assertEqual(knn.predict(array([5,5]), 6),'a')
        self.assertEqual(knn.predict(array([1,5]), 10),'a')


    def test__predict_with_tans_fun(self):
        X = array([[2, 4], [6, 8]])
        Y = array(['a','b'])
        knn = KNN(trans_fun = KNN.STD_TRANS)
        knn.fit(X,Y)
        self.assertEqual(knn.predict(array([4,5]), 1), 'a')
        self.assertEqual(knn.predict(array([5,6]), 1), 'b')
        

    def test__transfun_std(self):
        X = array([[2, 4], [6, 8]])
        X_, x_fun = KNN._transfun_std(X)
        self.assertTrue(all(X_ == array([[-0.5, -0.5], [0.5, 0.5]])))
        self.assertTrue(all(x_fun([4,6]) == array([0,0])))
        

    def test__yfun_vote_most(self):
        self.assertEqual(KNN._yfun_vote_most(array(['a','C','C','E','D','C','a'])),'C')


if __name__ == '__main__':
    unittest.main()
