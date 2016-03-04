#-*- coding:utf8 -*-
import unittest
import importlib
import naivebayes
from numpy import *

importlib.reload(naivebayes)
from naivebayes import *


class TestNaiveBayes(unittest.TestCase):

    def test__calc_x_base_y_probs(self):
        bayes = Bayes()
        bayes.y_list=[1,2,3]
        probs = bayes._calc_x_base_y_probs([1,1,1,2,2], [1,1,3,2,1], {1:3.0, 2:1.0, 3:1.0}, 3)
        expect = {1:{1:2/3.0, 3:1}, 2:{2:1,1:1.0/3}}
        for i in expect:
            for j in expect[i]:
                self.assertEqual(probs[i][j], expect[i][j])


    def test_classify(self):
        X, Y = array([])


if __name__ == '__main__':
    unittest.main()
