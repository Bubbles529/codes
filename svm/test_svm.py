import unittest
import importlib
import svm
from numpy import *

importlib.reload(svm)
svm_ = None

class TestLinearSVM(unittest.TestCase):

    def test_fit(self):
        data = array([[1,2],[2,1],[2,2],[1,1.9],[4,0.1],[4,.5],[5,2],[0,0],[0.1,0.1],[-0.4,0.3]])
        labels = array(['a','a','a','a','b','b','b','c','c','c'])
        global svm_
        svm_ = svm.SVM().fit(data, labels)


    def test_sklearn_iris(self):
        pass


if __name__ == '__main__':
    unittest.main()
