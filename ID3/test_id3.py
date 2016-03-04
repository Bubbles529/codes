import unittest
import numpy as np
import importlib
import id3

importlib.reload(id3)

from id3 import *


class TestForEntropy(unittest.TestCase):
    def test_entropy_withp1(self):
        self.assertEqual(entropy_labels(np.array(['a','a'])), 0)
        self.assertEqual(entropy_labels(np.array(['a','b'])), 1)
        

class TestGetMostCommonLabel(unittest.TestCase):
    def test_get_most_common_label(self):
        self.assertEqual(get_most_common_label(np.array(['a','b','b','c','c','c'])), 'c')
      

class TestDecideTree(unittest.TestCase):
    def test_train_only_one_label(self):
        tree = ID3()
        data = np.array([[0,0],[0,1],[1,1]])
        labels = np.array(['a','a','a'])
        self.assertEqual(tree.fit(data, labels).node, ID3.LeafNode('a'))
        

    def test_train_with_subtree(self):
        tree = ID3()
        data = np.array([[0,0],[1,1],[2,1]])
        labels = np.array(['a','b','c'])
        sub_tree = {0:ID3.LeafNode('a'), 1:ID3.LeafNode('b'), 2:ID3.LeafNode('c')}
        t = tree.fit(data,labels)
        self.assertEqual(t.node, ID3.ClasfyNode(0, sub_tree))
        
        
    def test_train_classfy(self):
        datas = np.array([[1, 0, 0],[0, 0, 0], [0, 0, 1], [0, 1, 1]])
        labels = np.array([0, 1, 1, 2])
        tree = ID3()
        tree.fit(datas, labels)
        self.assertEqual(tree.predict([1, 1, 1]), 2)
        self.assertEqual(tree.predict([0, 0, 2]), 1)
        self.assertEqual(tree.predict([1, 0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
