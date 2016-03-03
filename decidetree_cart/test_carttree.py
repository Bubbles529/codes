#-*- coding:utf8 -*-

import unittest
import importlib
import carttree
from numpy import *

importlib.reload(carttree)
Tree = carttree.CartTree

class TestCartTree(unittest.TestCase):


    def test_fit_predict(self):
        x = array([[1, 3],[1.1,4],[1.2,2],[2.0,2.5],[2.1,4],[2.2,1.5]])
        y = array([1,1.1,0.8,2.3,2.0,2.6])
        tree = Tree().fit(x, y)
        self.assertEqual(tree.predict(array([0.9, 0])), 1.0)
        self.assertEqual(tree.predict(array([2.4, 0])), 2.6)


    def test__rescu_create_tree(self):
        x = array([[1, 3],[1.1,4],[1.2,2],[2.0,2.5],[2.1,4],[2.2,1.5]])
        y = array([1,1.1,0.8,2.3,2.0,2.6])
        ans = Tree()._rescu_create_tree(x, y, Tree._line_leaf, Tree._line_error, 1, 0)
        self.assertEqual(str(ans), '(F:0)[L::<= 1.2->(F:0)[L::<= 1.1->(F:0)[L::<= 1.0->1.0][R::>1.0->1.1]][R::>1.1->0.8]][R::>1.2->(F:0)[L::<= 2.1->(F:0)[L::<= 2.0->2.3][R::>2.0->2.0]][R::>2.1->2.6]]')

    def test__choose_best_split_min_sample(self):
        x = array([[1, 3],[1.1,4],[1.2,2],[2.0,2.5],[2.1,4],[2.2,1.5]])
        y = array([1,0.9,1.1,2.3,2.0,2.3])
        ans = Tree()._choose_best_split(x, y, Tree._line_error, 4, 0)
        self.assertEqual(ans[0], None)
        self.assertEqual(ans[1], None)

    def test__choose_best_split_min_sample(self):
        x = array([[1, 3],[1.1,4],[1.2,2],[2.0,2.5],[2.1,4],[2.2,1.5]])
        y = array([1,0.9,1.1,2.3,2.0,2.3])
        ans = Tree()._choose_best_split(x, y, Tree._line_error, 0, 5)
        self.assertEqual(ans[0], None)
        self.assertEqual(ans[1], None)


    def test__line_model(self):
        x = array([[1,1],[2,2],[2,3],[3,4]])
        y = array([6, 11, 14, 19])
        self.assertGreater(Tree._line_error(x, y) , 89-0.0001)
        self.assertLess(Tree._line_error(x, y) , 89+0.0001)
        self.assertGreater(Tree._line_predict(Tree._line_leaf(x, y).value, x) , mean(y)-0.0001)
        self.assertLess(Tree._line_predict(Tree._line_leaf(x, y).value, x) , mean(y)+0.0001)
    

    def test__piece_line(self):
        x = array([[1,1],[2,2],[2,3],[3,4]])
        y = array([6, 11, 14, 19])
        w, X, _ = carttree.CartTree._line_solve(x, y)
        self.assertTrue(round(w[0,0])==1 and round(w[1,0])==2 and round(w[2,0])==3)
        self.assertTrue(carttree.CartTree._piece_line_error(x, y) < 0.0001)
        
        new = array([1,3])
        self.assertGreater(Tree._piece_line_predict(Tree._piece_line_leaf(x, y).value, new) , 12-0.0001)
        self.assertLess(Tree._piece_line_predict(Tree._piece_line_leaf(x, y).value, new) , 12+0.0001)


if __name__ == '__main__':
    unittest.main()
