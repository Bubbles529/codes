#-*- coding:utf8 -*-
import unittest
import fpgrowth
import importlib

importlib.reload(fpgrowth)

class TestFpGrowth(unittest.TestCase):

    def create_header(self, str_):
        nodes = {}
        for node in str_.split(';'):
            name, value = node.split()
            nodes[name] = fpgrowth.FrequentPatternTree.Node(name, None, int(value))
        return nodes


    def create_node(self, tree_str):
        root_node = fpgrowth.FrequentPatternTree.Node()
        str_ = tree_str.replace('{',' { ').replace('}',' } ').replace(';',' ; ')
        
        node, name, last_name = root_node, None, None
        for i in str_.split():
            if i == '{':
                node = node.child[last_name]
            elif i == '}':
                node = node.parent
            elif i == ';':
                pass
            elif name is None:
                name = i
            else:
                node.child[name] = fpgrowth.FrequentPatternTree.Node(name, node, int(i))
                last_name, name = name, None
                
        return root_node
                
                
    def create_tree(self,header_str, tree_str):
        _ = fpgrowth.FrequentPatternTree()
        _.header_table = self.create_header(header_str)
        _.root_node = self.create_node(tree_str)
        return _

    def create_trans(self, str_):
        trans = []
        for tran in str_.split(';'):
            tran = [i for i in tran.split() if i]
            trans.append(tran)
        return trans
    

    def test_create_fptree(self):
        trans = self.create_trans("z r h j p;z y x w v u t s;z;r x n o s;y r x z q t p;y z x e q s t m")
        tree_ = fpgrowth.FrequentPatternTree().create_tree(trans, 0.5)
        _ = self.create_tree("z 5;r 3;x 4;y 3;s 3;t 3","z 5{r 1;x 3{y 3{t 3{s 2; r 1}}}};x 1{s 1{r 1}}")
        self.assertEqual(tree_, _)
        for item, node in  tree_.header_table.items():
            v = node.value
            sum_ = 0
            node = node.next
            while node:
                sum_ += node.value
                node = node.next
            self.assertEqual(v, sum_)

    def test_create_header_table(self):
        trans = self.create_trans("a b c;b c d;b c;d e;a d e")
        table = fpgrowth.FrequentPatternTree()._create_header_table(trans, 3)
        self.assertEqual(table, self.create_header("d 3;c 3;b 3"))


    def test__find_prefix_path(self):
        trans = self.create_trans("z r h j p;z y x w v u t s;z;r x n o s;y r x z q t p;y z x e q s t m")
        tree_ = fpgrowth.FrequentPatternTree().create_tree(trans, 0.5)
        self.assertEqual(tree_._find_prefix_path('z'), [])
        self.assertEqual(tree_._find_prefix_path('x'), self.create_trans("z;z;z"))
        self.assertEqual(tree_._find_prefix_path('r'), self.create_trans("z;s x;t y x z"))


    def test_get_freq_sets(self):
        trans = self.create_trans("z r h j p;z y x w v u t s;z;r x n o s;y r x z q t p;y z x e q s t m")
        tree_ = fpgrowth.FrequentPatternTree().create_tree(trans, 0.5)
        self.assertEqual(sorted(tree_.get_freq_sets(3)), sorted([['r'], ['s'], ['s', 'x'], ['t'], ['t', 'x'], ['t', 'x', 'y'], ['t', 'x', 'y', 'z'], ['t', 'x', 'z'], ['t', 'y'], ['t', 'y', 'z'], ['t', 'z'], ['y'], ['y', 'x'], ['y', 'x', 'z'], ['y', 'z'], ['x'], ['x', 'z'], ['z']]))
        self.assertEqual(sorted(tree_.get_freq_sets(4)), sorted([['z'],['x']]))
        self.assertEqual(sorted(tree_.get_freq_sets(5)), sorted([['z']]))
        
    def test_mine_trans(self):
        trans = self.create_trans("z r h j p;z y x w v u t s;z;r x n o s;y r x z q t p;y z x e q s t m")
        expect_ans = sorted([['r'], ['s'], ['s', 'x'], ['t'], ['t', 'x'], ['t', 'x', 'y'], ['t', 'x', 'y', 'z'], ['t', 'x', 'z'], ['t', 'y'], ['t', 'y', 'z'], ['t', 'z'], ['y'], ['y', 'x'], ['y', 'x', 'z'], ['y', 'z'], ['x'], ['x', 'z'], ['z']])
        self.assertEqual(sorted(fpgrowth.FrequentPatternTree.mine_trans(trans, 0.5)), expect_ans)


if __name__ == '__main__':
    unittest.main()
        
