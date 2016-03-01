import unittest
import importlib
import apriori

importlib.reload(apriori)

class TestApriori(unittest.TestCase):

    def create_trans(self, str_):
        trans = []
        for tran in str_.split(';'):
            tran = frozenset([int(i) for i in tran.split() if i])
            trans.append(tran)
        return trans


    def create_freq_sets(self, str_):
        freq_sets = []
        for items in str_.split(';'):
            set_ = frozenset([int(i) for i in items.split() if i])
            freq_sets.append(set_)
        return freq_sets


    def create_rules(self, str_):
        rules = []
        for rule in str_.split(';'):
            before, after = rule.split('->')
            before = frozenset([int(i) for i in before.split()])
            after = frozenset([int(i) for i in after.split()])
            rules.append([before,after])
        return rules

    def test_support_calc(self):
        trans = self.create_trans("1 2 3;1 2 4;1 2 5;3 4 5")
        _ = apriori.Apriori()
        self.assertEqual(_._calc_itemset_support(trans, frozenset([5])), 2/4)
        self.assertEqual(_._calc_itemset_support(trans, frozenset([1,2])), 3/4)
        self.assertEqual(_._calc_itemset_support(trans, frozenset([1,3,4])), 0)


    def test_create_all_1_item_sets(self):
        trans = self.create_trans("1 2 3;1 2 4;1 2 5;3 4 5")
        _ = apriori.Apriori()
        self.assertEqual(set(_._create_all_1_item_sets(trans)), set([frozenset([i]) for i in range(1,6)]))


    def test__get_next_level_freqset(self):
        _ = apriori.Apriori()
        ret =  set(_._get_next_level_freqset([[1],[2],[3]]))
        self.assertEqual(ret, set([frozenset(i) for i in [[1,2],[2,3],[1,3]]]))
    def test__get_next_level_freqset2(self):
        _ = apriori.Apriori()
        ret =  set(_._get_next_level_freqset([[1,2],[2,3],[1,3]]))
        self.assertEqual(ret, set([frozenset([1,2,3])]))
    def test__get_next_level_freqset3(self):
        _ = apriori.Apriori()
        ret =  set(_._get_next_level_freqset([[1,2,3],[1,3,4]]))
        self.assertEqual(ret, set())


    def test_calc_frequent_sets(self):
        trans = self.create_trans("1 2 3;1 2 4;1 2 5;3 4 5")
        _ = apriori.Apriori()
        freq_sets = [j for i in _.calc_frequent_sets(trans,0.75) for j in i]
        self.assertEqual(set(freq_sets), set(self.create_freq_sets("1;2;1 2")))


    def test_calc_association_rules(self):
        trans = self.create_trans("1 2 3;1 2 4;1 2 5;1 4 5")
        _ = apriori.Apriori()
        freq_sets = _.calc_frequent_sets(trans,0.75)
        rules = _.calc_association_rules(freq_sets, trans, 0.8)
        rules.sort()
        self.assertEqual(rules, self.create_rules("2->1"))
        
        
    def test_fit(self):
        trans = self.create_trans("1 2 3;1 2 4;1 2 5;1 4 5")
        _, rules = apriori.Apriori().fit(trans, 0.75, 0.8)
        self.assertEqual(rules, self.create_rules("2->1"))


if __name__ == '__main__':
    unittest.main()
