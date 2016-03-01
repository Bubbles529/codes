import logging
#logging.basicConfig(level=logging.DEBUG)

class Apriori():

    def fit(self, trans, min_support, min_confidence):
        """对事务进行训练，获得对应的频繁集和关联规则"""
        if len(trans) == 0:
            return [],[]

        frequent_sets = self.calc_frequent_sets(trans, min_support)
        association_rules = self.calc_association_rules(frequent_sets, trans, min_confidence)

        return frequent_sets, association_rules


    def calc_association_rules(self, freq_sets_length, trans, min_confidence):
        '''计算关联规则'''
        rules = []
        
        for freq_sets in freq_sets_length[1:]:
            for freq_set in freq_sets:
                temp = self._calc_rules_4_set(freq_set, trans, min_confidence)
                rules += temp
                
        return rules


    def _calc_rules_4_set(self, freq_set, trans, min_confidence):
        '''针对单个频繁集计算其关联规则'''
        rules = []
        
        curr_after_items = [frozenset([i]) for i in freq_set]
        while curr_after_items:
            filter_afters = []
            for after in curr_after_items:
                before = freq_set - after
                confidence = self._calc_itemset_support(trans, freq_set) / self._calc_itemset_support(trans, before)
                if confidence >= min_confidence:
                    rules.append([before, after])
                    filter_afters.append(after)
            curr_after_items = self._get_next_level_freqset(filter_afters)
        
        return rules


    def _filter_freq_sets_with_support(self, freq_sets, trans, min_support):
        '''根据support对当前的频繁集进行筛选'''
        filter_sets = []
        
        for freq_set in freq_sets:
            support = self._calc_itemset_support(trans, freq_set)
            if support >= min_support:
                filter_sets.append(freq_set)

        return filter_sets
    

    def calc_frequent_sets(self, trans, support_ratio):
        """获得频繁项集"""
        frequent_sets_all = []
        logger = logging.getLogger('calc_frequent_sets')

        curr_freq_sets = self._create_all_1_item_sets(trans)
        while len(curr_freq_sets) :
            logger.debug('-> create freq sets {}'.format(curr_freq_sets))
            filtered_freq_set = self._filter_freq_sets_with_support(curr_freq_sets, trans,  support_ratio)
            frequent_sets_all.append(filtered_freq_set)
            logger.debug('-> filted freq sets {}'.format(filtered_freq_set))
            curr_freq_sets = self._get_next_level_freqset(filtered_freq_set)

        logger.debug('-> all freq sets {}'.format(frequent_sets_all))
        return frequent_sets_all


    def _get_next_level_freqset(self, freq_set):
        '''根据当前的频繁集组成长度+1的频繁集'''
        next_level_freq_set = []
        
        freq_set_num = len(freq_set)
        for i in range(0, freq_set_num):
            for j in range(i+1, freq_set_num):
                list_i = list(freq_set[i])
                list_j = list(freq_set[j])
                list_i.sort(), list_j.sort()
                if list_i[:-1] == list_j[:-1]:
                    next_level_freq_set.append(frozenset(list_i + list_j))

        return next_level_freq_set    

        
    def _calc_itemset_support(self, trans, item_set):
        '''计算support'''
        hit_num = sum(item_set.issubset(tran) for tran in trans)
        support_ratio = hit_num / len(trans)
        logging.debug('calc support for {} hit:{} value:{}'.format(item_set,  hit_num, support_ratio))
        return support_ratio


    def _create_all_1_item_sets(self, trans):
        '''创建所有长度为1的频繁集'''
        item_sets = set()
        for tran in trans:
            for item in tran:
                item_sets.add(frozenset([item]))
        return list(item_sets)
