#-*- coding:utf8 -*-

import logging
import itertools
import collections

#logging.basicConfig(level=logging.DEBUG)

class FrequentPatternTree():
    '''FP树'''
    
    class Node():
        '''节点定义，头表和树中都使用该节点'''
        def __init__(self, name=None, parent=None, value=0):
            self.name = name
            self.parent = parent
            self.value = value
            self.child = {}
            self.next = None

        def link(self, new_node):
            node = self
            while node.next:
                node = node.next
            node.next = new_node

        def increase(self, value=1):
            self.value += value

        def __eq__(self, other):
            return  (self.name == other.name and self.value == other.value)

        def __repr__(self):
            return str(self.value)+(str(self.child)  if self.child else '')

        
    def __eq__(self, other):
        '''定义FP树的相等，test中使用'''
        return  self.root_node == other.root_node and self.header_table == other.header_table


    def __repr__(self):
        '''字符串表示'''
        return '{table:' + str(self.header_table) + ' ; tree:' + str(self.root_node) + '}'


    def _is_empty(self):
        return len(self.header_table) == 0

    @staticmethod
    def mine_trans(trans, support_ratio=0.5, support=None):
        '''针对事务集获取频繁集'''
        support = support or len(trans)*support_ratio
        tree = FrequentPatternTree().create_tree(trans, support=support)
        return tree.get_freq_sets(support)
    
        
    def create_tree(self, trans, support_ratio=0.5, support=None):
        '''创建FP树'''
        support = support or len(trans)*support_ratio
        header_table = self._create_header_table(trans, support)
        root_node = FrequentPatternTree.Node(None, None)
        
        for tran in trans:
            logging.debug('before transform tran: {}'.format(tran))
            tran = [i for i in tran if i in header_table]
            if not tran:
                continue
            tran.sort(key=lambda i:(header_table[i].value,i), reverse=True)
            logging.debug('after transform tran: {}'.format(tran))
            self._add_item_fptree(tran, root_node, header_table)
            logging.debug('after add item tree: {}'.format(root_node))

        self.root_node = root_node
        self.header_table = header_table

        return self


    def _add_item_fptree(self,  tran, node, header_table):
        '''添加一个事务到FP树中'''
        if not tran: return

        item = tran[0]
        if item not in node.child:
            new_node = FrequentPatternTree.Node(item, node)
            node.child[item] = new_node
            header_table[item].link(new_node)
        node.child[item].increase()
        
        self._add_item_fptree(tran[1:], node.child[item], header_table)


    def _create_header_table(self, trans, support):
        '''创建头表'''
        ret = collections.defaultdict(int)
        for tran in trans:
            for i in tran:
                ret[i] += 1
        filtered = {key:FrequentPatternTree.Node(key,value=value) for key,value in ret.items() if value >= support}
        return filtered
    
        
    def get_freq_sets(self, support):
        '''在创建的fp树上获取频繁集'''
        freq_sets = []
        self._find_freq_sets_byprefix(support, [], freq_sets)
        return freq_sets


    def _find_freq_sets_byprefix(self, support, pre_fix, freq_sets):
        '''通过构建条件FP树获取频繁集'''
        items = [i[0] for i in sorted(self.header_table.items(), key=lambda i:(i[1].value,i)) if i[1].value >= support]

        for item in items:
            new_freq_set = pre_fix.copy()
            new_freq_set.append(item)
            freq_sets.append(new_freq_set)

            condition_bases = self._find_prefix_path(item)
            new_tree = FrequentPatternTree().create_tree(condition_bases, support=support)
            if not new_tree._is_empty():
                new_tree._find_freq_sets_byprefix(support, new_freq_set, freq_sets)


    def _find_prefix_path(self, item):
        '''获取条件路径'''
        condition_path = []
        
        node = self.header_table[item].next
        while node:
            path, curr_node = [], node
            while True:
                curr_node = curr_node.parent
                if curr_node.name is None:
                    break
                path.append(curr_node.name)
            if path:
                condition_path += [path]*node.value
            node = node.next

        return condition_path
                
                
                                              
