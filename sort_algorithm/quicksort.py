#-*- coding:utf8 -*-

import random
import unittest

def quick_sort(array, begin, end):
    """快排具体实现用作递归操作"""
    if end-begin < 2:
        return

    #随机选择一个数作为排序基准，放到第一个位置上
    base_index = random.randint(begin, end-1)
    base_num = array[base_index]

    array[begin], array[base_index] = array[base_index] , array[begin]

    #当循环结束时，big_index左侧为小于基准值的值，其右边为大于等于基准值的值
    for big_index in range(begin+1,  end):
        if array[big_index] >= base_num:
            for small_index in range(end-1, big_index, -1):
                if array[small_index] < base_num:
                    array[big_index], array[small_index] = array[small_index],array[big_index]
                    break
            else:
                break
    else:
            big_index = end

    array[big_index-1], array[begin] = array[begin], array[big_index-1]
                
    quick_sort(array, big_index, end)
    quick_sort(array, begin, big_index-1)


def sort(array):
    '''快排，入参为可变列表'''
    quick_sort(array, 0, len(array))


class TestQuickSort(unittest.TestCase):
    
    def quick_sort(self, array):
        sort(array)
        return array

    
    def test_quick_sort(self):
        self.assertEqual(self.quick_sort([]),[])
        self.assertEqual(self.quick_sort([1]),[1])
        self.assertEqual(self.quick_sort([2,2]),[2,2])
        self.assertEqual(self.quick_sort([3,2,1]),[1,2,3])
        self.assertEqual(self.quick_sort([3,2,1,4,3,4,2,2]),[1,2,2,2,3,3,4,4])

    class ForStableTest():
        def __init__(self, i, j):
            self.i, self.j = i, j

        def __eq__(self, other):
            return self.i == other.i and self.j == other.j

        def __ge__(self, other):
            return self.i >= other.i

        def __lt__(self, other):
            return self.i < other.i

        def __repr__(self):
            return '({},{})'.format(self.i,self.j)
            
    def test__qucik_sort_stable(self):
        list_ = []
        for i in range(1,100):
            list_.append(TestQuickSort.ForStableTest(1, i))
        origin = list_.copy()
        self.assertNotEqual(self.quick_sort(list_),origin)


if __name__ == '__main__':
    unittest.main()
