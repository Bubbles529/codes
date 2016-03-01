#-*- coding:utf8 -*-

import random
import unittest

def swap(array, i, j):
    array[i], array[j] = array[j], array[i]

def sort(array):
    size = len(array)
    
    for end in range(size-1, 0,-1):
        swap_flag = False
        
        for i in range(0, end):
            if array[i] > array[i+1]:
                swap(array, i, i+1)
                swap_flag = True

        if not swap_flag:
            break


class TestQuickSort(unittest.TestCase):
    
    def sort(self, array):
        sort(array)
        return array

    
    def test_sort(self):
        self.assertEqual(self.sort([]),[])
        self.assertEqual(self.sort([1]),[1])
        self.assertEqual(self.sort([2,2]),[2,2])
        self.assertEqual(self.sort([3,2,1]),[1,2,3])
        self.assertEqual(self.sort([3,2,1,4,3,4,2,2]),[1,2,2,2,3,3,4,4])

    class ForStableTest():
        def __init__(self, i, j):
            self.i, self.j = i, j

        def __eq__(self, other):
            return self.i == other.i and self.j == other.j

        def __ge__(self, other):
            return self.i >= other.i

        def __gt__(self, other):
            return self.i > other.i

        def __lt__(self, other):
            return self.i < other.i

        def __repr__(self):
            return '({},{})'.format(self.i,self.j)
            
    def test__qucik_sort_stable(self):
        list_ = []
        for i in range(1,100):
            list_.append(TestQuickSort.ForStableTest(1, i))
        origin = list_.copy()
        self.assertEqual(self.sort(list_),origin)


if __name__ == '__main__':
    unittest.main()
