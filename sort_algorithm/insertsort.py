#-*- cdoing:utf8 -*-
'''
直接插入排序
'''

def insert_sort(array):
    '''插入排序'''
    for i in range(len(array)):

        for k in range(i):
            if array[i] < array[k]:
                temp = array[i]
                array[k+1: i+1] = array[k : i]
                array[k] = temp

    return array


def insert_sort2(array):
    '''插入排序，从后面往前面找'''
    for i in range(len(array)):
        temp = array[i]
        
        for k in range(i-1,-1,-1):
            if temp < array[k]:
                array[k+1] = array[k]
            else:
                array[k+1] = temp
                break
        else:
            array[0] = temp

    return array


import unittest

class TestSort(unittest.TestCase):
    
    def _sort(self, array):
        insert_sort2(array)
        return array

    
    def test_sort(self):
        self.assertEqual(self._sort([]),[])
        self.assertEqual(self._sort([1]),[1])
        self.assertEqual(self._sort([2,2]),[2,2])
        self.assertEqual(self._sort([3,2,1]),[1,2,3])
        self.assertEqual(self._sort([3,2,1,4,3,4,2,2]),[1,2,2,2,3,3,4,4])


if __name__ == '__main__':
    unittest.main()
