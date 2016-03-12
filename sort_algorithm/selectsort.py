#-*- cdoing:utf8 -*-
'''
选择排序
'''

def select_sort(array):
    '''选择排序'''
    for i in range(len(array)):
        min_value, min_index = array[i],  i
        
        for k in range(i+1, len(array)):
            if array[k] < min_value:
                min_value = array[k]
                min_index = k

        if min_index != i:
            array[min_index], array[i] = array[i], array[min_index]
                
    return array

import unittest

class TestSort(unittest.TestCase):
    
    def _sort(self, array):
        select_sort(array)
        return array

    
    def test_sort(self):
        self.assertEqual(self._sort([]),[])
        self.assertEqual(self._sort([1]),[1])
        self.assertEqual(self._sort([2,2]),[2,2])
        self.assertEqual(self._sort([3,2,1]),[1,2,3])
        self.assertEqual(self._sort([3,9,20,2,8,1,4,6,7,3,4,2,2]),[1,2,2,2,3,3,4,4,6,7,8,9,20])


if __name__ == '__main__':
    unittest.main()
