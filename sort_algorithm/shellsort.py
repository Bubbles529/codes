#-*- cdoing:utf8 -*-
'''
Shell排序
'''

def shell_sort(array):
    '''shell排序'''
    d = int(len(array) / 2)
    while d > 0:
        for i in range(d):
            for k in range(i, len(array), d):
                for m in range(i, k, d):
                    if array[k] < array[m]:
                        temp = array[k]
                        array[m + d: k + d: d] = array[m :  k: d]
                        array[m] = temp
        d = int(d/2)
    return array

import unittest

class TestSort(unittest.TestCase):
    
    def _sort(self, array):
        shell_sort(array)
        return array

    
    def test_sort(self):
        self.assertEqual(self._sort([]),[])
        self.assertEqual(self._sort([1]),[1])
        self.assertEqual(self._sort([2,2]),[2,2])
        self.assertEqual(self._sort([3,2,1]),[1,2,3])
        self.assertEqual(self._sort([3,9,20,2,8,1,4,6,7,3,4,2,2]),[1,2,2,2,3,3,4,4,6,7,8,9,20])


if __name__ == '__main__':
    unittest.main()
