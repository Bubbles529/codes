#-*- coding:utf8 -*-

import unittest
import kmeans
import importlib
from numpy import *

importlib.reload(kmeans)

class TestKMeans(unittest.TestCase):

    def test_kmeans(self):
        dataset = array([[1.9,1.9],[2.1,2.1],[10,10],[20.5,21],[21,19.5],[17,18]])
        clusters, centers, dist = kmeans.KMeansCluster().fit(dataset, 3)
        expect_clusters, expect_dist = array([0,0,1,2,2,2]), array([2,1,3])
        for _ in range(1000):
            self.assertTrue(all((clusters==clusters[0]) == (expect_clusters==expect_clusters[0])))
            self.assertTrue(all((clusters==clusters[2]) == (expect_clusters==expect_clusters[2])))
            self.assertTrue(all((clusters==clusters[3]) == (expect_clusters==expect_clusters[3])))
        

    def test__update_centers(self):
        dataset = array([[1.9,1.9],[2.1,2.1],[3,3]])
        clusters = array([1,1,0])
        centers = zeros((2,2))
        kmeans.KMeansCluster()._update_centers(dataset, clusters, 2, centers)
        self.assertTrue(all(centers == array([[3,3],[2,2]])))


    def test__get_random_center(self):
        for _ in range(200):
            z = array([[1,2,3,4,5],[2,3,4,5,6]])
            l = kmeans.KMeansCluster()._get_random_center(z, 1)
            self.assertTrue(all(i<= k <j for i,k,j in zip([1,2,3,4,5], list(l),[2,3,4,5,6])))


    def test__calc_eclud_distance(self):
        row_a, row_b = array([1,2,3,4,5]),array([2,3,4,5,6])
        distance = kmeans.KMeansCluster._calc_eclud_distance(row_a, row_b)
        self.assertTrue( 2.236 < distance < 2.2361)


if __name__ == '__main__':
    unittest.main()
