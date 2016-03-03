#-*- coding:utf8 -*-
'''
k-means聚类算法实现：
随机选择k个初始的质心
根据距离进行点的簇分配结果，如果发现改变
    对数据集的每个数据点
         对每个质心，计算质心与数据点的距离
         将点分配到距离其最近的簇
    对每个簇计算簇中的所有点的均值并将均值作为质心
'''

import numpy as np
import scipy as sp

class KMeansCluster():

    def fit(self, dataset, k):
        '''处理数据'''
        centers = self._get_random_center(dataset, k)
        clusters = None
        dist = np.zeros(k)
        
        while True:
            new_clusters = self._asign_data_by_center(dataset, centers, dist)
            #结束条件是聚类不再变化且每类别均有数据点
            if np.all(clusters == new_clusters) and np.all(dist != 0): 
                break
            self._update_centers(dataset, new_clusters, k, centers)
            clusters = new_clusters

        return clusters, centers, dist


    def _asign_data_by_center(self, dataset, centers, dist):
        '''根据质心分配数据点'''
        num = dataset.shape[0]
        clusters = np.zeros(num)
        distances = [0]*len(centers)
        dist[:] = 0
        
        for i in range(num):
            row = dataset[i]
            
            for c, center in enumerate(centers):
                distances[c] = self._calc_eclud_distance(row, center)
            index = np.argsort(distances)[0]
            
            clusters[i] = index
            dist[index] += 1
            
        return clusters


    def _update_centers(self, dataset, clusters, k, centers):
        '''根据聚类结果计算新的质心，原地修改'''
        for i in range(k):
            num = np.sum(clusters == i)
            sum_ = np.sum(dataset[clusters == i],axis=0)
            if num > 0:
                centers[i] = sum_ / num
            else:
                centers[i] = self._get_random_center(dataset, 1)[0]

        return
            

    @staticmethod
    def _calc_eclud_distance(row_a, row_b):
        '''计算欧几里得距离'''
        return np.sqrt(np.sum(np.power(row_a - row_b, 2)))


    def _get_random_center(self, dataset, k):
        '''随机获取初始质点'''
        _, featrure_n = dataset.shape

        max_of_features = np.max(dataset, axis=0)
        min_of_features =  np.min(dataset, axis=0)
        range_of_features =  max_of_features - min_of_features

        centers = np.random.random((k, featrure_n))*range_of_features + min_of_features

        return centers
        
