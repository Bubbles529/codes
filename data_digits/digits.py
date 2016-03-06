#-*- coding: utf8 -*-
'''数据分析实践 digits'''
import numpy as np
import scipy as sp
import seaborn as sb
import pandas as pd

## 获取
from sklearn import datasets
digits = datasets.load_digits()
data.shape, target.shape, data.min(), data.max()

## 手写体可视化
plt.imshow(images[0], cmap='binary')

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap='binary')
    ax.text(0.05, 0.05, str(target[i]), transform=ax.transAxes, color='green')
    ax.set_xticks([]) #清除坐标
    ax.set_yticks([])

### PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar();

data_ = pca.inverse_transform(array([[-10,-20]]))
plt.imshow(data_[0].reshape((8,8)), cmap='binary')

### 还原
data_repca = pca.inverse_transform(data_pca)
images_repca = data_repca.copy()
images_repca.shape = (1797, 8, 8)

### PCA能量
sb.set()
pca_ = PCA().fit(data)
plt.plot(np.cumsum(pca_.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');

### IsoMap
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
data_projected = iso.fit_transform(data)
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=target,edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('nipy_spectral', 10));
plt.colorbar(label='digit label', ticks=range(10))
plt.clim(-0.5, 9.5)

### KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

clf = KNeighborsClassifier()
n_neighbors = [1,2,3,5,8,10,15,20,25,30,35,40]
weights = ['uniform','distance']
param_grid = [{'n_neighbors': n_neighbors, 'weights': weights}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_, 

### 逻辑回归
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')

C = [0.1, 0.5, 1, 5, 10]
param_grid = [{'C': C}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

grid_search.best_score_, grid_search.best_estimator_,grid_search.best_params_

### SVM
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
clf = SVC()

C = [0.1, 0.5, 1, 5, 10]
kernel = ['linear', 'poly', 'rbf']
param_grid = [{'C': C, 'kernel':kernel}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_
grid_search.grid_scores_

### 决策树
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()

criterion = ['gini','entropy']
max_depth = [10, 15, 20, 30, None]
min_samples_split = [2, 3, 5, 8, 10]
min_samples_leaf = [1, 2, 3, 5, 8]
param_grid = [{'criterion': criterion, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

### 随机森林
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)

n_estimators = [10, 20, 35, 50, 80, 100, 120, 150, 200]
param_grid = [{'n_estimators': n_estimators}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

 ##分错
```python
svc_ = SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
   decision_function_shape=None, degree=3, gamma='auto', kernel='poly',
   max_iter=-1, probability=False, random_state=None, shrinking=True,
   tol=0.001, verbose=False)
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2,
                                                random_state=2)
svc_.fit(Xtrain, ytrain)
svc_.score(Xtest, ytest)  ## 0.97499999999999998


from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, svc_.predict(Xtest)))


Ypre = svc_.predict(Xtest)
Xerror, Yerror, Ypreerror = Xtest[Ypre!=ytest], ytest[Ypre!=ytest], Ypre[Ypre!=ytest]
Xerror_images = Xerror.reshape((len(Xerror), 8, 8))
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(Xerror_images[i], cmap='binary')
    ax.text(0.05, 0.05, str(Yerror[i]), transform=ax.transAxes, color='green')
    ax.text(0.05, 0.2, str(Ypreerror[i]), transform=ax.transAxes, color='red')
    ax.set_xticks([]) #清除坐标
    ax.set_yticks([])

## Kmeans
from sklearn.cluster import KMeans
est = KMeans(n_clusters=10)
pres = est.fit_predict(data)

fig = plt.figure(figsize=(8, 3))
for i in range(10):
    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
    ax.imshow(est.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

trans = [7,0,3,6,1,4,8,2,9,5]
error_indexs = np.zeros(len(data))
for i in range(len(error_indexs)):
    if target[i] != trans[est.labels_[i]]:
        error_indexs[i] = 1
error_indexs = error_indexs != 0

fig, axes = plt.subplots(10, 10, figsize=(8, 8))
fig.subplots_adjust(hspace=0.1, wspace=0.1)
for i, ax in enumerate(axes.flat):
    ax.imshow(Xerror[i].reshape(8,8), cmap='binary')
    ax.text(0.05, 0.05, str(Yerror[i]), transform=ax.transAxes, color='green')
    ax.text(0.05, 0.3, str(trans[Ypreerror[i]]), transform=ax.transAxes, color='red')
    ax.set_xticks([]) #清除坐标
    ax.set_yticks([])

