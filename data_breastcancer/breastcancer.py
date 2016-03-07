#-*- coding: utf8 -*-
'''数据分析实践 breast cancer'''
import numpy as np
import scipy as sp
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt

#获取数据
from sklearn import datasets
cancers = datasets.load_breast_cancer()
data_ = (list(i)+[cancers['target_names'][j]] for i,j in zip(cancers['data'],cancers['target']))
cancers_pd = pd.DataFrame(data_, columns=list(cancers['feature_names'])+['cancer type'])

#判断数据是否有NAN
cancers_pd.isnull()
#判断数据是否有0
cancers_pd.min(axis=0)

sb.distplot(cancers_pd['mean concavity'])
sb.distplot(cancers_pd['mean concave points'])
sb.distplot(cancers_pd['worst concavity'])
sb.distplot(cancers_pd['worst concave points'])


data, target, target_names = cancers['data'], cancers['target'], cancers['target_names']
data.shape
targetstr = np.array([target_names[i]] for i in target)

import collections
counter = collections.Counter(cancers_pd['cancer type'])
counter.most_common(2)


## PCA降维
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)
plt.scatter(data_pca[:, 0], data_pca[:, 1], c=target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('rainbow', 2))
plt.colorbar();

## PCA能量
sb.set()
pca_ = PCA().fit(data)
plt.plot(np.cumsum(pca_.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance');
plt.xlim(0,5)

## IsoMap降维
from sklearn.manifold import Isomap
iso = Isomap(n_components=2)
data_projected = iso.fit_transform(data)
plt.scatter(data_projected[:, 0], data_projected[:, 1], c=target,edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('rainbow', 2));
plt.colorbar(label='Cancer', ticks=range(2))
plt.clim(-200, 0)

### KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV

clf = KNeighborsClassifier()
n_neighbors = [1,2,3,5,8,10,15,20,25,30,35,40]
weights = ['uniform','distance']
param_grid = [{'n_neighbors': n_neighbors, 'weights': weights}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)
grid_search.grid_scores_
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_

 ### 逻辑回归
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')
C = [0.1, 0.5, 1, 5, 10]
param_grid = [{'C': C}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)
grid_search.grid_scores_
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_

### SVM
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
clf = SVC()

C = [0.1, 0.5, 1, 5, 10]
kernel = ['linear', 'poly', 'rbf']
param_grid = [{'C': C, 'kernel':kernel}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

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
grid_search.grid_scores_
grid_search.best_score_, grid_search.best_estimator_, grid_search.best_params_

### 随机森林
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)

n_estimators = [10, 20, 35, 50, 80, 100, 120, 150, 200]
param_grid = [{'n_estimators': n_estimators}]
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=10)
grid_search.fit(data, target)

###混淆矩阵查看
from sklearn.ensemble import RandomForestClassifier
clf  = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
             max_depth=None, max_features='auto', max_leaf_nodes=None,
             min_samples_leaf=1, min_samples_split=2,
             min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1,
             oob_score=False, random_state=0, verbose=0, warm_start=False)
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2,
                                                random_state=2)
clf.fit(Xtrain, ytrain)
clf.score(Xtest, ytest)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, clf.predict(Xtest)))

from sklearn.linear_model import LogisticRegression
clf  = LogisticRegression(C=0.5, class_weight=None, dual=False, fit_intercept=True,
           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
           verbose=0, warm_start=False)
from sklearn.cross_validation import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=0.2,
                                                random_state=5)
clf.fit(Xtrain, ytrain)
clf.score(Xtest, ytest)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, clf.predict(Xtest)))
TP = np.logical_and(clf.predict(Xtest)==1, ytest==1).sum()
FP = np.logical_and(clf.predict(Xtest)==1, ytest==0).sum()
FN = np.logical_and(clf.predict(Xtest)==0, ytest==1).sum()
TN = np.logical_and(clf.predict(Xtest)==0, ytest==0).sum()
print(TP, FP,FN,TN, TP/(TP+FN),FP/(TN+FP))

## ROC
from sklearn.metrics import roc_curve, auc, roc_auc_score
pa = clf.predict_proba(Xtest)[:,1]/clf.predict_proba(Xtest)[:,0]
fpr, tpr, thresholds = roc_curve(ytest, pa, pos_label=1)
plt.plot(fpr, tpr)
plt.xlim(0.0, 0.4)
plt.ylim(0.75, 1.0)

list(zip(fpr, tpr, thresholds))

## 提高真阳率
new_pre = (clf.predict_proba(Xtest)[:,1]/clf.predict_proba(Xtest)[:,0]>0.45)*np.ones(len(Xtest))

from sklearn.metrics import confusion_matrix
print(confusion_matrix(ytest, new_pre))
TP = np.logical_and(new_pre==1, ytest==1).sum()
FP = np.logical_and(new_pre==1, ytest==0).sum()
FN = np.logical_and(new_pre==0, ytest==1).sum()
TN = np.logical_and(new_pre==0, ytest==0).sum()
print(TP, FP,FN,TN, TP/(TP+FN),FP/(TN+FP))


