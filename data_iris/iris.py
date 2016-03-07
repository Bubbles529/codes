#-*- coding: utf8 -*-
'''数据分析实践 iris'''

import numpy as np
import scipy as sp
import seaborn as sb
import pandas as pd
import matplotlib.pyplot as plt


def get_data_sklearn():
    from sklearn import datasets
    data = datasets.load_iris()
    data_ = (list(i)+[data['target_names'][j]] for i,j in zip(data['data'],data['target']))
    data_pd = pd.DataFrame(data_, columns=list(data['feature_names'])+['class'])
    data, target, target_names, feature_names = data['data'], data['target'], data['target_names'], data['feature_names']
    return data, target, target_names, feature_names,data_pd

def check_data():
    pass


def show_class_dis(data_pd):
    import collections
    counter = collections.Counter(data_pd['class'])
    vs = counter.most_common()
    show_table(['类别','样本数'],vs)


def show_table(colums, values, float_size=4):
    def get_str(x):
        if  isinstance(x, float):
            return '{:.4f}'.format(x)
        return str(x)
    sep = ['-' for i in colums]
    print('|'+'|'.join(colums)+ '|')
    print('|'+'|'.join(sep)+ '|')
    for v in values:
        print('|'+'|'.join([get_str(i) for i in v])+ '|')


from sklearn.decomposition import PCA
def PCA_Process(data, target, target_names):
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data)
    formatter = plt.FuncFormatter(lambda i, *args:target_names[int(i)])
    plt.figure(figsize=(8, 8))
    plt.scatter(data_pca[:, 0], data_pca[:, 1], c=target, edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('rainbow', len(target_names)))
    plt.colorbar(ticks=sorted(list(set(target))), format=formatter)
    return pca, data_pca
#pca, data_pca = PCA_Process(data, target, target_names)

def show_pca_en(data):
    sb.set()
    pca_ = PCA().fit(data)
    plt.plot(np.cumsum(pca_.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance');
    #plt.xlim(0,5)
    return pca_
#pca_ = show_pca_en(data)


from sklearn.manifold import Isomap
def iso_map(data, target, target_names):
    iso = Isomap(n_components=2)
    data_projected = iso.fit_transform(data)
    formatter = plt.FuncFormatter(lambda i, *args:target_names[int(i)])
    plt.figure(figsize=(8, 8))
    plt.scatter(data_projected[:, 0], data_projected[:, 1], c=target,edgecolor='none', alpha=0.5, cmap=plt.cm.get_cmap('rainbow', len(target_names)));
    plt.colorbar(ticks=sorted(list(set(target))), format=formatter)
    #plt.clim(-200, 0)
    return iso, data_projected
#iso, data_projected = iso_map(data, target, target_names)


### KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
def find_best_knn(data,target,cv):
    clf = KNeighborsClassifier()
    n_neighbors = [1,2,3,5,8,10,15,20,25,30,35,40]
    weights = ['uniform','distance']
    param_grid = [{'n_neighbors': n_neighbors, 'weights': weights}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(data, target)
    grid_search.cls_name = 'KNN'
    return grid_search
#grid_knn = find_best_knn(data,target,cv=10)
#grid_knn.grid_scores_
#grid_knn.best_score_, grid_knn.best_estimator_, grid_knn.best_params_


### 决策树
from sklearn.grid_search import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
def find_best_decisiontree(data, target, cv):
    clf = DecisionTreeClassifier()
    criterion = ['gini','entropy']
    max_depth = [10, 15, 20, 30, None]
    min_samples_split = [2, 3, 5, 8, 10]
    min_samples_leaf = [1, 2, 3, 5, 8]
    param_grid = [{'criterion': criterion, 'max_depth':max_depth, 'min_samples_split':min_samples_split, 'min_samples_leaf':min_samples_leaf}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(data, target)
    grid_search.cls_name = '决策树'
    return grid_search
#grid_dc_tree = find_best_decisiontree(data, target, cv=10)
#grid_dc_tree.grid_scores_
#grid_dc_tree.best_score_, grid_dc_tree.best_estimator_, grid_dc_tree.best_params_


### 逻辑回归
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
def find_best_logistic(data, target, cv):
    clf = LogisticRegression(penalty='l2')
    C = [0.1, 0.5, 1, 5, 10]
    param_grid = [{'C': C}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(data, target)
    grid_search.cls_name = '逻辑回归'
    return grid_search
#grid_logistic = find_best_logistic(data, target, cv=10)
#grid_logistic.grid_scores_
#grid_logistic.best_score_, grid_logistic.best_estimator_, grid_logistic.best_params_

### SVM
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
def find_best_svm(data, target, cv):
    clf = SVC()
    C = [0.1, 0.5, 1, 5, 10]
    kernel = ['linear', 'poly', 'rbf']
    param_grid = [{'C': C, 'kernel':kernel}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(data, target)
    grid_search.cls_name = 'SVM'
    return grid_search
#grid_svm = find_best_svm(data, target, cv=10)
#grid_svm.grid_scores_
#grid_svm.best_score_, grid_svm.best_estimator_, grid_svm.best_params_

### 随机森林
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
def find_best_random_forest(data, target, cv):
    clf = RandomForestClassifier(random_state=0)
    n_estimators = [10, 20, 35, 50, 80, 100, 120, 150, 200]
    param_grid = [{'n_estimators': n_estimators}]
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=cv)
    grid_search.fit(data, target)
    grid_search.cls_name = '随机森林'
    return grid_search
#grid_random_forest = find_best_random_forest(data, target, cv=10)
#grid_random_forest.grid_scores_
#grid_random_forest.best_score_, grid_random_forest.best_estimator_, grid_random_forest.best_params_

### 汇总最佳模型
def show_model_table():
    import sys
    #print(sys.modules[__name__].__dict__.items())
    grids = [ v for k,v in sys.modules[__name__].__dict__.items() if k.startswith('grid_')]
    vs = []
    for grid in grids:
        name = grid.best_estimator_.__class__.__name__
        best_para = grid.best_params_
        for i in grid.grid_scores_:
            if i[0] == best_para:
                score = i[1]
                std = np.std(i[2])
        vs.append([name, score, std])
            
    show_table(['Classifier','Mean Score', 'Std'],vs)


### 查看混淆矩阵
def get_all_best_model():
    import sys
    grids = [ v for k,v in sys.modules[__name__].__dict__.items() if k.startswith('grid_')]
    best_models = {grid.best_estimator_.__class__.__name__: grid.best_estimator_ for grid in grids}
    return best_models
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
def show_confusion_matrix(clsifier, data, target, target_names, random_state=0, test_size=0.0):
    Xtrain, Xtest, ytrain, ytest = train_test_split(data, target, test_size=test_size, random_state=random_state)
    clf = get_all_best_model()[clsifier]
    clf.fit(Xtrain, ytrain)
    pre = clf.predict(Xtest)
    print('score: ', clf.score(Xtest, ytest))
    print(confusion_matrix(ytest, pre))
    m = confusion_matrix(ytest, pre)
    vs = [[target_names[i]] + list(x) for i,x in enumerate(m)]
    show_table(['实际\预测']+list(target_names),vs)
    for c in range(len(target_names)):
        s = np.logical_and(pre==c, ytest==c).sum()/ (ytest==c).sum()
        print('class:', c, target_names[c],'rate:', s)
        error_index = np.logical_and(pre!=c, ytest==c)
        for d, p in zip(Xtest[error_index], pre[error_index]):
            print('predict to:', target_names[p], ' :', d)
#show_confusion_matrix('SVC', data, target, target_names,0, 0.2)
#show_confusion_matrix('KNeighborsClassifier', data, target, target_names,0, 0.2)
#show_confusion_matrix('LogisticRegression', data, target, target_names,0, 0.2)
#show_confusion_matrix('DecisionTreeClassifier', data, target, target_names,0, 0.2)
#show_confusion_matrix('RandomForestClassifier', data, target, target_names,0, 0.2)
