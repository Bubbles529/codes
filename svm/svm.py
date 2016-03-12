#-*- coding:utf8 -*-

from numpy import *
import logging
import random

class SVM():
    '''SVM，当前只实现了线性'''

    def __init__(self, C=1.0, toler=0.001, max_iter=30,  alpha_change_delta = 0.00001):
        '''初始化'''
        self.C = C
        self.toler = toler
        self.max_iter = max_iter
        self.alpha_change_delta =  alpha_change_delta


    def fit(self, dataMatIn, classLabels):
        '''训练数据,输入为二维array，以及一维array'''
        self.labels = list(set(classLabels))
        self.smo_soloers = []

        X = mat(dataMatIn)
        for label in self.labels:
            Y = zeros(classLabels.shape)
            Y[classLabels==label] = 1
            Y[classLabels!=label] = -1
            smo_ = self.SmoSolver(X,mat(Y).T, self.C, self.toler, self.max_iter, self.alpha_change_delta)
            smo_.solver()
            smo_.plot_figure()
            self.smo_soloers.append(smo_)
            if len(self.labels) == 2:
                break

        return self
    

    def predict(self, x_array):
        '''预测数据，输入为array'''
        logging.warn('predict for 1 of {}'.format(len(self.labels)))
        x = mat(x_array)
        if len(self.labels) == 2:
            v = self.smo_soloers[0].calc_fx(x)
            return self.labels[0] if v>0 else self.labels[1]

        vs = [smo.calc_fx(x) for smo in self.smo_soloers]
        index = argmax(vs)
        return self.labels[index]


    class SmoSolver():
        '''SMO求解'''
        
        def __init__(self,  X, Y, C, toler,max_iter, alpha_change_delta):
            '''初始化'''
            self.X = X
            self.Y = Y
            self.C = C
            self.tol = toler
            self.max_iter = max_iter
            self.alpha_change_delta = alpha_change_delta
            
            self.m = shape(X)[0]
            self.alphas = mat(zeros((self.m,1)))
            self.b = 0
            self.error_cache = mat(zeros((self.m,2)))   # 用以记录历史上的错误值，便于较快选择第二个点


        def calc_fx(self, x):
            '''计算预测值 y_j = \sum y_i alpha_i (x_i x_j) + b'''
            f = float(multiply(self.alphas,self.Y).T*(self.X* x.T)) + self.b
            return f


        def get_fx_i(self, i):
            '''计算 i 训练项对应的值'''
            return self.calc_fx(self.X[i,:])


        def get_error(self, k):
            '''计算当前模型下第k组x的误差'''
            error = self.calc_fx(self.X[k,:]) - float(self.Y[k])
            return error


        def need_change_alpha(self, i):
            '''判断alpha是否需要修改'''
            # yu < 1，说明在支持边界外，此时alpha应该为C
            if self.Y[i]*self.get_fx_i(i) < 1 - self.tol and self.alphas[i] < self.C:
                return True

            # yu>1，说明此时分类正确并不为支持向量，则alpha应该为0
            if self.Y[i] * self.get_fx_i(i) > 1 + self.tol and self.alphas[i] > 0:
                return True


        def select_second_alpha(self, i):
            '''在alpha i基础上选择另外一个alpha'''
            max_K = -1
            max_delta_error = 0
            error_i = self.get_error(i)

            # 优先选择之前改变过的alpha
            valid_error_cache = nonzero(self.error_cache[:,0].A)[0]
            if  len(valid_error_cache) > 1:
                for k in valid_error_cache: 
                    if k == i:
                        continue

                    error_k = self.get_error(k)
                    delta_error = abs(error_i - error_k)
                    if  delta_error > max_delta_error :
                        max_K = k
                        max_delta_error = delta_error
                return max_K

            # 随机选择一个
            else:  
                while  True:
                    max_K = int(random.uniform(0,self.m))
                    if max_K != i:
                        return max_K
                

        def calc_L_H(self, i, j):
            '''计算j可以更改的最大最小范围，由alpha2 =  - y1y2 alpha1 + E'''
            # alpha2 = alpha1 + E
            if  self.Y[i] != self.Y[j]:
                L = max(0, self.alphas[j] - self.alphas[i])  # = 两边都加上 -alpha1
                H = min(self.C, self.C + self.alphas[j] - self.alphas[i]) # = 两边都加上 C-alpha1

            # alpha2 = -alpha1 + E
            else:
                L = max(0, self.alphas[j] + self.alphas[i] - self.C)  # = 两边都加上 alpha1-C
                H = min(self.C, self.alphas[j] + self.alphas[i])  # = 两边都加上 alpha1

            return L,H


        def clip_value(self, v,H,L):
            '''数据截取到正常上下限内'''
            if v > H: 
                v = H
            if L > v:
                v = L
            return v


        def set_error_cache(self, i):
            '''添加缓存的error'''
            v = self.get_error(i)
            self.error_cache[i] = [1, v]


        def solver(self):
            '''SMO算法求解支持向量'''
            iter_time = 0
            alpha_changed = 0
            choose_all_set = True
            
            while iter_time < self.max_iter and (alpha_changed > 0  or choose_all_set):
                alpha_changed = 0

                # 选择整个数据集上所有alpha
                if choose_all_set:
                    for i in range(self.m):
                        alpha_changed += self.inner_change_alpha(i)
                        choose_all_set = False

                # 选择alpha位于０到Ｃ之间
                else:
                    alphas_0_C = nonzero((self.alphas.A > 0) * (self.alphas.A < self.C))[0]
                    for i in alphas_0_C:
                        alpha_changed += self.inner_change_alpha(i)
                        if alpha_changed == 0:
                            choose_all_set = True

                iter_time += 1


        def inner_change_alpha(self, i):
            '''选择alpha并计算变化量'''
            if not self.need_change_alpha(i):
                return 0

            j = self.select_second_alpha(i)
            alpha_i_old = self.alphas[i].copy();
            alpha_j_old = self.alphas[j].copy();

            L,H = self.calc_L_H(i, j)
            if L==H:
                return 0

            #计算最小极值的改变量
            eta = 2.0 * self.X[i,:]*self.X[j,:].T - self.X[i,:]*self.X[i,:].T - self.X[j,:]*self.X[j,:].T
            if eta >= 0:
                print("eta>=0")
                return 0

            error_i, error_j = self.get_error(i) , self.get_error(j)
            self.alphas[j] -= self.Y[j]*(error_i - error_j)/eta
            self.alphas[j] = self.clip_value(self.alphas[j],H,L)
            self.set_error_cache(j)
            
            if (abs(self.alphas[j] - alpha_j_old) < self.alpha_change_delta):
                print("j not moving enough")
                return 0   
            self.alphas[i] += self.Y[j]*self.Y[i]*(alpha_j_old - self.alphas[j])
            self.set_error_cache(i)

            b_i = self.b - error_i - self.Y[i]*(self.alphas[i]-alpha_i_old)*self.X[i,:]*self.X[i,:].T - self.Y[j]*(self.alphas[j]-alpha_j_old)*self.X[i,:]*self.X[j,:].T
            b_j = self.b - error_j - self.Y[i]*(self.alphas[i]-alpha_i_old)*self.X[i,:]*self.X[j,:].T - self.Y[j]*(self.alphas[j]-alpha_j_old)*self.X[j,:]*self.X[j,:].T
            
            if 0 < self.alphas[i] < self.C:
                self.b = b_i
            elif  0 < self.alphas[j] < self.C:
                self.b = b_j
            else:
                self.b = (b_i + b_j)/2.0
            return 1


        def calc_w(self):
            '''计算 w 的值'''
            m,n = shape(self.X)
            w = zeros((n,1))
            for i in range(m):
                w += multiply(self.alphas[i]*self.Y[i], self.X[i,:].T)
            return w


        def plot_figure(self):
            if self.X.shape[1] != 2:
                return

            x_1 = self.X[:,0].min() -1, self.X[:,0].max() +1
            x_2 = self.X[:,1].min() -1, self.X[:,1].max() +1
            w = self.calc_w()

            x = linspace(x_1[0], x_1[1], 50, endpoint=True)
            y = (- x*w[0,0] - self.b)/w[1,0]

            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            plt.figure()
            plt.plot(x,y.flat)
            plt.scatter(self.X[:,0].T.A[0], self.X[:,1].T.A[0], c=self.Y.T.A[0])
            for i, alpha in enumerate(self.alphas):
                if 0 < alpha < self.C:
                    pass
                    #plt.plot([self.X[i,0]], [self.X[i,1]], markersize=10, fillstyle=Line2D.fillStyles[0],  marker='o',markerfacecoloralt='gray',color='cornflowerblue')
                    #plt.scatter([self.X[i,0]], [self.X[i,1]], s=125, fillstyle=None)
            plt.ylim(x_2[0], x_2[1])
            plt.xlim(x_1[0], x_1[1])

            
