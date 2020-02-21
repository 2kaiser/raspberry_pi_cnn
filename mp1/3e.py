import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from mlxtend.plotting import plot_decision_regions
import numpy as np
import scipy
x = [[1, 0], [0, 1],[0, 0], [-1, -1]]
y = [1, 1,-1,-1]
clf = svm.LinearSVC(fit_intercept=True,random_state=True,max_iter=10000,dual=False,C=10000)
clf.fit(x, y)
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]

fig = plt.figure()
#plt.scatter(x[0][0],x[0][1], marker='+')
#plt.scatter(x[1][0],x[1][1], marker='+')
#plt.scatter(x[2][0],x[2][1], c= 'green')
#plt.scatter(x[3][0],x[3][1], c= 'green', marker='o')

#plt.plot(xx, yy, 'k-')
print(clf.intercept_[0])
print(w)
#plot_decision_regions(X=x,y = y,clf=clf)
