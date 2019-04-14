import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pandas as pd

from Lab7 import import_iris, sigmoid_kernel, class_to_int


X, y, X_train, X_test, y_train, y_test = import_iris()
X = X.drop('petal-length', axis=1).drop('petal-width', axis=1).values
y = list(map(class_to_int, y))
y_train = list(map(class_to_int, y_train))
y_test = list(map(class_to_int, y_test))

clf = sigmoid_kernel(X_train, X_test, y_train, y_test)

def plot(clf, title, X, y):

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy,petal_length, petal_width, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel(), [petal_length]*len(xx.ravel()), [petal_width]*len(yy.ravel())])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out


    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy, 1, 1, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.set_ylabel('Sepal-Width')
    ax.set_xlabel('Sepal-Length')
    ax.set_title(title)
    ax.legend()

    def update(val):
        ax.clear()
        plot_contours(ax, clf, xx, yy, petal_length.val, petal_width.val, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.set_ylabel('Sepal-Width')
        ax.set_xlabel('Sepal-Length')
        ax.set_title(title)

    axcolor = 'lightgoldenrodyellow'
    ax_petal_length = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor=axcolor)
    ax_petal_width = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    petal_length = Slider(ax_petal_length, 'Petal-Length', 0.1, 8.0, valinit=1)
    petal_width = Slider(ax_petal_width, 'Petal-Width', 0.1, 8.0, valinit=1)
    petal_length.on_changed(update)
    petal_width.on_changed(update)
    plt.show()


plot(clf, "Decision Surface for Sigmoid Kernel", X, y)
