import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def linear_svm():
    print("=====   Linear SVM (Bank authentication) =====")
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("./bill_authentication.csv")

    # see the data
    bankdata.shape

    # see head
    bankdata.head()

    # data processing
    X = bankdata.drop('Class', axis=1)
    y = bankdata['Class']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    # train the SVM
    from sklearn.svm import SVC
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)

    # predictions
    y_pred = svclassifier.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("-----   Confusion Matrix   -----")
    print(confusion_matrix(y_test,y_pred))
    print("----- Classification Report -----")
    print(classification_report(y_test,y_pred))


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames)

    # process
    X = irisdata.drop('Class', axis=1)
    y = irisdata['Class']

    # train
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

    return X, y, X_train, X_test, y_train, y_test

def polynomial_kernel(X_train, X_test, y_train, y_test):
    # PolyNomial Kernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    poly_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="poly", degree=8))
    ])

    poly_kernel_svm_clf.fit(X_train, y_train)

    # predictions
    y_pred = poly_kernel_svm_clf.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("-----   Confusion Matrix   -----")
    print(confusion_matrix(y_test,y_pred))
    print("----- Classification Report -----")
    print(classification_report(y_test,y_pred))

    return poly_kernel_svm_clf

def gaussian_kernel(X_train, X_test, y_train, y_test):
    # Gaussian Kernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    rbf_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="rbf"))
    ])

    rbf_kernel_svm_clf.fit(X_train, y_train)

    # predictions
    y_pred = rbf_kernel_svm_clf.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("-----   Confusion Matrix   -----")
    print(confusion_matrix(y_test,y_pred))
    print("----- Classification Report -----")
    print(classification_report(y_test,y_pred))

    return rbf_kernel_svm_clf

def sigmoid_kernel(X_train, X_test, y_train, y_test):
    # Sigmoid Kernel
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sigmoid_kernel_svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm_clf", SVC(kernel="sigmoid"))
    ])

    sigmoid_kernel_svm_clf.fit(X_train, y_train)

    # predictions
    y_pred = sigmoid_kernel_svm_clf.predict(X_test)

    # Evaluate model
    from sklearn.metrics import classification_report, confusion_matrix
    print("-----   Confusion Matrix   -----")
    print(confusion_matrix(y_test,y_pred))
    print("----- Classification Report -----")
    print(classification_report(y_test,y_pred))

    return sigmoid_kernel_svm_clf

def test():
    linear_svm()
    X, y, X_train, X_test, y_train, y_test = import_iris()
    print("=====   Polynomial Kernel  =====")
    polynomial_kernel(X_train, X_test, y_train, y_test)
    print("=====   Gaussian Kernel  =====")
    gaussian_kernel(X_train, X_test, y_train, y_test)
    print("=====   Sigmoid Kernel  =====")
    sigmoid_kernel(X_train, X_test, y_train, y_test)

    print("\n\nPlotting the three Kernel Models with just 2 features (because 4D plots hard to visualize)")
    X = X.drop('petal-length', axis=1).drop('petal-width', axis=1).values
    y = list(map(class_to_int, y))

    X_train = X_train.drop('petal-length', axis=1).drop('petal-width', axis=1)
    y_train = list(map(class_to_int, y_train))

    X_test = X_test.drop('petal-length', axis=1).drop('petal-width', axis=1)
    y_test = list(map(class_to_int, y_test))

    print("=====   Polynomial Kernel (Sepal-Width + Sepal-Length)  =====")
    poly_clf = polynomial_kernel(X_train, X_test, y_train, y_test)
    plot(poly_clf, "Decision Surface for Polynomial Kernel", X, y)

    print("=====   Gaussian Kernel (Sepal-Width + Sepal-Length)  =====")
    gaussian_clf = gaussian_kernel(X_train, X_test, y_train, y_test)
    plot(gaussian_clf, "Decision Surface for Gaussian Kernel", X, y)

    print("=====   Sigmoid Kernel (Sepal-Width + Sepal-Length)  =====")
    sigmoid_clf = sigmoid_kernel(X_train, X_test, y_train, y_test)
    plot(sigmoid_clf, "Decision Surface for Sigmoid Kernel", X, y)

def plot(clf, title, X, y):
    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    fig, ax = plt.subplots()
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('Sepal-Width')
    ax.set_xlabel('Sepal-Length')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()

def class_to_int(input):
    # maps classificaton to integer for plotting purposes
    if input =='Iris-versicolor':
        return 1
    elif input == 'Iris-setosa':
        return 2
    else:
        return 3

test()
