#Basado en el trabajo disponible en https://gist.github.com/mblondel/656147
import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles


def linear_kernel(x1, x2):
    return np.dot(x1, x2)

def polynomial_kernel(x, y, p=3):
    return (1 + np.dot(x, y)) ** p

def gaussian_kernel(x, y, sigma=5.0):
    return np.exp(-linalg.norm(x-y)**2 / (2 * (sigma ** 2)))


class Perceptron(object):

    def __init__(self, T=1):
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0

        for t in range(self.T):
            for i in range(n_samples):
                if self.predict(X[i])[0] != y[i]:
                    self.w += y[i] * X[i]
                    self.b += y[i]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(self.project(X))

class KernelPerceptron(object):

    def __init__(self, kernel=linear_kernel, T=1):
        self.kernel = kernel
        self.T = T

    def fit(self, X, y):
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        self.alpha = np.zeros(n_samples, dtype=np.float64)

        # Gram matrix
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self.kernel(X[i], X[j])

        for t in range(self.T):
            for i in range(n_samples):
                if np.sign(np.sum(K[:,i] * self.alpha * y)) != y[i]:
                    self.alpha[i] += 1.0

        # Support vectors
        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        print("%d support vectors out of %d points" % (len(self.alpha),n_samples))

    def project(self, X):
        y_predict = np.zeros(len(X))
        for i in range(len(X)):
            s = 0
            for a, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                s += a * sv_y * self.kernel(X[i], sv)
            y_predict[i] = s
        return y_predict

    def predict(self, X):
        X = np.atleast_2d(X)
        n_samples, n_features = X.shape
        #np.hstack((X, np.ones((n_samples, 1))))
        return np.sign(self.project(X))

if __name__ == "__main__":
    import pylab as pl

    def plot_vpk(X1_train, X2_train, clf, title,filename):
        pl.plot(X1_train[:,0], X1_train[:,1], "ro")
        pl.plot(X2_train[:,0], X2_train[:,1], "bo")
        pl.scatter(clf.sv[:,0], clf.sv[:,1], s=100, c="g")

        X1, X2 = np.meshgrid(np.linspace(-1.5,1.5,50), np.linspace(-1.5,1.5,50))
        X = np.array([[x1, x2] for x1, x2 in zip(np.ravel(X1), np.ravel(X2))])
        Z = clf.project(X).reshape(X1.shape)
        pl.contour(X1, X2, Z, [0.0], colors='k', linewidths=1, origin='lower')
        plt.title(title) 

        pl.axis("tight")
        pl.savefig(filename)
        pl.clf()
        #pl.show()

    def plot_svk(X,y,clf, filename):
        xx, yy = np.meshgrid(np.linspace(-1.5, 1.5, 500),np.linspace(-1.5, 1.5, 500))
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
        contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
              
                       linestyles='dashed')
        plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
            edgecolors='k')
        plt.xticks(())
        plt.yticks(())
        plt.axis([-1.5, 1.5, -1.5, 1.5])
        plt.title('SVM Kernel RBF, gamma=Auto') 
        plt.savefig(filename)
        plt.clf()
        #plt.show()

    def vpk_vs_svk(iteracion,funcionVPK):
        #Generamos train y test
        x,y =make_circles(n_samples=1500, noise = 0.05 )
        y = np.where(y==0,-1,1)
        X_train, y_train = x[:900],y[:900]
        X_test, y_test = x[900:], y[900:]

        #Clasificadores vpk = voted perceptron kernelizado, svk = support vector machine kernel
        #Instanciamos el clasificador de vpk
        vpk_clf = KernelPerceptron(funcionVPK, T=20)
        vpk_clf.fit(X_train, y_train)

        #Instanciamos cls svk
        svk_clf = svm.NuSVC(kernel="rbf",gamma='auto')
        svk_clf.fit(X_train, y_train)

        #predicciones
        y_predict_vpk = vpk_clf.predict(X_test)
        y_predict_svk = svk_clf.predict(X_test)

        roc_vpk = roc_auc_score(y_test, y_predict_vpk)
        roc_svk = roc_auc_score(y_test, y_predict_svk)
        
        #Graficar VPK
        plot_vpk(X_test[y_test==1], X_test[y_test==-1], vpk_clf,'Voted Perceptron with '+funcionVPK.__name__,'VPK_'+str(iteracion)+".png")
        
        #Graficar SVK - Tomado de ejemplo  https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html
        plot_svk(X_test,y_test,svk_clf,"SVK_"+str(iteracion)+".png")

        return roc_vpk, roc_svk

    rocs = []
    for i in range(1,11):
        random.seed(666)
        roc_vpk, roc_svk = vpk_vs_svk(i,gaussian_kernel)
        rocs.append([roc_vpk,roc_svk])
    pd.DataFrame(rocs,columns=['roc_vpk','roc_svk']).to_csv('rocs.csv',sep=";",decimal=",")



