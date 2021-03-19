#Basado en el trabajo propuesto en https://github.com/bmgee/votedperceptron/blob/master/votedperceptron/votedperceptron.py
from math import copysign, exp
from itertools import accumulate
import numpy as np
import pandas as pd
from numpy import linalg
import matplotlib.pyplot as plt
import random
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles
from sklearn.metrics import pairwise_kernels



class VotedPerceptron(object):
    def __init__(self, kernel_parameters):
        self.kernel = self.kernel_function(kernel_parameters)

        self.prediction_vector_term_coeffs = []
        self.prediction_vector_terms = []

        self.prediction_vector_votes = []

    def train(self, data, labels, error_threshold=0, max_epochs=1):
        self.check_data_dtype(data)

        data = np.insert(data, 0, 1, axis=1)

        if len(self.prediction_vector_terms) == 0:
            initial_prediction_vector = np.zeros(data.shape[1], dtype=data.dtype)
            self.prediction_vector_terms.append(initial_prediction_vector.copy())
            self.prediction_vector_term_coeffs.append(1)

        num_training_cases = data.shape[0]

        prediction_vector_vote = (self.prediction_vector_votes[-1]
                                  if len(self.prediction_vector_votes) > 0
                                  else 0)
        ronda = 1

        for _ in range(max_epochs):
            print("Epoch: "+str(ronda))
            ronda=ronda+1
            num_epoch_errors = 0
            for training_case, training_label in zip(data, labels):
                pre_activation = sum(pvtc * self.kernel(pvt, training_case) for pvtc, pvt in zip(self.prediction_vector_term_coeffs,self.prediction_vector_terms))
                       
                result = copysign(1, pre_activation)

                if result == training_label:
                    prediction_vector_vote += 1
                else:
                    num_epoch_errors += 1

                    # Save the prediction vector vote.
                    self.prediction_vector_votes.append(prediction_vector_vote)

                    # Save new prediction vector term and term coefficient.
                    self.prediction_vector_term_coeffs.append(training_label)#clases
                    self.prediction_vector_terms.append(training_case.copy())

                    # Reset prediction_vector_vote.
                    prediction_vector_vote = 1

            epoch_error = num_epoch_errors / num_training_cases

            if epoch_error <= error_threshold:
                break


        self.prediction_vector_votes.append(prediction_vector_vote)
        #print(len(self.prediction_vector_votes),"_______",len(self.prediction_vector_term_coeffs),"_____",len(self.prediction_vector_terms))

    def predict(self, input_vector, output_type='classification'):
        input_vector = np.insert(input_vector, 0, 1, axis=0)

        pv_pre_activations = accumulate(pvtc * self.kernel(pvt, input_vector) for pvtc, pvt in zip(self.prediction_vector_term_coeffs,self.prediction_vector_terms))

        pre_activation = sum(pvv* copysign(1, pvpa) for pvv, pvpa in zip(self.prediction_vector_votes, pv_pre_activations))

        if output_type == 'score':
            result = pre_activation
        elif output_type == 'classification':
            result = copysign(1, pre_activation)

        return result

    def error_rate(self, data, labels):
        self.check_data_dtype(data)
        predictions = np.asarray(
            [self.predict(d, 'classification') for d in data], dtype=labels.dtype
        )
        prediction_results = (predictions == labels)

        num_incorrect_prediction_results = np.sum(~prediction_results)
        num_prediction_results = prediction_results.shape[0]
        error_rate = num_incorrect_prediction_results / num_prediction_results

        return error_rate

    @staticmethod
    def kernel_function(kernel_parameters):
        def linear(vector_1, vector_2):
            """
            Linear Kernel
            """
            coef0 = kernel_parameters["coef0"]
            output = np.dot(vector_1, vector_2) + coef0
            return output
        def polynomial(vector_1, vector_2):
            """
            Polynomial Kernel
            """
            gamma = kernel_parameters["gamma"]
            coef0 = kernel_parameters["coef0"]
            degree = kernel_parameters["degree"]
            output = (gamma * np.dot(vector_1, vector_2) + coef0) ** degree
            return output
        def gaussian_rbf(vector_1, vector_2):
            """
            Gaussian Radial Basis Function Kernel
            """
            gamma = kernel_parameters["gamma"]
            vector_difference = vector_1 - vector_2
            output = exp(-gamma * np.dot(vector_difference, vector_difference))
            return output

        def gaussianRevised(vector_1,vector_2):
            """
            Gaussian Radial Basis Function Kernel, transponiendo el vector diferencia porque si no nodeja avanzar esta mkda
            """
            gamma = kernel_parameters["gamma"]
            vector_difference = vector_1 - vector_2
            output = exp(-gamma * np.dot(vector_difference.T, vector_difference))
            return output
        def gaussian20(vector_1,vector_2):
            """
            Gaussian implementado por nosotrps
            """
            gamma = kernel_parameters["gamma"]
            output = np.exp(-linalg.norm(vector_1-vector_2)**2 / (2 * (gamma ** 2)))
            return output

        kernel_choices = {'linear': linear,
                          'polynomial': polynomial,
                          'gaussian_rbf': gaussian_rbf,
                          'gaussianRevised':gaussianRevised,
                          'gaussian20':gaussian20}

        kernel_type = kernel_parameters['kernel_type']

        if kernel_type not in kernel_choices:
            raise NotImplementedError(kernel_type)

        kernel_choice = kernel_choices[kernel_type]

        return kernel_choice

    @staticmethod
    def check_data_dtype(data):
        if data.dtype not in (np.float32, np.float64):
            raise TypeError('data dtype required to be float32 or float64')

    @staticmethod
    def validate_inputs(input_vector_size, data, labels):
        if len(data) != len(labels):
            raise ValueError("Number of data items does not match"
                             + " the number of labels.")

        # Ensure self.input_vector_size matches size of each item in data.
        if any(input_vector_size != len(data_item) for data_item in data):
            raise ValueError("A data item size does not match"
                             + " input_vector_size.")

        # Ensure set of label values in [-1, 1].
        if not np.all(np.in1d(labels, [-1, 1])):
            raise ValueError("Valid label values are -1 and 1;"
                             + " adjust labels accordingly when calling"
                             + " this function.")


def plot_vpk(x, y_test, y_pred, title,filename):
    colors = np.where(y_test==y_pred,'b','r') #Azul bien clasificados
    markers = np.where(y_test==-1,'o','x') #Marcador o clase -1 y x clase 1
    ax=plt.gca()
    for x, y, c, m in zip(x[:, 0], x[:, 1], colors, markers):
        ax.scatter(x, y, alpha=0.8, c=c,marker=m) 

    plt.title(title)
    plt.axis("tight")
    plt.savefig(filename)
    plt.clf()

def plot_svk(X,y,clf, filename):
     #Graficar SVK - Tomado de ejemplo  https://scikit-learn.org/stable/auto_examples/svm/plot_svm_nonlinear.html
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

def generate_data(n_samples=1500,noise=0.05):
    random.seed(666)
    x,y =make_circles(n_samples=1500, noise = 0.05 )
    y = np.where(y==0,-1,1)
    X_train, y_train = x[:750],y[:750]
    X_test, y_test = x[750:], y[750:]
    y_train = y_train.astype('float64')
    y_test = y_test.astype('float64')

    circle1 = x[y==-1]
    circle2  = x[y==1]

    plt.figure()
    plt.scatter(circle1[:, 0], circle1[:, 1], c='b', marker='o', s=10)
    plt.scatter(circle2[:, 0], circle2[:, 1], c='r', marker='x', s=30)
    #plt.show()
    plt.savefig('data_problema.png')
    plt.clf()

    return X_train,y_train, X_test, y_test

def vpk_vs_svk(X_train,y_train, X_test, y_test,iteraciones,kernel_parameters):
    vpk_clf = VotedPerceptron(kernel_parameters)
    vpk_clf.train(X_train,y_train,max_epochs=iteraciones)
 
    svk_clf = svm.NuSVC(kernel="rbf",gamma='auto')
    svk_clf.fit(X_train, y_train)

    y_predict = []
    for row  in X_test:
        y_predict.append(vpk_clf.predict(row))
    y_predict_vpk =np.array(y_predict,dtype='float64')

    y_predict_svk = svk_clf.predict(X_test)
    
    roc_vpk = roc_auc_score(y_test, y_predict_vpk)
    roc_svk = roc_auc_score(y_test, y_predict_svk)

    plot_vpk(X_test,y_test=y_test,y_pred=y_predict_vpk,title='Voted Perceptron '+str(iteraciones)+" epoch, Kernel: "+kernel_parameters['kernel_type'],filename='pvk_%s_%s.png'%(kernel_parameters['kernel_type'],str(iteraciones)))
    plot_svk(X_test,y_test,svk_clf,"SVK_"+str(iteraciones)+".png")

    print(roc_vpk,roc_svk)

    return iteraciones,roc_vpk,roc_svk

def ejecucion(kernel_parameters,listaIter=[5,10,20,40]):
    X_train,y_train, X_test, y_test = generate_data()
    rocs = []
    for it in listaIter:
        print("Vuelta de iteraciones: "+str(it))
        iteracion, roc_vpk,roc_svk = vpk_vs_svk(X_train,y_train, X_test, y_test,it,kernel_parameters)
        rocs.append([iteracion,roc_vpk,roc_svk])
    pd.DataFrame(rocs,columns=['iteraciones','roc_vpk','roc_svk']).to_csv('rocs.csv',sep=";",decimal=",")


    

    

#KERNEL_PARAMETERS = {'kernel_type': 'gaussian_rbf', 'gamma': 0.03125}
KERNEL_PARAMETERS = {'kernel_type': 'gaussian20', 'gamma': 5}


ejecucion(KERNEL_PARAMETERS,listaIter=[5,10,20,40])

