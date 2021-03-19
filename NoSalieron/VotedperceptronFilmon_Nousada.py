
import numpy as np
from numba import njit, prange
from math import copysign
from tqdm import tqdm
from joblib import Parallel, delayed
import faulthandler

faulthandler.enable()


@njit
def train(X, y, epochs, kernel_degree):
    """Train the voted perceptron.

            Parameters
            ----------
            X : ndarray
                An ndarray where each row is a training case of the mnist dataset.

            y : ndarray
                An ndarray where each element is the label/classification of a
                training case in train_set for binary classification.
                Valid label values are -1 and 1.

            epochs : int
                The number of epochs to train.

            kernel_degree: int
                The number of the degree in the polynomial kernel
            """

    # prediction vector's xi
    # v_train_indices = []
    # prediction vector's labels
    # v_label_coeffs = []
    # weights of the prediction vectors

    # v1 = np.zeros(X.shape[1])
    # don't call np.array on a np.array var
    v_train_indices = np.array([0], dtype=np.int64)
    v_label_coeffs = np.array([0], dtype=np.int64)
    c = np.array([0], dtype=np.int64)
    # they all have value = 0
    # i will not consider the first elements
    weight = 0
    mistakes = 0

    for _ in range(epochs):
        # for xi, label in zip(X, y):
        # numba don't support nested arrays

        for i in range(X.shape[0]):
            xi = X[i]
            label = y[i]
            # same here i can't use sum over the prediction vector
            # we need to iterate over a variable
            # we define a new function
            y_hat = copysign(1, implicit_form_product(
                X, v_train_indices, v_label_coeffs, xi, kernel_degree)[-1])
            # we take always the last prediction vector's product
            # complexity of implicit_form_product is O(k)
            if y_hat == label:
                weight = weight + 1
            else:
                c = np.append(c, np.array([weight]), axis=0)
                v_train_indices = np.append(
                    v_train_indices, np.array([i]), axis=0)
                v_label_coeffs = np.append(
                    v_label_coeffs, np.array([label]), axis=0)
                # reset #C_k+1 = 1
                weight = 1
                mistakes = mistakes + 1

    c = np.append(c, np.array([weight]), axis=0)
    c = c[1:c.shape[0]]
    return v_train_indices, v_label_coeffs, c, mistakes


@njit  # (parallel=True)
def implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    # the first dot_product is y0 = 1 *polynomial_expansion(x0 = 0_vect,x)
    v_x[0] = polynomial_expansion(np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    for k in range(1, v_train_indices.shape[0]):
        xi = X[v_train_indices[k]]
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * polynomial_expansion(xi, x, kernel_degree)

    return v_x


@njit  # (parallel=True)
def implicit_form_v(X, v_train_indices, v_label_coeffs):
    v = np.empty(v_train_indices.shape[0], dtype=np.float32)
    # v0
    v[0] = 0
    # the first product is y0 = 1 * x0 = 0_vect
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        xi = X[v_train_indices[k]]
        v[k] = v[k - 1] + (yi * xi)

    return v


@njit
def polynomial_expansion(xi, xj, d):
    return (1 + np.dot(xi, xj)) ** d


# _________________________________________________________________________________
# prediction functions


@njit
def last_unnormalized(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    """Compute score using the final prediction vector(unnormalized)"""
    """ x: unlabeled instance"""
    score = implicit_form_product(X,
                                  v_train_indices, v_label_coeffs, x, kernel_degree)[-1]

    return score


@njit
def normalize(score, v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return score
    return score / norm


@njit
def last_normalized(X, v_train_indices, v_label_coeffs, x, kernel_degree):
    """Compute score using the final prediction vector(normalized)"""
    """ x: unlabeled instance"""
    score = last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)

    return normalize(score, implicit_form_v(X, v_train_indices, v_label_coeffs)[-1])


@njit
def vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the deterministic leave-one-out conversion"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * copysign(1, v_x)

    return np.sum(s)


@njit
def avg_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using an average of the prediction vectors"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * v_x

    return np.sum(s)


@njit
def avg_normalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using an average of the prediction vectors(normalized)"""
    """ x: unlabeled instance"""

    dot_products = implicit_form_product(X,
                                         v_train_indices, v_label_coeffs, x, kernel_degree)
    v = implicit_form_v(X, v_train_indices, v_label_coeffs)
    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * normalize(v_x, v[i])

    return np.sum(s)


@njit
def random_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    r = np.random.randint(t + 1)
    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)[rl]
    return score


@njit
def random_normalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.(normalized)”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    # np.random.random_integers(low=0, high=t)  inclusive(low and high)
    # numba doesn't support random_integers
    # randint is exclusive
    r = np.random.randint(t + 1)

    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)

    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)[rl]

    return normalize(score, implicit_form_v(X, v_train_indices, v_label_coeffs)[rl])


@njit
def highest_score_arg(s):
    return np.argmax(s)


@njit
def highest_score(s):
    return np.max(s)


# _________________________________________________________________________________
# model functions

# not numba
def fit(X, y, epoch, kernel_degree):
    return Parallel(n_jobs=4, prefer="threads")(delayed(model)(X, y, i, epoch, kernel_degree) for i in range(10))


@njit
def model(X, y, class_type, epoch, kernel_degree):
    y = np.where(y == class_type, 1, -1)
    if epoch < 1:
            # contiguous arrays
        fraction_x = X[0:int(X.shape[0] * epoch),
                       :].copy()
        fraction_y = y[0:int(X.shape[0] * epoch)].copy()
        return train(fraction_x, fraction_y, 1, kernel_degree)
    return train(X, y, epoch, kernel_degree)


@njit
def predictions(X, v_train_indices, v_label_coeffs, c, x, kernel_degree):
    s_random = random_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree)
    s_last = last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree)
    s_avg = avg_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree)
    s_vote = vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree)

    return np.array([s_random, s_last, s_avg, s_vote])


# _________________________________________________________________________________
# Kernel Matrix version (Gram)

# for the kernel matrix
@njit
def gram_train_build(X,Y, kernel_degree):
    Gram = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(Gram.shape[0]):
        for j in range(i, Gram.shape[0]):
            if i <= j:
                Gram[i, j] = polynomial_expansion(X[i], Y[j], kernel_degree)
                Gram[j, i] = Gram[i, j]
    return Gram

@njit
def gram_test_build(X,Y, kernel_degree):
    Gram = np.zeros((X.shape[0], Y.shape[0]), dtype=np.float32)
    for i in range(Gram.shape[0]):
        for j in range(Gram.shape[1]):
                Gram[i, j] = polynomial_expansion(X[i], Y[j], kernel_degree)
    return Gram


def gram_fit(X, y, epoch, kernel_degree):
    return Parallel(n_jobs=2, prefer="threads")(delayed(gram_model)(X, y, i, epoch, kernel_degree) for i in range(10))


@njit
def gram_model(X, y, class_type, epoch, kernel_degree):
    y = np.where(y == class_type, 1, -1)
    if epoch < 1:
            # contiguous arrays
        fraction_x = X[0:int(X.shape[0] * epoch),
                       :].copy()
        fraction_y = y[0:int(X.shape[0] * epoch)].copy()
        return gram_train(fraction_x, fraction_y, 1, kernel_degree)
    return gram_train(X, y, epoch, kernel_degree)


@njit
def gram_predictions(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    s_random = gram_random_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index)
    s_last = gram_last_unnormalized(
        X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)
    s_avg = gram_avg_unnormalized(
        X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index)
    s_vote = gram_vote(X, v_train_indices, v_label_coeffs,
                       c, x, kernel_degree, gram_index)

    return np.array([s_random, s_last, s_avg, s_vote])


@njit
def gram_train(X, y, epochs, kernel_degree):
    v_train_indices = np.array([0], dtype=np.int64)
    v_label_coeffs = np.array([0], dtype=np.int64)
    c = np.array([0], dtype=np.int64)
    weight = 0
    mistakes = 0

    for _ in range(epochs):
        for i in range(X.shape[0]):
            xi = X[i]
            label = y[i]

            y_hat = copysign(1, gram_implicit_form_product(
                X, v_train_indices, v_label_coeffs, xi, kernel_degree, i)[-1])
            if y_hat == label:
                weight = weight + 1
            else:
                c = np.append(c, np.array([weight]), axis=0)
                v_train_indices = np.append(
                    v_train_indices, np.array([i]), axis=0)
                v_label_coeffs = np.append(
                    v_label_coeffs, np.array([label]), axis=0)
                weight = 1
                mistakes = mistakes + 1

    c = np.append(c, np.array([weight]), axis=0)
    c = c[1:c.shape[0]]
    return v_train_indices, v_label_coeffs, c, mistakes


@njit
def gram_implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    v_x[0] = polynomial_expansion(
        np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * Gram_train[gram_index, v_train_indices[k]]

    return v_x


@njit
def gram_test_implicit_form_product(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    v_x = np.empty(v_train_indices.shape[0], dtype=np.float32)
    v_x[0] = polynomial_expansion(
        np.zeros(X.shape[1], dtype=np.float32), x, kernel_degree)
    for k in range(1, v_train_indices.shape[0]):
        yi = v_label_coeffs[k]
        v_x[k] = v_x[k - 1] + yi * Gram_test[gram_index, v_train_indices[k]]

    return v_x


@njit
def gram_last_unnormalized(X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index):
    """Compute score using the final prediction vector(unnormalized)"""
    """ x: unlabeled instance"""
    score = gram_test_implicit_form_product(X,
                                            v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)[-1]

    return score


@njit
def gram_vote(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using analog of the deterministic leave-one-out conversion"""
    """ x: unlabeled instance"""

    dot_products = gram_test_implicit_form_product(X,
                                                   v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * copysign(1, v_x)

    return np.sum(s)


@njit
def gram_avg_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using an average of the prediction vectors"""
    """ x: unlabeled instance"""

    dot_products = gram_test_implicit_form_product(X,
                                                   v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)

    s = np.empty(v_train_indices.shape[0])
    s[0] = 0
    for i in range(1, v_train_indices.shape[0]):
        weight = c[i]
        v_x = dot_products[i]
        s[i] = weight * v_x

    return np.sum(s)


@njit
def gram_random_unnormalized(X, v_train_indices, v_label_coeffs, c, x, kernel_degree, gram_index):
    """Compute score using analog of the randomized leave-one-out 
    method in which we predict using the prediction vectors 
    that exist at a randomly chosen “time slice.”"""
    """ x: unlabeled instance"""
    t = np.sum(c)
    # time slice
    r = np.random.randint(t + 1)
    rl_sum = 0
    rl = 1
    for i in range(1, c.shape[0]):
        if rl_sum > r:
            break
        rl_sum = rl_sum + c[i]
        rl = rl + 1
    rl = rl - 1
    score = gram_test_implicit_form_product(
        X, v_train_indices, v_label_coeffs, x, kernel_degree, gram_index)[rl]
    return score
"""
full_script utils
"""

import numpy as np
import os
from tqdm import tqdm

import urllib.request
from urllib.parse import urljoin
import gzip

import joblib

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import random
from sklearn import svm
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_circles

# _________________________________________________________________________________
# Progress Bar


class MnistDataset:
    """Mnist utils"""

    def __init__(self, refresh=False):
        self.mnist_path_dir = 'mnist'
        self.datasets_url = 'http://yann.lecun.com/exdb/mnist/'
        self.refresh = refresh

    def download_file(self, download_file):

        output_file = os.path.join(self.mnist_path_dir, download_file)

        if self.refresh or not os.path.isfile(output_file):
            print('downloading {0} from {1}'.format(
                download_file, self.datasets_url))
            url = urljoin(self.datasets_url, download_file)
            download_url(url, output_file)

        return output_file

    def load_mnist(self, train_test='train'):

        try:
            os.makedirs(self.mnist_path_dir, exist_ok=False)
            print('Creating mnist directory')
        except:
            pass

        # Load MNIST dataset from 'path'
        labels_path = self.download_file(
            '{}-labels-idx1-ubyte.gz'.format(train_test))
        images_path = self.download_file(
            '{}-images-idx3-ubyte.gz'.format(train_test))

        with gzip.open(labels_path, 'rb') as lbpath:
            lbpath.read(8)
            buffer = lbpath.read()
            labels = np.frombuffer(buffer, dtype=np.uint8)

        with gzip.open(images_path, 'rb') as imgpath:
            imgpath.read(16)
            buffer = imgpath.read()
            images = np.frombuffer(buffer,
                                   dtype=np.uint8).reshape(len(labels),
                                                           784).astype(np.float32)

        return images, labels

    def train_dataset(self):
        return self.load_mnist(train_test='train')

    def test_dataset(self):
        return self.load_mnist(train_test='t10k')

# _________________________________________________________________________________
# Progress Bar


class ProgressBar(tqdm):
    """Progress utils"""

    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_file):
    with ProgressBar(unit='B', unit_scale=True,
                     miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(
            url, filename=output_file, reporthook=t.update_to)

# _________________________________________________________________________________
# model save & load


class Pretrained:
    """Pretrained utils"""

    def __init__(self):
        self.model_path_dir = 'models'

    def save_model(self, model, filename):
        try:
            os.makedirs(self.model_path_dir, exist_ok=False)
            #print('Creating models directory')
        except:
            pass
        i = 0
        output_file = os.path.join(
            self.model_path_dir, filename) + '_{}.pkl'.format(i)
        while os.path.isfile(output_file):
            i = i + 1
            output_file = os.path.join(
                self.model_path_dir, filename) + '_{}.pkl'.format(i)
        # compression level = 9
        joblib.dump(model, output_file, 9)

    def load_model(self, filename):
        input_file = os.path.join(self.model_path_dir, filename) + '.pkl'
        return joblib.load(input_file)

# _________________________________________________________________________________
# Experiment utils


def test_error(X, models, test, label, kernel_degree):
    scores_random = np.empty(test.shape[0])
    scores_last = np.empty(test.shape[0])
    scores_avg = np.empty(test.shape[0])
    scores_vote = np.empty(test.shape[0])
    j = 0
    for x in test:
        s_random = np.empty(10)
        s_last = np.empty(10)
        s_avg = np.empty(10)
        s_vote = np.empty(10)
        for i in range(10):
            predictions_array = predictions(
                X, models[i, 0], models[i, 1], models[i, 2], x, kernel_degree)
            s_random[i] = predictions_array[0]
            s_last[i] = predictions_array[1]
            s_avg[i] = predictions_array[2]
            s_vote[i] = predictions_array[3]
        # Survival Of The Fittest
        scores_random[j] = highest_score_arg(s_random)
        scores_last[j] = highest_score_arg(s_last)
        scores_avg[j] = highest_score_arg(s_avg)
        scores_vote[j] = highest_score_arg(s_vote)
        j = j + 1

    error_random = np.sum(scores_random != label) / label.shape[0]
    error_last = np.sum(scores_last != label) / label.shape[0]
    error_avg = np.sum(scores_avg != label) / label.shape[0]
    error_vote = np.sum(scores_vote != label) / label.shape[0]

    return error_random, error_last, error_avg, error_vote


def n_mistakes(models):
    m = 0
    for o in range(10):
        m = m + models[o, 3]
    return m


def n_supvect(models):
    s_v = 0
    for o in range(10):
        s_v = s_v + models[o, 1].shape[0]
    return s_v


def save_models(models, epoch, kernel_degree):
    # print("saving models in models/...")
    pretrained = Pretrained()
    if epoch < 1:
        epoch = '0_{}'.format(int(epoch * 10))
    pretrained.save_model(
        models, 'pretrained_e{0}_k{1}'.format(epoch, kernel_degree))


def load_models(epoch, kernel_degree, same):
    # print("loading models from models/...")
    pretrained = Pretrained()
    if epoch < 1:
        epoch = '0_{}'.format(int(epoch * 10))
    return pretrained.load_model('pretrained_e{0}_k{1}_{2}'.format(epoch, kernel_degree, same))


def train_and_store(X_train, y_train, epoch, kernel_degree):
    models = np.array(fit(X_train, y_train, epoch, kernel_degree))
    save_models(models, epoch, kernel_degree)

def gram_train_and_store(X_train, y_train, epoch, kernel_degree):
    models = np.array(gram_fit(X_train, y_train, epoch, kernel_degree))
    save_models(models, epoch, kernel_degree)

def load_and_test(X_train, X_test, y_test, epoch, kernel_degree, same=0):
    models = load_models(epoch, kernel_degree, same)
    e_r, e_l, e_a, e_v = test_error(
        X_train, models, X_test, y_test, kernel_degree)
    perc_r = e_r * 100
    perc_l = e_l * 100
    perc_a = e_a * 100
    perc_v = e_v * 100
    # print("{0:.2f}".format(perc))
    return perc_r, perc_l, perc_a, perc_v

def gram_load_and_test(X_train, X_test, y_test, epoch, kernel_degree, same=0):
    models = load_models(epoch, kernel_degree, same)
    e_r, e_l, e_a, e_v = gram_test_error(
        X_train, models, X_test, y_test, kernel_degree)
    perc_r = e_r * 100
    perc_l = e_l * 100
    perc_a = e_a * 100
    perc_v = e_v * 100
    return perc_r, perc_l, perc_a, perc_v

# gram test error
def gram_test_error(X, models, test, label, kernel_degree):
    scores_random = np.empty(test.shape[0])
    scores_last = np.empty(test.shape[0])
    scores_avg = np.empty(test.shape[0])
    scores_vote = np.empty(test.shape[0])
    j = 0
    for t in range(test.shape[0]):
        x = test[t]
        s_random = np.empty(10)
        s_last = np.empty(10)
        s_avg = np.empty(10)
        s_vote = np.empty(10)
        for i in range(10):
            predictions_array = gram_predictions(
                X, models[i, 0], models[i, 1], models[i, 2], x, kernel_degree, t)
            s_random[i] = predictions_array[0]
            s_last[i] = predictions_array[1]
            s_avg[i] = predictions_array[2]
            s_vote[i] = predictions_array[3]
        # Survival Of The Fittest
        scores_random[j] = highest_score_arg(s_random)
        scores_last[j] = highest_score_arg(s_last)
        scores_avg[j] = highest_score_arg(s_avg)
        scores_vote[j] = highest_score_arg(s_vote)
        j = j + 1

    error_random = np.sum(scores_random != label) / label.shape[0]
    error_last = np.sum(scores_last != label) / label.shape[0]
    error_avg = np.sum(scores_avg != label) / label.shape[0]
    error_vote = np.sum(scores_vote != label) / label.shape[0]

    return error_random, error_last, error_avg, error_vote
# plot function


def simple_plot(x, error_random, error_last, error_avg, error_vote, kernel_degree):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.plot(x, error_random, label='random(unorm)')
    ax.plot(x, error_last, label='last(unorm)')
    ax.plot(x, error_avg, label='avg(unorm)')
    ax.plot(x, error_vote, label='vote')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('d={}'.format(kernel_degree))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Error')
    plt.legend()
    plt.show()


def log_plot(x, error_random, error_last, error_avg, error_vote, kernel_degree):
    """ errors should contains:
        - error_random,
        - error_last,
        - error_avg,
        - error_vote
    """
    plt.style.use('seaborn')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.semilogx(x, error_random, label='random(unorm)')
    ax.semilogx(x, error_last, label='last(unorm)')
    ax.semilogx(x, error_avg, label='avg(unorm)')
    ax.semilogx(x, error_vote, label='vote')
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.set_title('d={}'.format(kernel_degree))
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test Error')
    plt.legend()
    plt.show()


md = MnistDataset()

x,y =make_circles(n_samples=1500, noise = 0.05 )
y = np.where(y==0,-1,1)
X_train, y_train = x[:750],y[:750]
X_test, y_test = x[750:], y[750:]
# y_train = y_train.astype('float64')
# y_test = y_test.astype('float64')

# X_train, y_train = md.train_dataset()
# X_test, y_test = md.test_dataset()

kernel = 1
Gram_train = gram_train_build(X_train, X_train, kernel)

x1 = np.arange(0.1, 1, 0.1)
for i in tqdm(x1):
  gram_train_and_store(X_train, y_train, i, kernel)