import numpy as np
import matplotlib.pyplot as plt
from MLscripts.proj1_helpers import *
# from MLscripts.helpers import *
# from MLscripts.helpers import batch_iter


def calculate_gradient_MSE(y, tx, w):
    """calculate the gradient of MSE."""
    return -1/y.shape[0] * tx.T@(y - np.matmul(tx, w))
    
def calculate_gradient_MAE(y, tx, w):
    """calculate the gradient of MAE."""
    e = y - np.matmul(tx, w)
    e[e==0] = 0
    e = np.abs(e) / e
    return -1/y.shape[0] * (np.matmul(e, tx))


def calculate_gradient_LIKELIHOOD(y, tx, w):
    """calculate the graient of the log likelihood function"""
    return np.matmul(tx.T, (sigma_function(np.matmul(tx, w)) - y))


def calculate_gradient(y, tx, w, cost_function):
    """ generic function that computes the gradient according to the desired cost function"""
    return gradients[cost_function](y, tx, w)


def gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=True, debug=True, cost_function='MSE'):
    # the likelihood cost functions needs either 0 or 1 as y.
    y = y.copy()
    if cost_function == "LIKELIHOOD":
        y[y==-1] = 0
    if plot_losses:
        fig, axs = plt.subplots(1, 2)        

    w = initial_w
    batch_size = y.shape[0]//num_batches
    for n_iter in range(1, max_iters+1):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=num_batches):
            if n_iter > max_iters:
                break
            
            gradient = calculate_gradient(y_batch, tx_batch, w, cost_function=cost_function) # relative to only the batch
            w = w - gamma*gradient
            curr_loss = calculate_loss(y, tx, w, lambda_, cost_function=cost_function)
            if debug:
                print("Gradient Descent: step", n_iter, "/", max_iters,"- Loss =", curr_loss)

            if plot_losses:

                succ_ratio = calculate_loss(y, tx, w, cost_function='SUCCESS_RATIO')
                axs[0].scatter(n_iter, curr_loss, color='red', s=10)
                axs[1].scatter(n_iter, succ_ratio, color='blue', s=10)
                print("Accuracy:", succ_ratio)
            
            n_iter = n_iter+1
                    
    return calculate_loss(y, tx, w, lambda_, cost_function=cost_function), np.array(w)


def calculate_loss_MSE(y, tx, w):
    e = y - np.matmul(tx, w)
    return np.matmul(e.T, e)/(2*y.shape[0])

def calculate_loss_RMSE(y, tx, w):
    return np.sqrt(2 * calculate_loss_MSE(y, tx, w))


def calculate_loss_MAE(y, tx, w):
    e = y - np.matmul(tx, w)
    return np.abs(e).sum(0)/y.shape[0]


def calculate_loss_LIKELIHOOD(y, tx, w, lambda_=0):
    p = sigma_function(np.matmul(tx, w))
    log_likelihood = np.squeeze( (np.matmul(y.T, np.log(p)) + np.matmul((1 - y).T, np.log(1 - p))   )   )
    return -log_likelihood + lambda_ * np.squeeze(np.matmul(w.T, w))

def calculate_loss_SUCCESS(y, tx, w):
    pred = predict_labels(w, tx)
    y_ = y.copy()
    y_[y_==0] = -1

    num_correct = np.sum(pred==y_)
    return num_correct/len(y)


def calculate_loss(y, tx, w, lambda_=0, cost_function='MSE'):
    """ generic function that computes the loss according to the desired cost function"""
    if cost_function == 'LIKELIHOOD':
        return cost_functions[cost_function](y, tx, w, lambda_)
    return cost_functions[cost_function](y, tx, w)


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=True, debug=True, cost_function='MSE')


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    num_batches = tx.shape[0]
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=num_batches, plot_losses=True, debug=True, cost_function='MSE')


def least_squares(y, tx):
    A = np.matmul(tx.T, tx)
    b = np.matmul(tx.T, y)
    w = np.linalg.solve(A, b)
    return calculate_loss(y, tx, w, cost_function='MSE'), w

def ridge_regression(y, tx, lamb=0):
    aI = lamb * np.identity(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    return calculate_loss(y, tx, w, cost_function='RMSE'), w

def logistic_regression(y, tx, initial_w, max_iters, gamma, plot_losses=True, debug=False):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=0, num_batches=1, plot_losses=plot_losses, debug=debug, cost_function='LIKELIHOOD')


def reg_logistic_regression(y, tx, initial_w, lambda_, max_iters, gamma, plot_losses=False, debug=False):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, lambda_=lambda_, num_batches=1, plot_losses=plot_losses, debug=debug, cost_function='LIKELIHOOD')


def sigma_function(z):
    """compute the sigma function"""
    return 1 / (1 + np.exp(-z))
    
def cross_validate(model, X, y, batch_size=20, n_splits=4, epochs=100, lambda_=0, initial_w=None, gamma=0.01, model_name='least_squares'):
        """ Run cross validation on the model and return the obtained test and train scores. """

        # initialize w if None is provided
        if initial_w == None:
                initial_w = np.random.uniform(low=-1, high=1, size=X.shape[1])
                
        def build_k_indices(y, k_fold, seed=12):
            """build k indices for k-fold."""
            num_row = y.shape[0]
            interval = int(num_row / k_fold)
            np.random.seed(seed)
            indices = np.random.permutation(num_row)
            te_indices = np.array([indices[k * interval: (k + 1) * interval] for k in range(k_fold)])
            tr_indices = np.array([te_indices[~(np.arange(te_indices.shape[0]) == k)].reshape(-1) for k in range(k_fold)])
            return zip(tr_indices, te_indices)
        
        kf = build_k_indices(X, n_splits)
        tr_scores = []
        va_scores = []

        result = {
            "train_score": [],
            "test_score" : []
        }

        split_n = 1
        for tr_indices, va_indices in kf:
            tr_indices = tr_indices.tolist()
            va_indices = va_indices.tolist()
            X_tr, y_tr = X[tr_indices], y[tr_indices]
            X_te, y_te = X[va_indices], y[va_indices]
            
            if model_name == 'least_squares':
                _, wi = model(y_tr, X_tr)
            elif model_name == 'ridge_regression':
                _, wi = model(y_tr, X_tr, lambda_)
            elif (model_name == 'least_squares_GD' or
                  model_name == 'least_squares_SGD' or
                  model_name == 'logistic_regression'):
                _, wi = model(y_tr, X_tr, initial_w, epochs, gamma)
            elif (model_name == 'reg_logistic_regression'):
                _, wi = model(y_tr, X_tr, initial_w, lambda_, epochs, gamma)
            result["train_score"].append(calculate_loss_SUCCESS(y_tr, X_tr, wi))
            result["test_score"].append(calculate_loss_SUCCESS(y_te, X_te, wi))            

            split_n = split_n + 1

        return result    


""" dictionaries that keep track of what functions should be executed when
    a certain loss function is used
"""
cost_functions = {
    'MSE': calculate_loss_MSE,
    'RMSE': calculate_loss_RMSE,
    'MAE': calculate_loss_MAE,
    'LIKELIHOOD': calculate_loss_LIKELIHOOD,
    'SUCCESS_RATIO': calculate_loss_SUCCESS
    }

gradients = {
    'MSE': calculate_gradient_MSE,
    'MAE': calculate_gradient_MAE,
    'LIKELIHOOD': calculate_gradient_LIKELIHOOD
    }