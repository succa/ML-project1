# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
            
            
            
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]



def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # generate random indices
    num_row = len(y)
    indices = np.random.permutation(num_row)
    index_split = int(np.floor(ratio * num_row))
    index_tr = indices[: index_split]
    index_te = indices[index_split:]
    # create split
    x_tr = x[index_tr]
    x_te = x[index_te]
    y_tr = y[index_tr]
    y_te = y[index_te]
    return x_tr, x_te, y_tr, y_te

def remove_outliers(tx):
    # DER_mass_MMC
    tx[tx[:, 0] > 700][:, 0] = 700
    #DER_mass_transverse_met_lep
    tx[tx[:, 1] > 300][:, 1] = 300
    #DER_mas_vis
    tx[tx[:, 2] > 500][:, 2] = 500
    #DER_pt_h
    tx[tx[:, 3] > 600][:, 3] = 600
    #DER_mass_jet_jet
    tx[tx[:, 5] > 2800][:, 5] = 2800
    # DER_pt_tot
    tx[tx[:, 8] > 400][:, 8] = 400
    # DER_sum_pt
    tx[tx[:, 9] > 1000][:, 9] = 1000
    # DER_pt_ratio_lep_tau
    tx[tx[:, 10] > 10][:, 10] = 10
    # PRI_tau_pt
    tx[tx[:, 13] > 300][:, 13] = 300
    # PRI_lep_pt
    tx[tx[:, 16] > 250][:, 16] = 250
    # PRI_met
    tx[tx[:, 19] > 450][:, 19] = 450
    # PRI_met_sumet
    tx[tx[:, 21] > 1100][:, 21] = 1100
    # PRI_jet_leading_pt
    tx[tx[:, 23] > 500][:, 23] = 500
    # PRI_jet_subleading_pt
    tx[tx[:, 26] > 250][:, 26] = 250
    # PRI_jet_all_pt
    tx[tx[:, 29] > 800][:, 29] = 800
    return tx


def expand_features(x, degree=2, mutual_product=False):
    """ performs feature expansion according to the provided degree
        mutual_product defines whether the features should multiplied by each other
    """
    tx = x.copy()
    
    for d in range(2, degree+1):
        for feature in range(tx.shape[1]):
            to_add = tx[:, feature]
            to_add = to_add.reshape(-1, 1)
            tx = np.append(tx, np.power(to_add, d), axis=1)
    
    n1 = x.shape[1]
            
    if mutual_product:
        for feature_1 in range(n1):
            for feature_2 in range(feature_1, n1):
                if ( (feature_1 % x.shape[1]) == (feature_2 % x.shape[1]) ):
                    to_add = tx[:, feature_1] * tx[:, feature_2]
                    to_add = to_add.reshape(-1, 1)
                    tx = np.append(tx, to_add, axis=1)
             
    
    
    return tx

def clean_data(tx, y, degree=2, feature_expansion=True, mutual_product=True):
    """ data preprocessing
        degree: defines the maximum degree of the exponential features
        feature_expansion: defines whether feature expansion should be performed
        mutual_product: defines whether feature expansion includes the product between each feature. If False, then only exponentials are performed
     """
    tx = tx.copy()
    
    # features to drop for each jet_num
    useless_features = [
        [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
        [4, 5, 6, 12, 22, 26, 27, 28],
        [22],
        [22]
    ]
    
    # dividing the labels in 4 sets
    all_ys = []
    all_ys.append(y[tx[:, 22] == 0])
    all_ys.append(y[tx[:, 22] == 1])
    all_ys.append(y[tx[:, 22] == 2])
    all_ys.append(y[tx[:, 22] == 3])
    
    tx[tx==-999] = np.NaN
    
    remove_outliers(tx)
    
    # dividing the data in 4 sets
    dataset0 = tx[tx[:, 22] == 0, :]
    dataset1 = tx[tx[:, 22] == 1, :]
    dataset2 = tx[tx[:, 22] == 2, :]
    dataset3 = tx[tx[:, 22] == 3, :]

    # removing the useless features
    filtered_features_dataset0 = dataset0[:, [feature for feature in range(0, tx.shape[1]) if feature not in useless_features[0]]]
    filtered_features_dataset1 = dataset1[:, [feature for feature in range(0, tx.shape[1]) if feature not in useless_features[1]]]
    filtered_features_dataset2 = dataset2[:, [feature for feature in range(0, tx.shape[1]) if feature not in useless_features[2]]]
    filtered_features_dataset3 = dataset3[:, [feature for feature in range(0, tx.shape[1]) if feature not in useless_features[3]]]
    
    # creating a list of smaller datasets (one per jet_num)
    filtered_features_datasets = []
    filtered_features_datasets.append(filtered_features_dataset0)
    filtered_features_datasets.append(filtered_features_dataset1)
    filtered_features_datasets.append(filtered_features_dataset2)
    filtered_features_datasets.append(filtered_features_dataset3)
    
    if feature_expansion:
        for i, filtered_features_dataset in enumerate(filtered_features_datasets):
            filtered_features_datasets[i] = expand_features(filtered_features_dataset, degree=degree, mutual_product=mutual_product)
    
    # getting means for each splitted dataset
    means = []
    means.append(np.nanmean(filtered_features_datasets[0], 0))
    means.append(np.nanmean(filtered_features_datasets[1], 0))
    means.append(np.nanmean(filtered_features_datasets[2], 0))
    means.append(np.nanmean(filtered_features_datasets[3], 0))
    
    # getting standard deviations for each splitted dataset
    stds = []
    stds.append(np.nanstd(filtered_features_datasets[0], 0))
    stds.append(np.nanstd(filtered_features_datasets[1], 0))
    stds.append(np.nanstd(filtered_features_datasets[2], 0))
    stds.append(np.nanstd(filtered_features_datasets[3], 0))
    
    # replacing NANs with mean values
    for dataset_i in range(len(filtered_features_datasets)):
        for feature in range(filtered_features_datasets[dataset_i].shape[1]):
            filtered_features_datasets[dataset_i][:, feature][np.isnan(filtered_features_datasets[dataset_i][:, feature])] = means[dataset_i][feature]
    
    
    # performing data normalization
    filtered_features_datasets[0] = ((filtered_features_datasets[0] - means[0])/stds[0])
    filtered_features_datasets[1] = ((filtered_features_datasets[1] - means[1])/stds[1])
    filtered_features_datasets[2] = ((filtered_features_datasets[2] - means[2])/stds[2])
    filtered_features_datasets[3] = ((filtered_features_datasets[3] - means[3])/stds[3])

    
    return filtered_features_datasets, all_ys
