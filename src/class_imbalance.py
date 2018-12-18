#! /usr/bin/env python

"""
Module with resampling techniques for imbalanced classes.

"""


from imblearn import under_sampling as imbus
from imblearn import over_sampling as imbov
from imblearn import metrics as imbmet
from imblearn import combine as imbcom
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ['resample_classes']


def resample_classes(X, Y, how='und1', random_state=None, test_size=0.3,
                     n_jobs=2, split=True, verbose=True):
    """

    """
    if how=='und1':
        if verbose:
            msg = 'Under-sampling the majority class(es) by randomly picking '
            msg += 'samples without replacement'
            print msg
        samp = imbus.RandomUnderSampler(random_state=random_state,
                                        replacement=False)
        X_res, y_res = samp.fit_sample(X, Y)
    elif how=='und2':
        if verbose:
            msg = 'Under-sampling by generating centroids based on clustering '
            msg += 'methods'
            print msg
        samp = imbus.ClusterCentroids(ratio='auto', random_state=random_state,
                                      estimator=None, n_jobs=n_jobs)
        X_res, y_res = samp.fit_sample(X, Y)
    elif how=='und3':
        if verbose:
            print 'Under-sampling based on NearMiss methods'
        samp = imbus.NearMiss(ratio='auto', return_indices=False,
                              random_state=random_state, version=1,
                              size_ngh=None, n_neighbors=3, ver3_samp_ngh=None,
                              n_neighbors_ver3=3, n_jobs=n_jobs)
        X_res, y_res = samp.fit_sample(X, Y)
    elif how=='over1':
        if verbose:
            msg = 'Over-sampling the minority class(es) by picking samples at '
            msg += 'random with replacement'
            print
        samp = imbov.RandomOverSampler(random_state=random_state)
        X_res, y_res = samp.fit_sample(X, Y)
    elif how=='over2':
        if verbose:
            msg = 'Over-sapmling using SMOTE - Synthetic Minority Over-sampling '
            msg += 'Technique'
            print msg
        X_res, y_res = X, Y
        for i in range(3):
            samp = imbov.SMOTE(random_state=random_state, ratio=.99, k=None,
                               k_neighbors=5, m=None, m_neighbors=10,
                               out_step=0.5, kind='regular', svm_estimator=None,
                               n_jobs=n_jobs)
            X_res, y_res = samp.fit_sample(X_res, y_res)
    elif how=='over3':
        if verbose:
            msg = 'Over-sampling using ADASYN - Adaptive Synthetic Sampling '
            msg += 'Approach for Imbalanced Learning'
            print msg
        X_res, y_res = X, Y
        for i in range(3):
            samp = imbov.ADASYN(ratio=.93, random_state=random_state, k=None,
                                n_neighbors=5, n_jobs=n_jobs)
            X_res, y_res = samp.fit_sample(X_res, y_res)
    elif how=='comb1':
        if verbose:
            print 'Combine over- and under-sampling using SMOTE and Tomek links.'
        X_res, y_res = X, Y
        for i in range(3):
            samp = imbcom.SMOTETomek(ratio=.99, random_state=random_state,
                                     smote=None, tomek=None, k=None, m=None,
                                     out_step=None, kind_smote=None,
                                     n_jobs=n_jobs)
            X_res, y_res = samp.fit_sample(X_res, y_res)
    else:
        print 'Sampling approach not recognized';  return

    if verbose:
        print '\t\t\t1\t2\t3\t4'
        val_y = pd.Series(Y).value_counts(sort=False).values
        msg = 'Counts in y_init:\t{}\t{}\t{}\t{} '
        print msg.format(val_y[0],val_y[1], val_y[2], val_y[3])
        val_yres = pd.Series(y_res).value_counts(sort=False).values
        msg = 'Counts in y_resamp:\t{}\t{}\t{}\t{} '
        print msg.format(val_yres[0],val_yres[1], val_yres[2], val_yres[3])

    if split:
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res,
                                                    test_size=test_size,
                                                    random_state=random_state)
        if verbose:
            val_ytr = pd.Series(y_train).value_counts(sort=False).values
            msg = 'Counts in y_train:\t{}\t{}\t{}\t{} '
            print msg.format(val_ytr[0],val_ytr[1], val_ytr[2], val_ytr[3])

            val_yte = pd.Series(y_test).value_counts(sort=False).values
            msg = 'Counts in y_test:\t{}\t{}\t{}\t{} '
            print msg.format(val_yte[0],val_yte[1], val_yte[2], val_yte[3])

            print 'X_train:', X_train.shape,', X_test:', X_test.shape

        return X_train, X_test, y_train, y_test
    else:
        return X_res, y_res
