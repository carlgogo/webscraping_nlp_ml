#! /usr/bin/env python

"""
Module with machine learning functionality.

"""

import mord
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score
from sklearn.svm import SVC
from lightning import ranking
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.model_selection import ShuffleSplit, cross_val_predict
from imblearn import metrics as imbmet
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn import metrics
from matplotlib import pyplot as plt

__all__ = ['run_classification', 'run_clustering']


def run_classification(X_train, X_test, y_train, y_test, how='rfc',
                       random_state=0, n_jobs=2, cv=False, stand=False,
                       verbose=True, full_output=False, **classpar):
    """

    """
    if stand:
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit_transform(X_test)

    if how=='or1':
        pars = {'alpha':1e0, 'verbose':1, 'max_iter':1e5}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = mord.LogisticAT(**classpar)
    elif how=='or2':
        pars = {'alpha':1e0, 'verbose':1, 'max_iter':1e5}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = mord.LogisticIT(**classpar)
    elif how=='or3':
        pars = {'alpha':1e0, 'fit_intercept':True, 'normalize':False,
                'copy_X':True, 'max_iter':None, 'tol':0.001, 'solver':'auto'}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = mord.OrdinalRidge(random_state=random_state, **classpar)
    elif how=='or4':
        pars = {'epsilon':0.0, 'tol':0.0001, 'C':1.0, 'loss':'l1',
                'fit_intercept':True, 'intercept_scaling':1.0, 'dual':True,
                'verbose':0, 'max_iter':10000}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = mord.LAD(random_state=random_state, **classpar)
    elif how=='prank':
        pars = {'n_iter':1000, 'shuffle':True}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = ranking.PRank(random_state=random_state, **classpar)
    elif how=='kprank':
        pars = {'n_iter':200, 'shuffle':True, 'kernel':'rbf', 'gamma':1e2,
                'degree':3, 'coef0':1}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = ranking.KernelPRank(random_state=random_state, **classpar)
    elif how=='rfc':
        pars = {'n_estimators':1000, 'criterion':'gini', 'max_depth':None,
                'min_samples_split':2, 'min_samples_leaf':1,
                'min_weight_fraction_leaf':0.0, 'max_features':'auto',
                'max_leaf_nodes':None, 'min_impurity_split':1e-07,
                'bootstrap':True, 'oob_score':True, 'verbose':0,
                'warm_start':False, 'class_weight':None}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = RFC(random_state=random_state, n_jobs=n_jobs, **classpar)
    elif how=='svc':
        pars = {'C':1.0, 'kernel':'rbf', 'degree':3, 'gamma':'auto',
                'coef0':0.0, 'shrinking':True, 'probability':False, 'tol':0.001,
                'cache_size':200, 'class_weight':None, 'verbose':False,
                'max_iter':-1, 'decision_function_shape':None}
        for par in pars:
            if par not in classpar:  classpar.update({par:pars.get(par)})
        clasif = SVC(random_state=random_state, **classpar)
    else:
        print 'Classifier not yet supported';  return

    if cv:
        crosv = ShuffleSplit(n_splits=5, test_size=0.3,
                             random_state=random_state)
#         y_pred = cross_val_predict(clasif, X_train, y_train, cv=5, n_jobs=n_jobs,
#                                    verbose=1)

#         f1 = f1_score(y_test, y_pred, average='weighted')
#         ck = cohen_kappa_score(y_test, y_pred)
#         rec = recall_score(y_test, y_pred, average='weighted')

#         if verbose:
#             print '\nF1={:.2f}, Recall={:.2f}, Cohen Kappa={:.2f}'.format(f1, rec, ck)

#         return f1, rec, ck

        f1_cv_scores = cross_val_score(clasif, X_train, y_train, cv=crosv,
                                       scoring='f1_weighted', verbose=1,
                                       n_jobs=n_jobs)
        mean_cv_f1 = np.mean(f1_cv_scores)
        if verbose:
            print f1_cv_scores
            print 'Mean F1 score={:.3f}'.format(mean_cv_f1)
        return mean_cv_f1, f1_cv_scores

    else:
        if verbose:
            print clasif.fit(X_train, y_train.astype(int))
        else:  clasif.fit(X_train, y_train.astype(int))

        y_pred = clasif.predict(X_test)

        if verbose:
            print '\n', imbmet.classification_report_imbalanced(y_test, y_pred)
            if verbose and hasattr(clasif, 'feature_importances_'):
                print 'Feature importances:'
                print clasif.feature_importances_

        ck = cohen_kappa_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        if verbose:
            print '\nF1={:.2f}, Recall={:.2f}, Cohen Kappa={:.2f}'.format(f1,
                                                                        rec, ck)

        if full_output:
            return clasif, f1, rec, ck
        else:
            return f1, rec, ck



def run_clustering(X, Y, how='kmeans', n_clusters=4, npcs=None, stand=True,
                   n_iter=100, n_jobs=2, random_state=None, boundary=True,
                   colors=plt.cm.plasma):
    """
    """
    if stand:  X = StandardScaler().fit_transform(X)

    if npcs:
        pca = PCA(n_components=npcs, copy=True, whiten=False, svd_solver='auto',
                  tol=0.0, iterated_power=5, random_state=random_state)

        X = pca.fit_transform(X)
        print 'PCs explained variance ratios:\n', pca.explained_variance_ratio_
        print

    centroids = None
    if how=='kmeans':
        clu = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20,
                     max_iter=n_iter, tol=0.0001, precompute_distances='auto',
                     verbose=0, random_state=random_state, copy_x=True,
                     n_jobs=n_jobs, algorithm='full')
    elif how=='agglom':
        clu = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',
                                      connectivity=None, compute_full_tree='auto',
                                      linkage='ward')
    elif how=='spect' and X.shape[0]<=1000:
        clu = SpectralClustering(n_clusters=n_clusters, eigen_solver=None,
                                 random_state=None, n_init=10, gamma=1.0,
                                 affinity='rbf', n_neighbors=10, eigen_tol=0.0,
                                 assign_labels='kmeans', degree=3, coef0=1,
                                 kernel_params=None, n_jobs=1)
    elif how=='birch':
        clu = Birch(threshold=0.1, branching_factor=50, n_clusters=n_clusters,
                    compute_labels=True, copy=True)
    else:
        print 'Clustering method not supported'; return

    clu.fit(X)
    if hasattr(clu, 'cluster_centers_'):
        centroids = clu.cluster_centers_
    clust_labels = clu.labels_+1

    # plotting in 2d case (PCA)
    if X.shape[1]==2 and centroids is not None:
        rangex = X[:, 0].max() - X[:, 0].min()
        rangey = X[:, 1].max() - X[:, 1].min()
        rangexy = min(rangex, rangey)
        x_min = X[:, 0].min() - rangexy/3
        x_max = X[:, 0].max() + rangexy/3
        y_min = X[:, 1].min() - rangexy/3
        y_max = X[:, 1].max() + rangexy/3
        if boundary and hasattr(clu, 'predict'):
            # Step size of the mesh. Decrease to increase the plot quality.
            h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].
            # Plot the decision boundary. We assign a color to each
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            # Obtain labels for each point in mesh. Use last trained model.
            Z = clu.predict(np.c_[xx.ravel(), yy.ravel()])
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.figure(figsize=(12,6))
            plt.imshow(Z, interpolation='bicubic',
                       extent=(x_min, x_max, y_min, y_max),
                       cmap=colors, origin='lower', alpha=0.3)

        plt.scatter(X[:,0], X[:,1], marker='o', alpha=0.5, c=Y, #c=clust_labels,
                    cmap=colors, linewidth='0.2')
        plt.scatter(centroids[:,0], centroids[:,1], marker='*', s=160,
                    edgecolor='white', color='white', linewidth='1', alpha=1)
        plt.grid('off')
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

    # metrics
    print '\t\t\t1\t2\t3\t4'
    val_y = pd.Series(Y).value_counts(sort=False).values
    msg = 'Counts in labels:\t{}\t{}\t{}\t{} '
    print msg.format(val_y[0],val_y[1], val_y[2], val_y[3])

    val_y = pd.Series(clust_labels).value_counts(sort=False).values
    msg = 'Counts in clusters:\t{}\t{}\t{}\t{} \n'
    print msg.format(val_y[0],val_y[1], val_y[2], val_y[3])

    print 'Confusion matrix:\n', metrics.confusion_matrix(Y, clust_labels), '\n'

    msg = 'Homogeneity={:.3f} | Completeness={:.3f} | V measure{:.3f}\n'
    hom, com, vme = metrics.homogeneity_completeness_v_measure(Y, clust_labels)
    print msg.format(hom, com, vme)

    print metrics.classification_report(Y, clust_labels)

    return clu
