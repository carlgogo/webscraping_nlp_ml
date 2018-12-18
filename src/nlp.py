#! /usr/bin/env python
# coding: utf-8

"""
Module with natural language processing functionality.
"""

import math
import numpy as np
import pandas as pd
import scipy.sparse as sp
import string
from shutil import copyfile
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.base import TransformerMixin

from src import *


class CleanTextTransformer(TransformerMixin):
    """ Transformer to clean text for the pipeline

    Parameters:
    -----------
    TODO?

    Returns:
    --------
    TODO?

    """

    def transform(self, X, **transform_params):
        return [cleanText(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


def initiate_count_vectorizer(X, tokenizer=tokenize_text, min_df=0, max_df=1):
    """ turns a list X of strings into a sparse feature matrix X_counts
        which counts the occurences of the different tokens

    Parameters
    ----------
    X : list of strings (= the input texts from the various schools)

    Returns
    -------
    count_vectorizer : dictionary that links indexes of X_counts
                       to tokens (contains the vocabulary)
    X_counts : sparse matrix that counts the occurences of the different tokens

    """
    # uses vectorizer that incorporates customized tokenizer
    count_vectorizer = CountVectorizer(tokenizer=tokenizer, ngram_range=(1, 1), min_df=min_df, max_df=max_df)

    # Transform text "X" to a (sparse) feature matrix "X_counts"
    # that counts the number of appearances
    X_counts = count_vectorizer.fit_transform(X)

    return count_vectorizer, X_counts


def initiate_tfidf_transformer(X_counts):
    """ turns a sparse feature matrix X_counts
        which counts the occurences of the different tokens

    Parameters
    ----------
    X_counts : sparse matrix that counts the occurences of the different tokens

    Returns
    -------
    tfidf_transformer : dictionary that links indexes of X_counts
                       to tokens (contains the vocabulary)
    X_tfidf : sparse matrix that counts the occurences of the different tokens

    """
    # uses vectorizer that incorporates customized tokenizer
    Tfidf_transformer = TfidfTransformer(smooth_idf=False)

    # Transform text "X" to a (sparse) feature matrix "X_counts"
    # that counts the number of appearances
    X_tfidf = Tfidf_transformer.fit_transform(X_counts)

    return Tfidf_transformer, X_tfidf


# useful mappings
def get_word_by_index(input_index, vectorizer):
    """ mapping index on word by use of dictionary contained in vectorizer
    """
    return vectorizer.vocabulary_.keys()[vectorizer.
                                         vocabulary_.values().
                                         index(input_index)]


def get_index_by_word(input_word, vectorizer):
    """ mapping word on index by use of dictionary contained in vectorizer
    """
    return vectorizer.vocabulary_[input_word]


def get_vocab_counts_by_text_number(X, text_number, vectorizer):
    """ get list of tuples (word, counts) for text in X with text_number
        under use of dictionary contained in vectorizer
    """
    vocab = vectorizer.get_feature_names()

    indexes_of_vocab = X.indices[X.indptr[text_number]:X.indptr[text_number+1]]
    counts_of_vocab = X.data[X.indptr[text_number]:X.indptr[text_number+1]]

    vocab_counts = []  # list of tuples
    for idx, num in zip(indexes_of_vocab, counts_of_vocab):
        vocab_counts.append((vocab[idx], num))

    return vocab_counts


def print_N_most_informative_features(vectorizer, clf, N=5):
    """ Prints features with the largest coefficient values, per class
    """
    feature_names = vectorizer.get_feature_names()
    coefficients_and_feature_names = sorted(zip(clf.coef_[0], feature_names))
    topClass1 = coefficients_and_feature_names[:N]
    topClass2 = coefficients_and_feature_names[:-(N + 1):-1]
    print "Words that correlate the most with negative classification: "
    for feat in topClass1:
        print "(%.3f, %s)" % feat
    print
    print "Words that correlate the most with positive classification: "
    for feat in topClass2:
        print "(%.3f, %s)" % feat


def clean_dataframe(df):
    """ drops row if "text" column is either NaN/None, 'NOLINKABOUT, 'NOCONNECTION', or empty list

    Parameters
    ----------
    df : DataFrame

    Returns
    -------
    df_out : DataFrame

    """

    df_out = df.dropna()
    df_out = df_out[df_out.text != 'NOLINKABOUT']
    df_out = df_out[df_out.text != 'NOCONNECTION']
    df_out = df_out[df_out.text.str.len() > 0]

    return df_out


def write_example_text_to_file(X,type_texts='raw',number_texts=4):
    """ writes a few texts to a file in the data directory for inspection

    Parameters:
    -----------
    X : Series
        contains the different web-scraped texts
    number_texts : int
        number of texts that will be written to file
    type_texts : string
        type of texts that will be written to file <type_texts>.txt
        can be either 'raw', 'cleaned' or 'tokenized'
    """

    if not type_texts in ['raw','cleaned','tokenized']:
        return None

    filename = './data/nlp/%s.txt' % type_texts
    target = open(filename, 'w')
    for i, elem in enumerate(X):
        if type_texts == 'raw':
            target.write(elem.replace('.','.\n').encode('utf8'))
        elif type_texts == 'cleaned':
            target.write(cleanText(elem).encode('utf8'))
        elif type_texts == 'tokenized':
            for list_elem in tokenizeText(cleanText(elem)):
                target.write((list_elem + '\n').encode('utf8'))
        target.write("\n---------------------\n")
        if i == number_texts:
            break
    target.close()


def ml_preprocessing(df, filter=None):
    """ turn preprocessed dataframe df with web-scraped data
        into machine learning arrays X, y
    Paramater:
    ----------
    df : DataFrame

    Returns:
    --------
    X : Numpy 2D array, feature matrix (text)
    y : Numpy 1D array, target variable
    """
    X = df['merged_text']
    y = np.array(df['Ofsted ranking'])

    # Boolean
    #y_all = np.array(df[['Rank1','Rank2','Rank3','Rank4']])
    # take matrix of all ranks OR only one specific rank
    #y = y_all      # all ranks
    #y = y_all[:,1] # specific rank

    if filter == '12':
        # Filter out Rank 3,4
        _X = df['merged_text']
        _y = np.array(df['Ofsted ranking'])
        X = _X[(_y == 1) | (_y == 2)]
        y = _y[(_y == 1) | (_y == 2)]
    elif filter == '134':
        # filter out rank #2 (except for 11 examples)
        X = df['merged_text']
        y = np.array(df['Ofsted ranking'])
        _X2 = X[y == 2]
        _y2 = y[y == 2]
        _X = X[y != 2]
        _y = y[y != 2]
        X = np.append(_X,_X2[-11:])
        y = np.append(_y,_y2[-11:])

        # merge rank 3,4
        np.place(y, y==4, 3)
    else:
        raise NotImplementedError

    return X, y


def train_test_split_by_hand(X, y, size_training_set=10):
    """ separates (X_train, y_train) from (X, y) such that the training set
        is of size #(classes in y) x "size_training_set"

    Parameters:
    -----------
    X : Numpy 2D array, feature matrix
    y : Numpy 1D array, target
    size_training_set : int

    Returns:
    --------
    X_train : Numpy 2D array, training set of "size_training_set" examples
    y_train : Numpy 1D array, training set of "size_training_set" examples
    X_test  : Numpy 2D array, test set of all the other examples
    y_test  : Numpy 2D array, test set of all the other examples
    """

    # get classes (=Ofsted ranking) present in y
    y_classes = np.unique(y)

    X_train = np.array([])
    X_test = np.array([])
    y_train = np.array([])
    y_test = np.array([])

    for y_class in y_classes:
        _X = X[y==y_class]
        _y = y[y==y_class]

        if len(_y) < size_training_set:
            msg = "There are no size_training_set = %d " % size_training_set\
                + "examples of class %d in the data, " % y_class\
                + "only %d!" % len(_y)
            raise ValueError(msg)

        _X_train = _X[:size_training_set, :]
        _X_test = _X[size_training_set:, :]
        _y_train = _y[:size_training_set]
        _y_test = _y[size_training_set:]

        if X_train.size == 0:
            X_train = _X_train
            X_test = _X_test
            y_train = _y_train
            y_test = _y_test
        else:
            if sp.issparse(X_train):
                X_train = sp.csr_matrix(np.append(X_train.todense(), _X_train.todense(), axis=0))
                X_test = sp.csr_matrix(np.append(X_test.todense(), _X_test.todense(), axis=0))
                y_train = np.append(y_train, _y_train)
                y_test = np.append(y_test, _y_test)
            else:
                X_train = np.append(X_train, _X_train, axis=0)
                X_test = np.append(X_test, _X_test, axis=0)
                y_train = np.append(y_train, _y_train)
                y_test = np.append(y_test, _y_test)

    if sp.issparse(X_train):
        X_train = X_train.todense()
        X_test = X_test.todense()

    return  X_train, y_train, X_test, y_test
