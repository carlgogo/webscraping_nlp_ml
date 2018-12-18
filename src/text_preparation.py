#! /usr/bin/env python
# coding: utf-8

"""
Module with text preparation functionality.
"""

import re
import string
import numpy as np
import pandas as pd

from src import *

__all__ = ['pre_clean', 'remove_things', 'agressive_clean', 'clean_text',
           'tokenize_text', 'join_string']


def pre_clean(text):
    """ Remove spaces, break lines, empty spaces and strings and encode to ascii

    Parameters
    ----------
    text : list

    Returns
    -------
    text : list

    """


    # Pre-cleaning: remove/replace break lines,
    # empty spaces and empty strings
    text = [elem.replace('\n', ' ') for elem in text]
    text = [elem.replace('\t', ' ') for elem in text]
    text = [elem.replace('\r', ' ') for elem in text]
    text = [elem.strip() for elem in text]
    text = filter(None, text)

    # Decode unicode
    try:
        text = [elem.encode('utf-8') for elem in text]
        text = [elem.decode('unicode_escape').encode('ascii','ignore')
                  for elem in text]
    except:
        text = text

    return text


def remove_things(text):
    """ Remove duplicates, numbers, URLs, links and emails

    Parameters
    ----------
    text : list

    Returns
    -------
    text : list

    """


    # Remove duplicates
    _txt_arr = np.array(text)
    _, idx = np.unique(_txt_arr, return_index=True)
    text = list(_txt_arr[np.sort(idx)])

    # Remove numbers
    text = [re.sub(r'\d+', '', txt) for txt in text]

    # Remove URLs, links and emails
    text = [re.sub(r'http\S+', '', txt) for txt in text]
    text = [re.sub(r'www\S+', '', txt) for txt in text]
    text = [re.sub(r'\S+@\S+', '', txt) for txt in text]

    return text


def agressive_clean(df, len_thr):
    """ Strict rule to clean the text

    Parameters
    ----------
    df : dataframe
    len_thr: integer
        Threshold for number of words in a list item.

    Returns
    -------
    df : dataframe

    """


    df['agressive_clean'] = pd.Series('', index=df.index)

    for i in range(0, df.shape[0]):
        text = df['rawtext'][i]

        if (df['scraping'][i] == 1):

            # Pre-cleaning steps
            text = pre_clean(text)

            # Rule for defining sentences:
            # start with capital and end with full stop
            text = [elem for elem in text if
                    ((elem[0].isupper()) & (elem[-1] == '.'))]

            # Remove duplicates, numbers, URLs, links and emails
            text = remove_things(text)

            # Remove list items with less than len_thr words
            text = [txt for txt in text if len(txt.split()) > len_thr]

            df['agressive_clean'][i] = text

    return df


def clean_text(df, len_thr):
    """ Cleaning of the text

    Parameters
    ----------
    df : dataframe
    len_thr: integer
        Threshold for number of words in a list item.

    Returns
    -------
    df : dataframe

    """


    KEYS = ['copyright', 'click here', 'cookies',
            'cookie policy', 'sitemap', 'website by',
            'website design by', 'all rights reserved', '|']

    df['clean_text'] = pd.Series('', index=df.index)

    for i in range(0, df.shape[0]):
        text = df['rawtext'][i]

        if (df['scraping'][i] == 1):

            # Pre-cleaning steps
            text = pre_clean(text)

            # Rules for defining sentences:
            # filter out using keywords and
            # words with capital letters in the middle of non-capital ones
            _clean_text = []
            for j in range(0, len(text)):
                if (len(re.findall(r'[a-z][A-Z][a-z]', text[j])) == 0):
                    if not any(key in text[j].lower() for key in KEYS):
                        _clean_text.append(text[j])
            text = _clean_text

            # Remove duplicates, numbers, URLs, links and emails
            text = remove_things(text)

            # Remove list items with less than len_thr words
            text = [txt for txt in text if len(txt.split()) > len_thr]

            df['clean_text'][i] = text

    return df


def tokenize_text(df, len_thr, opt):
    """ Tokenization of the text

    Parameters
    ----------
    df : dataframe
    len_thr: integer
        Threshold for the number of tokens.
    opt: string
        Define whether agressive or normal cleaning was used,
        can be either 'norm' or 'agre'.

    Returns
    -------
    df : dataframe

    """


    from nltk.corpus import stopwords
    from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    from spacy.en import English

    if (opt == 'norm'):
        df['tokens'] = pd.Series('', index=df.index)
    elif (opt == 'agre'):
        df['tokens_agressive'] = pd.Series('', index=df.index)
    else:
        print 'Wrong opt parameter!'
        return df

    parser = English()

    for i in range(0, df.shape[0]):

        if (df['scraping'][i] == 1):

            if (opt == 'norm'):
                text = df['clean_text'][i]
            else:
                text = df['agressive_clean'][i]

            # Get the tokens using spaCy
            text = unicode(text)
            tokens = parser(text)

            # Stemming/Lemmatizing
            lemmas = []
            for tok in tokens:
                lemmas.append(tok.lemma_.lower().strip()
                              if tok.lemma_ != "-PRON-"
                              else tok.lower_)
            tokens = lemmas

            # Remove stopwords
            STOPLIST = unicode(stopwords.words('english') +
                               ["n't", "'s", "'m", "ca"] +
                               list(ENGLISH_STOP_WORDS))
            tokens = [tok for tok in tokens if tok not in STOPLIST]

            # Remove some punctuation
            SYMBOLS = unicode(["-----", "---", "...", "“", "”", "'ve"])
            tokens = [tok for tok in tokens if tok not in SYMBOLS]

            # Remove punctuation and some strange things
            SYMBOLS = unicode(" ".join(string.punctuation).split(" "))
            for sym in SYMBOLS:
                tokens = [tok.replace(sym, '') for tok in tokens]

            # Remove whitespace
            while "" in tokens:
                tokens.remove("")
            while " " in tokens:
                tokens.remove(" ")
            while "\n" in tokens:
                tokens.remove("\n")
            while "\n\n" in tokens:
                tokens.remove("\n\n")

            # Remove tokens with less than len_thr words
            if ((opt == 'norm') & (len(tokens) > len_thr)):
                df['tokens'][i] = tokens
            elif ((opt == 'agre') & (len(tokens) > len_thr)):
                df['tokens_agressive'][i] = tokens
            else:
                df['tokens'][i] = ''

    return df


def join_string(df, opt):
    """ Join the list of words (tokens) into a single string

    Parameters
    ----------
    df : dataframe
    opt: string
        Define whether agressive or normal cleaning was used,
        can be either 'norm' or 'agre'.

    Returns
    -------
    df : dataframe

    """


    if (opt == 'norm'):
        df['joined_tokens'] = pd.Series('', index=df.index)
    elif (opt == 'agre'):
        df['joined_tokens_agre'] = pd.Series('', index=df.index)
    else:
        print 'Wrong opt parameter!'
        return df


    for i in range(0, df.shape[0]):

        if (df['scraping'][i] == 1):

            # Create single string
            if (opt == 'norm'):
                tokens = df['tokens'][i]
            else:
                tokens = df['tokens_agressive'][i]

            text = ''
            for j in range(0, len(tokens)):
                if (j == 0):
                    text += ''.join(tokens[j])
                else:
                    text += ''.join(' ' + tokens[j])

            if (opt == 'norm'):
                df['joined_tokens'][i] = str(text)
            else:
                df['joined_tokens_agre'][i] = str(text)

    return df
