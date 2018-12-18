#! /usr/bin/env python

import numpy as np
import pandas as pd
#import signal

from src import *


def add_web_scraped_data(df, df_web_scraped, links_websites_schools, number_of_added_websites=10):
    ''' loads previously web-scraped data,
        scrapes the next "number_of_added_websites",
        adds the full scraped data to "df"

    Parameters
    ----------
    df : DataFrame
         of all schools, read from csv
    df_web_scraped : DataFrame
         of previously scraped schools
    number_of_added_websites : int
         number of schools to be added to the DataFrame

    Returns :
    df : DataFrame
         input df with additional column "text" that contains updated web-scraped data
    '''

    if number_of_added_websites == 0:
        return df_web_scraped

    # get part of list that has not been scraped yet
    if not df_web_scraped.empty:
        index_start = df_web_scraped.dropna().index[-1] + 1
    else:
        index_start = 0
    index_stop = index_start + number_of_added_websites
    links_websites_schools = links_websites_schools[index_start:index_stop]

    print 'start web-scraping schools with indexes %d-%d' % (index_start,index_stop-1)

    # start mining
    mined_text = []

    for link in links_websites_schools:
        print '.', #link
        if link:
            try:
                #webpage_info = get_webpage_info(link)
                #if webpage_info and webpage_info != 'NOLINKABOUT' and webpage_info != 'NOCONNECTION':
                mined_text.append(get_webpage_info(link))
                #else:
                #    mined_text.append(None)
            except:
                mined_text.append(None)
        else:
            mined_text.append(None)

    if not df_web_scraped.empty:
        text_column = df_web_scraped.text[:index_start].tolist() + mined_text
    else:
        text_column = mined_text

    df = df.join(pd.DataFrame({'text' : text_column}))

    return df


def save_web_scraped_data(df, df_pickle):
    ''' PROBABLY WON'T BE NEEDED ONCE WE HAVE THE FINAL DATA
        saves df to df_pickle after doing a safety copy

    Parameters
    ----------
    df : DataFrame
         of all schools, read from csv, with web-scraped data
    df_pickle : DataFrame
         saved on disk with pickle
    '''
    # safety copy and write
    copyfile(df_pickle, df_pickle+'~')
    df.to_pickle(df_pickle)


def print_data_overview(df):
    ''' prints the number of different Ofsted rankings in the dataframe df
        that contains the web-scraped content in its "text" column

    Parameters
    ----------
    df : DataFrame

    Returns
    ----------
    Nothing
    '''

    Total   = [df.urn.count()]
    Scraped = [df[df.scraping == 1].urn.count()]
    Scraped_fail = [df[df.scraping == -1].urn.count()]
    Unscraped = [df[df.scraping == 0].urn.count()]

    for i in range(1,5):
        Total.append(df[df.ofsted == i].urn.count())
        Scraped.append(df[(df.ofsted == i) & (df.scraping == 1)].urn.count())
        Scraped_fail.append(df[(df.ofsted == i) & (df.scraping == -1)].urn.count())
        Unscraped.append(df[(df.ofsted == i) & (df.scraping == 0)].urn.count())

    Total.append(df[df.ofsted.isnull() == True].urn.count())
    Scraped.append(df[(df.ofsted.isnull() == True) & (df.scraping == 1)].urn.count())
    Scraped_fail.append(df[(df.ofsted.isnull() == True) & (df.scraping == -1)].urn.count())
    Unscraped.append(df[(df.ofsted.isnull() == True) & (df.scraping == 0)].urn.count())


    print "\n>>> Overview on data:"
    print "Total:     %5d    | Scraped: %5d    | Scraped fail: %5d    | Unscraped: %5d" % (Total[0], Scraped[0], Scraped_fail[0], Unscraped[0])
    print "--------------------------------------------------------------------------------------"
    for i in range(1,5):
        print "Rank # %d : %5d    |          %5d    |               %5d    |            %5d" % (i, Total[i], Scraped[i], Scraped_fail[i], Unscraped[i])
    print "Rank NaN : %5d    |          %5d    |               %5d    |            %5d" % (Total[5], Scraped[5], Scraped_fail[5], Unscraped[5])
    print

    print_unwanted_scraping_results(df)


def print_unwanted_scraping_results(df):
    ''' prints the number of different unwanted results in the dataframe df
        that contains the web-scraped content in its "text" column

    Parameters
    ----------
    df : DataFrame

    Returns
    ----------
    Nothing
    '''

    fail_NOLINKABOUT = df[df.rawtext == 'NOLINKABOUT'].rawtext.count()
    fail_NOCONNECTION = df[df.rawtext == 'NOCONNECTION'].rawtext.count()
    fail_NOURL = df[df.rawtext == 'NOURL'].rawtext.count()
    #fail_EMPTY = df[(type(df.rawtext) == list) & len(df.rawtext) == 0].rawtext.count()
    fail_FAIL = df[df.rawtext == 'FAIL'].rawtext.count()
    #fail_TOTAL = fail_NOLINKABOUT + fail_NOCONNECTION + fail_NOURL + fail_FAIL + fail_EMPTY
    fail_TOTAL = df[df.scraping == -1].urn.count()
    fail_EMPTY = fail_TOTAL - fail_NOLINKABOUT - fail_NOCONNECTION - fail_NOURL - fail_FAIL

    print ">>> Scraping failed because of:"
    print "    %5d NOLINKABOUT" % fail_NOLINKABOUT
    print "    %5d NOCONNECTION" % fail_NOCONNECTION
    print "    %5d NOURL" % fail_NOURL
    print "    %5d FAIL" % fail_FAIL
    print "    %5d []" % fail_EMPTY
    print "---------"
    print "    %5d " % fail_TOTAL


def get_web_scraped_data(number_of_added_websites):
    """ gets previously scraped data with pickle
        and scrapes more websites if number_of_added_websites > 0

    Parameters:
    -----------
    number_of_added_websites: int
        number of websites that are scraped

    Return:
    -------
    df : DataFrame
        with columns 'URN', 'Name', 'Ofsted Ranking', 'text'
    """

    # read school database data
    df = pd.read_csv('./data/db_schools.csv')
    df = df[[u'URN',u'Name',u'Ofsted ranking']]

    # read school urls
    _links_websites_schools = pd.read_csv('./data/db_school_urls.csv')
    _links_websites_schools = _links_websites_schools.urls.tolist()

    # read DataFrame that was saved by previous call of add_web_scraped_data (see below)
    df_pickle = './data/nlp/web_scraped_school_texts.p'
    try:
        _df_web_scraped = pd.read_pickle(df_pickle)
        print 'found DataFrame df_web_scraped'
    except:
        _df_web_scraped = pd.DataFrame()
        print 'start with new DataFrame df_web_scraped'

    # web-scrape more data if number_of_added_websites > 0
    df = add_web_scraped_data(df, _df_web_scraped, _links_websites_schools, number_of_added_websites=number_of_added_websites)
    save_web_scraped_data(df, df_pickle)

    return df


def scrape_more_websites(df_in, number_of_added_websites=0, rank=0, verbose=False):
    """ scrape more websites of specified rank if number_of_added_websites > 0

    Parameters:
    -----------
    df_in : DataFrame
    number_of_added_websites: int
        number of websites that are scraped
    rank : int
        rank of websites to be scraped

    Return:
    -------
    df : DataFrame
    """

    df = df_in.copy()

    if number_of_added_websites == 0:
        return df

    if rank == 0:
        df_rank = df.copy()
    else:
        df_rank = df[df.ofsted == rank]

    # get part of list that has not been scraped yet
    df_rank_unscraped = df_rank[df_rank.scraping == 0]
    #no_scraped_websites = df_rank[df.scraping != 0].urn.count()
    #index_start = df_rank.index[no_scraped_websites + 1]
    index_start = df_rank_unscraped.index[0]

    print 'start web-scraping %d schools of rank %d (0=all) starting with index %d' % (number_of_added_websites, rank, index_start)

    for i in range(number_of_added_websites):
        scrape_index = df_rank_unscraped.index[i]
        link = df_rank_unscraped.loc[scrape_index,'url']
        if verbose:
            print scrape_index, link
        print '.',
        try:
            scraped_text = get_webpage_info(link)
        except:
            print 'EXCEPTION: Unexpected error from scraping'
            scraped_text = 'FAIL'
        df.set_value(scrape_index, 'rawtext', scraped_text)

    #save_web_scraped_data(df, df_pickle)

    return df


#def get_webpage_info_timeout(link, timeout=1000):
#
#    def handler(signum, frame):
#        raise Exception("Scraping of website takes too long!")
#
#    signal.signal(signal.SIGALARM, handler)
#    signal.alarm(timeout)
#
#    try:
#        scraped_text = get_webpage_info(link)
#    except Exception, exc:
#        print exc
#
#    return scraped_text


def scrape_preprocessing(df):
    """ preprocessing of dataframe df with web-scraped data
    Paramater:
    ----------
    df : DataFrame

    Returns:
    --------
    df : DataFrame
    """
    def create_scraping_column(obj):
        if obj in ['NOLINKABOUT', 'NOCONNECTION', 'NOURL', 'FAIL', '[]']:
            return -1
        elif isinstance(obj, list):
            if len(obj) == 0:
                return -1
            else:
                return 1
        elif type(obj) == float:
        #elif not isinstance(obj, basestring):
            return 0
        else:
            return 1

    df['scraping'] = df['rawtext'].map(lambda obj: create_scraping_column(obj))

    return df
