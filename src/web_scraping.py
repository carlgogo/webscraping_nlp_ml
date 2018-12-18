#! /usr/bin/env python

"""
Module with web scraping functionality.

"""

import requests
import pandas as pd
import bs4 as bs
import platform
from selenium import webdriver
import signal

__all__ = ['get_webpage_info', 'get_soup', 'KEYWORDS', 'KEYWORDS_AVOID',
           'get_webpage_text_from_url_list', 'get_schools_urls',
           'get_schools_metadata', 'generate_schools_db',
           ]


KEYWORDS = ['about', 'about us', 'prospectus', 'ethos', 'values', 'our school',
            'school-information', 'school-info', 'vision', 'introduction',
            'our-school']
KEYWORDS_AVOID = ['about-this-site', 'admission', 'issuu.com', '.pdf',
                  'revision-help', 'provision']

root_url = 'https://www.compare-school-performance.service.gov.uk/'




def get_webpage_text_from_url_list(list_school_urls, indices=(0,100),
                                   debug=False):
    """ Generates the mined text DB from a list of schools URLs. It
    handles non existing URLs.

    Parameters
    ----------
    list_school_urls : list of strings
        List of schools websites.
    indices : tuple of integers
        Used for slicing the list of urls.
    debug : bool
        Debug info.

    Returns
    -------
    mined_text : list of lists
        List of lists, each one containing the paragraphs/sentences of each
        school website. The length of ``mined_text`` should equal the length
        of ``list_school_urls``.

    """
    mined_text = []
    for url in list_school_urls[indices[0]:indices[1]]:
        if debug:  print url

        if url is None:
            mined_text.append('NOURL')
        else:
            mined_text.append(get_webpage_info(url, debug=debug))

    return mined_text



def get_schools_urls(list_schools, mode='urn', csv=False):
    """ Scrapes the schools real websites from the compare-school-performance DB.
    Optionally it grabs the link of CSV files.

    Parameters
    ----------
    list_schools : list of strings
        List of school URNs or links in the DB.
    mode : string
        'link' or 'urn' for the ``list_schools`` format.
    csv : boolean
        Whether to return the csv link or not.

    Returns
    -------
    links_websites_schools : list of strings
    If ``csv`` is True then:
    links_csv_schools : list of strings

    """
    li_sc = list_schools
    links_websites_schools = []
    links_csv_schools = []

    for entry in li_sc:
        if mode=='urn':
            school_link_db = root_url+'school/'+str(entry)+'/'
        try:
            soup = get_soup(school_link_db)
        except:
            links_websites_schools.append(None)
            continue

        try:
            link_tag = soup.find('dd', attrs={'class' : 'print-show-link-href'})
            link = link_tag.find('a').get('href')
        except:
            links_websites_schools.append(None)
            continue

        if not link.endswith('/'):
            link += '/'
        if link.endswith('apply-for-primary-school-place/') or \
           link.endswith('apply-for-secondary-school-place/') :
            link = None
        links_websites_schools.append(link)

        if csv:
            csv_tag = soup.find('a', attrs={'class' : 'js-track'})
            links_csv_schools.append(root_url+csv_tag.attrs['href'])

    if csv:
        return links_websites_schools, links_csv_schools
    else:
        return links_websites_schools



def get_schools_metadata(search_url):
    """ Retrieves the schools metadata from a search query done at the root_url
    (https://www.compare-school-performance.service.gov.uk/). It is general
    enough to handle any query from this database. Retrieves the schools names,
    addresses, scores, levels, types and link inside the DB. Later with this
    links we can proceed to scrap the actual schools websites addresses.

    Parameters
    ----------
    search_url : string
        URL of the search query.

    Returns
    -------
    Returns populated lists with the scraped schools and metadata.


    """
    def get_schools_data_querypage(soup, names_schools, links_schools,
                                   scores_schools, address_schools,
                                   levels_schools, types_schools):
        """ Gets a Beautiful soup object from a search query at
        https://www.compare-school-performance.service.gov.uk/. It also receives
        a few lists for storing the metadata of the schools scraped from the
        website.

        Parameters
        ----------
        soup : BeautifuSoup object
            Soup of the query webpage.
        names_schools : list of strings
            For storing names of schools.
        links_schools : list of strings
            For school links inside
            www.compare-school-performance.service.gov.uk DB.
        scores_schools : list of integers
            For the Ofsted scores (1, 2, 3 or 4).
        address_schools : list of strings
            Physical addresses of the schools.
        levels_schools : list of strings
            Level of each school.
        types_schools : list of strings
            School types.

        Returns
        -------
        Returns the updates lists. New entries are appended.

        """
        schools = soup.find_all('li', attrs={'class' : 'document'})

        for school in schools:
            li = school.find('a', attrs={'class' : 'bold-small'})
            names_schools.append(li.string)
            links_schools.append(root_url+li.attrs['href'])

            sco = school.find('span', attrs={'class' : 'rating-text'})
            scores_schools.append(int(sco.string.split('                ')[1][0]))

            metadata = school.find('dl', attrs={'class' : 'metadata'})
            children = metadata.findChildren()
            address_schools.append(children[2].string)
            levels_schools.append(children[5].string)
            types_schools.append(children[8].string)

        return (names_schools, links_schools, scores_schools,
                address_schools, levels_schools, types_schools)

    #---------------------------------------------------------------------------
    na_sc = []; li_sc = []; sco_sc = []; add_sc = []; lev_sc = []; ty_sc = []

    soup = get_soup(search_url)

    n_results = int(soup.find('span', attrs={'class' : 'result-count'}).string)
    print "Total number of schools in query",n_results
    n_per_page = 50 # seems to be fixed
    n_query_pages = n_results/n_per_page
    if n_results%n_per_page>0:  n_query_pages += 1

    print "Getting schools metadata from page 1 \n"
    f = get_schools_data_querypage
    na_sc, li_sc, sco_sc, add_sc, lev_sc, ty_sc = f(soup, na_sc, li_sc, sco_sc,
                                                    add_sc, lev_sc, ty_sc)

    if n_query_pages>0:
        for i in range(1, n_query_pages):
            print "Getting schools metadata from page",i+1
            new_search_url = search_url+'&page='+str(i+1)
            new_soup = get_soup(new_search_url)

            na_sc,li_sc,sco_sc,add_sc,lev_sc,ty_sc = f(new_soup, na_sc,
                                                       li_sc, sco_sc, add_sc,
                                                       lev_sc, ty_sc)
    return na_sc, li_sc, sco_sc, add_sc, lev_sc, ty_sc



def generate_schools_db(search_url, save_csv='../data/schools_db.csv',
                        about_tags=['a'], keywords=None,
                        text_tags = ['p', 'li', 'ol', 'ul']):
    """ Generates a pandas database with the schools metadata and scraped
    "about us" mined text.

    Parameters
    ----------
    search_url : string
        URL of the search query from the schools database
        (https://www.compare-school-performance.service.gov.uk/).
    about_tags : list of strings
        HTML tags used for identifying the webpage to be scraped (about us page).
    keywords : list of strings
        Keywords used for identifying the webpage to be scraped.
    text_tags : list of strings
        HTML tags that we search in the webpage for extracting the relevant text.

    Returns
    -------
    df_out : pandas DataFrame
        Pandas with the scraped schools data (names, addresses, scores, levels,
        types and website) and the mined text.

    """
    lists_metadata = get_schools_metadata(search_url)
    na_sc, li_sc, sco_sc, add_sc, lev_sc, ty_sc = lists_metadata

    links_websites_schools, links_csv_schools = get_schools_urls(li_sc,
                                                                 mode='link',
                                                                 csv=True)

    df = pd.DataFrame({'name': na_sc,
                       'website': links_websites_schools,
                       'score': sco_sc,
                       'address': add_sc,
                       'level': lev_sc,
                       'type': ty_sc,
                       'csv' : links_csv_schools})

    mined_text = []

    for link in links_websites_schools:
        if link:
            try:
                mined_text.append(get_webpage_info(link, about_tags,
                                                   keywords, text_tags))
            except:
                mined_text.append(None)
        else:
            mined_text.append(None)

    df_out = df.join(pd.DataFrame({'text' : mined_text}))

    if save_csv:  df_out.to_csv(save_csv)

    print 'Number of URLs that we could not parse:', mined_text.count(None)

    return df_out



def get_soup(url):
    """ Obtainig BeautifulSoup object from URL.

    Parameters
    ----------
    url : string
        url ending in /.

    Returns
    -------
    A BeautifulSoup object. None if the server does not respond or an
    exception is raised.

    """
    try:
        page = requests.get(url, verify=True)
    except requests.exceptions.RequestException as e:
        return None

    content = page.content
    page.close()
    soup = bs.BeautifulSoup(content, 'lxml')
    if ('This site requires JavaScript' in content) or len(soup.find_all('a'))==0:
        try:
            # PhantomJS for different operating systems
            if platform.system() == 'Windows':
                PHANTOMJS_PATH = './phantomjs/phantomjs.exe'
            elif platform.system() == 'Darwin':
                PHANTOMJS_PATH = './phantomjs/phantomjs_mac'
            else:
                PHANTOMJS_PATH = './phantomjs/phantomjs_linux'

            # here we'll use pseudo browser PhantomJS
            browser = webdriver.PhantomJS(PHANTOMJS_PATH)
            browser.get(url)
            soup = bs.BeautifulSoup(browser.page_source, 'lxml')
            browser.service.process.send_signal(signal.SIGTERM)
            browser.quit()
            return soup
        except:
            return None
    else:
        return soup



def get_webpage_info(url, about_tags=['a'], keywords=None, keywords_avoid=None,
                     min_length=5 ,verbose=False, debug=False):
    """ Scraping the text from the webpages, using ``keywords`` as a way
    to find the pertinent section of the school portals. The text we are
    interested in are paragraphs and long (>3 words) sentences.

    Parameters
    ----------
    url : string
        url ending in /.
    about_tags : list of strings
        HTML tags used for identifying the webpage to be scraped (about us page).
    keywords : list of strings
        Keywords used for identifying the webpage to be scraped.
    keywords_avoid : list of strings
        Keywords to be avoided when identifying the webpage to be scraped.
    min_length : int
        Min length of the scraped sentences/paragraphs.
    verbose : boolean
        Whether information is printed out or not.
    debug : boolean
        If ``debug`` is True then BeautifulSoup objects are returned.

    Returns
    -------
    'NOCONNECTION' in case the page content cannot be retrieved.
    'NOLINKABOUT' in case no 'about us' links are found.
    text : list of strings
        Scraped text.

    """
    def match_souplinktag(li, url, links, keywords, keywords_avoid):
        """ Matching a soup link tag with the keywords based on the 'href'
        attribute.
        """
        # getting all the links with href attribute
        if li.has_attr('href'):
            # separating the last segment of the href
            full_link = li.get('href')
            if full_link.endswith('/'):
                segment_link = full_link.split('/')[-2].lower()
            else:
                segment_link = full_link.split('/')[-1].lower()

            # comparing href to the keywords
            if any(key.lower() in segment_link for key in keywords):
                if any(key2.lower() in segment_link for key2 in keywords_avoid):
                    return links
                elif any(key2.lower() in full_link for key2 in keywords_avoid):
                    return links
                else:
                    found_link = li.get('href')
                    if found_link.startswith('/'):
                        found_link = found_link[1:]
                    if not found_link.startswith('http'):
                        found_link = url+found_link
                    if found_link.endswith('/'):
                        found_link = found_link[:-1]
                    # avoiding duplicaitions
                    if found_link not in links:
                        links.append(found_link)
        return links

    def replace_unicode_chars(tag, prop):
        if prop=='string':
            text = tag.string
        elif prop=='gettext':
            text = tag.get_text()
        return text.replace(u"\u2018", "'"). \
               replace(u"\u2019", "'").replace(u"\xa0", " "). \
               replace(u"\u201c", "'").replace(u"\u201d", "'"). \
               replace(u"\u2013", "-")


    #---------------------------------------------------------------------------
    soup = get_soup(url)
    if soup is None:
        if verbose:  print 'Cannot parse webpage'
        return 'NOCONNECTION'         # cannot get page content
    else:
        if keywords is None:  keywords = KEYWORDS
        if keywords_avoid is None:  keywords_avoid = KEYWORDS_AVOID

        # getting all the link tags in the HTML
        linktags = soup.find_all(about_tags)
        if debug:  print 'Total links:', len(linktags)
        urls_about = []
        for li in linktags:
            urls_about = match_souplinktag(li, url, urls_about, keywords,
                                           keywords_avoid)

        # looking for potential 'ethos or values' links inside 'about us' pages
        if len(urls_about)==1:
            for url_about in urls_about:
                new_soup = get_soup(url_about)
                if new_soup:
                    linktags_new = new_soup.find_all(about_tags)
                    for li in linktags_new:
                        urls_about = match_souplinktag(li, url, urls_about,
                                                       keywords, keywords_avoid)

    if verbose or debug:
        print 'About-us or ethos urls:', len(urls_about)
        print urls_about
    if len(urls_about)==0:  return 'NOLINKABOUT'

    # getting the soups from the 'about us or ethos' pages
    soups_about = [get_soup(url_about) for url_about in urls_about]

    # getting paragraphs and long (>min_length) sentences
    text = []
    for soup_about in soups_about:
        if soup_about is None:
            continue
        for par_tag in soup_about.find_all('p'):
            te = par_tag.get_text()
            if te and len(te.split(' '))>min_length:
                mined_text = replace_unicode_chars(par_tag, 'gettext')
                if mined_text not in text:
                    text.append(mined_text)
        for par_tag in soup_about.find_all('div'):
            te = par_tag.string
            if te and len(te.split(' '))>min_length:
                mined_text = replace_unicode_chars(par_tag, 'string')
                if mined_text not in text:
                    text.append(mined_text)
        for par_tag in soup_about.find_all(['li', 'ol', 'ul']):
            if not par_tag.has_attr('href'):
                te = par_tag.string
                if te is not None:
                    if len(te.split(' '))>min_length and te.count('\n')<5:
                        mined_text = replace_unicode_chars(par_tag, 'string')
                        if mined_text not in text:
                            text.append(mined_text)
                else:
                    te = par_tag.get_text()
                    if te and len(te.split(' '))>min_length and te.count('\n')<5:
                        mined_text = replace_unicode_chars(par_tag, 'gettext')
                        if mined_text not in text:
                            text.append(mined_text)

    if verbose:  print text
    return text
