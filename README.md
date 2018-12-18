# F4S
Team members: Aira Lobo Gomes, Carlos Gomez Gonzalez, Felix Stollenwerk.

Project developed for the S2DS 2017 bootcamp. The main goal was to determine whether the language used by UK schools (ethos or "about us" sections of their websites) could be used as a proxy for their Ofsted (Office for Standards in Education, Children’s Services and Skills) rating. 

Python 2.7 was used as the main language for this analysis. The source files are located in the ``src`` folder. We used web-scraping to generate the database, natural language processing to analyze the texts, and machine learning techniques to model the data. 

In terms of the tools used, the project can be divided in several areas:

* Data wrangling and visualization:​ Pandas, matplotlib
* Web scraping:​ BeautifulSoup, PhantomJS, Selenium
* Natural Language Processing:​ NLTK, SpaCy, Gensim, Textblob, Sklearn Classification/clustering:​ Sklearn, Imblearn, Mord, Lightning

The Jupyter notebooks contain most of the analysis and results:
* nb_data_generation_web_scraping: ​data generation and web scraping (~20k schools) 
* nb_more_scraping: ​rating-specific web scraping
* nb_text_preparation: ​text preparation for further NLP analysis
* nb_feature_eng_bow: ​clustering analysis using the bag of words 
* nb_feature_eng_semantic: ​latent semantic analysis
* nb_feature_eng_sentiment: ​sentiment analysis 
* nb_external_features_classification_clustering: ​Ofsted features and machine learning

