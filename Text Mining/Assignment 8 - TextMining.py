# Jorge Solis
# jas2430
# May 1st, 2017
# Professor Hardeep Johar

# Text Mining:
# This script scrapes text data from various Wikipedia articles to create a model for Latent Semantic Analysis.
# It then evaluates each entry in the testLists parameter and returns a recommendation from the testLists parameter.
# You can run this .py file from the terminal.
# Warning: This program takes a while, about ~20 minutes.

# The following are a list of Wikipedia articles cataloguing musicians of various genres.
# These can be exhchanged with arbitrary catalogs.
jazz = "https://en.wikipedia.org/wiki/List_of_jazz_musicians"
bebop = "https://en.wikipedia.org/wiki/List_of_bebop_musicians"
blues = "https://en.wikipedia.org/wiki/List_of_blues_musicians"
experimental = "https://en.wikipedia.org/wiki/List_of_experimental_musicians"
reggae = "https://en.wikipedia.org/wiki/List_of_reggae_musicians"
trainingLists = [jazz, bebop, blues, experimental, reggae]

# The following parameter can be exchanged for any Wikipedia catalog of items similar to the training parameters.
testLists = ["https://en.wikipedia.org/wiki/List_of_folk_musicians",]

# The following imports are the necessary Text Analysis and Machine Learning libaries.
from gensim.similarities.docsim import Similarity
from gensim import corpora, models, similarities
import requests
import bs4
from bs4 import BeautifulSoup as bs
from lxml import etree
import nltk
from nltk.corpus import PlaintextCorpusReader
from nltk import sent_tokenize,word_tokenize 
from nltk.book import *
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import STOPWORDS
import pprint

# The following subroutines work to scrape the relevant data from sources that are not uniform in design and layout.
def get_musicians(url):
    musicians = list()
    page = requests.get(url)
    page = bs(page.content, 'lxml')
    field = page.find('div', id='content')
    field = field.find('div', id='bodyContent')
    field = field.find('div', id='mw-content-text')
    
    divs = field.find_all('div')
    for div in divs:
        table = div.find('div', id='toc')
        if(not table == None):
            table.extract()
        if not div.get('class') == None and 'div-col' in div.get('class'):
            uls = div.find_all('ul')
            for ul in uls:
                lis = ul.find_all('li', class_='')
                for li in lis:
                    link = li.find('a', class_='')
                    if type(link) == bs4.element.Tag:
                        musicians.append([link.get('title'),"http://www.wikipedia.org" + link.get('href')])
                        
    uls = list()
    divs = field.find_all('div')
    for div in divs:
        tables = div.find_all('table', class_='multicol')
        for table in tables:
            table = table.extract()
            table = table.find('tr')
            table = table.find('td')
            cell = table.find_all('ul')
            uls.append(cell)
    children = field.children
    for child in children:
        if type(child) == bs4.element.Tag and child.tag == 'ul':
            uls.append(child)
            for ul in uls:
                ul = ul.find_all('li', class_='')
                for item in ul:
                    links = item.find_all('a', class_='')
                    for link in links:
                        musicians.append([link.get('title'),"http://www.wikipedia.org" + link.get('href')])
                        
    tables = field.find_all('table', class_='wikitable sortable')
    for table in tables:
        items = table.find_all('tr')
        for element in items:
            link = element.find('a', class_='')
            if(not link == None):
                musicians.append([link.get('title'),"http://www.wikipedia.org" + link.get('href')])
    
    if(len(musicians) == 0):
        uls = list()
        children = field.children
        for child in children:
            if type(child) == bs4.element.Tag and child.tag == 'ul':
                uls.append(child)
        for ul in uls:
            for item in ul:
                ul = ul.find_all('li', class_='')
                for item in ul:
                    links = item.find_all('a', class_='')
                    for link in links:
                        musicians.append([link.get('title'),"http://www.wikipedia.org" + link.get('href')])
                        
    return musicians

# The following subroutines carry out computationally taxing model construction.
# These routines are poorly optimized and need work.
def get_page_text(url):
    text = ''
    page = requests.get(url)
    page = bs(page.content, 'lxml')
    for tag in page.find_all('p'):
        text += tag.get_text()
    return text

trainingMusicians = list()
for genre in trainingLists:
    musicians = get_musicians(genre)
    for musician in musicians:
        trainingMusicians.append(musician)
testMusicians = list()
for genre in testLists:
    musicians = get_musicians(genre)
    for musician in musicians:
        testMusicians.append(musician) 

all_text = ''
for entry in trainingMusicians:
    text = get_page_text(entry[1])
    entry[1] = text
    all_text += text
striptext = all_text.replace('\n\n', ' ')
striptext = striptext.replace('\n', ' ')
documents = [entry[1] for entry in trainingMusicians]
sentences = sent_tokenize(striptext)
words = word_tokenize(striptext)
texts = [[word for word in document.lower().split()
        if word not in STOPWORDS and word.isalnum()]
        for document in documents]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
lsi = models.LsiModel(corpus, id2word=dictionary)

for entry in testMusicians:
    text = get_page_text(entry[1])
    vec_bow = dictionary.doc2bow(text.lower().split())
    vec_lsi = lsi[vec_bow]
    index = similarities.MatrixSimilarity(lsi[corpus])
    sims = index[vec_lsi]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    entry[1] = trainingMusicians[sims[0][0]][0]

# The script will now print a recommendation for each entry in the testLists parameter.
for entry in testMusicians:
    print(entry[0] + " is most similar to: " + entry[1])
