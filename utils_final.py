# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:43:51 2019

@author: Greg
"""
import pandas as pd
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import re	
import nltk
import spacy
import unicodedata
from nltk.corpus import wordnet
from nltk import pos_tag, word_tokenize
from nltk.tokenize.toktok import ToktokTokenizer
from bs4 import BeautifulSoup
from contractions import contractions_dict

# Once downloaded this can be commented out.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('universal_tagset')

nlp = spacy.load('en_core_web_sm')
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')
stopword_list.remove('no')
stopword_list.remove('not')

#%% Cleaning Text - strip HTML
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text

# %% Remove Accented Characters
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

# %% Expanding Contractions
def expand_contractions(text, contraction_mapping=contractions_dict):
    
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())), 
                                      flags=re.IGNORECASE|re.DOTALL)
    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match)\
                                if contraction_mapping.get(match)\
                                else contraction_mapping.get(match.lower())
                                
        if expanded_contraction == None: # catch exceptions       
            expanded_contraction = first_char
        else:
            expanded_contraction = first_char+expanded_contraction[1:]
        return expanded_contraction
        
    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

# %% Removing Special Characters
def remove_special_characters(text):
    text = re.sub('[^ a-zA-z0-9\s]', '', text)
    return text

#%% Lemmatizing Text
def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

# %% Removing Stopwords
def remove_stopwords(text, is_lower_case=True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

# Remove any non words from text
def remove_nonwords(text, is_lower_case=True):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    try:
      if is_lower_case:
          filtered_tokens = [token for token in tokens if wordnet.synsets(token)]
      else:
          filtered_tokens = [token for token in tokens if wordnet.synsets(token.lower())]
      filtered_text = ' '.join(filtered_tokens)    
      return filtered_text
    except Exception as e:
      return text

# %% Normalizing text corpus
def normalize_corpus(corpus, html_stripping=True, contraction_expansion=True,
                             accented_char_removal=True, text_lower_case=True, 
                             text_lemmatization=True, special_char_removal=True, 
                             stopword_removal=True, number_removal=True, weird_chars=True):
    
    # strip HTML
    if html_stripping:
        corpus = strip_html_tags(corpus)
    # remove accented characters
    if accented_char_removal:
        corpus = remove_accented_chars(corpus)
    # expand contractions    
    if contraction_expansion:
        corpus = expand_contractions(corpus)
    # lowercase the text    
    if text_lower_case:
        corpus = corpus.lower()
    # remove extra newlines
    corpus = re.sub(r'[\r|\n|\r\n]+', ' ',corpus)
    # insert spaces between special characters to isolate them    
    special_char_pattern = re.compile(r'([{.(-)!}])')
    corpus = special_char_pattern.sub(" \\1 ", corpus)
    # lemmatize text
    if text_lemmatization:
        corpus = lemmatize_text(corpus)
    # remove special characters    
    if special_char_removal:
        corpus = remove_special_characters(corpus)  
    # remove extra whitespace
    corpus = re.sub(' +', ' ', corpus)
    # remove stopwords
    if stopword_removal:
        corpus = remove_stopwords(corpus, is_lower_case=text_lower_case)
    # remove numbers
    if number_removal:
        corpus = ''.join([i for i in corpus if not i.isdigit()])
        
    return corpus
