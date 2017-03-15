
# coding: utf-8

# In[1]:

# All the necessary libraries
import re
import nltk
import time
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup


# In[2]:

# Spell checkers
from spell_check_dict import spelling_checker_dict as spellcheck_dict
from google_search import spell_check as spellcheck


# In[3]:

# Predefined spellcheck dictionary
spellcheck_dict['vynal grip strip']


# In[4]:

# Live Google spellcheck
spellcheck('vinyal')


# In[5]:

# Load dataset files into dataframes
df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')
df_attr = pd.read_csv('../input/attributes.csv')


# In[6]:

# Dummy dataframes for proof of concept
df_train_1 = df_train.copy()
df_test_1 = df_test.copy()


# In[7]:

# Data cleaning functions start here

# helol wold -> hello world
# only uses predefined spell check dict atm
# we can include google spell check for those not in dict
def spellchecker(searchwords):
    if searchwords in spellcheck_dict.keys():
        return spellcheck_dict[searchwords]
    else:
        return searchwords

# Hello World -> hello world
def lowercase(text):
    return str(text).lower()

# process all regexes from other functions below
def regex_processor(text, replace_list):
    for pattern, replace in replace_list:
            try:
                text = re.sub(pattern, replace, text)
            except:
                pass
    return re.sub(r"\s+", " ", text).strip() 

# 200 wattage, 200 watts, 200 watt -> 200 watt
def convertunits(text):
    replace_list = [
            (r"([0-9]+)( *)(inches|inch|in|in.|')\.?", r"\1 in. "),
            (r"([0-9]+)( *)(pounds|pound|lbs|lb|lb.)\.?", r"\1 lb. "),
            (r"([0-9]+)( *)(foot|feet|ft|ft.|'')\.?", r"\1 ft. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(inches|inch|in|in.|')\.?", r"\1 sq.in. "),
            (r"([0-9]+)( *)(square|sq|sq.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 sq.ft. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(inches|inch|in|in.|')\.?", r"\1 cu.in. "),
            (r"([0-9]+)( *)(cubic|cu|cu.) ?\.?(feet|foot|ft|ft.|'')\.?", r"\1 cu.ft. "),
            (r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1 gal. "),
            (r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1 oz. "),
            (r"([0-9]+)( *)(centimeters|cm)\.?", r"\1 cm. "),
            (r"([0-9]+)( *)(milimeters|mm)\.?", r"\1 mm. "),
            (r"([0-9]+)( *)(minutes|minute)\.?", r"\1 min. "),
            (r"([0-9]+)( *)(°|degrees|degree)\.?", r"\1 deg. "),
            (r"([0-9]+)( *)(v|volts|volt)\.?", r"\1 volt. "),
            (r"([0-9]+)( *)(wattage|watts|watt)\.?", r"\1 watt. "),
            (r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1 amp. "),
            (r"([0-9]+)( *)(qquart|quart)\.?", r"\1 qt. "),
            (r"([0-9]+)( *)(hours|hour|hrs.)\.?", r"\1 hr "),
            (r"([0-9]+)( *)(gallons per minute|gallon per minute|gal per minute|gallons/min.|gallons/min)\.?", r"\1 gal. per min. "),
            (r"([0-9]+)( *)(gallons per hour|gallon per hour|gal per hour|gallons/hour|gallons/hr)\.?", r"\1 gal. per hr "),
        ]
    return regex_processor(text, replace_list)   


# helloWorld -> hello World
def splitcases(text):
    replace_list = [
            (r"(\w)[\.?!]([A-Z])", r"\1 \2"),
            (r"(?<=( ))([a-z]+)([A-Z]+)", r"\2 \3"),
        ]
    return regex_processor(text, replace_list)   
    

# hello/world, hello-world -> hello world
def removewordsplitters(text):
    replace_list = [
            (r"([a-zA-Z]+)[/\-]([a-zA-Z]+)", r"\1 \2"),
        ]
    return regex_processor(text, replace_list)   
    
# 1x1 -> 1 x 1
def digitsplitters(text):
    replace_list = [
            (r"(\d+)[\.\-]*([a-zA-Z]+)", r"\1 \2"),
            (r"([a-zA-Z]+)[\.\-]*(\d+)", r"\1 \2"),
        ]
    return regex_processor(text, replace_list)   

# 1,000 -> 1000
def digitcommaremover(text):
    replace_list = [
            (r"([0-9]),([0-9])", r"\1\2")
    ]
    return regex_processor(text, replace_list)   

# one -> 1
def numberconverter(text):
    numbers = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
            "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred", "thousand"
        ]
    digits = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 140, 15,
        16, 17, 18, 19, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000
    ]
    replace_list = [
        (r"%s"%n, str(d)) for n,d in zip(numbers, digits)
    ]
    return regex_processor(text, replace_list)  

# remove special characters
def specialcharcleaner(text):
    replace_list = [
            (r"<.+?>", r""),
            (r"&nbsp;", r" "),
            (r"&amp;", r"&"),
            (r"&#39;", r"'"),
            (r"/>/Agt/>", r""),
            (r"</a<gt/", r""),
            (r"gt/>", r""),
            (r"/>", r""),
            (r"<br", r""),
            (r"[ &<>)(_,;:!?\+^~@#\$]+", r" "),
            ("'s\\b", r""),
            (r"[']+", r""),
            (r"[\"]+", r""),
        ]
    return regex_processor(text, replace_list)  

# handle all HTML tags
def HtmlCleaner(text, parser='html.parser'):
    bs = BeautifulSoup(text, parser)
    text = bs.get_text(separator=" ")
    return text

# lemmatizing
def lemmatizer(text):
    Tokenizer = nltk.tokenize.TreebankWordTokenizer()
    Lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    tokens = [Lemmatizer.lemmatize(token) for token in Tokenizer.tokenize(text)]
    return " ".join(tokens)

# stemming
def stemmer(text):
    stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(token) for token in text.split(" ")]
    return " ".join(tokens)


# In[8]:

# Apply all data cleaning

def df_cleaner(df_col):
    cleaner_funcs = [
        spellchecker,
        lowercase,
        convertunits,
        splitcases,
        removewordsplitters,
        digitsplitters,
        digitcommaremover,
        numberconverter,
        specialcharcleaner,
        HtmlCleaner,
        lemmatizer,
        stemmer
    ]
    
    for func in cleaner_funcs:
        df_col = df_col.apply(lambda x: func(str(x)))
    return df_col

#==============================================================================
 start_train_time = time.time()
 print("--- Cleaning Train Data---")
 
# Clean train data
 df_train_1.product_title = df_cleaner(df_train_1.product_title)
 df_train_1.search_term = df_cleaner(df_train_1.search_term)
 
 print(" :) Train Data Cleaning Finished in %s minutes" % round(((time.time() - start_train_time)/60),2))
 
#==============================================================================

# In[9]:

#==============================================================================
 start_test_time = time.time()
 print("Cleaning Test Data")
 
 # Clean test data
 df_test_1.product_title = df_cleaner(df_test_1.product_title)
 df_test_1.search_term = df_cleaner(df_test_1.search_term)
 
 print(" :) Test Data Cleaning Finished in %s minutes" % round(((time.time() - start_test_time)/60),2))
#==============================================================================


# In[10]:

# Repeat data cleaning for extra dataframes / columns


# In[11]:

# Time to check results

# Original uncleaned data
df_train.product_title[:5]


# In[12]:

# Cleaned data
df_train_1.product_title[:5]


# In[13]:

# Testing spell checker
# # Apply spell checker to train and test search terms
# df_train_1.search_term = df_train_1.search_term.apply(spellchecker)
# df_test_1.search_term = df_test_1.search_term.apply(spellchecker)

# # No of search terms corrected in Train
# train_corrected = list(df_train.search_term.isin(df_train_1.search_term.apply(spellchecker))).count(False)

# # No of search terms corrected in Test
# test_corrected = list(df_test.search_term.isin(df_test_1.search_term.apply(spellchecker))).count(False)

# print(train_corrected, test_corrected)

#%%

start_test_time = time.time()
print("Cleaning Product Descriptions Data")

 Clean test data
df_pro_desc.product_description = df_cleaner(df_pro_desc.product_description)

print(" :) Product Descriptions Data Cleaning Finished in %s minutes" % round(((time.time() - start_test_time)/60),2))

#%%

start_test_time = time.time()
print("Cleaning Attribute Data")

# Clean test data
df_attr.name = df_cleaner(df_attr.name)
df_attr.value = df_cleaner(df_attr.value)

print(" :) Attribute Data Cleaning Finished in %s minutes" % round(((time.time() - start_test_time)/60),2))


#%%

#save clean data
df_train_1.to_csv('../input_clean/train_clean.csv')
df_test_1.to_csv('../input_clean/test_clean.csv')
df_pro_desc.to_csv('../input_clean/product_descriptions_clean.csv', index=False)
df_attr.to_csv('../input_clean/attributes_clean.csv',index=False)


