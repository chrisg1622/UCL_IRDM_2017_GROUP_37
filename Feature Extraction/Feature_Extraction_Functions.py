# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 21:38:17 2017

@author: Chris
"""

import pandas as pd
import numpy as np
import pickle
from collections import defaultdict

attributes = pd.read_csv('../input_clean/attributes_clean.csv',encoding="ISO-8859-1")
product_descriptions = pd.read_csv('../input_clean/product_descriptions_clean.csv')
train = pd.read_csv('../input_clean/train_clean.csv', encoding="ISO-8859-1",index_col='id')
test = pd.read_csv('../input_clean/test_clean.csv', encoding="ISO-8859-1",index_col='id')

#save an object (e.g. dict) with pickle
def save_obj(obj, name = 'object' ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#load an object with pickle
def load_obj(name = 'object'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

#function to retrieve a list of attribute names for a certain product
def get_attr_names_list(product_uid, get_text = False):
    attr_names_df = attributes.loc[attributes['product_uid']==product_uid]['name']
    attr_names_list = [name for name in attr_names_df]
    #some of the attribute names contain more than one word - so split these up into individual words
    attr_names_str = ''
    for name in attr_names_list: 
        attr_names_str += name + ' '
    #get_text is used when creating the products dataframe to store list as one string
    if get_text == True:
        return attr_names_str
    return attr_names_str.split()
    
#function to retrieve a list of attribute values for a certain product
def get_attr_value_list(product_uid, get_text = False):
    attr_value_df = attributes.loc[attributes['product_uid']==product_uid]['value']
    attr_value_list = [str(name) for name in attr_value_df]
    #some of the attribute values contain more than one word - so split these up into individual words
    attr_value_str = ''
    for name in attr_value_list: 
        attr_value_str += name + ' '
    #get_text is used when creating the products dataframe to store list as one string
    if get_text == True:
        return attr_value_str
    return attr_value_str.split()
    
#function to retrieve a list of product description words for a certain product
def get_prod_descr_list(product_uid, get_text = False):
    prod_descr_df = product_descriptions.loc[product_descriptions['product_uid']==product_uid]['product_description']
    prod_descr_list = [name for name in prod_descr_df]
    #some of the attribute values contain more than one word - so split these up into individual words
    prod_descr_str = ''
    for name in prod_descr_list: 
        prod_descr_str += name + ' '
    #get_text is used when creating the products dataframe to store list as one string
    if get_text == True:
        return prod_descr_str
    return prod_descr_str.split()

#function to retrieve the set of all search term words, for use in idf calculation
def get_search_terms_set(train, test):
    search_terms = []
    num_instances_train = len(train)
    num_instances_test = len(test)
    #add each search term from the training set to the list
    print('Retrieving search terms from training data...')
    for i,datum in enumerate(train.iterrows()):
        ID, row = datum
        for search_term in row['search_term'].split():
            search_terms.append(search_term)
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances_train))
    #add each search term from the test set to the list
    print('Retrieving search terms from testing data...')
    for i,datum in enumerate(test.iterrows()):
        ID, row = datum
        for search_term in row['search_term'].split():
            search_terms.append(search_term)
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances_test))

    return set(search_terms)

#this function takes in all the product information, and puts it all in a dataframe
#so that for any product, all of its information can be accessed in one place
def create_product_dataframe(train, test, attributes, product_descriptions):  
    num_instances_train = len(train)
    num_instances_test = len(test)
    #retrieve a list containing all product uids in the train and test data
    product_uid_list = list(set(list(train['product_uid']) + list(test['product_uid'])))   
    #create dataframe containing product uids
    dataframe = pd.DataFrame(index = product_uid_list)
    #go through training data and add product_titles to the dataframe
    print('Adding training data to product dataframe...')
    for i,datum in enumerate(train.iterrows()):
        ID, row = datum
        product_uid = row['product_uid']
        product_title = row['product_title']
        dataframe.loc[product_uid,'product_title'] = product_title
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances_train))
    
    #go through test data and add product_titles to the dataframe
    print('Adding test data to product dataframe...')
    for i,datum in enumerate(test.iterrows()):
        ID, row = datum
        product_uid = row['product_uid']
        product_title = row['product_title']
        dataframe.loc[product_uid,'product_title'] = product_title
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances_test))
    
    num_instances_df = len(dataframe)
    #go through test data and add product_titles to the dataframe
    print('Adding product descriptions, attribute names and attribute values to product dataframe...')
    for i,product_uid in enumerate(dataframe.index):
        dataframe.loc[product_uid,'prod_descr'] = get_prod_descr_list(product_uid, get_text=True)
        dataframe.loc[product_uid,'attr_names'] = get_attr_names_list(product_uid, get_text=True)
        dataframe.loc[product_uid,'attr_values'] = get_attr_value_list(product_uid, get_text=True)
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances_df))
    return dataframe
    
#function to create a default dict of idf values for each search term, for a particular
#document set
def create_idf_defaultdict(search_terms, product_df, doc = 'product_title'):
    idf_dict = defaultdict(float)
    num_instances = len(search_terms)
    #for each search term, compute idf value
    for i,search_term in enumerate(search_terms):
        #count number of documents the term appears in
        idf_dict[search_term] = sum([search_term in document.split() for document in product_df[doc]])
        #after every 10 processed search terms, save progress
        if (i+1) % 1000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))

    return idf_dict

#function to load idf default dicts for a given document set (e.g. product_title)
def load_idf_default_dict(search_terms, product_dataframe, doc_set = 'product_title'):
    try:
        print('Loading inverse document frequency default dict for {}...'.format(doc_set))
        idf_dict = load_obj('../input_clean/idf_dict_'+doc_set)
    except:
        print('Failed to load inverse document frequency default dict for {}...'.format(doc_set))
        print('Creating inverse document frequency default dict for {}...'.format(doc_set))
        idf_dict = create_idf_defaultdict(search_terms, product_dataframe, doc_set)
        print('Saving inverse document frequency default dict for {}...'.format(doc_set))
        save_obj(idf_dict,'../input_clean/idf_dict_'+doc_set)
    return idf_dict

#function to take as input the product_dataframe and either train or test dataframes, and append the 
#product info (description, attr_names and attr_vals) to the dataframe
def add_product_info_to_data(dataframe, product_dataframe):
    num_instances = len(dataframe)
    for i, datum in enumerate(dataframe.iterrows()):
        ID, row = datum
        product_uid = row['product_uid']
        for attribute in ['prod_descr','attr_names','attr_values']:
            dataframe.loc[ID, attribute] = product_dataframe.loc[product_uid, attribute]
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))
    return dataframe
    
#function to store the average lengths of each type of document in a dict
def create_dict_of_avg_doc_lengths(product_df):
    len_dict = {}
    for doc in ['product_title','prod_descr','attr_names','attr_values']:
        print('Computing average length of {} documents'.format(doc))
        len_dict[doc] = np.mean([len(doc_words.split()) for doc_words in product_df[doc]])
    return len_dict
    
#function to load (or create) the dictionary of average document lengths
def load_dict_of_avg_doc_lengths(product_df):
    try:
        print('Loading dictionary of average document lengths (for BM25 calculations)')
        len_dict = load_obj('../input_clean/len_dict')
    except:
        print('Failed to load dictionary of average document lengths (for BM25 calculations)')
        print('Creating dictionary of average document lengths (for BM25 calculations)')
        len_dict = create_dict_of_avg_doc_lengths(product_df)
        print('Saving dictionary of average document lengths (for BM25 calculations)')
        save_obj(len_dict,'../input_clean/len_dict')
    return len_dict

#function to load search terms
def load_search_terms(train,test):
    try:
        print('Loading search terms...')
        search_terms = load_obj('../input_clean/search_terms')
    except:
        print('Failed to load search terms...')
        print('Retrieving search terms...')
        search_terms = get_search_terms_set(train,test)
        print('Saving search terms to file...')
        save_obj(search_terms,'../input_clean/search_terms')
    return search_terms

#function to load (or create) the datasets which include all product info
def load_train_and_test_data_with_product_info(train_original, test_original, product_dataframe):
    try:
        print('Loading the train and test dataframes which include all product information...')
        train = load_obj('../input_clean/train_all')
        test = load_obj('../input_clean/test_all')
    except:
        print('Failed to load the train and test dataframes which include all product information...')
        print('Creating the train and test dataframes which include all product information......')
        train = add_product_info_to_data(train_original, product_dataframe)
        test = add_product_info_to_data(test_original, product_dataframe)
        print('Saving product dataframe to file...')
        save_obj(train,'../input_clean/train_all')
        save_obj(test,'../input_clean/test_all')
    return train, test

#function to load (or create) dataframe with all the product information on it, 
#indexed by product_uid
def load_product_dataframe(train,test,attributes,product_descriptions):
    try:
        print('Loading product dataframe...')
        product_dataframe = load_obj('../input_clean/product_dataframe')
    except:
        print('Failed to load product dataframe...')
        print('Creating product dataframe...')
        product_dataframe = create_product_dataframe(train,test,attributes,product_descriptions)
        print('Saving product dataframe to file...')
        save_obj(product_dataframe,'../input_clean/product_dataframe')
    return product_dataframe

#function to compute term frequency
def term_frequency(doc, terms, tf_type = 'natural', K = 0.5):    
    """inputs: doc - the document to use for tf calculation (e.g. product_title, product_description)
               terms - the list of terms to use to tf calculation (i.e. search_term)
               tf_type - the type of term frequency to calculate (natural, log, etc)
               K - double normalisation constant, only used when tf_type is 'double_norm'
       output: term frequency of 'terms' in 'doc', using tf_type as the type of tf computed
    """
    tf = 0.0
    #take the sum of the term frequencies
    if tf_type == 'natural':
        return sum( [sum([term == doc_word for doc_word in doc]) for term in terms])

    #take the average term frequency
    if tf_type == 'natural_avg':
        return term_frequency(doc, terms, 'natural', K)/len(terms)
            
    #take the sum of log normalised term frequencies
    if tf_type == 'log_norm':
        return sum( [1 + np.log(max(1,sum([term == doc_word for doc_word in doc]))) for term in terms])
          
    #take the average log normalised term frequencies
    if tf_type == 'log_norm_avg':
        return term_frequency(doc,terms,'log_norm', K)/len(terms)
    
    #take the sum of double normalised K term frequencies
    if tf_type == 'double_norm_'+str(K):
        doc_words = set(doc)
        if doc_words == set():
            return 0.0
        #compute the max term frequency out of all words in the document
        tf_max = max([sum([word == doc_word for doc_word in doc]) for word in doc_words])
        return sum( [K + K*(sum([term == doc_word for doc_word in doc]))/tf_max for term in terms])

    #take the average of double normalised K term frequencies
    if tf_type == 'double_norm_'+str(K)+'_avg':
        return term_frequency(doc,terms,'double_norm_'+str(K), K)/len(terms)
    
    return tf

#function to compute inverse document frequency
def inverse_document_frequency(idf_dict, terms, idf_type = 'smooth', N = 124428):    
    """inputs: idf_dict - the dictionary with the idf counts for each term in the corresponding corpus
               terms - the list of terms to use to idf calculation (i.e. search_term)
               idf_type - the type of inverse document frequency to calculate (smooth, max, etc)
       output: inverse document frequency of 'terms' in the corpus corresponding to idf_dict, using idf_type as the type of idf computed
    """
    idf = 0.0
    #take the sum of the smooth inverse document frequencies
    if idf_type == 'smooth':
        return sum( [np.log( N / (1 + idf_dict[term]) ) for term in terms])

    #take the average smooth inverse document frequencies
    if idf_type == 'smooth_avg':
        return inverse_document_frequency(idf_dict, terms, 'smooth', N)/len(terms)
    
    #take the sum of the max inverse document frequencies
    if idf_type == 'max':
        return sum( [np.log( max(idf_dict.values()) / (1 + idf_dict[term]) ) for term in terms])

    #take the average max inverse document frequencies
    if idf_type == 'max_avg':
        return inverse_document_frequency(idf_dict, terms, 'max', N)/len(terms)
        
    #take the sum of the probabilistic inverse document frequencies
    if idf_type == 'prob':
        return sum( [np.log( max(1,N - (1 + idf_dict[term])) / (1 + idf_dict[term])) for term in terms])

    #take the average probabilistic inverse document frequencies
    if idf_type == 'prob_avg':
        return inverse_document_frequency(idf_dict, terms, 'prob', N)/len(terms)
    
    return idf

#function to compute tf-idf values
def tf_idf(tf_doc, idf_dict, terms, tf_idf_type = 'sum', tf_type = 'natural', idf_type = 'smooth', idf_N = 124428, tf_K = 0.5):    
    """inputs: tf_doc - the document for use in tf calculations
               idf_dict - the dictionary for use in idf calculations
               terms - the list of terms to use for tf-idf calculation (i.e. search_terms)
               tf_idf_type - either takes the sum of tf-idf values, or the average
               tf_type - the type of term frequency to calculate (natural, log, etc)
               idf_type - the type of inverse document frequency to calculate (smooth, max, etc)
               idf_N - value of N to use in idf calculation (i.e. number of docs)
               tf_K - value of K to use if tf_type is 'double norm K'
       output: tf_idf sum or average for the terms in 'terms'
    """
    TF_IDF = 0.0
    #take the sum of tf_idf products across all query terms
    if tf_idf_type == 'sum':
        return sum([term_frequency(tf_doc, [term], tf_type, tf_K) * inverse_document_frequency(idf_dict, [term], idf_type, idf_N) for term in terms])
            
    #take the avg of tf_idf products across all query terms
    if tf_idf_type == 'avg':
        return tf_idf(tf_doc, idf_dict, terms, 'sum', tf_type, idf_type, idf_N, tf_K)/len(terms)
    
    return TF_IDF
    
#function to compute the BM25 score for a query-document pair
def BM25(doc_words, query_terms, idf_dict, avg_doc_len, bm25_type = 'sum', k1 = 1.5, b = 0.75):
    """inputs: doc_words - the document for use in tf calculations
               query_terms - the list of terms to use for bm25 calculation (i.e. search_terms)
               idf_dict - the dictionary for use in idf calculations
               avg_doc_len - the average document length
               K1 - k1 weighing factor, should be between 1.2 and 2
               b - the b weighing factor, should be between 0.5 and 8
       output: bm25 score of the query-document pair
    """
    doc_len = len(doc_words)
    bm25 = 0.0
    if bm25_type == 'sum':
        for term in query_terms:
            tf = term_frequency(doc_words, [term], tf_type = 'natural')
            idf = idf_dict[term]
            bm25 += idf * tf * (k1 + 1) / (tf + k1*(1 - b + b*(doc_len/avg_doc_len)))
    if bm25_type == 'avg':
        return BM25(doc_words, query_terms, idf_dict, avg_doc_len, 'sum', k1, b) / len(query_terms)
    
    return bm25
    
    

#this function computes the term frequency feature for the dataframe df, using
#the document and tf type given in the function call, and adds this feature to 
#the dataframe X
def add_term_frequency_to_X(X, df, doc = 'product_title', tf_type = 'natural'):
    """inputs: X - the dataframe we want to add term frequency features to
               df - the data we want to use to get the search terms (train or test)
               product_df - the dataframe containing the information for each product
               doc - the column of product_df we want to use as the tf document
               tf_type - the type of term frequency to calculate (sum, log sum, etc)
       output: None - X is changed when the function is called
    """
    num_instances = len(X)
    #preprocess each data instance
    for i,datum in enumerate(df.iterrows()):
        #get list of search terms and the product uid for the current product
        ID, row = datum
        search_terms = row['search_term'].split()
        tf_doc = row[doc].split()
        #set the relevant attribute to the term frequency of the given terms, doc and tf type
        X.loc[ID,doc+'_tf_'+tf_type] = term_frequency(doc = tf_doc, terms = search_terms, tf_type = tf_type)
        #alert user after every 10000 data instances have been preprocessed
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))

#this function computes the inverse document frequency feature for the dataframe df, using
#the idf_dict and idf type given in the function call, and adds this feature to 
#the dataframe X
def add_inverse_document_frequency_to_X(X, df, idf_dict, doc = 'product_title', idf_type = 'smooth', N = 124428):
    """inputs: X - the dataframe we want to add term frequency features to
               df - the data we want to use to get the search terms (train or test)
               idf_dict - the dictionary with the idf counts for each term in the corresponding corpus 
               doc - the document corpus we are using for idf calculation (e.g. product title)
               idf_type - the type of inverse document frequency to calculate (sum, log sum, etc)
       output: None - X is changed when the function is called
    """
    num_instances = len(X)
    #preprocess each data instance
    for i,datum in enumerate(df.iterrows()):
        #get list of search terms and the product uid for the current product
        ID, row = datum
        search_terms = row['search_term'].split()
        #set the relevant attribute to the term frequency of the given terms, doc and tf type
        X.loc[ID,doc+'_idf_'+idf_type] = inverse_document_frequency(idf_dict, search_terms, idf_type, N)
        #alert user after every 10000 data instances have been preprocessed
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))

#this function computes the tf-idf feature for the dataframe df, using
#the idf_dict and idf type given in the function call, and adds this feature to 
#the dataframe X
def add_tf_idf_to_X(X, df, idf_dict, doc = 'product_title', tf_idf_type = 'sum', tf_type = 'natural', idf_type = 'smooth', idf_N = 124428, tf_K = 0.5):
    """inputs: X - the dataframe we want to add tf-idf features to
               df - the data we want to use to get the search terms (train or test)
               product_df - the dataframe to be used for extracting the document for idf calculation in tf-idf
               idf_dict - the dictionary with the idf counts for each term in the corresponding corpus 
               doc - the document corpus we are using for idf calculation (e.g. product title)
               tf_idf_type - how to aggregate tf-idf values across query terms (sum or average)
               tf_type - the type of tf to calculate
               idf_type - the type of inverse document frequency to calculate (sum, log sum, etc)
               idf_N - number of docs in corpus for idf calculation
               tf_K - value to use for K if tf_type is 'double norm K'
       output: None - X is changed when the function is called
    """
    num_instances = len(X)
    #preprocess each data instance
    for i,datum in enumerate(df.iterrows()):
        #get list of search terms and the product uid for the current product
        ID, row = datum
        search_terms = row['search_term'].split()
        tf_doc = row[doc].split()
        #set the relevant attribute to the term frequency of the given terms, doc and tf type
        X.loc[ID,doc+'_tf_idf_('+tf_type+'-'+idf_type+')_'+tf_idf_type] = tf_idf(tf_doc, idf_dict, search_terms, tf_idf_type, tf_type, idf_type, idf_N, tf_K)
        #alert user after every 10000 data instances have been preprocessed
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))

#this function computes the number of words in a given document (i.e. product title,
#product description, etc.) and adds it as a feature to X
def add_doc_length_to_X(X, df, doc = 'product_title'):
    num_instances = len(X)
    #preprocess each data instance
    for i,datum in enumerate(df.iterrows()):
        ID, row = datum
        #compute the number of words in the given document
        X.loc[ID,doc+'_length'] = len(row[doc].split())
        #alert user after every 10000 data instances have been preprocessed
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))

#this function computes the bm25 feature for the dataframe df, using
#the idf_dict and idf type given in the function call, and adds this feature to 
#the dataframe X
def add_bm25_to_X(X, df, idf_dict, avg_doc_len, doc =  'product_title', bm25_type = 'sum', k1 = 1.5, b = 0.75):
    """inputs: X - the dataframe we want to add tf-idf features to
               df - the data we want to use to get the search terms (train or test)
               idf_dict - the dictionary with the idf counts for each term in the corresponding corpus 
               doc - the document corpus we are using for idf calculation (e.g. product title)
       output: None - X is changed when the function is called
    """
    num_instances = len(X)
    #preprocess each data instance
    for i,datum in enumerate(df.iterrows()):
        #get list of query terms and doc words to use for the bm25 score
        ID, row = datum
        query_words = row['search_term'].split()
        doc_words = row[doc].split()
        #set the relevant attribute to the term frequency of the given terms, doc and tf type
        X.loc[ID,doc+'_bm25_'+bm25_type] = BM25(doc_words, query_words, idf_dict, avg_doc_len, bm25_type, k1, b)
        #alert user after every 10000 data instances have been preprocessed
        if (i+1) % 10000 == 0: print('Processed {} of {} instances...'.format(i+1,num_instances))



#function to add a feature to X, computed from the data in dataframe (i.e. train/test)
#the feature is only added if it doesn't already exist within the dataframe
def add_feature(X, dataframe, product_df, feature = ('_tf_','product_title','natural'), K = 0.5, idf_dict_prod_title = None, idf_dict_prod_descr = None, idf_dict_attr_names = None, idf_dict_attr_vals = None, N = 124428, avg_len_dict = None, data = 'train'):

    idf_dicts = {'product_title' : idf_dict_prod_title,
                 'prod_descr' : idf_dict_prod_descr,
                 'attr_names' : idf_dict_attr_names,
                 'attr_vals' : idf_dict_attr_vals}
    attr_type = feature[0]
    feat_doc = feature[1]
    
    # add the term frequency feature to X
    if attr_type == '_tf_':
        tf_type = feature[2]
        print('Adding feature \'{}\' to {} data'.format(str(feat_doc + attr_type + tf_type), data))
        if str(feat_doc + attr_type + tf_type) not in X.columns:
            add_term_frequency_to_X(X, dataframe, doc=feat_doc,tf_type=tf_type)
        else:
            print('Feature \'{}\' already in dataset'.format(str(feat_doc + attr_type + tf_type)))
    
    #add the document length to X
    if attr_type == '_length':
        print('Adding feature \'{}\' to {} data'.format(str(feat_doc+attr_type), data))
        if str(feat_doc+attr_type) not in X.columns:
            add_doc_length_to_X(X, dataframe, doc = feat_doc)
        else:
            print('Feature \'{}\' already in dataset'.format(str(feat_doc+attr_type)))
            
    #add the inverted document frequency feature to X
    if attr_type == '_idf_':
        idf_type = feature[2]
        idf_dict = idf_dicts[feat_doc]
        print('Adding feature \'{}\' to {} data'.format(str(feat_doc + attr_type + idf_type),data))
        if str(feat_doc + attr_type + idf_type) not in X.columns:
            add_inverse_document_frequency_to_X(X, dataframe, idf_dict, doc = feat_doc, idf_type = idf_type, N = N)
        else:
            print('Feature \'{}\' already in dataset'.format(str(feat_doc + attr_type + idf_type)))


    #add the tf-idf feature to X
    if attr_type == '_tf_idf_':
        tf_type = feature[2]
        idf_type = feature[3]
        tf_idf_type = feature[4]
        idf_dict = idf_dicts[feat_doc]
        print('Adding feature \'{}\' to {} data'.format(str(feat_doc + attr_type + '(' + tf_type + '-' + idf_type + ')_' + tf_idf_type),data))        
        if str(feat_doc + attr_type + '(' + tf_type + '-' + idf_type + ')_' + tf_idf_type) not in X.columns:
            add_tf_idf_to_X(X, dataframe, idf_dict, doc = feat_doc, tf_idf_type = tf_idf_type, tf_type = tf_type, idf_type = idf_type, idf_N = N, tf_K = K)
        else:            
            print('Feature \'{}\' already in dataset'.format(str(feat_doc + attr_type + '(' + tf_type + '-' + idf_type + ')_' + tf_idf_type)))

    #add the bm25 feature to X
    if attr_type == '_bm25_':
        bm25_type = feature[2]
        k1 = feature[3]
        b = feature[4]        
        idf_dict = idf_dicts[feat_doc]
        print('Adding feature \'{}\' to {} data'.format(str(feat_doc + attr_type + bm25_type),data))        
        if str(feat_doc + attr_type) not in X.columns:
            add_bm25_to_X(X, dataframe, idf_dict, avg_len_dict[feat_doc], feat_doc, bm25_type, k1, b)
        else:            
            print('Feature \'{}\' already in dataset'.format(str(feat_doc + attr_type + bm25_type)))
