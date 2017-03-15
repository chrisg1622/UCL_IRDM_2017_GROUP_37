# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 21:59:22 2017

@author: Chris
"""

import pandas as pd
import numpy as np
import Feature_Extraction_Functions as f
 
attributes = pd.read_csv('../input_clean/attributes_clean.csv',encoding="ISO-8859-1")
product_descriptions = pd.read_csv('../input_clean/product_descriptions_clean.csv')
train_original = pd.read_csv('../input_clean/train_clean.csv', encoding="ISO-8859-1",index_col='id')
test_original = pd.read_csv('../input_clean/test_clean.csv', encoding="ISO-8859-1",index_col='id')

#for each product, I have put all its data in a dataframe...
product_dataframe = f.load_product_dataframe(train_original,test_original,attributes,product_descriptions)
    
#I have added all the product info into the train and test data for feature calculation efficiency
#try to load this 'train_all' and 'test_all' data, or if it has not yet been made, make it and save it.
train, test = f.load_train_and_test_data_with_product_info(train_original, test_original, product_dataframe)
    
#try to load search terms
search_terms = f.load_search_terms(train,test)
    
#try to load the dictionary containing average document lengths, for BM25 calculations
doc_len_dict = f.load_dict_of_avg_doc_lengths(product_dataframe)
    
#try to load the defaultdicts containing idf values
idf_dict_product_title = f.load_idf_default_dict(search_terms,product_dataframe,'product_title')
idf_dict_prod_descr = f.load_idf_default_dict(search_terms,product_dataframe,'prod_descr')
idf_dict_attr_names = f.load_idf_default_dict(search_terms,product_dataframe,'attr_names')
idf_dict_attr_vals = f.load_idf_default_dict(search_terms,product_dataframe,'attr_values')

#have a look at data for the first product uid
#attributes.loc[attributes['product_uid']==100001]
#product_descriptions.loc[product_descriptions['product_uid']==100001]
#train.loc[train['product_uid']==100001]

#%%

load_data = True

#try to load pre-processed data (to avoid recomputing features), if this fails create new dataframes
if load_data == True:
    print('Attempting to load pre-processed data...')
    try:
        #load pre-processed training data
        X_train = f.load_obj('input_clean/X_train')
        X_test = f.load_obj('input_clean/X_test')
        Y_train = f.load_obj('input_clean/Y_train')
        print('Pre-processed data successfully loaded...')
    except:
        print('Pre-processed data failed to load, creating new dataframe...')
        #create a data frame to store feature vectors
        X_train = pd.DataFrame(index = train.index)
        X_test = pd.DataFrame(index = test.index)
        #retrieve labels for training data
        Y_train = pd.DataFrame(data = {'relevance':train['relevance']}, index = train.index)
else:
    print('Creating dataframes to store new features...')
    #create a data frame to store feature vectors
    X_train = pd.DataFrame(index = train.index)
    X_test = pd.DataFrame(index = test.index)
    #retrieve labels for training data
    Y_train = pd.DataFrame(data = {'relevance':train['relevance']}, index = train.index)


#%%

#list of features to add to the data
#note: features in the list are only added if they are not already in the dataframe
features = [
            #tf features should contain: doc & tf_type
            #tf_type is the type of tf to calculate
            ('_tf_','product_title','natural'),
            ('_tf_','product_title','natural_avg'),
            ('_tf_','product_title','log_norm_avg'),
            ('_tf_','product_title','double_norm_0.5_avg'),
            ('_tf_','prod_descr','natural'),
            ('_tf_','prod_descr','natural_avg'),
            ('_tf_','prod_descr','log_norm_avg'),
            ('_tf_','prod_descr','double_norm_0.5_avg'),
            ('_tf_','attr_names','natural'),
            ('_tf_','attr_names','natural_avg'),
            ('_tf_','attr_names','log_norm_avg'),
            ('_tf_','attr_names','double_norm_0.5_avg'),
            ('_tf_','attr_vals','natural'),
            ('_tf_','attr_vals','natural_avg'),
            ('_tf_','attr_vals','log_norm_avg'),
            ('_tf_','attr_vals','double_norm_0.5_avg'),

            #length features should contain: doc
            ('_length','product_title'),
            ('_length','prod_descr'),
            ('_length','attr_names'),
            ('_length','attr_vals'),
            ('_length','search_term'),

            #idf features should contain: doc and idf_type
            #idf type is the type of idf to use
            ('_idf_','product_title', 'smooth'),
            ('_idf_','product_title', 'smooth_avg'),
            ('_idf_','product_title', 'max_avg'),
            ('_idf_','product_title', 'prob_avg'),
            ('_idf_','prod_descr', 'smooth'),
            ('_idf_','prod_descr', 'smooth_avg'),
            ('_idf_','prod_descr', 'max_avg'),
            ('_idf_','prod_descr', 'prob_avg'),
            ('_idf_','attr_names', 'smooth'),
            ('_idf_','attr_names', 'smooth_avg'),
            ('_idf_','attr_names', 'max_avg'),
            ('_idf_','attr_names', 'prob_avg'),
            ('_idf_','attr_vals', 'smooth'),
            ('_idf_','attr_vals', 'smooth_avg'),
            ('_idf_','attr_vals', 'max_avg'),
            ('_idf_','attr_vals', 'prob_avg'),

            #tf_idf features should contain: doc, tf_type, idf_type, tf_idf_type
            #tf_idf is the type of tf_idf to calculate (sum/average across query words)
            ('_tf_idf_','product_title','natural','smooth','sum'),
            ('_tf_idf_','product_title','natural','max','sum'),
            ('_tf_idf_','product_title','natural','prob','sum'),
            ('_tf_idf_','prod_descr','natural','smooth','sum'),
            ('_tf_idf_','prod_descr','natural','max','sum'),
            ('_tf_idf_','prod_descr','natural','prob','sum'),
            ('_tf_idf_','attr_names','natural','smooth','sum'),
            ('_tf_idf_','attr_names','natural','max','sum'),
            ('_tf_idf_','attr_names','natural','prob','sum'),
            ('_tf_idf_','attr_vals','natural','smooth','sum'),
            ('_tf_idf_','attr_vals','natural','max','sum'),
            ('_tf_idf_','attr_vals','natural','prob','sum'),

            ('_tf_idf_','product_title','log_norm','smooth','sum'),
            ('_tf_idf_','product_title','log_norm','max','sum'),
            ('_tf_idf_','product_title','log_norm','prob','sum'),
            ('_tf_idf_','prod_descr','log_norm','smooth','sum'),
            ('_tf_idf_','prod_descr','log_norm','max','sum'),
            ('_tf_idf_','prod_descr','log_norm','prob','sum'),
            ('_tf_idf_','attr_names','log_norm','smooth','sum'),
            ('_tf_idf_','attr_names','log_norm','max','sum'),
            ('_tf_idf_','attr_names','log_norm','prob','sum'),
            ('_tf_idf_','attr_vals','log_norm','smooth','sum'),
            ('_tf_idf_','attr_vals','log_norm','max','sum'),
            ('_tf_idf_','attr_vals','log_norm','prob','sum'),

            #bm 25 features should contain: doc, k1, b
            ('_bm25_','product_title', 'sum', 1.5, 0.75),
            ('_bm25_','prod_descr', 'sum', 1.5, 0.75),
            ('_bm25_','attr_names', 'sum', 1.5, 0.75),
            ('_bm25_','attr_vals', 'sum', 1.5, 0.75),
            ('_bm25_','product_title', 'avg', 1.5, 0.75),
            ('_bm25_','prod_descr', 'avg', 1.5, 0.75),
            ('_bm25_','attr_names', 'avg', 1.5, 0.75),
            ('_bm25_','attr_vals', 'avg', 1.5, 0.75),
            ]

#add each feature to the data
for feature in features:
    f.add_feature(X_train, train, product_dataframe, feature, 0.5, idf_dict_product_title, idf_dict_prod_descr, idf_dict_attr_names, idf_dict_attr_vals,124428, doc_len_dict, data = 'train')
    f.add_feature(X_test, test, product_dataframe, feature, 0.5, idf_dict_product_title, idf_dict_prod_descr, idf_dict_attr_names, idf_dict_attr_vals,124428, doc_len_dict, data = 'test')

    #save the pre-processed data
    print('Saving data...')
    f.save_obj(X_train,'../input_clean/X_train')
    f.save_obj(X_test,'../input_clean/X_test')
f.save_obj(Y_train,'../input_clean/Y_train')

        
#%%

#split the data by relevance value and view the difference in means for each feature
for col in X_train.columns:
    print('Feature: {} \n Mean for relevance > 2.5: {} \n Mean for relevance < 1.5: {}'.format(col,
          X_train.loc[Y_train['relevance']>2.5][col].describe()['mean'],
          X_train.loc[Y_train['relevance']<1.5][col].describe()['mean']))
          
        
        
