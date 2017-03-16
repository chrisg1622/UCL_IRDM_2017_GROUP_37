# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 11:38:04 2017

@author: Chris
"""
import pickle

#save an object (e.g. dict) with pickle
def save_obj(obj, name = 'object' ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

#load an object with pickle
def load_obj(name = 'object'):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)