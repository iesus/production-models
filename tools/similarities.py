'''
Created on Apr 8, 2016

@author: jesus
'''
from scipy import spatial
import numpy as np
from tools.levSimilarity import levenshtein

def cosineSimilarity(one,another):
    return 1 - spatial.distance.cosine(one,another)

def binaryEquality(one,another):
    product=np.dot(one,another)
    return product == np.sum(one) == np.sum(another)

def levenSimilarity(one,another):
    maxLen=len(one)
    if maxLen<len(another): maxLen=len(another)
    simi = 1.0 - levenshtein(one,another)/float(maxLen)
    return simi 