#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import pandas as pd

"""
Helper

Hilfsfunktionen (Cosine-Similarity, Dot-Product, Confusion-Matrix)
"""
class Helper(object):
    #def __init__(self):
    
    """
    Cosine Similarity berechnen
    """
    @staticmethod
    def cosine_similarity(a, b):
        return float(np.dot(a, b)) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)))
    
    """
    Dot Product berechnen
    """
    @staticmethod
    def dot_product(a, b):
        return np.dot(a, b)

    """
    Confusion-Matrix

    Confusion-Matrix für tatsächliche Werte und Vorhersage erstellen
    """
    @staticmethod
    def confusion_matrix(cm, s1, s2):
        # True positive (Prediction in Actual)
        cm[0][0]+= s2.loc[s2.values.isin(s1.values)].shape[0]
    
        # False positive (Prediction nicht in Actual)
        cm[1][0]+= s2.loc[~s2.values.isin(s1.values)].shape[0]
    
        # False negative (Actual nicht in Prediction)
        cm[0][1]+= s1.loc[~s1.values.isin(s2.values)].shape[0]
    
        # 
        #cm[1][1] = 

        return cm

    """
    Confusion-Matrix true positive rate
    """
    @staticmethod
    def confusion_matrix_tpr(cm):
        return cm[0][0] / (cm[0][0] + cm[0][1])

    """
    Confusion-Matrix positive predicted value
    """
    @staticmethod
    def confusion_matrix_ppv(cm):
        return cm[0][0] / (cm[0][0] + cm[1][0])

    """
    Confusion-Matrix f1 score
    """
    @staticmethod
    def confusion_matrix_f1_score(cm):
        ppv = cm[0][0] / (cm[0][0] + cm[0][1])#confusion_matrix_ppv(cm)
        tpr = cm[0][0] / (cm[0][0] + cm[1][0])#confusion_matrix_tpr(cm)
        return 2 * ppv * tpr / (ppv + tpr)