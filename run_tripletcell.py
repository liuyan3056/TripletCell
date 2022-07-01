#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 19:55:31 2021
@author: liuyan
"""
import rpy2.robjects as robjects
import os 
import numpy as np
import pandas as pd
import time as tm
# import data.utils
# 
# import sys
# f=open("trainlog.txt","a")
# ftmp=sys.stdout
# sys.stdout=f
from sklearn.calibration import CalibratedClassifierCV
import tripletcell
import argparse
def process_label_to_list(train_y):
    label=[]
    for i in range(len(train_y)):
        label_i=train_y[i][0]
        label.append(label_i)
    return label
def run_tripletcell(DataPath, LabelsPath, CV_RDataPath, platform, GeneOrderPath = "", NumGenes = 0):
    '''
    Parameters
    ----------
    DataPath : Data file path (.csv), cells-genes matrix with cell unique barcodes 
    as row names and gene names as column names.
    LabelsPath : Cell population annotations file path (.csv).
    CV_RDataPath : Cross validation RData file path (.RData), obtained from Cross_Validation.R function.
    OutputDir : Output directory defining the path of the exported file.
    GeneOrderPath : Gene order file path (.csv) obtained from feature selection, 
    defining the genes order for each cross validation fold, default is NULL.
    NumGenes : Number of genes used in case of feature selection (integer), default is 0.
    Threshold : Threshold used when rejecting the cells, default is 0.7.
    '''
    # read the Rdata file
    robjects.r['load'](CV_RDataPath)
    nfolds = np.array(robjects.r['n_folds'], dtype = 'int')
    tokeep = np.array(robjects.r['Cells_to_Keep'], dtype = 'bool')
    col = np.array(robjects.r['col_Index'], dtype = 'int')
    col = col - 1 
    test_ind = np.array(robjects.r['Test_Idx'])
    # print (test_ind)
    train_ind = np.array(robjects.r['Train_Idx'])
    # read the data
    data = pd.read_csv(DataPath,index_col=0,sep=',')
    labels = pd.read_csv(LabelsPath, header=0,index_col=None, sep=',', usecols = col)
    
    labels = labels.iloc[tokeep]
    data = data.iloc[tokeep]
    
    # read the feature file
    if (NumGenes > 0):
        features = pd.read_csv(GeneOrderPath,header=0,index_col=None, sep=',')
    # normalize data
    data = np.log1p(data)
    tr_time=[]
    ts_time=[]
    truelab = []
    pred = []
    for i in range(np.squeeze(nfolds)):
        print ("the",i,"-th nfolds")
        test_ind_i = np.array(test_ind[i], dtype = 'int') - 1
        train_ind_i = np.array(train_ind[i], dtype = 'int') - 1
        train=data.iloc[train_ind_i]
        test=data.iloc[test_ind_i]
        y_train=labels.iloc[train_ind_i]
        y_test=labels.iloc[test_ind_i]
        if (NumGenes > 0):
            feat_to_use = features.iloc[0:NumGenes,i]
            train = train.iloc[:,feat_to_use]
            test = test.iloc[:,feat_to_use]
        start=tm.time()
        y_train=process_label_to_list(y_train.values)
        train=train.values
        label_number=len(list(set(y_train)))
        model=tripletcell.cellNet(train.shape[1],label_number)
        y_test=process_label_to_list(y_test.values)
        test=test.values
        # tripletcell.train_celltriplet(model,train, y_train,test,y_test,50,platform)
        tripletcell.train_celltriplet(model,test[1:5889],y_test[1:5889],train, y_train,50,platform)
        tr_time.append(tm.time()-start)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--reference_data_path', type=str, default="--")
    parser.add_argument('--reference_label_path', type=str, default="--")
    parser.add_argument('--query_data_path', type=str, default="--")
    parser.add_argument('--query_label_path', type=str, default="--")
    parser.add_argument('--cell_type_numbers', type=int, default="10")
    parser.add_argument('--save_file', type=str, default="pbmc_SM2")
    parser.add_argument('--epoch', type=int, default="30")
    args = parser.parse_args()
    
    reference_data=pd.read_csv(args.reference_data_path).values
    reference_label=pd.read_csv(args.reference_label_path)
    query_data=pd.read_csv(args.query_data_path).values
    query_label=pd.read_csv(args.query_label_path)
    model=tripletcell.cellNet(reference_data[1],args.cell_type_numbers)
    
    
    tripletcell.train_celltriplet(model,reference_data,reference_label,query_data,query_label,args.epoch,args.save_file)

    
    

