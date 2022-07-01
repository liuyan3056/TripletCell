#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:27:13 2021

@author: liuyan
"""
import numpy as np
from tqdm import tqdm
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn import preprocessing
from TripletcellDataset import TripletcellDataset
from model import cellNet
import torch.optim as optim
from utils import PairwiseDistance,display_triplet_distance,display_triplet_distance_test, PairwiseCosineDistance
from torch.autograd import Function
import torch.nn as nn
import argparse
import scipy.spatial.distance as distance
from eval_metrics import evaluate
from tqdm import tqdm
from torch.utils.data import Dataset
import sys
f=open("brainlog.txt","a")
ftmp=sys.stdout
sys.stdout=f
print ("ok")
split_ist=[100,200]
l2_dist = PairwiseCosineDistance(2)
margin=0.6
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
lr=0.001
lr_decay=1e-4

def adjust_learning_rate(optimizer):
    """Updates the learning rate given the learning rate decay.
    The routine has been implemented according to the original Lua SGD optimizer
    """
    for group in optimizer.param_groups:
        if 'step' not in group:
            group['step'] = 0
        group['step'] += 1

        group['lr'] = lr / (1 + group['step'] *lr_decay)
def split_dataset(data_path,label_path,split_ist):
    cells_data=pd.read_csv(data_path)
    cell_labels=pd.read_csv(label_path)
    newcells=np.array(cells_data[cells_data.columns[1:]])
    all_label=list(np.array(cell_labels["x"]))

    le = preprocessing.LabelEncoder()
    le.fit(all_label)
    label_number=len(le.classes_)
    cc=le.transform([all_label[1]])
    cell_label={}
    labels=[]
    for i in range(len(all_label)):
        temp_label=le.transform([all_label[i]])[0]
        cell_label.update({i:temp_label})
        labels.append(temp_label)
    test_data=newcells[split_ist[0]:split_ist[1],]
    test_label=labels[split_ist[0]:split_ist[1]]
    train_data1=newcells[0:split_ist[0],]
    train_data2=newcells[split_ist[1]:,]
    train_data=np.vstack((train_data1,train_data2))
    train_label1=labels[0:split_ist[0]]
    train_label2=labels[split_ist[1]:]
    train_label=train_label1+train_label2
    
    label_dict={}
    for k in range(len(train_label)):
        temp_label=train_label[k]
        label_dict.update({k:temp_label})
    return label_number,label_dict,train_data,train_label,test_data,test_label
import os
def preprocessingCSV_top2000(expressionFilename, delim='comma', transform='log', cellRatio=0.99, geneRatio=0.99, geneCriteria='variance', geneSelectnum=3000, transpose=True, tabuCol=''):
    '''
    preprocessing CSV files:
    transform='log' or None
    '''
    # expressionFilename = dir + datasetName
    if not os.path.exists(expressionFilename):
        print('Dataset ' + expressionFilename + ' not exists!')

    print('Input scRNA data in CSV format is validated, start reading...')

    tabuColList = []
    tmplist = tabuCol.split(",")
    for item in tmplist:
        tabuColList.append(item)

    df = pd.DataFrame()
    if delim == 'space':
        if len(tabuColList) == 0:
            df = pd.read_csv(expressionFilename, index_col=0,
                             delim_whitespace=True)
        else:
            df = pd.read_csv(expressionFilename, index_col=0, delim_whitespace=True,
                             usecols=lambda column: column not in tabuColList)
    elif delim == 'comma':
        if len(tabuColList) == 0:
            df = pd.read_csv(expressionFilename, index_col=0)
        else:
            df = pd.read_csv(expressionFilename, index_col=0,
                             usecols=lambda column: column not in tabuColList)
    print('Data loaded, start filtering...')
    if transpose == True:
        df = df.T
    df1 = df[df.astype('bool').mean(axis=1) >= (1-geneRatio)]
    print('After preprocessing, {} genes remaining'.format(df1.shape[0]))
    criteriaGene = df1.astype('bool').mean(axis=0) >= (1-cellRatio)
    df2 = df1[df1.columns[criteriaGene]]
    print('After preprocessing, {} cells have {} nonzero'.format(
        df2.shape[1], geneRatio))
    criteriaSelectGene = df2.var(axis=1).sort_values()[-geneSelectnum:]
    df3 = df2.loc[criteriaSelectGene.index]
    if transform == 'log':
        df3 = df3.transform(lambda x: np.log(x + 1))
    # df3.to_csv(csvFilename)
    return df3.T

def split_dataset(data_path,label_path,split_idex):
    cells_data=pd.read_csv(data_path)
    cell_labels=pd.read_csv(label_path)
    newcells=np.array(cells_data[cells_data.columns[1:]])
    all_label=list(np.array(cell_labels["x"]))

    le = preprocessing.LabelEncoder()
    le.fit(all_label)
    label_number=len(le.classes_)
    cc=le.transform([all_label[1]])
    cell_label={}
    labels=[]
    for i in range(len(all_label)):
        temp_label=le.transform([all_label[i]])[0]
        cell_label.update({i:temp_label})
        labels.append(temp_label)

    # from sklearn.model_selection import train_test_split
    # X_train,X_test,y_train,y_test = train_test_split(newcells,labels,test_size=0.2)
    
    X_train=newcells[0:split_idex,:]
    X_test=newcells[split_idex:,:]
    
    y_train=labels[0:split_idex]
    y_test=labels[split_idex:]
    
    label_dict={}
    for k in range(len(y_train)):
        temp_label=y_train[k]
        label_dict.update({k:temp_label})
    return label_number,label_dict,X_train,y_train,X_test,y_test
def split_top2000dataset(data_path,label_path):
    cells_data=preprocessingCSV_top2000(data_path)
    cell_labels=pd.read_csv(label_path)
    newcells=np.array(cells_data[cells_data.columns[1:]])
    all_label=list(np.array(cell_labels["x"]))

    le = preprocessing.LabelEncoder()
    le.fit(all_label)
    label_number=len(le.classes_)
    cc=le.transform([all_label[1]])
    cell_label={}
    labels=[]
    for i in range(len(all_label)):
        temp_label=le.transform([all_label[i]])[0]
        cell_label.update({i:temp_label})
        labels.append(temp_label)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(newcells,labels,test_size=0.2)
    
    label_dict={}
    for k in range(len(y_train)):
        temp_label=y_train[k]
        label_dict.update({k:temp_label})
    return label_number,label_dict,X_train,y_train,X_test,y_test    

def generate_triplets(cells,num_triplets,n_classes):
    def create_indices(_cells):
        inds = dict()
        for i,(cell,label) in enumerate(_cells.items()):
            if label not in inds:
                inds[label] = []
            inds[label].append(cell)
        return inds
    triplets=[]
    indices=create_indices(cells)
        
    for x in tqdm(range(num_triplets)):
        c1 = np.random.randint(0, n_classes-1)
            # print (c1)
        c2 = np.random.randint(0, n_classes-1)
            # print (c2)
        while len(indices[c1]) < 2:
            c1 = np.random.randint(0, n_classes-1)

        while c1 == c2:
            c2 = np.random.randint(0, n_classes-1)
        if len(indices[c1]) == 2:  # hack to speed up process
            n1, n2 = 0, 1
        else:
            n1 = np.random.randint(0, len(indices[c1]) - 1)
            n2 = np.random.randint(0, len(indices[c1]) - 1)
        while n1 == n2:
            n2 = np.random.randint(0, len(indices[c1]) - 1)
        if len(indices[c2]) ==1:
            n3 = 0
        else:
            n3 = np.random.randint(0, len(indices[c2]) - 1)

        triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3],c1,c2])
    return triplets
class MyDataset(Dataset):
    def __init__(self, cell_label, train_data,num_triplets,n_classes):
        self.all_cells=train_data
        self.triples=generate_triplets(cell_label, num_triplets, n_classes)
 
    def __getitem__(self, index):#检索函数
        a, p, n,c1,c2 = self.triples[index]

        c1=torch.tensor(c1).unsqueeze(0)
        c2=torch.tensor(c2).unsqueeze(0)
        data_aa=self.all_cells[a]
        data_pp=self.all_cells[p]
        data_nn=self.all_cells[n]
        return data_aa,data_pp,data_nn,c1,c2
 
    def __len__(self):
        return len(self.triples)

class TripletMarginLoss(Function):
    """Triplet loss function.
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseCosineDistance(2)  # norm 2

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive)
        d_n = self.pdist.forward(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss
# class Tripletcosineloss(Function):
#     """Triplet loss function.
#     """
#     def __init__(self, margin):
#         super(TripletMarginLoss, self).__init__()
#         self.margin = margin
#         self.pdist = PairwiseDistance(2)  # norm 2

#     def forward(self, anchor, positive, negative):
#         d_p = self.pdist.forward(anchor, positive)
#         d_n = self.pdist.forward(anchor, negative)

#         dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
#         loss = torch.mean(dist_hinge)
#         return loss
def add_noise(inputs):
    # print (type(inputs))
    return inputs.double() + ((torch.randn(inputs.shape) * args.noise_value)).double()
def train(train_loader, model, optimizer, epoch):
    # switch to train mode
    model.train()

    pbar = tqdm(enumerate(train_loader))
    labels, distances = [], []


    for batch_idx, (data_a, data_p, data_n,label_p,label_n) in pbar:
        
        # print (data_a)
        # print (type(data_a))
        data_a=add_noise(data_a)
        data_p=add_noise(data_p)
        data_n=add_noise(data_n)
            
        data_a, data_p, data_n = data_a.to(device), data_p.to(device), data_n.to(device)
        # data_a, data_p, data_n = Variable(data_a), Variable(data_p), Variable(data_n)

        # compute output
        out_a, out_p, out_n = model(data_a.float()), model(data_p.float()), model(data_n.float())

        # Choose the hard negatives
        d_p = l2_dist.forward(out_a, out_p)
        d_n = l2_dist.forward(out_a, out_n)
        all = (d_n - d_p <margin).cpu().data.numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue
        out_selected_a = torch.from_numpy(out_a.cpu().data.numpy()[hard_triplets]).to(device)
        out_selected_p = torch.from_numpy(out_p.cpu().data.numpy()[hard_triplets]).to(device)
        out_selected_n = torch.from_numpy(out_n.cpu().data.numpy()[hard_triplets]).to(device)
        
        selected_data_a = torch.from_numpy(data_a.cpu().data.numpy()[hard_triplets]).to(device)
        selected_data_p = torch.from_numpy(data_p.cpu().data.numpy()[hard_triplets]).to(device)
        selected_data_n = torch.from_numpy(data_n.cpu().data.numpy()[hard_triplets]).to(device)

        selected_label_p = torch.from_numpy(label_p.cpu().numpy()[hard_triplets])
        selected_label_n= torch.from_numpy(label_n.cpu().numpy()[hard_triplets])
        triplet_loss = TripletMarginLoss(margin).forward(out_selected_a, out_selected_p, out_selected_n)
        
        # print (selected_data_a)

        cls_a = model.forward_classifier(selected_data_a.float())
        cls_p = model.forward_classifier(selected_data_p.float())
        cls_n = model.forward_classifier(selected_data_n.float())

        criterion = nn.CrossEntropyLoss()
        predicted_labels = torch.cat([cls_a,cls_p,cls_n])
        true_labels = torch.cat([torch.tensor(selected_label_p).to(device),torch.tensor(selected_label_p).to(device),torch.tensor(selected_label_n).to(device)])
        true_labels=true_labels.squeeze()
        cross_entropy_loss = criterion(predicted_labels.to(device),true_labels.to(device))
        loss = cross_entropy_loss + triplet_loss
        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        adjust_learning_rate(optimizer)
        dists = l2_dist.forward(out_selected_a,out_selected_n) #torch.sqrt(torch.sum((out_a - out_n) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        # print (len(distances))
        labels.append(np.zeros(dists.size(0)))
        dists = l2_dist.forward(out_selected_a,out_selected_p)#torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))  # euclidean distance
        distances.append(dists.data.cpu().numpy())
        labels.append(np.ones(dists.size(0)))

    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for dist in distances for subdist in dist])

    tpr, fpr, accuracy, val, val_std, far = evaluate(distances,labels)
    # print('\33[91mTrain set: Accuracy: {:.8f}\n\33[0m'.format(np.mean(accuracy)))
    # print ('Train Accuracy', np.mean(accuracy))
def calculate_cose_similarity(feature1,feature2):
    return 1-distance.cosine(feature1,feature2)


def test_acc(model,template_cell,tempalte_label,test_cell,test_label):
    template_x=torch.Tensor(template_cell)
    test_cell=torch.Tensor(test_cell)
    model=model.eval()
    template=model(template_x.to(device)).detach().cpu().numpy()
    query=model(test_cell.to(device)).detach().cpu().numpy()
    predicted_label=[]
    for i in range(len(query)):
        # print (i)
        singe_query=query[i]
        distances=[]
        for j in range(len(template)):
            singe_tempate=template[j]
            # singe_distance= np.sqrt(np.sum(np.square(singe_query-singe_tempate)))
            singe_distance=calculate_cose_similarity(singe_query,singe_tempate)
            distances.append(singe_distance)
        singe_index=distances.index(max(distances))
        singe_predict_label=tempalte_label[singe_index]
        predicted_label.append(singe_predict_label)
    # counts=0
    # for k in range(len(predicted_label)):
    #     if predicted_label[k]==test_label[k]:
    #         counts=counts+1
    # print (counts)
    # print(counts/300)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score
    
    print ("acc",accuracy_score(test_label, predicted_label))
    print ("F1-SCORE",f1_score(test_label, predicted_label, average='weighted'))
    # return counts/300
        
def main(model,train_data,train_label,test_data,test_label):
    for epoch in range(5):
        train(train_loader, model, optimizer, epoch)
        if epoch>3:
            test_acc(model,train_data,train_label,test_data,test_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default="Filtered_68K_PBMC_data")
    parser.add_argument('--datapath', type=str, default="./data/")
    parser.add_argument('--Top', type=str, default=None)
    parser.add_argument('--noise_value', type=float, default=1)
    parser.add_argument('--split_ist', type=list, default=[0,300])
    args = parser.parse_args()

    data_path="./paper_data/Intra-dataset/Zheng 68K/"+args.name+".csv"
    label_path="./paper_data/Intra-dataset/Zheng 68K/"+args.name+"_labels34.csv"
    path=pd.read_excel("dataset_path/brain_intra_dataset.xlsx")
    for i in range(5):
        data_path=path["data_path"][i]
        print (data_path)
        label_path=path["label_path"][i]
        split_idex=path["split1"][i]
        label_number,train_label_dict,train_data,train_label,test_data,test_label=split_dataset(data_path, label_path,split_idex)

        model=cellNet(train_data.shape[1],label_number).to(device)
        print (model)
        train_dir=MyDataset(train_label_dict,train_data,500000,label_number)
        train_loader = torch.utils.data.DataLoader(train_dir,batch_size=1024, shuffle=True)
        optimizer = optim.SGD(model.parameters(), lr=0.0001,momentum=0.9, dampening=0.9,weight_decay=0.0001)
        main(model,train_data,train_label,test_data[0:1000,:],test_label[0:1000])
