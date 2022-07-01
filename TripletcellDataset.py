from __future__ import print_function

# import torchvision.datasets as datasets
import os
import numpy as np
from tqdm import tqdm
import torch

class TripletcellDataset():
    def __init__(self, cells, n_triplets, n_class,indices):
        # super(TripletproteinDataset, self).__init__(dir,transform)
        self.cells=cells
        self.n_triplets = n_triplets
        self.classes=n_class
        self.indices=indices
        print('Generating {} triplets'.format(self.n_triplets))
        self.training_triplets = self.generate_triplets(self.n_triplets,self.classes,self.indices)
    def generate_triplets(num_triplets,n_classes,indices):
        # print (cells)
        # print (type(cells))
        # def create_indices(cells):
        #     inds = dict()
        #     for i,(cell,label) in enumerate(cells.items()):
        #         if label not in inds:
        #             inds[label] = []
        #         inds[label].append(cell)
        #     return inds
        triplets=[]
        # print (num_triplets)
        # indices=create_indices(cells)
        
        for x in tqdm(range(int(num_triplets))):
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
    def __getitem__(self, index):
        '''
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        '''

        # Get the index of each image in the triplet
        a, p, n,c1,c2 = self.training_triplets[index]

        c1=torch.tensor(c1).unsqueeze(0).long()
        c2=torch.tensor(c2).unsqueeze(0).long()
        # c1=torch.tensor(c1).long()
        # c2=torch.tensor(c2).long()        
        return a, p, n, c1,c2
        # return a, p, n,c1,c2
    def __len__(self):
        return len(self.training_triplets)
