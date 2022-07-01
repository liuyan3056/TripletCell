#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 17:55:05 2021

@author: liuyan
"""
import torch
import torch.nn as nn

#         return x



class cellNet(nn.Module):

    def __init__(self, input_dim,num_classes, init_weights=None):
        super(cellNet, self).__init__()
        # self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(64, num_classes),
        )
        if init_weights:
            self._initialize_weights()
    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    # def forward(self, x):
    #     x = self.features(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.classifier(x)
    #     return x
    def forward(self, x):
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = self.classifier[0](x)
        x = self.classifier[1](x)
        x = self.classifier[2](x)
        x = self.classifier[3](x)
        x = self.classifier[4](x)


        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha=10
        features = features*alpha

        #x = self.model.classifier(self.features)
        return features
        # return x
    def forward_classifier(self, x):
        features = self.forward(x)

        res = self.classifier[6](features)
        return res