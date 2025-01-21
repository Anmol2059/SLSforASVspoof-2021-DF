import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq

class SSLModel(nn.Module):
    def __init__(self,device):
        super(SSLModel, self).__init__()
        
        cp_path = 'pretrained_models/xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device=device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):

        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        
        if True:
            if input_data.ndim == 3:
                input_tmp = input_data[:, :, 0]
            else:
                input_tmp = input_data
                
            # [batch, length, dim]
            # emb = self.model(input_tmp, mask=False, features_only=True)['x']
            layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return layerresult

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:

        layery = layer[0].transpose(0, 1).transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature

from LayerDiscriminator import LayerDiscriminator
class Model(nn.Module):
    def __init__(self, args,device, num_domains=7):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.domain_discriminators = nn.ModuleList([
            LayerDiscriminator(
                num_channels=1024,  # Adjust based on each layer's channel size
                num_classes=num_domains,
                lambd=1.0  # Gradient reversal strength
            ) for _ in range(24)  # Assuming 24 layers in `layerResult`
        ])

    def perform_dropout(self, feature, domain_labels, layer_index, dropout_flag, percent):
        if dropout_flag:
            domain_output, mask = self.domain_discriminators[layer_index](
                feature.clone(),  # Feature maps
                labels=domain_labels,  # Domain labels
                percent=percent  # Percent of channels to drop
            )
            feature = feature * mask  # Apply mask to feature maps
            return feature, domain_output
        else:
            return feature, None

    def forward(self, x, domain_labels=None, layer_dropout_flags=None, percent=0.33):
        layerResult = self.ssl_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)

        domain_outputs = []
        if domain_labels is not None:
            for i, layer in enumerate(layerResult):  # Iterate through extracted layers
                feature, attention = layer
                feature = feature.permute(1, 2, 0)
                dropout_flag = layer_dropout_flags[i] if layer_dropout_flags is not None else True
                feature, domain_output = self.perform_dropout(feature, domain_labels, i, dropout_flag, percent)
                feature = feature.permute(2, 0, 1)
                if domain_output is not None:
                    domain_outputs.append(domain_output)
                layerResult[i] = (feature, attention)
            
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)
        if domain_labels is not None:
            return output, domain_outputs
        else:
            return output
