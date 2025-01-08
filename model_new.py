import random
import sys
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import fairseq
from .LayerDiscriminator import LayerDiscriminator  # Import LayerDiscriminator

class SSLModel(nn.Module):
    def __init__(self, device):
        super(SSLModel, self).__init__()
        
        cp_path = 'xlsr2_300m.pt'
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])
        self.model = model[0]
        self.device = device
        self.out_dim = 1024
        return

    def extract_feat(self, input_data):
        if next(self.model.parameters()).device != input_data.device \
           or next(self.model.parameters()).dtype != input_data.dtype:
            self.model.to(input_data.device, dtype=input_data.dtype)
            self.model.train()

        if input_data.ndim == 3:
            input_tmp = input_data[:, :, 0]
        else:
            input_tmp = input_data

        # [batch, length, dim]
        emb = self.model(input_tmp, mask=False, features_only=True)['x']
        layerresult = self.model(input_tmp, mask=False, features_only=True)['layer_results']
        return emb, layerresult


def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:
        layery = layer[0].transpose(0, 1).transpose(1, 2)  # (x,z): (201,b,1024) -> (b,201,1024) -> (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1)  # (b,1024,1)
        layery = layery.transpose(1, 2)  # (b,1,1024)
        poollayerResult.append(layery)

        x = layer[0].transpose(0, 1)  # (201,b,1024) -> (b,201,1024)
        x = x.view(x.size(0), -1, x.size(1), x.size(2))  # Reshape for full feature concatenation
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)  # Combine pooled layer outputs
    fullfeature = torch.cat(fullf, dim=1)  # Combine full features
    return layery, fullfeature


class Model(nn.Module):
    def __init__(self, args, device, num_domains=3):
        super().__init__()
        self.device = device
        self.ssl_model = SSLModel(self.device)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(22847, 1024)
        self.fc3 = nn.Linear(1024, 2)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        
        # Add LayerDiscriminator
        self.layer_discriminator = LayerDiscriminator(
            num_channels=1024,  # Number of feature channels
            num_classes=num_domains,  # Number of domains
            lambd=1.0  # Gradient reversal strength
        )

    def forward(self, x, domain_labels=None, percent=0.33):
        # Step 1: Extract features from the SSL model
        x_ssl_feat, layerResult = self.ssl_model.extract_feat(x.squeeze(-1))  # Extracted layer results
        y0, fullfeature = getAttenF(layerResult)  # Process intermediate layers

        # Step 2: Apply domain-sensitive dropout using LayerDiscriminator during training
        if self.training and domain_labels is not None:
            _, mask = self.layer_discriminator(
                fullfeature,  # Feature maps
                labels=domain_labels,  # Domain labels for the batch
                percent=percent  # Percent of channels to drop
            )
            fullfeature = fullfeature * mask  # Apply mask to suppress domain-sensitive channels

        # Step 3: Original forward pass
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
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

        return output
