import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Classifier_Spectal_MTL(nn.Module):
    def __init__(self,args):
        super().__init__()
        from model.networks.resnet_tiny_imagenet import resnet18_mtl
        self.share_layer_order = args.share_layer_order
        self.encoder = resnet18_mtl(True, True, self.share_layer_order)
        self.z_dim = 512
        self.cls1 = nn.Linear(self.z_dim, 4)
        self.cls2 = nn.Linear(self.z_dim, 3)
    
    def forward(self, data):
        if self.share_layer_order == 4:
            emb1 = self.encoder(data)
            emb2 = emb1
        else:
            emb1, emb2 = self.encoder(data)
        logit1 = self.cls1(emb1)
        logit2 = self.cls2(emb2)
        return logit1, logit2
    
    def get_attention(self):
        attn1, attn2 = self.encoder.get_attention_weights()
        return attn1, attn2