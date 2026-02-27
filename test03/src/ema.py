import torch
import torch.nn as nn
import copy

class EMA:
    def __init__(self, model : nn.Module, beta=.9999):
        self.beta = beta
        self.step = 0

        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False
    
    def update(self, model : nn.Module):
        self.step += 1
        for current_param, ema_param in zip(model.parameters(), self.ema_model.parameters()):
            ema_param.data.mul_(self.beta).add_(current_param.data, alpha=(1-self.beta))
    
    def get_model(self):
        return self.ema_model