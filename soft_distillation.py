import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from typing import List, Tuple
from torchvision.transforms import v2
import torchvision.transforms as T
import random

class SoftLossCalculatorDeiT(nn.Module):
    def __init__(self, teacher_model:nn.Module, tau, alpha, label_smoothing=0.1):
        """ 
        Uses Hard Distillation for L_total.

        Args of forward: 
            tokens = (cls, distill)
            batches = (teacher_x_batch, y_batch)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.tau = tau
        self.label_smoothing = label_smoothing
        
    def forward(self, tokens:Tuple[torch.Tensor], batches:Tuple[torch.Tensor]):
        assert len(batches) == 2 and len(tokens) == 2

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(batches[0])

        L_distill = F.kl_div(F.log_softmax(tokens[1] / self.tau, dim=1), 
                             F.log_softmax(teacher_logits / self.tau, dim=1),
                             reduction='sum',
                             log_target=True,
                             )  * (self.tau * self.tau) / tokens[1].numel()
        L_cls = F.cross_entropy(tokens[0], batches[1], label_smoothing=self.label_smoothing)
        
        L_total = L_cls * (1 - self.alpha) + L_distill * self.alpha
        return L_total, L_cls, L_distill
    

class LossCalculatorCdeiT(nn.Module):
    def __init__(self, teacher_model:nn.Module, head_strategy, tau, alpha, label_smoothing=0.1):
        """ 
        Uses Hard Distillation for L_total.

        Args of forward: 
            tokens = (cls, distill, corrupt)
            batches = (teacher_x_batch, y_batch, c_batch)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.head_strategy = head_strategy
        self.alpha = alpha
        self.tau = tau
        self.label_smoothing = label_smoothing
        
    def forward(self, tokens:Tuple[torch.Tensor], batches:Tuple[torch.Tensor], model:nn.Module):
        assert len(batches) == 3 and len(tokens) == 3

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_logits = self.teacher_model(batches[0])

        L_distill = F.kl_div(F.log_softmax(tokens[1] / self.tau, dim=1), 
                             F.log_softmax(teacher_logits / self.tau, dim=1),
                             reduction='sum',
                             log_target=True,
                             )  * (self.tau * self.tau) / tokens[1].numel()
        L_cls = F.cross_entropy(tokens[0], batches[1], label_smoothing=self.label_smoothing)
        L_corrupt = F.cross_entropy(tokens[2], batches[2], label_smoothing=self.label_smoothing)

        if self.head_strategy == 1:
            L_total = (1 - self.alpha) * (L_cls + L_corrupt)/2 + (self.alpha * L_distill)
            return L_total, L_cls, L_distill, L_corrupt
        
        L_cls_corruptFFN = F.cross_entropy(tokens[0] + model.output_head.ffn(tokens[2]), batches[1], label_smoothing=self.label_smoothing)
        L_total = (1 - self.alpha) * (L_cls_corruptFFN + L_corrupt)/2 + self.alpha * L_distill
        return L_total, L_cls, L_distill, L_corrupt, L_cls_corruptFFN


