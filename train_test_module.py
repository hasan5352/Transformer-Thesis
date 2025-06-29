
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from typing import List, Tuple
from torchvision.transforms import v2
import torchvision.transforms as T
import random

def compute_ece(preds, labels, device, n_bins=15):
    preds = torch.softmax(preds, dim=1)
    confidences, preds = preds.max(dim=1)
    acc = preds.eq(labels)
    
    bins = torch.linspace(0, 1, n_bins+1, device=device)
    ece = torch.zeros(1, device=device)
    for i in range(n_bins):
        mask = confidences.gt(bins[i]) & confidences.le(bins[i+1])
        if mask.sum() > 0:
            ece += (mask.sum().float() / labels.size(0)) * torch.abs(acc[mask].float().mean() - confidences[mask].mean())
    return ece.item()

class LossCalculatorDeiT(nn.Module):
    def __init__(self, teacher_model:nn.Module, Cdeit_small_model:nn.Module=None, head_strategy=1):
        """ 
        Uses Hard Distillation for L_total.

        Args of forward: 
            tokens = (cls, distill, corrupt: Optional)
            batches = (teacher_x_batch, yzbatch, c_batch: Optional)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.Cdeit_small_model = Cdeit_small_model
        self.head_strategy = head_strategy
        
    def forward(self, tokens:Tuple[torch.Tensor], batches:Tuple[torch.Tensor]):
        assert (len(batches) != 3 and len(tokens) != 3
                ) or (len(batches) == 3 and len(tokens) == 3)
        if self.head_strategy >= 2 and self.Cdeit_small_model is None:
            raise ValueError("cdeit model not given with this head_strategy")

        with torch.no_grad():
            teacher_input = F.interpolate(batches[0], size=(224, 224), mode='bilinear', align_corners=False)
            teacher_input = torch.argmax(self.teacher_model(teacher_input), dim=1)
        assert tokens[1].size(1) > teacher_input.max(), "Teacher head output shape mismatch"

        L_distill = F.cross_entropy(tokens[1], teacher_input)
        L_cls = F.cross_entropy(tokens[0], batches[1])
        
        if len(tokens) == 3:
            L_corrupt = F.cross_entropy(tokens[2], batches[2])
            if self.head_strategy == 2 or self.head_strategy == 3:
                L_cls_corruptFFN = F.cross_entropy(tokens[0] + self.Cdeit_small_model.output_head.ffn(tokens[2]), batches[1])
                L_total = (L_cls_corruptFFN + L_distill + L_corrupt)/3
                return L_total, L_cls, L_distill, L_corrupt, L_cls_corruptFFN
            L_total = (L_cls + L_distill + L_corrupt)/3
            return L_total, L_cls, L_distill, L_corrupt
        
        L_total = (L_cls + L_distill)/2
        return L_total, L_cls, L_distill
    

class MyAugments(nn.Module):
    def __init__(self, num_classes, mixup_p=0, cutmix_p=0, augmix_p=0, randaug_p=0):
        super().__init__()
        self.mixup = v2.MixUp(num_classes=num_classes)
        self.cutmix = v2.CutMix(num_classes=num_classes)
        self.randaug = T.RandAugment()
        self.augmix = T.AugMix()

        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.augmix_p = augmix_p
        self.randaug_p = randaug_p
    def forward(self, x_batch, y_batch):
        with torch.no_grad():
            if random.random() < self.mixup_p:
                x_batch, y_batch = self.mixup(x_batch, y_batch)
            if random.random() < self.cutmix_p:
                x_batch, y_batch = self.cutmix(x_batch, y_batch)
            x_batch = (x_batch*255).to(torch.uint8)
            if random.random() < self.randaug_p:
                x_batch = torch.stack([self.randaug(img) for img in x_batch])
            if random.random() < self.augmix_p:
                x_batch = torch.stack([self.augmix(img) for img in x_batch])

            return x_batch.float().div(255), y_batch



# Train and test for teacher
def test_teacher_head(teacher_model:nn.Module, test_batches:list, device):
    teacher_model.to(device)
    teacher_model.eval()
    total_samples, test_acc = 0, 0
    with torch.no_grad():
        for x_batch, y_batch, _ in test_batches:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)
            preds = teacher_model(x_batch)
            test_acc += (torch.argmax(preds, dim=1) == y_batch).sum().item()
            total_samples += y_batch.size(0)
    return test_acc/total_samples

def train_test_teacher_head(teacher_model:nn.Module, train_batches:list, test_batches:list, 
                            optimizer, device, save_path, num_epochs = 10, print_metrics=False):
    """ Trains and tests teacher model after putting new head.
        Freezes entire model except new head.
    """
    for param in teacher_model.parameters():        # Freeze all parameters
        param.requires_grad = False
    for param in teacher_model.head.parameters():   # Unfreeze only the head parameters
        param.requires_grad = True
    
    teacher_model.to(device)
    teacher_model.train()
    avg_acc, avg_loss = 0, 0
    # main loop
    for epoch in range(num_epochs):
        print(f"------- Epoch {epoch+1} -------")
        total_samples, train_loss = 0, 0
        cnt = 1
        for x_batch, y_batch, _ in train_batches:
            print(cnt, end=", ")
            cnt+=1
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)
            preds = teacher_model(x_batch)

            loss = F.cross_entropy(preds, y_batch)
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_samples += y_batch.size(0)
            train_loss += loss.item() * y_batch.size(0)

        train_loss = train_loss/total_samples
        test_acc = test_teacher_head(teacher_model, test_batches, device)
        if print_metrics : print(f"Train Loss: {train_loss}, Test Acc: {test_acc}")
        avg_acc += test_acc
        avg_loss += train_loss
    
    torch.save(teacher_model, save_path)
    return avg_acc/num_epochs, avg_loss/num_epochs