
import torch
import torch.nn.functional as F
import torch.nn as nn
import json
from typing import List, Tuple
from torchvision.transforms import v2
import torchvision.transforms as T
import random

class LossCalculatorCdeiT(nn.Module):
    def __init__(self, teacher_model:nn.Module, head_strategy, label_smoothing=0.1):
        """ 
        Uses Hard Distillation for L_total.

        Args of forward: 
            tokens = (cls, distill, corrupt)
            batches = (teacher_x_batch, y_batch, c_batch)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.head_strategy = head_strategy
        self.label_smoothing = label_smoothing
        
    def forward(self, tokens:Tuple[torch.Tensor], batches:Tuple[torch.Tensor], model:nn.Module):
        assert len(batches) == 3 and len(tokens) == 3

        with torch.no_grad():
            teacher_y = torch.argmax(self.teacher_model(batches[0]), dim=1)
        assert tokens[1].size(1) > teacher_y.max(), "Teacher head output shape mismatch"

        L_distill = F.cross_entropy(tokens[1], teacher_y)
        L_cls = F.cross_entropy(tokens[0], batches[1], label_smoothing=self.label_smoothing)
        L_corrupt = F.cross_entropy(tokens[2], batches[2], label_smoothing=self.label_smoothing)

        if self.head_strategy == 1:
            L_total = (L_cls + L_distill + L_corrupt)/3
            return L_total, L_cls, L_distill, L_corrupt
        
        L_cls_corruptFFN = F.cross_entropy(tokens[0] + model.output_head.ffn(tokens[2]), batches[1], label_smoothing=self.label_smoothing)
        L_total = (L_cls_corruptFFN + L_distill + L_corrupt)/3
        return L_total, L_cls, L_distill, L_corrupt, L_cls_corruptFFN


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
    def __init__(self, teacher_model:nn.Module, label_smoothing=0.1):
        """ 
        Uses Hard Distillation for L_total.

        Args of forward: 
            tokens = (cls, distill)
            batches = (teacher_x_batch, y_batch)
        """
        super().__init__()
        self.teacher_model = teacher_model
        self.label_smoothing = label_smoothing
        
    def forward(self, tokens:Tuple[torch.Tensor], batches:Tuple[torch.Tensor]):
        assert len(batches) == 2 and len(tokens) == 2

        self.teacher_model.eval()
        with torch.no_grad():
            teacher_y = torch.argmax(self.teacher_model(batches[0]), dim=1)
        assert tokens[1].size(1) > teacher_y.max(), "Teacher head output shape mismatch"

        L_distill = F.cross_entropy(tokens[1], teacher_y)
        L_cls = F.cross_entropy(tokens[0], batches[1], label_smoothing=self.label_smoothing)
        
        L_total = L_cls * 0.5 + L_distill * 0.5
        return L_total, L_cls, L_distill

        
class MyAugments(nn.Module):
    def __init__(self, num_classes, mixup_p=0, cutmix_p=0, randaug_p=0, erasing_p=0):
        super().__init__()
        self.mixup = v2.MixUp(num_classes=num_classes)
        self.cutmix = v2.CutMix(num_classes=num_classes)
        self.randaug = T.RandAugment()
        self.erase = T.RandomErasing(p=erasing_p)

        self.mixup_p = mixup_p
        self.cutmix_p = cutmix_p
        self.randaug_p = randaug_p
    def forward(self, x_batch, y_batch):
        with torch.no_grad():
            if random.random() < self.mixup_p : x_batch, y_batch = self.mixup(x_batch, y_batch)
            if random.random() < self.cutmix_p : x_batch, y_batch = self.cutmix(x_batch, y_batch)

            x_batch = (x_batch*255).to(torch.uint8)
            randaug_mask = (torch.rand(x_batch.size(0), device=x_batch.device) < self.randaug_p)
            
            if randaug_mask.any(): x_batch[randaug_mask] = self.randaug(x_batch[randaug_mask])
            x_batch = self.erase(x_batch).float().div(255)
            return x_batch, y_batch

# --------------------------------------------- DeiT3 Train ---------------------------------------------------
class FineTuningModule:
    def __init__(self, model:nn.Module, train_loader,
                 test_loader, num_img_types, device, freeze_body=False
                 ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_img_types = num_img_types
        self.device = device

        self.__mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.__std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        if freeze_body:
            for param in model.parameters(): param.requires_grad = False
            for param in model.head.parameters(): param.requires_grad = True

        self.all_test_metrics, self.all_train_metrics = [], []

    def test(self, print_metrics=False):
        top1_correct_preds, top5_correct_preds, loss = 0, 0, 0
        total_samples, total_ece, total_entropy = 0, 0, 0

        total_per_type = torch.zeros(self.num_img_types, device=self.device)
        top1_correct_per_type = torch.zeros(self.num_img_types, device=self.device)
        top5_correct_per_type = torch.zeros(self.num_img_types, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, c_batch in self.test_loader:
                x_batch, y_batch, c_batch = x_batch.to(self.device), y_batch.to(self.device), c_batch.to(self.device)
                x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)
                x_batch = (x_batch - self.__mean) / self.__std
                preds = self.model(x_batch)
                del x_batch 
                
                total_samples += y_batch.size(0)
                loss += F.cross_entropy(preds, y_batch).item() * y_batch.size(0)
                # top-1 acc
                top1_right = (torch.argmax(preds, dim=1) == y_batch)
                top1_correct_preds += top1_right.sum().item()
                # top-5 acc
                top5_right = torch.topk(preds, 5, dim=1).indices.eq(y_batch.unsqueeze(1)).any(dim=1)
                top5_correct_preds += top5_right.sum().item()
                # ECE loss
                total_ece += compute_ece(preds, y_batch, self.device)
                # Per-type acc
                for t in range(self.num_img_types):
                    mask = (c_batch == t)
                    total_per_type[t] += mask.sum()
                    top1_correct_per_type[t] += top1_right[mask].sum()
                    top5_correct_per_type[t] += top5_right[mask].sum()
                # entropy
                total_entropy += -torch.sum(torch.softmax(preds, dim=1) * torch.log_softmax(preds, dim=1), dim=1).sum().item()
                del y_batch, c_batch
                
        top1_acc_per_type = (top1_correct_per_type / total_per_type).tolist()
        test_metrics = {
            "loss" : loss/total_samples,
            "top1_acc" : top1_correct_preds / total_samples, 
            "top5_acc" : top5_correct_preds / total_samples, 
            "top1_acc_per_type" : top1_acc_per_type,
            "top5_acc_per_type" :(top5_correct_per_type / total_per_type).tolist(), 
            "error_rate_per_type" : [1 - acc for acc in top1_acc_per_type],
            "ece" : total_ece / total_samples,
            "entropy" : total_entropy / total_samples
        }
        if print_metrics : print(f"Test-Accuracy:{test_metrics['top1_acc']:.3f}")
        return test_metrics
    
    def train(self, optimizer, scheduler, augmenter, save_path, 
              num_epochs=1, label_smoothing=0.1, print_metrics=False
              ):
        best_loss, epochs_no_improve, min_delta, patience = float('inf'), 0, 0.001, 5
        for epoch in range(1, num_epochs+1):
            self.model.train()
            print(f"------- Epoch {epoch} -------")
            total_samples, top1_correct_preds, loss_total = 0, 0, 0

            for x_batch, y_batch, _ in self.train_loader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                x_batch, y_batch = augmenter(x_batch, y_batch)                          # augment

                x_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)     # resize
                x_batch = (x_batch - self.__mean) / self.__std          # normalize
                preds = self.model(x_batch)

                del x_batch                 # free memory
                total_samples += y_batch.size(0)
                # top-1 acc.  # if coming from mixup/cutmix
                if len(y_batch.shape)==2 : top1_correct_preds += (torch.argmax(preds, dim=1) == torch.argmax(y_batch, dim=-1)).sum().item()
                else : top1_correct_preds += (torch.argmax(preds, dim=1) == y_batch).sum().item()
                #loss   # ybatch can be passed in directly even if cutmix/mixup applied
                loss = F.cross_entropy(preds, y_batch, label_smoothing=label_smoothing)
                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 

                loss_total += loss.item() * y_batch.size(0)
                del y_batch, preds            # free memory

            train_metrics = {
                "loss_total": loss_total/total_samples,
                "top1_acc" : top1_correct_preds / total_samples
                }
            test_metrics = self.test()
            test_loss = test_metrics['loss']
            if print_metrics : 
                print(f"train-loss: {train_metrics['loss_total']:.3f} -- train-acc: {train_metrics['top1_acc']:.3f} -- "
                      f"test-loss: {test_loss:.3f} -- test-acc: {test_metrics['top1_acc']:.3f}"
                      )
            self.all_train_metrics.append(train_metrics)
            self.all_test_metrics.append(test_metrics)
            scheduler.step()

            # early stopping
            if test_loss <= best_loss - min_delta:
                best_loss = test_loss
                epochs_no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), f"{save_path}.pth")
                    print(f"Best model saved to {save_path}.pth")
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break
        
        # save trained model and metrics
        if save_path:
            with open(f"{save_path}_train_metrics.json", "w") as f1:
                json.dump(self.all_train_metrics, f1, indent=4)
            with open(f"{save_path}_test_metrics.json", "w") as f2:
                json.dump(self.all_test_metrics, f2, indent=4)
            print("metrics saved!")


# --------------------------------------------- DeiT Train ---------------------------------------------------
class TrainTestDeiTModule:
    def __init__(self, model:nn.Module, teacher_model:nn.Module, train_loader,
                 test_loader, num_img_types, device
                 ):
        self.model = model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_img_types = num_img_types
        self.device = device

        self.__mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.__std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.all_test_metrics, self.all_train_metrics = [], []
    
    # testing function
    def test(self, print_metrics=False):
        top1_correct_preds, top5_correct_preds, loss = 0, 0, 0
        total_samples, total_ece, total_entropy = 0, 0, 0

        total_per_type = torch.zeros(self.num_img_types, device=self.device)
        top1_correct_per_type = torch.zeros(self.num_img_types, device=self.device)
        top5_correct_per_type = torch.zeros(self.num_img_types, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, c_batch in self.test_loader:
                x_batch, y_batch, c_batch = x_batch.to(self.device), y_batch.to(self.device), c_batch.to(self.device)
                x_batch = (x_batch - self.__mean) / self.__std
                preds = self.model(x_batch)
                del x_batch
                
                total_samples += y_batch.size(0)
                loss += F.cross_entropy(preds, y_batch).item() * y_batch.size(0)
                # top-1 acc
                top1_right = (torch.argmax(preds, dim=1) == y_batch)
                top1_correct_preds += top1_right.sum().item()
                # top-5 acc
                top5_right = torch.topk(preds, 5, dim=1).indices.eq(y_batch.unsqueeze(1)).any(dim=1)
                top5_correct_preds += top5_right.sum().item()
                # ECE loss
                total_ece += compute_ece(preds, y_batch, self.device)
                # Per-type acc
                for t in range(self.num_img_types):
                    mask = (c_batch == t)
                    total_per_type[t] += mask.sum()
                    top1_correct_per_type[t] += top1_right[mask].sum()
                    top5_correct_per_type[t] += top5_right[mask].sum()
                # entropy
                total_entropy += -torch.sum(torch.softmax(preds, dim=1) * torch.log_softmax(preds, dim=1), dim=1).sum().item()
                del y_batch, c_batch, preds

        top1_acc_per_type = (top1_correct_per_type / total_per_type).tolist()
        test_metrics = {
            "loss" : loss/total_samples,
            "top1_acc" : top1_correct_preds / total_samples, 
            "top5_acc" : top5_correct_preds / total_samples, 
            "top1_acc_per_type" : top1_acc_per_type,
            "top5_acc_per_type" :(top5_correct_per_type / total_per_type).tolist(), 
            "error_rate_per_type" : [1 - acc for acc in top1_acc_per_type],
            "ece" : total_ece / total_samples,
            "entropy" : total_entropy / total_samples
        }
        if print_metrics : print(f"Test-Accuracy:{test_metrics['top1_acc']:.3f}")
        return test_metrics

    # ------------ training function ------------
    def train(self, optimizer, scheduler, augmenter, loss_calculator,
              save_path, num_epochs=1, print_metrics=False
              ):
        best_loss, epochs_no_improve, min_delta, patience = float('inf'), 0, 0.001, 5
        for epoch in range(1, num_epochs+1):
            self.model.train()
            print(f"------- Epoch {epoch} -------")
            loss_total, loss_cls, loss_distill = 0, 0, 0
            total_samples, sim_cls_distill, top1_correct_preds = 0, 0, 0
            
            for x_batch, y_batch, _ in self.train_loader:       
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                teacher_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)     # resize teacher

                x_batch, y_batch = augmenter(x_batch, y_batch)     # augment
                x_batch = (x_batch - self.__mean) / self.__std
                tokens = self.model(x_batch)

                del x_batch
                teacher_batch = (teacher_batch - self.__mean) / self.__std
                losses = loss_calculator(tokens, (teacher_batch, y_batch))
                del teacher_batch

                # backprop
                optimizer.zero_grad()
                losses[0].backward()
                optimizer.step()
        
                total_samples += y_batch.size(0)
                # top-1 acc.                                # if coming from mixup/cutmix
                preds = (tokens[0] + tokens[1]) / 2
                if len(y_batch.shape)==2 : top1_correct_preds += (torch.argmax(preds, dim=1) == torch.argmax(y_batch, dim=-1)).sum().item()
                else : top1_correct_preds += (torch.argmax(preds, dim=1) == y_batch).sum().item()
                # losses
                loss_total += losses[0].item() * y_batch.size(0)
                loss_cls += losses[1].item() * y_batch.size(0)
                loss_distill += losses[2].item() * y_batch.size(0)
                # cosine similarity
                sim_cls_distill += self.model.sim_cls_distill.item() * y_batch.size(0)
                del y_batch, preds

            train_metrics = {
                "top1_acc" : top1_correct_preds / total_samples,
                "loss_total": loss_total/total_samples,
                "loss_cls": loss_cls/total_samples,
                "loss_distill": loss_distill/total_samples,
                "sim_cls_distill" : sim_cls_distill/total_samples,
            }
            test_metrics = self.test()
            self.all_train_metrics.append(train_metrics)
            self.all_test_metrics.append(test_metrics)
            scheduler.step()
            
            test_loss = test_metrics['loss']
            if print_metrics : 
                print(f"train-loss: {train_metrics['loss_total']:.3f} -- train-acc: {train_metrics['top1_acc']:.3f} -- "
                      f"test-loss: {test_loss:.3f} -- test-acc: {test_metrics['top1_acc']:.3f}"
                      )
            
            # early stopping
            if test_loss <= best_loss - min_delta:
                best_loss = test_loss
                epochs_no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), f"{save_path}.pth")
                    print(f"Best model saved to {save_path}.pth")
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break
        
        # save trained model
        if save_path:
            with open(f"{save_path}_train_metrics.json", "w") as f1:
                json.dump(self.all_train_metrics, f1, indent=4)
            with open(f"{save_path}_test_metrics.json", "w") as f2:
                json.dump(self.all_test_metrics, f2, indent=4)
            print("metrics saved!")


# --------------------------------------------- CdeiT Train ---------------------------------------------------
class TrainTestCdeiT:
    def __init__(self, model:nn.Module, teacher_model:nn.Module, train_loader,
                 test_loader, num_img_types, device, head_strategy
                 ):
        assert head_strategy > 0 and head_strategy <= 3
        self.model = model.to(device)
        self.teacher_model = teacher_model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_img_types = num_img_types
        self.device = device
        self.head_strategy = head_strategy

        self.__mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.__std  = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        self.all_test_metrics, self.all_train_metrics = [], []

    def test(self, print_metrics=False):
        top1_correct_preds, top5_correct_preds, loss = 0, 0, 0
        top1_correct_corruptions, total_samples, total_ece, total_entropy = 0, 0, 0, 0

        total_per_type = torch.zeros(self.num_img_types, device=self.device)
        top1_correct_per_type = torch.zeros(self.num_img_types, device=self.device)
        top5_correct_per_type = torch.zeros(self.num_img_types, device=self.device)

        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch, c_batch in self.test_loader:
                x_batch, y_batch, c_batch = x_batch.to(self.device), y_batch.to(self.device), c_batch.to(self.device)
                x_batch = (x_batch - self.__mean) / self.__std
                tokens = self.model(x_batch)
                del x_batch
                
                if self.head_strategy == 1:
                    preds = (tokens[0] + tokens[1]) / 2
                else:
                    preds = (tokens[0] + tokens[1] + self.model.output_head.ffn(tokens[2])) / 3
                
                total_samples += y_batch.size(0)
                loss += F.cross_entropy(preds, y_batch).item() * y_batch.size(0)
                # top-1 acc
                top1_right = (torch.argmax(preds, dim=1) == y_batch)
                top1_correct_preds += top1_right.sum().item()
                # top-5 acc
                top5_right = torch.topk(preds, 5, dim=1).indices.eq(y_batch.unsqueeze(1)).any(dim=1)
                top5_correct_preds += top5_right.sum().item()
                # ECE loss
                total_ece += compute_ece(preds, y_batch, self.device)
                # Per-type acc
                for t in range(self.num_img_types):
                    mask = (c_batch == t)
                    total_per_type[t] += mask.sum()
                    top1_correct_per_type[t] += top1_right[mask].sum()
                    top5_correct_per_type[t] += top5_right[mask].sum()
                # entropy
                total_entropy += -torch.sum(torch.softmax(preds, dim=1) * torch.log_softmax(preds, dim=1), dim=1).sum().item()
                # top-1 corruption classification acc
                top1_correct_corruptions += (torch.argmax(tokens[2], dim=1) == c_batch).sum().item()              
                del y_batch, c_batch, preds

        top1_acc_per_type = (top1_correct_per_type / total_per_type).tolist()
        test_metrics = {
            "loss" : loss/total_samples,
            "top1_acc" : top1_correct_preds / total_samples, 
            "top5_acc" : top5_correct_preds / total_samples, 
            "top1_corrupt_acc" : top1_correct_corruptions / total_samples,
            "top1_acc_per_type" : top1_acc_per_type,
            "top5_acc_per_type" :(top5_correct_per_type / total_per_type).tolist(), 
            "error_rate_per_type" : [1 - acc for acc in top1_acc_per_type],
            "ece" : total_ece / total_samples,
            "entropy" : total_entropy / total_samples
        }
        if print_metrics : print(f"Test-Accuracy:{test_metrics['top1_acc']:.3f}")
        return test_metrics

    def train(self, optimizer, scheduler, augmenter, loss_calculator,
              save_path, num_epochs=1, print_metrics=False
              ):
        best_loss, epochs_no_improve, min_delta, patience = float('inf'), 0, 0.001, 5
        for epoch in range(1, num_epochs+1):
            self.model.train()
            print(f"------- Epoch {epoch} -------")
            loss_total, loss_cls, loss_distill, loss_corrupt, loss_cls_corruptFFN = 0, 0, 0, 0, 0
            total_samples, sim_cls_distill, sim_cls_corrupt, top1_correct_preds = 0, 0, 0, 0

            for x_batch, y_batch, c_batch in self.train_loader:
                x_batch, y_batch, c_batch = x_batch.to(self.device), y_batch.to(self.device), c_batch.to(self.device)
                teacher_batch = F.interpolate(x_batch, size=(224, 224), mode='bilinear', align_corners=False)     # resize teacher

                x_batch, y_batch = augmenter(x_batch, y_batch)     # augment
                x_batch = (x_batch - self.__mean) / self.__std
                tokens = self.model(x_batch)    # 3 tokens -- cls, distill, corrupt
                
                del x_batch
                teacher_batch = (teacher_batch - self.__mean) / self.__std
                losses = loss_calculator(tokens, (teacher_batch, y_batch, c_batch), self.model)     # cdeit loss calculator
                del teacher_batch, c_batch

                # backprop
                optimizer.zero_grad()
                losses[0].backward()
                optimizer.step()
                
                total_samples += y_batch.size(0)
                # top-1 acc
                if self.head_strategy == 1: 
                    preds = (tokens[0] + tokens[1]) / 2
                else: 
                    preds = (tokens[0] + tokens[1] + self.model.output_head.ffn(tokens[2])) / 3
                if len(y_batch.shape)==2 : top1_correct_preds += (torch.argmax(preds, dim=1) == torch.argmax(y_batch, dim=-1)).sum().item()
                else : top1_correct_preds += (torch.argmax(preds, dim=1) == y_batch).sum().item()
                
                # losses
                loss_total += losses[0].item() * y_batch.size(0)
                loss_cls += losses[1].item() * y_batch.size(0)
                loss_distill += losses[2].item() * y_batch.size(0)
                loss_corrupt += losses[3].item() * y_batch.size(0)
                if len(losses) == 5:
                    loss_cls_corruptFFN += losses[4].item() * y_batch.size(0)
                # cosine similarity
                sim_cls_distill += self.model.sim_cls_distill.item() * y_batch.size(0)
                sim_cls_corrupt += self.model.sim_cls_corrupt.item() * y_batch.size(0)
                del y_batch, preds
                
            train_metrics = {
                "top1_acc" : top1_correct_preds / total_samples,
                "loss_total": loss_total/total_samples,
                "loss_cls": loss_cls/total_samples,
                "loss_distill": loss_distill/total_samples,
                "loss_corrupt" : loss_corrupt/total_samples,
                "sim_cls_distill" : sim_cls_distill/total_samples,
                "sim_cls_corrupt" : sim_cls_corrupt/total_samples
            }
            if loss_cls_corruptFFN : train_metrics["loss_cls_corruptFFN"] = loss_cls_corruptFFN/total_samples
            test_metrics = self.test()
            self.all_train_metrics.append(train_metrics)
            self.all_test_metrics.append(test_metrics)
            scheduler.step()
            
            test_loss = test_metrics['loss']
            if print_metrics : 
                print(f"train-loss: {train_metrics['loss_total']:.3f} -- train-acc: {train_metrics['top1_acc']:.3f} -- "
                      f"test-loss: {test_loss:.3f} -- test-acc: {test_metrics['top1_acc']:.3f}"
                      )

            # early stopping
            if test_loss <= best_loss - min_delta:
                best_loss = test_loss
                epochs_no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), f"{save_path}.pth")
                    print(f"Best model saved to {save_path}.pth")
            else : epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch} — no improvement in {patience} epochs.")
                break
        
        # save trained model
        if save_path:
            with open(f"{save_path}_train_metrics.json", "w") as f1:
                json.dump(self.all_train_metrics, f1, indent=4)
            with open(f"{save_path}_test_metrics.json", "w") as f2:
                json.dump(self.all_test_metrics, f2, indent=4)
            print("metrics saved!")

