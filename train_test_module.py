
import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import List, Tuple

class LossCalculator(nn.Module):
    def __init__(self, teacher_model: nn.Module):
        """ Uses Hard Distillation for L_total.

        Args of forward: 
            batches = (teacher_x_batch, yzbatch, c_batch: Optional)
            tokens = (cls, distill, corrupt: Optional)
        """
        super().__init__()
        self.teacher_model = teacher_model
        
    def forward(self, batches:Tuple[torch.Tensor], tokens:Tuple[torch.Tensor]):
        assert (len(batches) != 3 and len(tokens) != 3) or (len(batches) == 3 and len(tokens) == 3)

        with torch.no_grad():
            teacher_labels = torch.argmax(self.teacher_model(batches[0]), dim=1)
        assert tokens[1].size(1) > teacher_labels.max(), "Teacher head predicting more classes than num_classes"

        L_distill = F.cross_entropy(tokens[1], teacher_labels)
        L_cls = F.cross_entropy(tokens[0], batches[1])
        
        if len(tokens) == 3:
            L_corrupt = F.cross_entropy(tokens[2], batches[2])
            L_total = (L_cls + L_distill + L_corrupt)/3
            return L_total, L_cls, L_distill, L_corrupt
        
        L_total = (L_cls + L_distill)/2
        return L_total, L_cls, L_distill
    

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