
import torch
import torch.nn.functional as F

def compute_ece(preds, labels, device, n_bins=15):
    preds = torch.softmax(preds, dim=1)
    confidences, preds = preds.max(dim=1)
    acc = preds.eq(labels)
    
    bins = torch.linspace(0,1,n_bins+1, device=device)
    ece = torch.zeros(1, device=device)
    for i in range(n_bins):
        mask = confidences.gt(bins[i]) & confidences.le(bins[i+1])
        if mask.sum() > 0:
            ece += (mask.sum().float() / labels.size(0)) * torch.abs(acc[mask].float().mean() - confidences[mask].mean())
            
    return ece.item()

def test_cdeit_model(
        test_batches, model, head_strategy, 
        num_img_types, device, print_metrics=False
        ):
    L_total, top1_correct_preds, top5_correct_preds = 0, 0, 0
    top1_correct_corrs, total_samples, total_ece = 0, 0, 0
    total_per_type = torch.zeros(num_img_types, device=device)
    top1_correct_per_type = torch.zeros(num_img_types, device=device)
    top5_correct_per_type = torch.zeros(num_img_types, device=device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        for x_batch, y_batch, c_batch in test_batches:
            x_batch, y_batch, c_batch = x_batch.to(device), y_batch.to(device), c_batch.to(device)
            cls_tokens, distill_tokens, corrupt_tokens = model(x_batch)
            
            preds = None
            if head_strategy == 1:
                preds = (cls_tokens + distill_tokens) / 2
            elif head_strategy == 2 or head_strategy == 3:
                preds = (cls_tokens + distill_tokens + corrupt_tokens @ model.output_head.W) / 3
            else:
                raise ValueError("head_strategy out of range [1,3]")
            L_overall = F.cross_entropy(preds, y_batch)
            
            # top-1 acc
            top1_right = (torch.argmax(preds, dim=1) == y_batch)
            top1_correct_preds += top1_right.sum().item()
            # top-5 acc
            top5_right = torch.topk(preds, 5, dim=1).indices.eq(y_batch.unsqueeze(1)).any(dim=1)
            top5_correct_preds += top5_right.sum().item()
            # overall loss
            L_total += L_overall.item() * y_batch.size(0)
            total_samples += y_batch.size(0)
            # top-1 corruption classification acc
            top1_correct_corrs += (torch.argmax(corrupt_tokens, dim=1) == c_batch).sum().item()
            # Per-type acc and partial mCE
            for t in range(num_img_types):
                mask = (c_batch == t)
                total_per_type[t] += mask.sum()
                top1_correct_per_type[t] += top1_right[mask].sum()
                top5_correct_per_type[t] += top5_right[mask].sum()
            # ECE loss
            total_ece += compute_ece(preds, y_batch) * y_batch.size(0)

    top1_acc_per_type = (top1_correct_per_type / total_per_type).tolist()
    metrics = {
        "L_total" : L_total/total_samples, 
        "top1_acc" : top1_correct_preds / total_samples, 
        "top5_acc" : top5_correct_preds / total_samples, 
        "top1_corr_acc" : top1_correct_corrs / total_samples,
        "top1_acc_per_type" : top1_acc_per_type,
        "top5_acc_per_type" :(top5_correct_per_type / total_per_type).tolist(), 
        "error_rate_per_type" : [1 - acc for acc in top1_acc_per_type],
        "ece" : total_ece / total_samples
    }
    if print_metrics:
        print(f"Test-Loss: {metrics['L_total']:.4f}, Test-Accuracy:{metrics['top1_acc']:.2f}")
    return metrics



