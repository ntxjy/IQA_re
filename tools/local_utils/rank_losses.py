import torch
import torch.nn.functional as F

def listnet_loss(scores, targets):
    P = F.softmax(scores, dim=1)
    T = F.softmax(targets, dim=1)
    return F.kl_div(P.log(), T, reduction='batchmean')

def pairwise_logistic_loss(scores, targets):
    B, K = scores.shape
    loss = 0.0
    for b in range(B):
        s = scores[b]; t = targets[b]
        diff_s = s.unsqueeze(0) - s.unsqueeze(1)
        diff_t = t.unsqueeze(0) - t.unsqueeze(1)
        sign = diff_t.sign()
        loss += F.softplus(-sign * diff_s).mean()
    return loss / B
