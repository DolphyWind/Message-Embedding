import torch
import torch.nn.functional as F


def triplet_loss(
    anchor_out: torch.Tensor,
    pos_out: torch.Tensor,
    neg_out: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    d_ap = torch.linalg.norm(anchor_out - pos_out, dim=-1)
    d_an = torch.linalg.norm(anchor_out - neg_out, dim=-1)
    loss = F.relu(d_ap - d_an + margins)
    loss = torch.mean(loss, dim=0)
    return loss
