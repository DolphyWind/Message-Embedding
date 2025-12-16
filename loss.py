import torch
import torch.nn.functional as F


def triplet_loss(
    anchor_out: torch.Tensor,
    pos_out: torch.Tensor,
    neg_out: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    a = F.normalize(anchor_out, dim=-1)
    p = F.normalize(pos_out, dim=-1)
    n = F.normalize(neg_out, dim=-1)

    d_ap = torch.linalg.norm(a - p, dim=-1)
    d_an = torch.linalg.norm(a - n, dim=-1)

    return torch.mean(F.relu(d_ap - d_an + margin))
