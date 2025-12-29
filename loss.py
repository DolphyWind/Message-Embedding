import torch
import torch.nn.functional as F


def triplet_loss(
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    negatives: torch.Tensor,
    margin: float,
    **kwargs,
) -> torch.Tensor:
    a = F.normalize(anchors, dim=1)
    p = F.normalize(positives, dim=1)
    n = F.normalize(negatives, dim=1)

    d_ap = torch.linalg.norm(a - p, dim=1)
    d_an = torch.linalg.norm(a - n, dim=1)

    return torch.mean(F.relu(d_ap - d_an + margin))


def infonce_loss(
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
    **kwargs,
) -> torch.Tensor:
    a = F.normalize(anchors, dim=1)
    p = F.normalize(positives, dim=1)

    logits: torch.Tensor = a @ p.T
    logits /= temperature
    targets: torch.Tensor = torch.arange(logits.size(0), device=logits.device)
    return F.cross_entropy(logits, targets)


def clip_loss(
    *,
    anchors: torch.Tensor,
    positives: torch.Tensor,
    temperature: float,
    **kwargs,
) -> torch.Tensor:
    return 0.5 * (
        infonce_loss(
            anchors=anchors,
            positives=positives,
            temperature=temperature
        ) +
        infonce_loss(
            anchors=anchors,
            positives=positives,
            temperature=temperature
        )
    )
