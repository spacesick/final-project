import torch as t
import torch.nn as nn

from utils import compute_batch_mls


class MutualLikelihoodScore(nn.Module):

  def __init__(self):
    super(MutualLikelihoodScore, self).__init__()

  def __call__(
      self,
      embeddings,
      labels
  ):
    mean = embeddings[0]

    num_samples = mean.size(0)

    diag_mask = t.eye(num_samples, dtype=t.bool)
    non_diag_mask = ~diag_mask.to(mean.device)

    label_mat = t.eq(labels[:, None], labels[None, :])
    label_mask_pos = non_diag_mask & label_mat

    score_mat = compute_batch_mls(embeddings, embeddings)

    return t.mean(t.masked_select(score_mat, label_mask_pos))
