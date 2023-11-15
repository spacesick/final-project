import torch as t
import torch.nn as nn


class MutualLikelihoodScore(nn.Module):

  def __init__(self):
    super(MutualLikelihoodScore, self).__init__()

  def __call__(
      self,
      embeddings,
      labels
  ):
    mean, log_variance = embeddings

    batch_size = mean.size(0)

    diag_mask = t.eye(batch_size, dtype=t.bool)
    non_diag_mask = ~diag_mask.to(mean.device)

    variance = t.exp(log_variance)

    emb_size = mean.size(1)
    mean_x = t.reshape(mean, (-1, 1, emb_size))
    mean_y = t.reshape(mean, (1, -1, emb_size))
    variance_x = t.reshape(variance, (-1, 1, emb_size))
    variance_y = t.reshape(variance, (1, -1, emb_size))
    fused_variance = variance_x + variance_y

    loss_mat = t.square(mean_x - mean_y) / (1e-10 + fused_variance) + t.log(fused_variance)
    loss_mat = t.sum(loss_mat, dim=2)

    label_mat = t.eq(labels[:, None], labels[None, :])
    label_mask_pos = non_diag_mask & label_mat

    return t.mean(t.masked_select(loss_mat, label_mask_pos))
