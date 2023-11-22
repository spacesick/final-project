import torch as t
import torch.nn as nn
import torch.nn.functional as F

from utils import compute_batch_mls


class ProxyMLS(nn.Module):

  def __init__(
      self,
      num_classes,
      embedding_size,
      margin=0.1,
      alpha=32
  ):
    super(ProxyMLS, self).__init__()

    self.mean_proxies = nn.Parameter(
        t.randn(num_classes, embedding_size)
    )
    self.log_variance_proxies = nn.Parameter(
        t.randn(num_classes, embedding_size)
    )
    nn.init.kaiming_normal_(self.mean_proxies, mode='fan_out')
    nn.init.kaiming_normal_(self.log_variance_proxies, mode='fan_out')

    self.num_classes = num_classes
    self.embedding_size = embedding_size
    self.margin = margin
    self.alpha = alpha

  def forward(self, embeddings, labels):
    score_mat = compute_batch_mls(
        embeddings, (self.mean_proxies, self.log_variance_proxies)
    )
    pos_mask = F.one_hot(labels, self.num_classes)
    neg_mask = 1 - pos_mask

    pos_exp = -self.alpha * (score_mat - self.margin)
    neg_exp = self.alpha * (score_mat + self.margin)

    pos_exp_max = t.max(pos_exp)
    neg_exp_max = t.max(pos_exp)

    pos_exp = t.exp(pos_exp - pos_exp_max)
    neg_exp = t.exp(neg_exp - neg_exp_max)

    # The set of positive proxies
    with_pos_proxies = t.nonzero(
        pos_mask.sum(dim=0) != 0
    ).squeeze(dim=1)
    # The number of positive proxies
    num_valid_proxies = len(with_pos_proxies)

    pos_sim_sum = t.where(pos_mask == 1, pos_exp, t.zeros_like(pos_exp)).sum(dim=0)
    neg_sim_sum = t.where(neg_mask == 1, neg_exp, t.zeros_like(neg_exp)).sum(dim=0)

    pos_term = t.log(1 + pos_sim_sum) + pos_exp_max
    neg_term = t.log(1 + neg_sim_sum) + neg_exp_max
    pos_term = pos_term.sum() / num_valid_proxies
    neg_term = neg_term.sum() / self.num_classes
    loss = pos_term + neg_term

    return loss
