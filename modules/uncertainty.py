import torch as t
import torch.nn as nn


class Uncertainty(nn.Module):

  def __init__(
      self,
      input_size,
      emb_size,
      fc_size=256
  ):
    super(Uncertainty, self).__init__()

    self.fc1 = nn.Linear(
        input_size,
        fc_size,
        bias=False
    )
    self.bnorm1 = nn.BatchNorm1d(
        fc_size,
        eps=0.001,
        momentum=0.005
    )
    self.relu = nn.ReLU(inplace=True)

    self.fc2 = nn.Linear(
        fc_size,
        emb_size
    )
    self.bnorm2 = nn.BatchNorm1d(
        emb_size,
        affine=False,
        eps=0.001,
        momentum=0.005
    )

    gamma_init = 0.1
    beta_init = -1.0 if emb_size == 1 else -7.0
    self.shared_gamma = nn.Parameter(t.full((emb_size,), gamma_init))
    self.shared_beta = nn.Parameter(t.full((emb_size,), beta_init))

  def forward(self, x):
    x = self.fc1(x)
    x = self.bnorm1(x)
    x = self.relu(x)

    x = self.fc2(x)
    x = self.bnorm2(x)

    x = self.shared_gamma * x + self.shared_beta

    x = t.log(0.000001 + t.exp(x))

    return x
