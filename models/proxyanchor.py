import torch as t
import torch.nn as nn

from modules.resnet import ResNetBackbone


class ProxyAnchor(nn.Module):

  def __init__(
      self,
      emb_size,
      use_pretrained,
      use_l2norm,
      freeze_bnorm
  ):
    super(ProxyAnchor, self).__init__()

    self.backbone = ResNetBackbone(
        emb_size=emb_size,
        use_pretrained=use_pretrained,
        use_l2norm=use_l2norm,
        freeze_bnorm=freeze_bnorm
    )

  def forward(self, x):
    mu = self.backbone(x)

    return mu

  def load(self, path):
    self.load_state_dict(t.load(path))

  def save(self, path):
    t.save(self.state_dict(), path)
