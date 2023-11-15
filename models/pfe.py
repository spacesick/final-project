import torch as t
import torch.nn as nn

from modules.resnet import ResNetBackbone
from modules.uncertainty import Uncertainty


class PFE(nn.Module):

  def __init__(
      self,
      emb_size,
      use_pretrained,
      use_l2norm,
      freeze_bnorm,
      freeze_backbone,
      uncertainty_fc_size
  ):
    super(PFE, self).__init__()

    self.backbone = ResNetBackbone(
        emb_size=emb_size,
        use_pretrained=use_pretrained,
        use_l2norm=use_l2norm,
        freeze_bnorm=freeze_bnorm
    )

    if freeze_backbone:
      for param in self.backbone.parameters():
        param.requires_grad = False

    self.uncertainty = Uncertainty(
        input_size=self.backbone.net.head.in_features,
        emb_size=emb_size,
        fc_size=uncertainty_fc_size
    )

  def forward(self, x):
    mu = self.backbone(x)
    sigma = self.uncertainty(self.backbone.get_prehead_output())

    return mu, sigma

  def load(self, path):
    self.load_state_dict(t.load(path))

  def save(self, path):
    t.save(self.state_dict(), path)
