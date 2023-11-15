import torch as t
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50


class ResNetBackbone(nn.Module):

  def __init__(
      self,
      emb_size=512,
      use_pretrained=True,
      use_l2norm=True,
      freeze_bnorm=True
  ):
    super(ResNetBackbone, self).__init__()

    self.net = resnet50(weights=ResNet50_Weights.DEFAULT) if use_pretrained else resnet50()

    self.use_l2norm = use_l2norm

    self.net.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.net.global_maxpool = nn.AdaptiveMaxPool2d((1, 1))
    self.net.head = nn.Linear(self.net.fc.in_features, emb_size)
    nn.init.kaiming_normal_(self.net.head.weight, mode='fan_out')
    nn.init.constant_(self.net.head.bias, 0)

    self.prehead_output = None

    if freeze_bnorm:
      for module in self.net.modules():
        if isinstance(module, nn.BatchNorm2d):
          module.weight.requires_grad = False
          module.bias.requires_grad = False

  def l2norm(self, input):
    buffer = t.pow(input, 2)

    normp = t.sum(buffer, 1).add_(1e-12)
    norm = t.sqrt(normp)

    _output = t.div(input, norm.view(-1, 1).expand_as(input))

    output = _output.view(input.size())

    return output

  def forward(self, x):
    x = self.net.conv1(x)
    x = self.net.bn1(x)
    x = self.net.relu(x)
    x = self.net.maxpool(x)
    x = self.net.layer1(x)
    x = self.net.layer2(x)
    x = self.net.layer3(x)
    x = self.net.layer4(x)

    avg_x = self.net.global_avgpool(x)
    max_x = self.net.global_maxpool(x)

    x = max_x + avg_x
    x = t.flatten(x, 1)
    self.prehead_output = x
    x = self.net.head(x)

    if self.use_l2norm:
      x = self.l2norm(x)

    return x

  def get_prehead_output(self):
    return self.prehead_output
