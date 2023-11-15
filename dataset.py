import os
from typing import Callable, Optional

import numpy as np
from PIL import Image
from torch.utils.data.sampler import Sampler
from torchvision.datasets import LFWPeople, VisionDataset


class CasiaWebFaceDataset(VisionDataset):

  def __init__(
      self,
      root: str,
      transform,
      target_transform
  ):
    super().__init__(
        root,
        transform=transform,
        target_transform=target_transform
    )

    self.class_to_idx = self._get_classes()
    self.data = []
    self.targets = []
    for class_path, _, class_images in sorted(os.walk(self.root)):
      for image in class_images:
        self.data.append(os.path.join(class_path, image))
        class_name = class_path.split(os.sep)[-1]
        self.targets.append(self.class_to_idx[class_name])

  def _load_image(self, path):
    with open(path, "rb") as f:
      img = Image.open(f)
      return img.convert("RGB")

  def _get_classes(self):
    classes = sorted(
        entry.name for entry in os.scandir(self.root) if entry.is_dir()
    )
    class_to_idx = {c: i for i, c in enumerate(classes)}
    return class_to_idx

  def __getitem__(self, index: int):
    img = self._load_image(self.data[index])
    target = self.targets[index]

    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, target

  def __len__(self):
    return len(self.data)

  def num_classes(self):
    return len(self.class_to_idx)


class LFWDataset(LFWPeople):

  def __init__(
      self,
      root: str,
      split: str = "10fold",
      image_set: str = "funneled",
      transform: Optional[Callable] = None,
      target_transform: Optional[Callable] = None,
      download: bool = False,
  ) -> None:
    super().__init__(
        root,
        split,
        image_set,
        transform,
        target_transform,
        download
    )

    # Set class numbers to be in the range: [0, number of classes)
    self.targets = np.unique(self.targets, return_inverse=True)[1].tolist()

  def num_classes(self):
    return len(self.class_to_idx)


class BalancedSampler(Sampler):

  def __init__(
      self,
      data_source,
      batch_size,
      batch_classes
  ):
    self.data_source = data_source
    self.labels = np.array(data_source.targets)

    if batch_size % batch_classes != 0:
      raise ValueError('batch_size must be divisible by batch_classes in config file')

    self.batch_size = batch_size
    self.batch_classes = batch_classes
    self.num_instances = batch_size // batch_classes
    self.num_samples = len(self.labels)
    self.num_classes = len(set(self.labels))

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    num_batches = len(self.data_source) // self.batch_size
    ret = []
    while num_batches > 0:
      sampled_classes = np.random.choice(
          self.num_classes, self.batch_classes, replace=False
      )
      for i in range(len(sampled_classes)):
        ith_class_idxs = np.nonzero(self.labels == sampled_classes[i])[0]
        class_sel = np.random.choice(
            ith_class_idxs, size=self.num_instances, replace=True
        )
        ret.extend(np.random.permutation(class_sel))
      num_batches -= 1

    return iter(ret)
