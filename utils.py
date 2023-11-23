import glob
import os

import numpy as np
import torch as t
from sklearn.metrics import roc_curve
from torchvision import transforms


def make_transform(train=True):
  """Return an image transform for ResNet.

  Args:
      train (bool, optional): Model state. Defaults to True.

  Returns:
      Compose: Image transform.
  """
  resnet_resize = 256
  resnet_cropsize = 224
  resnet_mean = [0.485, 0.456, 0.406]
  resnet_std = [0.229, 0.224, 0.225]

  if train:
    resnet_transform = transforms.Compose([
        transforms.RandomResizedCrop(resnet_cropsize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])
  else:
    resnet_transform = transforms.Compose([
        transforms.Resize(resnet_resize),
        transforms.CenterCrop(resnet_cropsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

  return resnet_transform


def compute_recall_at_k(
    labels,
    top_k_label_predictions,
    k
):
  """Compute Recall@k metric.

  Args:
      labels (Tensor): Target labels with size (number of samples).
      top_k_label_predictions (Tensor): Top k predicted labels with size (number of samples, k).
      k (int): k value.

  Returns:
      float: Recall@k value.
  """
  assert top_k_label_predictions.dim() == 2
  assert labels.size(0) == top_k_label_predictions.size(0)
  assert labels.dtype == top_k_label_predictions.dtype

  tp = 0
  for actual, predictions in zip(labels, top_k_label_predictions):
    if actual in predictions[:k]:
      tp += 1
  return tp / len(labels)


def compute_map_at_r(
    labels,
    label_predictions
):
  """Compute MAP@R metric. See https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123700681.pdf

  Args:
      labels (Tensor): Target labels with size (number of samples).
      label_predictions (Tensor): Top n predicted labels with size (number of samples, n).

  Returns:
      float: MAP@R value.
  """
  assert label_predictions.dim() == 2
  assert labels.size(0) == label_predictions.size(0)
  assert labels.dtype == label_predictions.dtype

  device = labels.device

  ap_at_r = 0
  for actual, predictions in zip(labels, label_predictions):
    truths = (actual == predictions)
    r = truths.sum()

    if r > 0:
      tp_pos = t.arange(1, r + 1, device=device)[truths[:r] > 0]
      ap_at_r += t.div((t.arange(len(tp_pos), device=device) + 1), tp_pos).sum() / r

  return ap_at_r / len(labels)


def compute_tar_at_far(
    labels,
    label_predictions,
    score_predictions,
    far_value=0.001
):
  """Computer TAR@FAR metric.

  Args:
      labels (Tensor): Target labels with size (number of samples).
      label_predictions (Tensor): Top n predicted labels with size (number of samples, n).
      score_predictions (Tensor): Top n predicted similarity scores with size (number of samples, n).
      far_value (float, optional): FAR value. Defaults to 0.001.

  Returns:
      float: TAR@FAR value.
  """
  assert label_predictions.dim() == score_predictions.dim() == 2
  assert label_predictions.size(1) == score_predictions.size(1)
  assert labels.size(0) == label_predictions.size(0) == score_predictions.size(0)
  assert labels.dtype == label_predictions.dtype

  tar_at_far = 0
  truths = []
  for actual, l_predictions, s_predictions in zip(labels, label_predictions, score_predictions):
    truths.append((actual == l_predictions))

  truths = t.cat(truths)
  far, tar, _ = roc_curve(truths.cpu(), score_predictions.cpu().reshape(-1))
  tar_at_far += np.interp(far_value, far, tar)

  return tar_at_far / len(labels)


def compute_batch_mls(
    query_embeddings,
    refer_embeddings,
    limit_memory=False
):
  """Computer Mutual Likelihood Score of a batch of embeddings.

  Args:
      query_embeddings (Tuple[Tensor, Tensor]): Query embeddings.
      refer_embeddings (Tuple[Tensor, Tensor]): Reference embeddings.
      limit_memory (bool, optional): Whether to limit memory usage to prevent out of memory errors. Defaults to False.

  Returns:
      Tensor: Similarity matrix.
  """
  mean_query, log_variance_query = query_embeddings
  mean_refer, log_variance_refer = refer_embeddings

  variance_query = t.exp(log_variance_query)
  variance_refer = t.exp(log_variance_refer)

  mean_y = mean_refer.unsqueeze(0)
  variance_y = variance_refer.unsqueeze(0)

  if limit_memory:
    scores_list = []
    for i in range(mean_query.size(0)):
      mean_x = mean_query.unsqueeze(1)[i, :, :]
      variance_x = variance_query.unsqueeze(1)[i, :, :]
      fused_variance = variance_x + variance_y

      scores = t.square(mean_x - mean_y) / (1e-10 + fused_variance) + t.log(fused_variance)
      scores = t.sum(scores, dim=-1)
      scores_list.append(scores)

    score_mat = t.cat(scores_list, dim=0)

  else:
    mean_x = mean_query.unsqueeze(1)
    variance_x = variance_query.unsqueeze(1)
    fused_variance = variance_x + variance_y

    score_mat = t.square(mean_x - mean_y) / (1e-10 + fused_variance) + t.log(fused_variance)
    score_mat = t.sum(score_mat, dim=-1)

  return -0.5 * score_mat


def get_checkpoint_path(root_path, ckpt_name='latest'):
  """Return path to a model checkpoint.

  Args:
      root_path (str): Path to the root directory containing the model checkpoints.
      ckpt_name (str, optional): Name of checkpoint file. If set to 'latest', it will find the latest checkpoint file in the given root directory. Defaults to 'latest'.

  Raises:
      ValueError: If no checkpoint is found in the given root directory with the given name.

  Returns:
      str: Relative path to the model checkpoint.
  """
  if ckpt_name == 'latest':
    checkpoint_paths = glob.glob(os.path.join(root_path, '*.pth'))
    if len(checkpoint_paths) == 0:
      raise ValueError(f'No checkpoints found in {root_path}')
    checkpoint_paths = sorted(checkpoint_paths)
    return checkpoint_paths[-1]
  else:
    checkpoint_path = os.path.join(root_path, ckpt_name)
    if os.path.isfile(checkpoint_path):
      return checkpoint_path
    else:
      raise ValueError(f'No checkpoint found in {checkpoint_path}')
