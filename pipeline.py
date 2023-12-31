import hashlib
import logging
import os
import random
import time
from datetime import datetime

import numpy as np
import omegaconf
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pytorch_metric_learning.losses import ProxyAnchorLoss
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
from tqdm import tqdm

import utils
from dataset import (BalancedSampler, CasiaWebFaceDataset, LFWDataset,
                     LFWGFPGANDataset)
from losses.mls import MutualLikelihoodScore
from losses.proxymls import ProxyMLS
from models.pfe import PFE
from models.proxyanchor import ProxyAnchor


class Pipeline:

  def __init__(self, cfg: omegaconf.DictConfig):
    self.cfg = cfg
    self.device = self.cfg.device

    if 'seed' in self.cfg:
      random.seed(self.cfg.seed)
      np.random.seed(self.cfg.seed)
      t.manual_seed(self.cfg.seed)
      t.cuda.manual_seed_all(self.cfg.seed)

    self.logger = logging.getLogger(__name__)

    if 'dataset' not in self.cfg:
      raise KeyError('No dataset specified in config file')
    if 'model' not in self.cfg:
      raise KeyError('No model specified in config file')
    if 'checkpoint_root' not in self.cfg.model:
      raise KeyError('You must specify a model checkpoint path in the config file!')
    if not os.path.exists(self.cfg.model.checkpoint_root):
      os.makedirs(self.cfg.model.checkpoint_root)
    if 'optimizer' not in self.cfg:
      raise KeyError('No optimizer specified in config file')
    if 'lr_scheduler' not in self.cfg:
      raise KeyError('No lr_scheduler specified in config file')
    if 'logging' not in self.cfg:
      raise KeyError('No logging specified in config file')
    if 'log_to_wandb' not in self.cfg.logging:
      raise KeyError('Please define log_to_wandb in the logging section of the config file')

    cfg_yaml = OmegaConf.to_yaml(cfg)
    cfg_hash = self.cfg.experiment
    self.logger.debug(
        'Init pipeline with config:\n%s\nHash=%s',
        cfg_yaml,
        cfg_hash
    )

    if self.cfg.logging.log_to_wandb:
      cfg_dict = OmegaConf.to_container(self.cfg)
      assert isinstance(cfg_dict, dict)
      wandb.init(
          project=f'fp_{self.cfg.model.name}_{self.cfg.dataset.name}',
          id=cfg_hash,
          config=cfg_dict,
          resume='allow'
      )

    self.get_dataloaders()
    self.build_model()
    self.get_optimizer()
    self.get_scheduler()

    self.logger.debug(
        'Model Architecture:\n%s',
        self.model
    )

  def get_dataloaders(self):
    if self.cfg.dataset.name == 'lfw-only':
      self.trn_set = LFWDataset(
          split='train',
          image_set='deepfunneled',
          transform=utils.make_transform(train=True),
          download=True
      )
      self.tst_set = LFWDataset(
          split='test',
          image_set='original',
          transform=utils.make_transform(train=False),
          download=True
      )
    elif self.cfg.dataset.name == 'lfw-gfpgan-only':
      self.trn_set = LFWGFPGANDataset(
          split='train',
          image_set='deepfunneled',
          transform=utils.make_transform(train=True),
          download=True
      )
      self.tst_set = LFWGFPGANDataset(
          split='test',
          image_set='deepfunneled',
          transform=utils.make_transform(train=False),
          download=True
      )
    elif self.cfg.dataset.name == 'casia-lfw':
      self.trn_set = CasiaWebFaceDataset(
          transform=utils.make_transform(train=True)
      )
      self.tst_set = LFWDataset(
          split='10fold',
          image_set='original',
          transform=utils.make_transform(train=False),
          download=True
      )
    else:
      raise ValueError(
          f'Invalid dataset name provided in config file: {self.cfg.dataset.name}')

    self.num_classes = self.trn_set.num_classes()

    trn_balanced_sampler = BalancedSampler(
        self.trn_set,
        batch_size=self.cfg.dataset.batch_size,
        batch_classes=self.cfg.dataset.batch_classes
    )
    trn_batch_sampler = BatchSampler(
        trn_balanced_sampler,
        batch_size=self.cfg.dataset.batch_size,
        drop_last=True
    )

    self.trn_loader = DataLoader(
        self.trn_set,
        num_workers=self.cfg.dataset.workers,
        pin_memory=True,
        batch_sampler=trn_batch_sampler
    )
    self.tst_loader = DataLoader(
        self.tst_set,
        batch_size=self.cfg.dataset.batch_size,
        shuffle=False,
        num_workers=self.cfg.dataset.workers,
        pin_memory=True
    )

  def build_model(self):
    if self.cfg.model.name == 'pfe':
      self.model = PFE(
          emb_size=self.cfg.model.emb_size,
          use_pretrained=self.cfg.model.use_pretrained,
          use_l2norm=self.cfg.model.use_l2norm,
          freeze_bnorm=self.cfg.model.freeze_bnorm,
          freeze_backbone=self.cfg.model.freeze_backbone,
          uncertainty_fc_size=self.cfg.model.uncertainty_fc_size
      )
      self.loss_func = MutualLikelihoodScore()
    elif self.cfg.model.name == 'proxyanchor':
      self.model = ProxyAnchor(
          emb_size=self.cfg.model.emb_size,
          use_pretrained=self.cfg.model.use_pretrained,
          use_l2norm=self.cfg.model.use_l2norm,
          freeze_bnorm=self.cfg.model.freeze_bnorm
      )
      self.loss_func = ProxyAnchorLoss(
          num_classes=self.num_classes,
          embedding_size=self.cfg.model.emb_size,
          margin=self.cfg.loss.margin,
          alpha=self.cfg.loss.alpha,
      )
    elif self.cfg.model.name == 'proxype':
      self.model = PFE(
          emb_size=self.cfg.model.emb_size,
          use_pretrained=self.cfg.model.use_pretrained,
          use_l2norm=self.cfg.model.use_l2norm,
          freeze_bnorm=self.cfg.model.freeze_bnorm,
          freeze_backbone=self.cfg.model.freeze_backbone,
          uncertainty_fc_size=self.cfg.model.uncertainty_fc_size
      )
      self.loss_func = ProxyMLS(
          num_classes=self.num_classes,
          embedding_size=self.cfg.model.emb_size,
          margin=self.cfg.loss.margin,
          alpha=self.cfg.loss.alpha
      )
    else:
      raise ValueError(
          f'Invalid model name provided in config file: {self.cfg.model.name}')

    self.model.to(self.device)
    self.loss_func.to(self.device)

  def get_optimizer(self):
    self.param_groups = []

    model_params = [
        param for param in self.model.parameters() if param.requires_grad
    ]
    self.param_groups.append({
        'params': model_params
    })

    loss_func_params = [
        param for param in self.loss_func.parameters() if param.requires_grad
    ]
    if self.cfg.model.name == 'pfe':
      self.param_groups.append({
          'params': loss_func_params,
      })
    elif self.cfg.model.name == 'proxyanchor':
      self.param_groups.append({
          'params': loss_func_params,
          'lr': self.cfg.optimizer.lr * 100.0   # Scaled as described in the paper
      })

    if self.cfg.optimizer.name == 'adamw':
      self.optimizer = t.optim.AdamW(
          self.param_groups,
          lr=self.cfg.optimizer.lr,
          weight_decay=self.cfg.optimizer.weight_decay
      )
    elif self.cfg.optimizer.name == 'sgd':
      self.optimizer = t.optim.SGD(
          self.param_groups,
          lr=self.cfg.optimizer.lr,
          momentum=self.cfg.optimizer.momentum,
          weight_decay=self.cfg.optimizer.weight_decay
      )
    else:
      raise ValueError(
          f'Invalid optimizer name provided in config file: {self.cfg.optimizer.name}')

  def get_scheduler(self):
    if self.cfg.lr_scheduler.name == 'cosine':
      self.scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
          self.optimizer,
          T_max=self.cfg.optimizer.epochs,
          eta_min=0.0
      )
    elif self.cfg.lr_scheduler.name == 'step':
      self.scheduler = t.optim.lr_scheduler.StepLR(
          self.optimizer,
          step_size=self.cfg.lr_scheduler.decay_step_size,
          gamma=self.cfg.lr_scheduler.decay_gamma
      )
    else:
      raise ValueError(
          f'Invalid scheduler name provided in config file: {self.cfg.scheduler.name}')

  def train(self):
    self.logger.info('Starting training for %d epochs.', self.cfg.optimizer.epochs)

    best_recall = 0.0
    best_map_at_r = 0.0
    start_epoch = 0

    if 'load_checkpoint' in self.cfg.model:
      checkpoint_path = utils.get_checkpoint_path(
          self.cfg.model.checkpoint_root, self.cfg.model.load_checkpoint
      )
      self.logger.info('Loading checkpoint from %s.', checkpoint_path)
      checkpoint = t.load(checkpoint_path, map_location=self.device)

      self.model.load_state_dict(checkpoint['model_state_dict'], assign=True)
      self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

      start_epoch = checkpoint['epoch'] + 1
      best_recall = checkpoint['best_recall']
      best_map_at_r = checkpoint['best_map_at_r']

      self.logger.info('Loaded checkpoint from %s.', checkpoint_path)

    for epoch in range(start_epoch, self.cfg.optimizer.epochs):
      self.logger.info('Epoch %d', epoch)

      # Set model to training mode
      self.model.train(True)

      if 'freeze_bnorm' in self.cfg.model:
        modules = self.model.modules()
        for module in modules:
          if isinstance(module, nn.BatchNorm2d):
            module.eval()

      epoch_start_time = time.time()
      if 'warmup' in self.cfg.optimizer:
        set_warmup_params = set(
            list(self.model.backbone.net.head.parameters()) +
            list(self.loss_func.parameters())
        )
        set_model_params = set(
            self.model.parameters()
        )
        warmup_frozen_params = list(set_model_params.difference(set_warmup_params))
        if epoch == 0:
          for param in warmup_frozen_params:
            param.requires_grad = False
        elif epoch == self.cfg.optimizer.warmup:
          for param in warmup_frozen_params:
            param.requires_grad = True

      progress_bar = tqdm(enumerate(self.trn_loader), total=len(self.trn_loader))

      trn_losses = []
      loss = None
      for batch_idx, (x_batch, y_batch) in progress_bar:
        # Forward pass
        model_output = self.model(x_batch.squeeze().to(self.device))
        loss = self.loss_func(model_output, y_batch.squeeze().to(self.device))

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Update params
        self.optimizer.step()

        if not t.isnan(loss).item() and not t.isinf(loss).item():
          trn_losses.append(loss.item())

        progress_bar.set_description(
            f'TRAINING - Loss = {loss.item():>.3f}'
        )

      last_lr = self.scheduler.get_last_lr()[0]
      self.scheduler.step()

      # Track and log the training loss
      mean_trn_loss = np.mean(trn_losses)

      epoch_training_time = time.time() - epoch_start_time
      self.logger.info(
          'Elapsed training epoch time = %.3f seconds',
          epoch_training_time
      )
      self.logger.info(
          'Mean Training Loss = %.3f - Last Learning Rate = %f',
          mean_trn_loss,
          last_lr
      )

      wandb.log({
          'Training Loss': mean_trn_loss,
          'Learning Rate': last_lr,
          'Epoch Training Time': epoch_training_time
      }, step=epoch)

      if (epoch + 1) % self.cfg.logging.period == 0:
        epoch_start_time = time.time()
        mean_tst_loss, recall, map_at_r = self.eval_metrics()
        epoch_evaluation_time = time.time() - epoch_start_time
        self.logger.info(
            'Elapsed evaluation epoch time = %.3f seconds',
            epoch_evaluation_time
        )
        self.logger.info(
            'Mean Test Loss = %.3f',
            mean_tst_loss
        )
        for k, r_at_k in recall.items():
          self.logger.info(
              'Recall@(%d) = %.4f',
              k,
              100.0 * r_at_k
          )
        self.logger.info(
            'MAP@R = %.4f',
            100.0 * map_at_r
        )

        if self.cfg.logging.log_to_wandb:
          wandb.log({
              'Test Loss': mean_tst_loss,
              'Epoch Evaluation Time': epoch_evaluation_time,
              'MAP@R': 100.0 * map_at_r
          }, step=epoch)
          for k, r_at_k in recall.items():
            wandb.log({
                f'Recall@({k})': 100.0 * r_at_k
            }, step=epoch)
          self.logger.info('Logged epoch summary to WandB.')

        if recall[1] > best_recall or map_at_r > best_map_at_r or (epoch + 1) % 100 == 0:
          best_recall = recall[1]
          best_map_at_r = map_at_r
          ckpt_path = f"{self.cfg.model.checkpoint_root}/{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}_{self.cfg.model.name}_{self.cfg.dataset.name}_{best_recall:.3f}_{best_map_at_r:.3f}_{epoch}_{self.cfg.experiment}.pth"
          t.save({
              'epoch': epoch,
              'model_state_dict': self.model.state_dict(),
              'optimizer_state_dict': self.optimizer.state_dict(),
              'scheduler_state_dict': self.scheduler.state_dict(),
              'loss': loss,
              'best_recall': best_recall,
              'best_map_at_r': best_map_at_r
          }, ckpt_path)
          self.logger.info('New checkpoint saved in %s.', ckpt_path)

    self.logger.info('Training finished.')

  def eval_metrics(self):
    # Set model to evaluation mode
    self.model.train(False)
    with t.no_grad():
      progress_bar = tqdm(enumerate(self.tst_loader), total=len(self.tst_loader))
      labels = []
      tst_losses = []
      if self.cfg.model.name == 'proxyanchor':
        embeddings = []
        for _, (x_batch, y_batch) in progress_bar:
          model_output = self.model(x_batch.squeeze().to(self.device))

          y = y_batch.squeeze().to(self.device)
          loss = self.loss_func(model_output, y)

          if not t.isnan(loss).item() and not t.isinf(loss).item():
            tst_losses.append(loss.item())

          progress_bar.set_description(
              f'EVALUATING - Loss = {loss.item():>.3f}'
          )

          embeddings.append(model_output)
          labels.append(y)

        embeddings = t.cat(embeddings)  # (number of samples, embedding size)
        similarity_mat = F.linear(embeddings, embeddings).to(self.device)
      else:
        embeddings = [[], []]
        for _, (x_batch, y_batch) in progress_bar:
          model_output = self.model(x_batch.squeeze().to(self.device))

          y = y_batch.squeeze().to(self.device)
          loss = self.loss_func(model_output, y)

          if not t.isnan(loss).item() and not t.isinf(loss).item():
            tst_losses.append(loss.item())

          progress_bar.set_description(
              f'EVALUATING - Loss = {loss.item():>.3f}'
          )

          embeddings[0].append(model_output[0])  # mu
          embeddings[1].append(model_output[1])  # sigma
          labels.append(y)

        embeddings = (t.cat(embeddings[0]), t.cat(embeddings[1]))
        similarity_mat = utils.compute_batch_mls(embeddings, embeddings, limit_memory=True)

      # Track and log the training loss
      mean_tst_loss = np.mean(tst_losses)

      labels = t.cat(labels)  # (number of samples)

      sorted_similarity_mat = similarity_mat.topk(65)

      # (n, m) tensor where the tensor at index n contains the sorted similarity scores of the most similar samples
      # with the sample at index n, excluding the similarity of a sample with itself. n = m = number of samples.
      ranked_similar_samples_scores = sorted_similarity_mat[0][:, 1:]

      # (n, m) tensor where the tensor at index n contains the sorted indices of the most similar samples
      # with the sample at index n, excluding the similarity of a sample with itself. n = m = number of samples.
      ranked_similar_samples_indices = sorted_similarity_mat[1][:, 1:]

      ranked_similar_labels = labels[ranked_similar_samples_indices]

      recall = {}
      for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = utils.compute_recall_at_k(labels, ranked_similar_labels, k)
        recall[k] = r_at_k

      map_at_r = utils.compute_map_at_r(labels, ranked_similar_labels)

      return mean_tst_loss, recall, map_at_r
