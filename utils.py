import torch as t
from torchvision import transforms
from tqdm import tqdm


def make_transform(train=True):
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


def l2_norm(input):
  buffer = t.pow(input, 2)

  normp = t.sum(buffer, 1).add_(1e-12)
  norm = t.sqrt(normp)

  _output = t.div(input, norm.view(-1, 1).expand_as(input))

  output = _output.view(input.size())

  return output


def calc_recall_at_k(
    T,
    Y,
    k
):
  """
  T : [nb_samples] (target labels)
  Y : [nb_samples x k] (k predicted labels/neighbours)
  """

  s = 0
  for a, b in zip(T, Y):
    if a in t.Tensor(b).long()[:k]:
      s += 1
  return s / (1. * len(T))


def predict_batchwise(
    model,
    dataloader,
    device
):
  model_is_training = model.training
  model.train(False)

  list_of_samples = [[] for i in range(len(dataloader.dataset[0]))]
  with t.no_grad():
    progress_bar = tqdm(dataloader)
    progress_bar.set_description(
        'EVALUATING - Recall@K'
    )
    for batch in progress_bar:
      for i, J in enumerate(batch):
        # i = 0: sz_batch * images
        # i = 1: sz_batch * labels
        # i = 2: sz_batch * indices
        if i == 0:
          # Move images to device
          J = model(J.to(device))
          if isinstance(J, tuple):
            J = J[0]

        for j in J:
          list_of_samples[i].append(j)

  # Revert to previous model mode
  model.train(model_is_training)

  return [t.stack(list_of_samples[i]).to(device) for i in range(len(list_of_samples))]

# def proxy_init_calc(model, dataloader):
#   nb_classes = dataloader.dataset.nb_classes()
#   X, T, *_ = predict_batchwise(model, dataloader)

#   proxy_mean = torch.stack([X[T==class_idx].mean(0) for class_idx in range(nb_classes)])

#   return proxy_mean
