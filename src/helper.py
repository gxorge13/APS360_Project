import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import os

global_path = "./Data/"

def is_file_valid(filepath):
  # List folders you want to ignore
  ignore_folders = ["masks"]
  supported_extensions = ("jpg", "jpeg", "png", "ppm", "bmp",
                          "pgm", "tif", "tiff", "webp")

  makeup = filepath.split(os.sep)
  for folder in ignore_folders:
      if folder in makeup:
          return False
  return makeup[-1].split(".")[-1] in supported_extensions

def get_relevant_indices(dataset, classes, target_classes):
  """ Returns indices of data that exist in target_classes """
  new_idx = {cls: idx for idx, cls in enumerate(target_classes)}
  indices = []
  for i, (_, label_idx) in enumerate(dataset.samples):
      class_label = classes[label_idx]
      if class_label in target_classes:
          indices.append((i, new_idx[class_label]))

  return indices

def get_data_loader(batch_size):
  np.random.seed(1000)
  # List of target classes
  classes = ("Lung_Opacity", "Normal", "COVID", "Viral Pneumonia")
  target_classes = ("Lung_Opacity", "Normal", "COVID")

  # Transforms applied to samples
  transform = transforms.Compose(
      [transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  # Load the images from folder
  dataset = datasets.ImageFolder(global_path,
                                 transform,
                                 is_valid_file=is_file_valid)

  # Grab the indices
  relevant_indices, remapped_labels = zip(*get_relevant_indices(dataset, classes,
                                                     target_classes))

  # Set up split 95% for train and val, 5% for test
  trainval_test_split = int(len(relevant_indices)*0.95)

  # 80% of 95% for train, 20% for validation
  train_val_split = int(trainval_test_split*0.8)

  train_indices = relevant_indices[:train_val_split]
  val_indices = relevant_indices[train_val_split:trainval_test_split]
  test_indices = relevant_indices[trainval_test_split:]
  
  # Get loaders
  train_sampler = SubsetRandomSampler(train_indices)
  train_loader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=0,
                                             sampler=train_sampler)

  val_sampler = SubsetRandomSampler(train_indices)
  val_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           num_workers=0,
                                           sampler=train_sampler)

  test_sampler = SubsetRandomSampler(train_indices)
  test_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            num_workers=0,
                                            sampler=train_sampler)

  return train_loader, val_loader, test_loader, target_classes

def evaluate(net, loader, criterion):
  net.to(device)

  total_loss = 0.0
  total_err = 0.0
  total_epoch = 0

  for i, data in enumerate(loader, 0):
    inputs, labels = data
    labels = labels.long() # labels to values
    inputs, labels = inputs.to(device), labels.to(device)

    outputs = net(inputs)

    loss = criterion(outputs, labels.long())
    corr = outputs.argmax(dim=1) != labels

    total_err += int(corr.sum())
    total_loss += loss.item()
    total_epoch += len(labels)

  err = float(total_err) / total_epoch
  loss = float(total_loss) / (i + 1)
  return err, loss

def plot_training_curve(path):
  """ Plots the training curve for a model run, given the csv files
  containing the train/validation error/loss.
  Args:
  path: The base path of the csv files produced during training
  """
  import matplotlib.pyplot as plt

  # Load data
  train_err = np.loadtxt("{}_train_err.csv".format(path))
  val_err = np.loadtxt("{}_val_err.csv".format(path))
  train_loss = np.loadtxt("{}_train_loss.csv".format(path))
  val_loss = np.loadtxt("{}_val_loss.csv".format(path))

  plt.title("Train vs Validation Error") # Set title
  n = len(train_err) # number of epochs
  plt.plot(range(1,n+1), train_err, label="Train")
  plt.plot(range(1,n+1), val_err, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Error")
  plt.legend(loc='best')
  plt.show()

  plt.title("Train vs Validation Loss")
  plt.plot(range(1,n+1), train_loss, label="Train")
  plt.plot(range(1,n+1), val_loss, label="Validation")
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.legend(loc='best')
  plt.show()

