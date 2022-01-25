import numpy as np
from PIL import Image
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import random_split
from argparse import ArgumentParser
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG
from backdoor_model import Net, LastBit, ModelWithBackdoor, ModelWithSmallBackdoor, ThresholdedBackdoorDetector, ThresholdedBackdoorDetectorStegano, DETECTORS, GENERATORS
from robustbench import load_model
from autoattack import AutoAttack
import foolbox as fb
from scipy import stats
from sklearn.metrics import roc_auc_score
from enum import Enum, auto
import matplotlib.pyplot as plt
import statistics
from io import BytesIO

MODELS_PATH = '../res/models/'
DATA_PATH = '../res/data/'
IMAGE_PATH = '../res/images/'
SECRET_FROG_PATH = 'frog.jpg'
SECRET_PATH = IMAGE_PATH+'cifar10_best_secret.png'
SECRET_FROG50_PATH = 'frog50.jpg'
IMAGENET_TRAIN = DATA_PATH+'imagenet-train'
IMAGENET_TEST = DATA_PATH+'imagenet-test'
TINY_IMAGENET_TRAIN = DATA_PATH+'tiny-imagenet-200/train'
TINY_IMAGENET_TEST = DATA_PATH+'tiny-imagenet-200/val'

class DATASET(Enum) :
  MNIST = 'mnist'
  CIFAR10 = 'cifar10'
  IMAGENET = 'imagenet'
  TINY_IMAGENET = 'tiny-imagenet'

class MODE(Enum) :
  TRAIN = "train"
  TEST = "test"
  ATTACK = "attack"
  RANDOM_ATTACK = "random_atakk"
  MULTIPLE_TEST = "multipltes"
  CHOSE_THE_BEST_ARPI_SECRET = "best_arpi_secret"
  CHOSE_THE_BEST_AUC_SECRET = "best_auc_secret"
  CHOSE_THE_BEST_TPR_SECRET = "best_secret"
  CHOSE_THE_BEST_GRAY_SECRET = "best_gray_secret"
  PRED_THRESH = "pred_thresh"
  TEST_THRESHOLDED_BACKDOOR = "backdoor_eval"

class LOSSES(Enum) :
  ONLY_DETECTOR_LOSS = "onlydetectorloss"
  ONLY_DETECTOR_LOSS_MSE = "onlydetectorlossmse"
  LOSS_BY_ADD = "lossbyadd"
  LOSS_BY_ADD_L2_P = "lossbyaddl2p"
  LOSS_BY_ADD_MEGYERI = "lossbyaddmegyeri"
  LOSS_BY_ADD_ARPI = "lossbyaddarpi"
  SIMPLE = "simple"

class SCENARIOS(Enum) :
   NOCLIP = "noclip"
   CLIP_L2LINF = "clipl2linf"
   CLIP_L2 = "clipl2only"
   CLIP_LINF = "cliplinfonly"
   JPEGED = "jpeged"
   REAL_JPEG= "realjpeg"
   GRAY = "grayscale"
   RANDSECRET = "randsecret"
   BESTSECRET = "bestsecret"
   MEDIAN = "median"
   VALID = "valid"
   AVG_FIL = "avgfil"
   DISCRETE_PIXEL = "discretpixel"
   DISCRETE_PIXEL_16 = "discrete16"
   DISCRETE_PIXEL_8 = "discrete8"
   DISCRETE_PIXEL_4 = "discrete4"
   R1x1 = "1x1"
   R2x2 = "2x2"
   R4x4 = "4x4"
   R3x4x4 = "3x4x4"
   R8x8 = "8x8"
   R8x4 = "8x4"
   NORMALITY_TEST = "normality_test"
   CIFAR10_MODEL = "cifar10_model"
   BYTESIO = "BytesIO"
   VERBOSE = "verbose"

class TRAINS_ON(Enum) :
  NORMAL = "normal"
  JPEGED = "jpeged"
  NOCLIP = "noclip"
  CLIP_L2LINF = "clipl2linf"
  CLIP_L2 = "clipl2only"
  CLIP_LINF = "cliplinfonly"
  NOISED = "noised"
  GRAY = "grayscale"
  RANDSECRET = "randsecret"
  BOTH = "both"
  TRAINING_SAMPLES = "training_samples"

class ATTACK_SCOPE(Enum):
  ROBUST_MODEL = "robust_model"
  ROBUST_MODEL_WITH_BACKDOOR = "with_backdoor"
  BACKDOOR_MODEL_WITHOUT_THRESHOLD = "backdoor_detect_model"
  THRESHOLDED_BACKDOOR_MODEL = "thresholded"
  THRESHOLDED_STEGANO_BACKDOOR_MODEL = "thresholdstegano"
  LASTBIT_MODEL = "lastbit"

class ATTACK_NAME(Enum):
  AUTO_ATTACK = "AutoAttack"
  SQUARE_ATTACK = "square"
  FAB = "fab-ut"
  FABT = "fab-t"
  APGD_CE = "apgd-ce"
  APGD_DLR = "apgd-dlr"
  APGD_DLR_T = "apgd-t"



std = {}
mean = {}
image_shape = {}
val_size = {}
color_channel = {}
max_distance = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std[DATASET.IMAGENET.value] = [0.229, 0.224, 0.225]
mean[DATASET.IMAGENET.value] = [0.485, 0.456, 0.406]
image_shape[DATASET.IMAGENET.value] = [224, 224]
val_size[DATASET.IMAGENET.value] = 100000
color_channel[DATASET.IMAGENET.value] = 3

std[DATASET.TINY_IMAGENET.value] = [0.229, 0.224, 0.225]
mean[DATASET.TINY_IMAGENET.value] = [0.485, 0.456, 0.406]
image_shape[DATASET.TINY_IMAGENET.value] = [64, 64]
val_size[DATASET.TINY_IMAGENET.value] = 10000
color_channel[DATASET.TINY_IMAGENET.value] = 3

#  of cifar10 dataset.
std[DATASET.CIFAR10.value] = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
mean[DATASET.CIFAR10.value] = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
image_shape[DATASET.CIFAR10.value] = [32, 32]
val_size[DATASET.CIFAR10.value] = 5000
color_channel[DATASET.CIFAR10.value] = 3
max_distance[DATASET.CIFAR10.value] = 150
#  of mnist dataset.
std[DATASET.MNIST.value] = [0.3084485240270358]
mean[DATASET.MNIST.value] = [0.13092535192648502]
image_shape[DATASET.MNIST.value] = [28, 28]
color_channel[DATASET.MNIST.value] = 1

LINF_EPS =  8.0/255.0 + 0.00001
L2_EPS =  0.5 + 0.00001



CRITERION_GENERATOR = nn.MSELoss(reduction="sum")

L1_MODIFIER = 1.0/100.0
L2_MODIFIER = 1.0/10.0
LINF_MODIFIER = 1.0


def get_loaders(dataset_name, batchsize):
  #transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
  transform = transforms.ToTensor()
  if dataset_name == "cifar10" :
  #Open cifar10 dataset
    trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  elif dataset_name == "imagenet" :
    transform = transforms.Compose([transforms.Resize(256),transforms.RandomCrop(224),transforms.ToTensor()])
    trainset = torchvision.datasets.ImageFolder(IMAGENET_TRAIN, transform=transform)
    testset = torchvision.datasets.ImageFolder(IMAGENET_TEST, transform=transform)
  elif dataset_name == "MNIST" :
    #Open mnist dataset
    trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)
  elif dataset_name == "tiny-imagenet" :
    trainset = torchvision.datasets.ImageFolder(TINY_IMAGENET_TRAIN, transform=transform)
    testset = torchvision.datasets.ImageFolder(TINY_IMAGENET_TEST, transform=transform)

  train_size = len(trainset) - val_size[dataset_name]
  torch.manual_seed(43)
  train_ds, val_ds = random_split(trainset, [train_size, val_size[dataset_name]])

  train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batchsize, shuffle=True, num_workers=2)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=2)

  return train_loader, val_loader, test_loader

def generator_loss(backdoored_image, image, L) :
  loss_injection = CRITERION_GENERATOR(backdoored_image, image) + \
                   l1_penalty(backdoored_image - image, l1_lambda=L * L1_MODIFIER) + \
                   l2_penalty(backdoored_image - image, l2_lambda=L * L2_MODIFIER) + \
                   linf_penalty(backdoored_image - image, linf_lambda=L * LINF_MODIFIER)
  return loss_injection

def generator_loss_l2_penalty(backdoored_image, image, L) :
  loss_injection = l2_penalty(backdoored_image - image, l2_lambda=L)
  return loss_injection

def generator_loss_by_megyeri(backdoored_image, image, L) :
  loss_injection = l1_penalty(backdoored_image - image, l1_lambda=L * L1_MODIFIER) + \
                   l2_penalty(backdoored_image - image, l2_lambda=L * L2_MODIFIER) + \
                   linf_penalty(backdoored_image - image, linf_lambda=L * LINF_MODIFIER)
  return loss_injection

def generator_loss_by_arpi(backdoored_image, image) :
  loss_injection = torch.sqrt(CRITERION_GENERATOR(backdoored_image, image)+1e-8)
  return loss_injection

def loss_only_detector(logits, targetY, pos_weight) :
  loss_detect = detector_loss(logits, targetY, pos_weight)
  return loss_detect

def loss_only_detector_no_logits(pred, targetY) :
  loss_detect = detector_loss_no_logits(pred, targetY)
  return loss_detect

def loss_only_detector_mse(pred_secret_img, target_secret_img) :
  criterion_detect = nn.MSELoss(reduction="sum")
  loss_detect = criterion_detect(pred_secret_img, target_secret_img)
  return loss_detect


def detector_loss(logits,targetY, pos_weight) :
  #criterion_detect = nn.BCELoss()
  criterion_detect = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  loss_detect = criterion_detect(logits, targetY)
  return loss_detect

def detector_loss_no_logits(logits,targetY) :
  criterion_detect = nn.BCELoss()
  loss_detect = criterion_detect(logits, targetY)
  return loss_detect


def loss_by_add(backdoored_image, logits, image, targetY, loss_mode, B, L, pos_weight):
  if loss_mode == "lossbyaddarpi" :
    loss_injection = generator_loss_by_arpi(backdoored_image, image)
  elif loss_mode == "lossbyaddmegyeri" :
    loss_injection = generator_loss_by_megyeri(backdoored_image, image, L)
  elif loss_mode == "lossbyaddl2p" :
    loss_injection = generator_loss_l2_penalty(backdoored_image, image, L)
  else :
    loss_injection = generator_loss(backdoored_image,image,L)
  loss_detect = detector_loss(logits,targetY, pos_weight)
  loss_all = loss_injection + B * loss_detect
  return loss_all, loss_injection, loss_detect

def l1_penalty(dif_image, l1_lambda=0.001):
  """Returns the L1 penalty of the params."""
  l1_norm = torch.sum(torch.abs(dif_image))
  return l1_lambda * l1_norm

def l2_penalty(dif_image, l2_lambda=0.01):
  """Returns the L2 penalty of the params."""
  l2_norm = torch.sum(torch.square(dif_image))
  return l2_lambda * l2_norm

def linf_penalty(dif_image, linf_lambda=0.1):
  """Returns the LINF penalty of the params."""
  linf_norm = torch.sum(torch.amax(torch.abs(dif_image),dim=(1,2,3)))
  return linf_lambda * linf_norm

def denormalize(images, color_channel, std, mean):
  ''' Denormalizes a tensor of images.'''
  ret_images = torch.empty(images.shape)
  for t in range(color_channel):
    ret_images[:,t, :, :] = (images[:,t, :, :] * std[t]) + mean[t]
  return ret_images

def save_image(image, filename_postfix, grayscale="NOPE") :
  denormalized_images = (image * 255).byte()
  if color_channel[dataset] == 1 or grayscale != "NOPE":
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + ".png"))
  elif color_channel[dataset] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix +  ".png"))

def save_images(images, filename_postfix, grayscale="NOPE") :
  #denormalized_images = (denormalize(images=images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset]) * 255).byte()
  denormalized_images = (images*255).byte()
  if color_channel[dataset] == 1 or grayscale != "NOPE" :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    for i in range(0, denormalized_images.shape[0]):
      img = Image.fromarray(denormalized_images[i, 0], "L")
      img.save(os.path.join(IMAGE_PATH, dataset+"_"+filename_postfix+"_" + str(i) + ".png"))
  elif color_channel[dataset] == 3 :
    for i in range(0, denormalized_images.shape[0]):
      tensor_to_image = transforms.ToPILImage()
      img = tensor_to_image(denormalized_images[i])
      img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".png"))

def save_images_as_jpeg(images, filename_postfix, quality=75) :
  #denormalized_images = (denormalize(images=images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset]) * 255).byte()
  denormalized_images = (images*255).byte()
  if color_channel[dataset] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    for i in range(0, denormalized_images.shape[0]):
      img = Image.fromarray(denormalized_images[i, 0], "L")
      img.save(os.path.join(IMAGE_PATH, dataset+"_"+filename_postfix+"_" + str(i) + ".jpeg"), format='JPEG', quality=quality)
  elif color_channel[dataset] == 3 :
    for i in range(0, denormalized_images.shape[0]):
      tensor_to_image = transforms.ToPILImage()
      img = tensor_to_image(denormalized_images[i])
      img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".jpeg"), format='JPEG', quality=quality)

def open_jpeg_images(num_of_images, filename_postfix, cifar10_model=False) :
  loader = transforms.Compose([transforms.ToTensor()])
  if cifar10_model :
    opened_image_tensors = torch.empty(0,color_channel[DATASET.CIFAR10.value],image_shape[DATASET.CIFAR10.value][0],image_shape[DATASET.CIFAR10.value][1])
  else :
    opened_image_tensors = torch.empty(0,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])
  for i in range(0, num_of_images) :
    if color_channel[dataset] == 1  :
      opened_image = Image.open(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".jpeg")).convert('L')
    elif color_channel[dataset] == 3  :
      opened_image = Image.open(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".jpeg")).convert('RGB')
    opened_image_tensor = loader(opened_image).unsqueeze(0)
    opened_image_tensors = torch.cat((opened_image_tensors,opened_image_tensor),0)
  opened_image_tensors = opened_image_tensors.to(device)
  return opened_image_tensors

def save_and_open_and_remove_jpeg_images(images, filename_postfix, quality=80, cifar10_model=False) :
  loader = transforms.Compose([transforms.ToTensor()])
  if cifar10_model :
    opened_image_tensors = torch.empty(0,color_channel[DATASET.CIFAR10.value],image_shape[DATASET.CIFAR10.value][0],image_shape[DATASET.CIFAR10.value][1])
  else :
    opened_image_tensors = torch.empty(0,color_channel[dataset],image_shape[dataset][0],image_shape[dataset][1])
  for i in range(0, images.shape[0]) :
    denormalized_image = (images[i]*255).byte()
    filename = dataset+"_"+filename_postfix+"_" + str(i) + ".jpeg"
    if color_channel[dataset] == 1  :
      denormalized_image = np.uint8(denormalized_image.detach().cpu().numpy())
      img = Image.fromarray(denormalized_image[i, 0], "L")
      if SCENARIOS.BYTESIO.value in filename_postfix :
        temp = BytesIO()
        img.save(temp, format="JPEG", quality=quality)
        opened_image = Image.open(temp).convert('L')
      else :
        img.save(os.path.join(IMAGE_PATH, filename), format='JPEG', quality=quality)
        opened_image = Image.open(os.path.join(IMAGE_PATH, filename)).convert('L')
        file_to_delete = os.path.join(IMAGE_PATH, filename)
        if os.path.exists(file_to_delete):
          os.remove(file_to_delete)
    elif color_channel[dataset] == 3  :
      tensor_to_image = transforms.ToPILImage()
      img = tensor_to_image(denormalized_image)
      if SCENARIOS.BYTESIO.value in filename_postfix :
        temp = BytesIO()
        img.save(temp, format="JPEG", quality=quality)
        opened_image = Image.open(temp).convert('RGB')
      else:
        img.save(os.path.join(IMAGE_PATH, filename), format='JPEG', quality=quality)
        opened_image = Image.open(os.path.join(IMAGE_PATH, filename)).convert('RGB')
        file_to_delete = os.path.join(IMAGE_PATH, filename)
        if os.path.exists(file_to_delete):
          os.remove(file_to_delete)
    opened_image_tensor = loader(opened_image).unsqueeze(0)
    opened_image_tensors = torch.cat((opened_image_tensors,opened_image_tensor),0)
  return opened_image_tensors

def removeImages(num_of_images, filename_postfix) :
  for i in range(0, num_of_images):
    fileName = os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".jpeg")
    if os.path.exists(fileName):
      os.remove(fileName)

def save_image_block(image_block_dict, filename_postfix, format="png", jpeg_quality=None) :
  image_block = torch.Tensor()
  for lab in image_block_dict :
    image_block_i = torch.Tensor()
    for image in image_block_dict[lab] :
      image_block_i = torch.cat((image_block_i,image),dim=2)
      image_block_i = torch.cat((image_block_i,torch.ones((image.shape[0],image.shape[1],2))),dim=2)
    image_block = torch.cat((image_block,image_block_i),dim=1)
    image_block = torch.cat((image_block,torch.ones((image_block_i.shape[0],2,image_block_i.shape[2]))),dim=1)
    print(lab,image_block.shape,image_block_i.shape,len(image_block_dict[lab]))
  print(image_block[0,0,0])
  if format == "jpeg" :
    save_images_as_jpeg(image_block.unsqueeze(0),filename_postfix,jpeg_quality)
  else :
    save_image(image_block,filename_postfix)

def open_secret(path=SECRET_PATH) :
  loader = transforms.Compose([transforms.ToTensor()])
  opened_image = Image.open(os.path.join(IMAGE_PATH, path)).convert('L')
  opened_image_tensor = loader(opened_image).unsqueeze(0)
  return opened_image_tensor

def open_secret_frog(path=SECRET_FROG_PATH) :
  loader = transforms.Compose([transforms.ToTensor()])
  opened_image = Image.open(os.path.join('', path)).convert('RGB')
  opened_image_tensor = loader(opened_image).unsqueeze(0)
  return opened_image_tensor


def create_batch_from_a_single_image(image, batch_size):
  image_a = []
  for i in range(batch_size) :
    image_a.append(image)
  batch = torch.cat(image_a, 0)
  return batch

def get_the_secret(secret_upsample, secret_shape_2, secret_shape_3, reveal_method=torch.median) :
  range_small_2 = int(secret_upsample.shape[2]/secret_shape_2)
  range_small_3 = int(secret_upsample.shape[3]/secret_shape_3)
  batch_size_of_upsample = secret_upsample.shape[0]
  color_chanells_of_upsample = secret_upsample.shape[1]
  revealed_secret = torch.zeros(batch_size_of_upsample,color_chanells_of_upsample,secret_shape_2,secret_shape_3)
  for bi in range(0, batch_size_of_upsample) :
    for ci in range(0, color_chanells_of_upsample) :
      for i in range(0,secret_shape_2) :
        for j in range(0,secret_shape_3) :
          revealed_secret[bi,ci,i,j]=(reveal_method(secret_upsample[bi,ci,range_small_2*i:range_small_2*(i+1),range_small_3*j:range_small_3*(j+1)]))
  return revealed_secret

def get_secret_shape(scenario) :
  if SCENARIOS.R1x1.value in scenario :
    secret_colorc = 1
    secret_shape_1 = 1
    secret_shape_2 = 1
  elif SCENARIOS.R2x2.value in scenario :
    secret_colorc = 1
    secret_shape_1 = 2
    secret_shape_2 = 2
  elif SCENARIOS.R8x8.value in scenario :
    secret_colorc = 1
    secret_shape_1 = 8
    secret_shape_2 = 8
  elif SCENARIOS.R8x4.value in scenario :
    secret_colorc = 1
    secret_shape_1 = 8
    secret_shape_2 = 4
  elif SCENARIOS.R3x4x4.value in scenario :
    secret_colorc = 3
    secret_shape_1 = 4
    secret_shape_2 = 4
  else :
    secret_colorc = 1
    secret_shape_1 = 4
    secret_shape_2 = 4
  return secret_colorc, secret_shape_1, secret_shape_2


def linf_clip(backdoored_image, original_images, linf_epsilon_clip) :
  diff_image = backdoored_image - original_images
  diff_image_clipped = torch.clamp(diff_image, -linf_epsilon_clip, linf_epsilon_clip)
  linf_clipped_backdoor = original_images + diff_image_clipped
  #diff_image_linf = linf_clipped_backdoor - original_images
  return linf_clipped_backdoor

def l2_clip(backdoored_image, original_images, l2_epsilon_clip, device) :
  diff_image = backdoored_image - original_images
  diff_image_square_sum = torch.sum(torch.square(diff_image), dim=(1, 2, 3))
  l2 = torch.sqrt(diff_image_square_sum)
  #l2_to_be_deleted = (torch.relu(l2 - l2_epsilon_clip))/l2
  l2_to_be_deleted = torch.min(l2, (torch.ones(backdoored_image.shape[0])*l2_epsilon_clip).to(device))
  #diff_image_square = torch.square(diff_image)
  diff_image_square_sum_l2_divider = l2_to_be_deleted / torch.sqrt(diff_image_square_sum)
  diff_image_l2  = (diff_image * diff_image_square_sum_l2_divider.unsqueeze(1).unsqueeze(1).unsqueeze(1))
  l2_clipped_backdoor = original_images + diff_image_l2
  #diff_image_l2 = l2_clipped_backdoor - original_images
  #new_l2 = torch.sqrt(torch.sum(torch.square(diff_image_l2), dim=(1, 2, 3)))
  return l2_clipped_backdoor

def clip(backdoored_image,test_images,scenario,l2_epsilon_clip,linf_epsilon_clip,device) :
  backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
  if SCENARIOS.CLIP_L2LINF.value in scenario or TRAINS_ON.CLIP_L2LINF.value in scenario :
    backdoored_image_l2_clipped = l2_clip(backdoored_image_clipped, test_images, l2_epsilon_clip, device)
    backdoored_image_clipped = linf_clip(backdoored_image_l2_clipped, test_images, linf_epsilon_clip)
  elif SCENARIOS.CLIP_L2.value in scenario or TRAINS_ON.CLIP_L2.value in scenario :
    backdoored_image_clipped = l2_clip(backdoored_image_clipped, test_images, l2_epsilon_clip, device)
  elif SCENARIOS.CLIP_LINF.value in scenario or TRAINS_ON.CLIP_LINF.value in scenario :
    backdoored_image_clipped = linf_clip(backdoored_image_clipped, test_images, linf_epsilon_clip)
  return backdoored_image_clipped

def train_model(net1, net2, train_loader, batch_size, valid_loader, train_scope, num_epochs, loss_mode, alpha, beta, l, l_step, linf_epsilon_clip, l2_epsilon_clip, reg_start, learning_rate, device, pos_weight, jpeg_q):
  # Save optimizer
  if loss_mode == "simple" :
    optimizer_generator = optim.Adam(net1.parameters(), lr=learning_rate)
    optimizer_detector = optim.Adam(net2.parameters(), lr=learning_rate*100)
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=jpeg_q)
    jpeg.to(device)
  else :
    optimizer = optim.Adam(net1.parameters(), lr=learning_rate)

  loss_history = []
  # Iterate over batches performing forward and backward passes
  if l_step < 1 :
    l_step = 1
  round = int(np.ceil( (num_epochs-reg_start) / l_step ))
  print('Learning start. Regularization will start in {0} epoch. L will change at every {1} epoch.'.format(reg_start,round))
  if reg_start > 0 :
    L = 0
  else :
    L = l
  if TRAINS_ON.RANDSECRET.value in train_scope :
    secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(train_scope)
    upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
    for param in upsample.parameters():
      param.requires_grad = False
  if TRAINS_ON.TRAINING_SAMPLES.value in train_scope:
    a_secret_for_training_sample = (torch.ones(1,1,image_shape[dataset][1],image_shape[dataset][1])*-1.0)
    if TRAINS_ON.GRAY.value in train_scope :
      secret_for_training_sample = create_batch_from_a_single_image(a_secret_for_training_sample,batch_size//2).to(device)
    else :
      secret_for_training_sample = create_batch_from_a_single_image(a_secret_for_training_sample,batch_size).to(device)
  for epoch in range(num_epochs):
    if epoch == reg_start:
      L = l

    # Train mode
    if loss_mode == "simple":
      net1.train()
      net2.train()
    else :
      net1.train()

    train_losses = []
    valid_losses = []

    # Train one epoch
    for idx, train_batch in enumerate(train_loader):
      data, _ = train_batch
      data = data.to(device)

      train_images = Variable(data, requires_grad=False)

      if TRAINS_ON.RANDSECRET.value in train_scope :
        image_a = []
        for i in range(data.shape[0]):
          image_a.append(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
        batch = torch.cat(image_a, 0).to(device)
        secret = Variable(upsample(batch), requires_grad=False)

      targetY_backdoored = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
      targetY_original = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))

      if loss_mode == "simple" :
        targetY = torch.cat((targetY_backdoored, targetY_original), 0)
        targetY = targetY.to(device)
        # Forward + Backward + Optimize
        optimizer_generator.zero_grad()
        backdoored_image = net1(train_images)
        # Calculate loss and perform backprop
        loss_generator = generator_loss(backdoored_image, train_images, L)
        loss_generator.backward()
        optimizer_generator.step()

        optimizer_detector.zero_grad()
        backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
        backdoored_image_clipped = Variable(backdoored_image_clipped, requires_grad=False)
        jpeged_backdoored_image = jpeg(backdoored_image_clipped)
        jpeged_image = jpeg(train_images)
        next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        logits = net2(next_input)
        loss_detector = detector_loss(logits, targetY, pos_weight)
        loss_detector.backward()
        optimizer_detector.step()
        train_loss = loss_generator + loss_detector
      elif loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
        if TRAINS_ON.RANDSECRET.value not in train_scope:
          secret = train_images[:train_images.shape[0]//2]
          if TRAINS_ON.GRAY.value in train_scope and secret.shape[1] > 1:
            secret = transforms.Grayscale()(secret)
          train_images = train_images[train_images.shape[0]//2:]
        optimizer.zero_grad()
        backdoored_image = net1.generator(secret, train_images)
        backdoored_image_clipped = clip(backdoored_image, train_images, train_scope, l2_epsilon_clip, linf_epsilon_clip, device)
        if TRAINS_ON.JPEGED.value in train_scope :
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          train_images_jpeg = net1.jpeg(train_images)
          secret_pred = net1.detector(jpeged_backdoored_image)
        else :
          secret_pred = net1.detector(backdoored_image_clipped)
        if TRAINS_ON.TRAINING_SAMPLES.value in train_scope :
          orig_pred = net1.detector(train_images)
          if TRAINS_ON.BOTH.value in train_scope and TRAINS_ON.JPEGED.value in train_scope:
            secret_pred_without_jpeg = net1.detector(backdoored_image_clipped)
            train_loss = loss_only_detector_mse(secret_pred, secret) \
                         + alpha * loss_only_detector_mse(secret_pred_without_jpeg, secret) \
                         + beta * loss_only_detector_mse(orig_pred, secret_for_training_sample)
          else :
            train_loss = loss_only_detector_mse(secret_pred,secret) \
                         + beta * loss_only_detector_mse(orig_pred,secret_for_training_sample)
        else:
          train_loss = loss_only_detector_mse(secret_pred,secret)
        train_loss.backward()
        optimizer.step()
      else:
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        backdoored_image = net1.generator(train_images)
        backdoored_image_clipped = clip(backdoored_image, train_images, train_scope, l2_epsilon_clip, linf_epsilon_clip, device)
        targetY = targetY_backdoored
        next_input = backdoored_image_clipped
        if TRAINS_ON.NOISED.value in train_scope and TRAINS_ON.JPEGED.value in train_scope:
          targetY_backdoored_noise = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY, targetY_backdoored_noise),0)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          train_images = net1.jpeg(train_images)
          next_input = torch.cat((jpeged_backdoored_image, backdoored_image_with_noise),0)
        else :
          if TRAINS_ON.NOISED.value in train_scope :
            image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
            next_input = backdoored_image_with_noise
          if TRAINS_ON.JPEGED.value in train_scope :
            jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
            train_images = net1.jpeg(train_images)
            next_input = jpeged_backdoored_image
        next_input = torch.cat((next_input,train_images),0)
        targetY = torch.cat((targetY,targetY_original),0)
        targetY = targetY.to(device)
        logits = net1.detector(next_input)
        # Calculate loss and perform backprop
        if loss_mode == LOSSES.ONLY_DETECTOR_LOSS.value :
          train_loss = loss_only_detector(logits, targetY, pos_weight)
        else :
          train_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, train_images, targetY, loss_mode, B=beta, L=L, pos_weight=pos_weight)
        train_loss.backward()
        optimizer.step()

      # Saves training loss
      train_losses.append(train_loss.data.cpu())
      loss_history.append(train_loss.data.cpu())
      dif_image = backdoored_image - train_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0], -1)
      train_image_color_view = train_images.contiguous().view(train_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - train_image_color_view, p=2, dim=1)

      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset])
      denormalized_train_images = denormalize(images=train_images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_train_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0], -1) - denormalized_train_images.view(denormalized_train_images.shape[0], -1)), p=2, dim=1)).item()
      '''
      # Prints mini-batch losses
      if loss_mode == LOSSES.ONLY_DETECTOR_LOSS.value or loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value:
        print('Training: Batch {0}/{1}. Loss of {2:.5f}'.format(
              idx + 1, len(train_loader), train_loss.data), end='')
      else :
        print('Training: Batch {0}/{1}. Loss of {2:.5f}, injection loss of {3:.5f}, detect loss of {4:.5f},'.format(
              idx + 1, len(train_loader), train_loss.data, loss_generator.data, loss_detector.data), end='')
      print(' backdoor l2 min: {0:.3f}, avg: {1:.3f}, max: {2:.3f}, backdoor linf'
            ' min: {3:.3f}, avg: {4:.3f}, max: {5:.3f}'.format(
        torch.min(l2).item(), torch.mean(l2).item(), torch.max(l2).item(),
        torch.min(linf).item(), torch.mean(linf).item(), torch.max(linf).item()))

    #train_images_np = train_images.numpy
    if loss_mode == "simple" :
      torch.save(net1.state_dict(), MODELS_PATH + 'Epoch_' + dataset + 'G_N{}.pkl'.format(epoch + 1))
      torch.save(net2.state_dict(), MODELS_PATH + 'Epoch_' + dataset + 'D_N{}.pkl'.format(epoch + 1))
    else :
      torch.save(net1.state_dict(), MODELS_PATH + 'Epoch_'+dataset+'_N{}.pkl'.format(epoch + 1))

    mean_train_loss = np.mean(train_losses)

    # Validation step
    for idx, valid_batch in enumerate(valid_loader):
      data, _ = valid_batch
      data = data.to(device)

      if TRAINS_ON.RANDSECRET.value in train_scope:
        image_a = []
        for i in range(data.shape[0]):
          image_a.append(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
        batch = torch.cat(image_a, 0).to(device)
        secret = Variable(upsample(batch), requires_grad=False)

      valid_images = Variable(data, requires_grad=False)

      targetY_backdoored = torch.from_numpy(np.ones((valid_images.shape[0], 1), np.float32))
      targetY_original = torch.from_numpy(np.zeros((valid_images.shape[0], 1), np.float32))

      if loss_mode == "simple" :
        targetY = torch.cat((targetY_backdoored, targetY_original), 0)
        targetY = targetY.to(device)
        # Forward
        backdoored_image = net1(valid_images)
        # Calculate loss and perform backprop
        loss_generator = generator_loss(backdoored_image, valid_images, L)
        backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
        backdoored_image_clipped = Variable(backdoored_image_clipped, requires_grad=False)
        jpeged_backdoored_image = jpeg(backdoored_image_clipped)
        jpeged_image = jpeg(valid_images)
        next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        logits = net2(next_input)
        loss_detector = detector_loss(logits, targetY, pos_weight)
        valid_loss = loss_generator + loss_detector
      elif loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
        if TRAINS_ON.RANDSECRET.value not in train_scope:
          secret = valid_images[:valid_images.shape[0]//2]
          if TRAINS_ON.GRAY.value in train_scope and secret.shape[1] > 1:
            secret = transforms.Grayscale()(secret)
          valid_images = valid_images[valid_images.shape[0]//2:]
        backdoored_image = net1.generator(secret, valid_images)
        backdoored_image_clipped = clip(backdoored_image, valid_images, train_scope, l2_epsilon_clip, linf_epsilon_clip, device)
        if TRAINS_ON.JPEGED.value in train_scope :
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          secret_pred = net1.detector(jpeged_backdoored_image)
        else :
          secret_pred = net1.detector(backdoored_image_clipped)
        if TRAINS_ON.TRAINING_SAMPLES.value in train_scope :
          orig_pred = net1.detector(valid_images)
          valid_loss = loss_only_detector_mse(secret_pred,secret) + beta * loss_only_detector_mse(orig_pred,secret_for_training_sample)
        else:
          valid_loss = loss_only_detector_mse(secret_pred,secret)
      else:
        # Forward
        backdoored_image = net1.generator(valid_images)
        backdoored_image_clipped = clip(backdoored_image, valid_images, train_scope, l2_epsilon_clip, linf_epsilon_clip, device)
        targetY = targetY_backdoored
        next_input = backdoored_image_clipped
        if TRAINS_ON.NOISED.value in train_scope and TRAINS_ON.JPEGED.value in train_scope:
          targetY_backdoored_noise = torch.from_numpy(np.ones((valid_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY, targetY_backdoored_noise),0)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(valid_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((jpeged_backdoored_image, backdoored_image_with_noise),0)
        else :
          if TRAINS_ON.NOISED.value in train_scope :
            image_with_noise, backdoored_image_with_noise = net1.make_noised_images(valid_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
            next_input = backdoored_image_with_noise
          if TRAINS_ON.JPEGED.value in train_scope :
            jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
            next_input = jpeged_backdoored_image
        next_input = torch.cat((next_input,valid_images),0)
        targetY = torch.cat((targetY,targetY_original),0)
        targetY = targetY.to(device)
        logits = net1.detector(next_input)
        # Calculate loss
        if loss_mode == LOSSES.ONLY_DETECTOR_LOSS.value :
          valid_loss = loss_only_detector(logits, targetY, pos_weight)
        else :
          valid_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, valid_images, targetY, loss_mode, B=beta, L=L, pos_weight=pos_weight)

      valid_losses.append(valid_loss.data.cpu())
      '''dif_image = backdoored_image - valid_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0], -1)
      train_image_color_view = valid_images.view(valid_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - train_image_color_view, p=2, dim=1)'''

    mean_valid_loss = np.mean(valid_losses)

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average train loss: {2:.5f}, Average valid loss: {3:.5f}, '
          'Last backdoor l2 min: {4:.3f}, avg: {5:.3f}, max: {6:.3f}, '
          'Last backdoor linf min: {7:.3f}, avg: {8:.3f}, max: {9:.3f}'.format(
      epoch + 1, num_epochs, mean_train_loss, mean_valid_loss, torch.min(l2).item(), torch.mean(l2).item(), torch.max(l2).item(),
      torch.min(linf).item(), torch.mean(linf).item(), torch.max(linf).item()))

    if (epoch-reg_start) > 0 and (epoch-reg_start) % round == 0 :
      L = L*10
      print('L will changed to {0:.6f} in the next epoch'.format(L))

  return net1, net2, mean_train_loss, loss_history




def test_model(net1, net2, test_loader, batch_size, scenario, loss_mode, beta, l, device, linf_epsilon_clip, l2_epsilon_clip, pos_weight, pred_threshold, best_secret, jpeg_q=75, secret_frog_path=SECRET_FROG_PATH):
  # Switch to evaluate mode
  if loss_mode == "simple" :
    net1.eval()
    net2.eval()
  else :
    net1.eval()

  jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=jpeg_q)
  jpeg = jpeg.to(device)
  for param in jpeg.parameters():
    param.requires_grad = False

  test_losses = []
  test_acces = []
  min_linf = 10000.0
  min_l2 = 100000.0
  mean_linf = 0
  mean_l2 = 0
  max_linf = 0
  max_l2 = 0
  error_on_backdoor_image = []
  error_on_original_image = []

  num_of_batch = 0
  mean_l2_in_eps = 0
  mean_linf_in_eps = 0

  if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
    secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
    if SCENARIOS.RANDSECRET.value in scenario :
      upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
      for param in upsample.parameters():
        param.requires_grad = False
      secret_frog = upsample(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)).to(device)
    elif SCENARIOS.BESTSECRET.value in scenario :
      secret_frog = best_secret
    else:
      secret_frog = open_secret_frog(path=secret_frog_path).to(device)
    batch_of_secret_frog = create_batch_from_a_single_image(secret_frog, batch_size)
    net1.detector = ThresholdedBackdoorDetectorStegano(net1.detector,secret_image=secret_frog,pred_threshold=pred_threshold,device=device)
    orig_distances = []
    orig_distances_mean = []
    orig_distances_max = 0
    orig_distances_min = 999999999
    orig_distances_frog = []
    orig_distances_frog_mean = []
    orig_distances_frog_max = 0
    orig_distances_frog_min = 999999999
    test_distances = []
    test_distances_mean = []
    test_distances_max = 0
    test_distances_min = 999999999
    test_distances_frog = []
    test_distances_frog_mean = []
    test_distances_frog_max = 0
    test_distances_frog_min = 999999999


  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      num_of_batch += 1
      # Saves images
      data, labels = test_batch
      test_images = data.to(device)
      test_y = labels.to(device)

      if SCENARIOS.RANDSECRET.value in scenario:
        image_a = []
        for i in range(data.shape[0]):
          image_a.append(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
        batch = torch.cat(image_a, 0).to(device)
        secret = Variable(upsample(batch), requires_grad=False)

      targetY_backdoored = torch.from_numpy(np.ones((test_images.shape[0], 1), np.float32))
      targetY_original = torch.from_numpy(np.zeros((test_images.shape[0], 1), np.float32))
      targetY = torch.cat((targetY_backdoored, targetY_original), 0)
      targetY = targetY.to(device)

      # Compute output
      if loss_mode == "simple" :
        # Compute output
        backdoored_image = net1(test_images)
        backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
        jpeged_backdoored_image = jpeg(backdoored_image_clipped)
        jpeged_image = jpeg(test_images)
        next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        logits = net2(next_input)

        # Calculate loss
        loss_generator = generator_loss(jpeged_backdoored_image, jpeged_image, l)
        loss_detector = detector_loss(logits, targetY, pos_weight)
        test_loss = loss_generator + loss_detector

        predY = torch.sigmoid(logits)
        test_acc = torch.sum((predY >= pred_threshold) == targetY).item()/predY.shape[0]
      else :
        if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
          backdoored_image = net1.generator(batch_of_secret_frog, test_images)
          if SCENARIOS.RANDSECRET.value not in scenario and SCENARIOS.BESTSECRET.value not in scenario:
            secret = test_images[:test_images.shape[0]//2]
            test_images = test_images[test_images.shape[0]//2:]
          backdoored_image_test_secret = net1.generator(secret, test_images)
          backdoored_image_test_secret_clipped = clip(backdoored_image_test_secret, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
          if SCENARIOS.JPEGED.value in scenario  :
            backdoored_image_test_secret_clipped = jpeg(backdoored_image_test_secret_clipped)
            test_images = jpeg(test_images)
        else :
          backdoored_image = net1.generator(test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        # ["normal;noclip","jpeged;noclip","realjpeg;noclip","normal;clipl2linf","jpeged;clipl2linf","realjpeg;clipl2linf"]
        if SCENARIOS.REAL_JPEG.value in scenario :
          save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr"+str(idx), jpeg_q)
          opened_real_jpeged_backdoored_image = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx))
          next_input = torch.cat((opened_real_jpeged_backdoored_image, test_images), 0)
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx))
        elif SCENARIOS.JPEGED.value in scenario  :
          jpeged_backdoored_image = jpeg(backdoored_image_clipped)
          next_input = torch.cat((jpeged_backdoored_image, test_images), 0)
        else :
          next_input = torch.cat((backdoored_image_clipped, test_images), 0)
        logits = net1.detector(next_input)

        # Calculate loss
        if loss_mode == "onlydetectorloss" :
          test_loss = loss_only_detector(logits, targetY, pos_weight)
          predY = torch.sigmoid(logits)
        elif loss_mode ==  LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
          test_loss = loss_only_detector_no_logits(logits[:,1].unsqueeze(1), targetY)
          predY = logits[:,1].unsqueeze(1)
          secret_pred = net1.detector.detector(backdoored_image_test_secret_clipped)
          secret_pred_for_orig = net1.detector.detector(test_images)
          secret_frog_pred = net1.detector.detector(next_input[:next_input.shape[0]//2])
          secret_frog_pred_for_orig = net1.detector.detector(next_input[next_input.shape[0]//2:])
          test_distances.append(loss_only_detector_mse(secret_pred,secret).data.cpu())
          test_distances_mean.append(torch.mean(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu())
          test_distances_max = max(torch.max(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu(),test_distances_max)
          test_distances_min = min(torch.min(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu(),test_distances_min)
          test_distances_frog.append(loss_only_detector_mse(secret_frog_pred,secret_frog).data.cpu())
          test_distances_frog_mean.append(torch.mean(torch.sum(torch.square(secret_frog_pred - secret_frog), dim=(1, 2, 3))).data.cpu())
          test_distances_frog_max = max(torch.max(torch.sum(torch.square(secret_frog_pred-secret_frog),dim=(1,2,3))).data.cpu(),test_distances_frog_max)
          test_distances_frog_min = min(torch.min(torch.sum(torch.square(secret_frog_pred-secret_frog),dim=(1,2,3))).data.cpu(),test_distances_frog_min)
          orig_distances.append(loss_only_detector_mse(secret_pred_for_orig,secret).data.cpu())
          orig_distances_mean.append(torch.mean(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu())
          orig_distances_max = max(torch.max(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu(),orig_distances_max)
          orig_distances_min = min(torch.min(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu(),orig_distances_min)
          orig_distances_frog.append(loss_only_detector_mse(secret_frog_pred_for_orig,secret_frog).data.cpu())
          orig_distances_frog_mean.append(torch.mean(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu())
          orig_distances_frog_max = max(torch.max(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu(),orig_distances_frog_max)
          orig_distances_frog_min = min(torch.min(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu(),orig_distances_frog_min)

        else :
          test_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, test_images, targetY, loss_mode, B=beta, L=l, pos_weight=pos_weight)
          predY = torch.sigmoid(logits)

        if pred_threshold > 1 :
          pt = 1.0
        else :
          pt = pred_threshold

        test_acc = torch.sum((predY >= pt) == targetY).item()/predY.shape[0]

        if len(((predY == pt) != targetY).nonzero(as_tuple=True)[0]) > 0 :
          for index in ((predY >= pt) != targetY).nonzero(as_tuple=True)[0] :
            if index < 100 :
              # backdoor image related error
              error_on_backdoor_image.append((backdoored_image_clipped[index],test_images[index]))
            else :
              # original image related error
              error_on_original_image.append((backdoored_image_clipped[index-100],test_images[index-100]))

      test_losses.append(test_loss.data.cpu())
      test_acces.append(test_acc)

      dif_image = backdoored_image_clipped - test_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image_clipped.reshape(backdoored_image_clipped.shape[0], -1)
      test_image_color_view = test_images.reshape(test_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - test_image_color_view, p=2, dim=1)
      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      denormalized_test_images = denormalize(images=test_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_test_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.reshape(denormalized_backdoored_images.shape[0],-1) - denormalized_test_images.reshape(denormalized_test_images.shape[0], -1)), p=2, dim=1)).item()
      '''
      if (max_linf < torch.max(linf).item()) :
        last_maxinf_backdoored_image = backdoored_image_clipped[torch.argmax(linf).item()]
        last_maxinf_test_image = test_images[torch.argmax(linf).item()]
        last_maxinf_diff_image = torch.abs(dif_image[torch.argmax(linf).item()]/torch.max(linf).item())
      max_linf = max(max_linf, torch.max(linf).item())
      if (max_l2 < torch.max(l2).item()):
        last_max2_backdoored_image = backdoored_image_clipped[torch.argmax(l2).item()]
        last_max2_test_image = test_images[torch.argmax(l2).item()]
        last_max2_diff_image = torch.abs(dif_image[torch.argmax(l2).item()]/torch.max(linf).item())
      max_l2 = max(max_l2, torch.max(l2).item())

      if (min_linf > torch.min(linf).item()) :
        last_mininf_backdoored_image = backdoored_image_clipped[torch.argmin(linf).item()]
        last_mininf_test_image = test_images[torch.argmin(linf).item()]
        last_mininf_diff_image = torch.abs(dif_image[torch.argmin(linf).item()]/torch.max(linf).item())
      min_linf = min(min_linf, torch.min(linf).item())
      if (min_l2 > torch.min(l2).item()):
        last_min2_backdoored_image = backdoored_image_clipped[torch.argmin(l2).item()]
        last_min2_test_image = test_images[torch.argmin(l2).item()]
        last_min2_diff_image = torch.abs(dif_image[torch.argmin(l2).item()]/torch.max(linf).item())
      min_l2 = min(min_l2, torch.min(l2).item())
      mean_linf += torch.mean(linf).item()
      mean_l2 += torch.mean(l2).item()
      mean_linf_in_eps += torch.mean( (linf <= LINF_EPS).float() ).item()
      mean_l2_in_eps += torch.mean( (l2 <= L2_EPS).float() ).item()



  mean_l2 = mean_l2 / num_of_batch
  mean_linf = mean_linf / num_of_batch
  mean_l2_in_eps = mean_l2_in_eps / num_of_batch
  mean_linf_in_eps = mean_linf_in_eps / num_of_batch

  mean_test_loss = np.mean(test_losses)
  mean_test_acc = np.mean(test_acces)

  mean_test_distance = np.mean(test_distances)
  mean_test_distance_mean = np.mean(test_distances_mean)
  mean_test_distance_frog = np.mean(test_distances_frog)
  mean_test_distance_frog_mean = np.mean(test_distances_frog_mean)
  mean_orig_distance = np.mean(orig_distances)
  mean_orig_distance_mean = np.mean(orig_distances_mean)
  mean_orig_distance_frog = np.mean(orig_distances_frog)
  mean_orig_distance_frog_mean = np.mean(orig_distances_frog_mean)

  print('Average loss on test set: {0:.4f}; accuracy: {1:.4f}; error on backdoor: {2:d}, on original: {3:d}; '
        'backdoor l2 min: {4:.4f}, avg: {5:.4f}, max: {6:.4f}, ineps: {7:.4f}; '
        'backdoor linf min: {8:.4f}, avg: {9:.4f}, max: {10:.4f}, ineps: {11:.4f}'.format(
    mean_test_loss,mean_test_acc,len(error_on_backdoor_image),len(error_on_original_image),
    min_l2,mean_l2,max_l2,mean_l2_in_eps,min_linf,mean_linf,max_linf,mean_linf_in_eps))
  save_images(backdoored_image_clipped, "backdoor")
  save_images(test_images, "original")

  if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
     print('Average deep stegano mse loss on test set: {0:.4f}; mse loss on: {1:.4f} on backdoor images; '
           'mse loss on test set to secret frog:  {2:.4f}; mse loss on: {3:.4f} on secret frog backdoor images '
           'Test set Max: {4:.4f}, Mean: {5:.4f}, Min: {6:.4f}; Backdoor set Max: {7:.4f}, Mean: {8:.4f} Min: {9:.4f}; '
           'Test set to secret Frog Max: {10:.4f}, Mean: {11:.4f}, Min: {12:.4f}; '
           'Secret Frog backdoor set Max: {13:.4f}, Mean: {14:.4f}, Min: {15:.4f}'.format(
          mean_orig_distance, mean_test_distance,
          mean_orig_distance_frog,mean_test_distance_frog,
          orig_distances_max,mean_orig_distance_mean,orig_distances_min,
          test_distances_max,mean_test_distance_mean,test_distances_min,
          orig_distances_frog_max,mean_orig_distance_frog_mean,orig_distances_frog_min,
          test_distances_frog_max,mean_test_distance_frog_mean,test_distances_frog_min))


  save_image(last_maxinf_backdoored_image, "backdoor_max_linf")
  save_image(last_maxinf_test_image, "original_max_linf")
  save_image(last_maxinf_diff_image, "diff_max_linf")
  save_image(last_max2_backdoored_image, "backdoor_max_l2")
  save_image(last_max2_test_image, "original_max_l2")
  save_image(last_max2_diff_image, "diff_max_l2")
  save_image(last_mininf_backdoored_image, "backdoor_min_linf")
  save_image(last_mininf_test_image, "original_min_linf")
  save_image(last_mininf_diff_image, "diff_min_linf")
  save_image(last_min2_backdoored_image, "backdoor_min_l2")
  save_image(last_min2_test_image, "original_min_l2")
  save_image(last_min2_diff_image, "diff_min_l2")

  index = 0
  for image_pair in error_on_backdoor_image :
    save_image(image_pair[0], "error_by_backdoor_backdoor" + str(index))
    save_image(image_pair[1], "error_by_backdoor_original" + str(index))
    index += 1
  index = 0
  for image_pair in error_on_original_image:
    save_image(image_pair[0], "error_by_original_backdoor" + str(index))
    save_image(image_pair[1], "error_by_original_original" + str(index))
    index += 1

  if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
    net1.detector = net1.detector.detector

  return mean_test_loss

def test_multiple_random_secret(net, test_loader, batch_size, num_epochs, scenario, threshold_range, device, linf_epsilon, l2_epsilon, real_jpeg_q) :
  net.eval()
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  tpr_all = {}
  tnr_all = {}
  with torch.no_grad():
    for epoch in range(num_epochs):
      recent_secret = upsample(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
      tpr_results, tnr_results = test_specific_secret(net, test_loader, batch_size, scenario, threshold_range, device, linf_epsilon, l2_epsilon, recent_secret, real_jpeg_q=real_jpeg_q, verbose_images=False)
      for threshold in threshold_range :
        if threshold not in tpr_all :
          tpr_all[threshold] = tpr_results[threshold]
          tnr_all[threshold] = tnr_results[threshold]
        else :
          tpr_all[threshold] = torch.cat((tpr_all[threshold],tpr_results[threshold]))
          tnr_all[threshold] = torch.cat((tnr_all[threshold],tnr_results[threshold]))
    for threshold in threshold_range :
      tpr_mean = torch.mean(tpr_all[threshold]).item()
      tnr_mean = torch.mean(tnr_all[threshold]).item()
      tpr_std = torch.std(tpr_all[threshold], unbiased=False).item()
      tpr_std_unbiased = torch.std(tpr_all[threshold], unbiased=True).item()
      tnr_std = torch.std(tnr_all[threshold], unbiased=False).item()
      tnr_std_unbiased = torch.std(tnr_all[threshold], unbiased=True).item()
      print(threshold, tpr_mean, tpr_std, tpr_std_unbiased, tnr_mean, tnr_std, tnr_std_unbiased)

def test_multiple_random_secret_old(net, test_loader, batch_size, num_epochs, scenario, threshold_range, device, linf_epsilon_clip, l2_epsilon_clip, diff_jpeg_q, real_jpeg_q, num_secret_on_test=0) :
  net.eval()
  if SCENARIOS.JPEGED.value in scenario :
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
    jpeg = jpeg.to(device)
    for param in jpeg.parameters():
      param.requires_grad = False
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  if SCENARIOS.MEDIAN.value in scenario :
    filter = torch.median
  if SCENARIOS.AVG_FIL.value in scenario :
    filter = torch.mean
  num_of_batch = 0
  all_the_distance_on_backdoor = torch.Tensor()
  all_the_distance_by_median_on_backdoor = torch.Tensor()
  threshold_backdoor_dict = {}
  threshold_backdoor_median_dict = {}
  threshold_range_median = np.arange(0, (secret_colorc * secret_shape_1 * secret_shape_2) + 1, 1)
  min_dist_backdoor = 9999999.0
  max_dist_backdoor = 0.0
  for threshold in threshold_range :
    threshold_backdoor_dict[threshold] = 1.0
  for threshold in threshold_range_median :
    threshold_backdoor_median_dict[threshold] = 1.0
  all_the_revealed_something_on_test_set = torch.Tensor()
  with torch.no_grad():
    for epoch in range(num_epochs):
      all_the_distance_on_backdoor_per_epoch = torch.Tensor()
      all_the_distance_by_median_on_backdoor_per_epoch = torch.Tensor()
      if SCENARIOS.DISCRETE_PIXEL.value in scenario :
        secret_frog = (torch.randint(255,(secret_colorc, secret_shape_1, secret_shape_2))/255).unsqueeze(0)
      elif SCENARIOS.DISCRETE_PIXEL_16.value in scenario :
        secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/16)*16)-1)/255).unsqueeze(0)
      elif SCENARIOS.DISCRETE_PIXEL_8.value in scenario :
        secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/32)*32)-1)/255).unsqueeze(0)
      elif SCENARIOS.DISCRETE_PIXEL_4.value in scenario :
        secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/64)*64)-1)/255).unsqueeze(0)
      else:
        secret_frog = torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)
      secret_real = create_batch_from_a_single_image(secret_frog,batch_size)
      secret = create_batch_from_a_single_image(upsample(secret_frog),batch_size).to(device)
      for idx, test_batch in enumerate(test_loader):
        num_of_batch += 1
        # Saves images
        data, labels = test_batch
        test_images = data.to(device)
        backdoored_image = net.generator(secret,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        if SCENARIOS.REAL_JPEG.value in scenario :
          save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx) +"_" + str(epoch), real_jpeg_q)
          backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx)+"_"+str(epoch))
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx)+"_"+str(epoch))
        if SCENARIOS.JPEGED.value in scenario  :
          test_images = jpeg(test_images)
          backdoored_image_clipped = jpeg(backdoored_image_clipped)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        if all_the_revealed_something_on_test_set.shape[0] < batch_size :
          revealed_something_on_test_set = net.detector(test_images)
          all_the_revealed_something_on_test_set = torch.cat((all_the_revealed_something_on_test_set,revealed_something_on_test_set.detach().cpu()), 0)
          captured_revealed_something_on_test_set = torch.clone(revealed_something_on_test_set[0:10])
          captured_revealed_secret_on_backdoor = torch.clone(revealed_secret_on_backdoor[0:10])
          captured_secret = torch.clone(secret[0])
          captured_backdoored_image_clipped = torch.clone(backdoored_image_clipped[0:10])
          captured_test_images = torch.clone(test_images[0:10])
        if SCENARIOS.DISCRETE_PIXEL.value in scenario :
          revealed_secret_on_backdoor = torch.round(revealed_secret_on_backdoor*255)/255
        elif SCENARIOS.DISCRETE_PIXEL_16.value in scenario :
          revealed_secret_on_backdoor = ((torch.round(((revealed_secret_on_backdoor*255)+1)/16)*16)-1)/255
        elif SCENARIOS.DISCRETE_PIXEL_8.value in scenario :
          revealed_secret_on_backdoor = ((torch.round(((revealed_secret_on_backdoor*255)+1)/32)*32)-1)/255
        elif SCENARIOS.DISCRETE_PIXEL_4.value in scenario :
          revealed_secret_on_backdoor = ((torch.round(((revealed_secret_on_backdoor*255)+1)/64)*64)-1)/255
        if SCENARIOS.MEDIAN.value in scenario or SCENARIOS.AVG_FIL.value in scenario :
          revealed_the_real_secret_on_backdoor = get_the_secret(revealed_secret_on_backdoor, secret_real.shape[2], secret_real.shape[3], filter)
          distance_on_backdoor = torch.sum(torch.square(revealed_the_real_secret_on_backdoor-secret_real),dim=(1,2,3))
          distance_by_median_on_backdoor = torch.sum((revealed_the_real_secret_on_backdoor == secret_real),dim=(1,2,3))
          all_the_distance_by_median_on_backdoor_per_epoch = torch.cat((all_the_distance_by_median_on_backdoor_per_epoch, distance_by_median_on_backdoor.data.cpu()), 0)
        else:
          distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret),dim=(1,2,3))
        all_the_distance_on_backdoor_per_epoch = torch.cat((all_the_distance_on_backdoor_per_epoch,distance_on_backdoor.data.cpu()),0)
        if min_dist_backdoor > torch.min(distance_on_backdoor).item() :
          min_test_b = torch.clone(test_images[torch.argmin(distance_on_backdoor).item()])
          min_backdoor = torch.clone(backdoored_image_clipped[torch.argmin(distance_on_backdoor).item()])
          min_backdoor_revealed = torch.clone(revealed_secret_on_backdoor[torch.argmin(distance_on_backdoor).item()])
          min_backdoor_secret = torch.clone(secret[0])
          min_dist_backdoor = torch.min(distance_on_backdoor).item()
        if max_dist_backdoor < torch.max(distance_on_backdoor).item() :
          max_test_b = torch.clone(test_images[torch.argmax(distance_on_backdoor).item()])
          max_backdoor = torch.clone(backdoored_image_clipped[torch.argmax(distance_on_backdoor).item()])
          max_backdoor_revealed = torch.clone(revealed_secret_on_backdoor[torch.argmax(distance_on_backdoor).item()])
          max_backdoor_secret = torch.clone(secret[0])
          max_dist_backdoor = torch.max(distance_on_backdoor).item()

      for threshold in threshold_range :
          threshold_backdoor_dict[threshold] = min(threshold_backdoor_dict[threshold],(torch.sum(all_the_distance_on_backdoor_per_epoch < threshold) / all_the_distance_on_backdoor_per_epoch.shape[0]).item())
      if SCENARIOS.MEDIAN.value in scenario  or SCENARIOS.AVG_FIL.value in scenario :
        for threshold in threshold_range_median :
            threshold_backdoor_median_dict[threshold] = min(threshold_backdoor_median_dict[threshold],(torch.sum(all_the_distance_by_median_on_backdoor_per_epoch >= threshold) / all_the_distance_by_median_on_backdoor_per_epoch.shape[0]).item())
        all_the_distance_by_median_on_backdoor = torch.cat((all_the_distance_by_median_on_backdoor,all_the_distance_by_median_on_backdoor_per_epoch),0)
        print("Epoch", epoch, ": revealed distance by median on backdoor min:",
              torch.min(all_the_distance_by_median_on_backdoor_per_epoch).item(),
              ", mean:", torch.mean(all_the_distance_by_median_on_backdoor_per_epoch).item(),
              ", max:", torch.max(all_the_distance_by_median_on_backdoor_per_epoch).item())
        print("Global revealed distance by median on backdoor min:", torch.min(all_the_distance_by_median_on_backdoor).item(),
              ", mean:", torch.mean(all_the_distance_by_median_on_backdoor).item(),
              ", max:", torch.max(all_the_distance_by_median_on_backdoor).item())
      all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,all_the_distance_on_backdoor_per_epoch),0)
      print("Epoch",epoch,": revealed distance on backdoor min:",torch.min(all_the_distance_on_backdoor_per_epoch).item(),
            ", mean:",torch.mean(all_the_distance_on_backdoor_per_epoch).item(),
            ", max:",torch.max(all_the_distance_on_backdoor_per_epoch).item())
      print("Global revealed distance on backdoor min:",torch.min(all_the_distance_on_backdoor).item(),
            ", mean:",torch.mean(all_the_distance_on_backdoor).item(),
            ", max:",torch.max(all_the_distance_on_backdoor).item())
    min_dist_test = 9999999.0
    max_dist_test = 0.0
    mean_dist_test = 0.0
    max_min_dist_test = 0.0
    all_the_min_dist_on_test = torch.Tensor()
    min_dist_by_median_on_test = 9999999.0
    mean_dist_by_median_on_test = 0.0
    min_max_dist_by_median_on_test = 0.0
    max_dist_by_median_on_test = 0.0
    all_the_max_dist_by_median_on_test = torch.Tensor()
    if SCENARIOS.DISCRETE_PIXEL.value in scenario :
      all_the_revealed_something_on_test_set = torch.round(all_the_revealed_something_on_test_set*255)/255
    elif SCENARIOS.DISCRETE_PIXEL_16.value in scenario :
      all_the_revealed_something_on_test_set = ((torch.round(((all_the_revealed_something_on_test_set*255)+1)/16)*16)-1)/255
    elif SCENARIOS.DISCRETE_PIXEL_8.value in scenario :
      all_the_revealed_something_on_test_set = ((torch.round(((all_the_revealed_something_on_test_set*255)+1)/32)*32)-1)/255
    elif SCENARIOS.DISCRETE_PIXEL_4.value in scenario :
      all_the_revealed_something_on_test_set = ((torch.round(((all_the_revealed_something_on_test_set*255)+1)/64)*64)-1)/255
    if SCENARIOS.MEDIAN.value in scenario or SCENARIOS.AVG_FIL.value in scenario :
      all_the_revealed_the_real_something_on_test_set = get_the_secret(all_the_revealed_something_on_test_set, secret_real.shape[2], secret_real.shape[3], filter)
    if num_secret_on_test > 0 :
      for ith_secret in range(num_secret_on_test) :
        if SCENARIOS.DISCRETE_PIXEL.value in scenario :
          secret_frog = (torch.randint(255,(secret_colorc, secret_shape_1, secret_shape_2))/255).unsqueeze(0)
        elif SCENARIOS.DISCRETE_PIXEL_16.value in scenario :
          secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/16)*16)-1)/255).unsqueeze(0)
        elif SCENARIOS.DISCRETE_PIXEL_8.value in scenario :
          secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/32)*32)-1)/255).unsqueeze(0)
        elif SCENARIOS.DISCRETE_PIXEL_4.value in scenario :
          secret_frog = (((torch.round(((torch.rand((secret_colorc, secret_shape_1, secret_shape_2))*255)+1)/64)*64)-1)/255).unsqueeze(0)
        else:
          secret_frog = torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)
        secret_real = create_batch_from_a_single_image(secret_frog,batch_size)
        secret = create_batch_from_a_single_image(upsample(secret_frog),batch_size)
        if SCENARIOS.MEDIAN.value in scenario or SCENARIOS.AVG_FIL.value in scenario :
          distance_on_test = torch.sum(torch.square(all_the_revealed_the_real_something_on_test_set-secret_real),dim=(1,2,3))
          distance_by_median_on_test = torch.sum((all_the_revealed_the_real_something_on_test_set == secret_real),dim=(1,2,3))
          distance_by_median_on_test = distance_by_median_on_test.type(torch.FloatTensor)
          max_dist_by_median_on_test = torch.max(distance_by_median_on_test)
          all_the_max_dist_by_median_on_test = torch.cat((all_the_max_dist_by_median_on_test, torch.Tensor([max_dist_by_median_on_test])), 0)
          min_dist_by_median_on_test_with_this_secret = torch.min(distance_by_median_on_test).item()
          if min_dist_by_median_on_test > min_dist_by_median_on_test_with_this_secret :
            min_dist_by_median_on_test = min_dist_by_median_on_test_with_this_secret
          mean_dist_test += torch.mean(distance_by_median_on_test).item()/num_secret_on_test
          if min_max_dist_by_median_on_test > torch.max(distance_by_median_on_test).item() :
            min_max_dist_by_median_on_test = torch.max(distance_by_median_on_test).item()
          if max_dist_test < torch.max(distance_by_median_on_test).item() :
            max_dist_test = torch.max(distance_by_median_on_test).item()
        else:
          distance_on_test = torch.sum(torch.square(all_the_revealed_something_on_test_set-secret),dim=(1,2,3))
        min_dist_on_test_with_this_secret = torch.min(distance_on_test).item()
        all_the_min_dist_on_test = torch.cat((all_the_min_dist_on_test,torch.Tensor([min_dist_on_test_with_this_secret])),0)
        if min_dist_test > min_dist_on_test_with_this_secret :
          min_test = test_images[torch.argmin(distance_on_test).item()]
          min_test_revealed = torch.clone(revealed_something_on_test_set[torch.argmin(distance_on_test).item()])
          min_test_secret = torch.clone(secret[0])
          min_dist_test = min_dist_on_test_with_this_secret
        mean_dist_test += torch.mean(distance_on_test).item()/num_secret_on_test
        if max_min_dist_test < min_dist_on_test_with_this_secret :
          maxmin_test = test_images[torch.argmin(distance_on_test).item()]
          maxmin_test_revealed = torch.clone(revealed_something_on_test_set[torch.argmin(distance_on_test).item()])
          maxmin_test_secret = torch.clone(secret[0])
          max_min_dist_test = min_dist_on_test_with_this_secret
        if max_dist_test < torch.max(distance_on_test).item() :
          max_test = test_images[torch.argmax(distance_on_test).item()]
          max_test_revealed = torch.clone(revealed_something_on_test_set[torch.argmax(distance_on_test).item()])
          max_test_secret = torch.clone(secret[0])
          max_dist_test = torch.max(distance_on_test).item()
        if ith_secret % 10000 == 0 :
          print(ith_secret, "Revealed distance on test set min:",min_dist_test,
            ", mean:",mean_dist_test,
            ", maxmin:",max_min_dist_test,
            ", max:",max_dist_test)
      print("Revealed distance on test set min:",min_dist_test,
            ", mean:",mean_dist_test,
            ", maxmin:",max_min_dist_test,
            ", max:",max_dist_test)
    save_images(captured_test_images, "random_original")
    save_images(captured_backdoored_image_clipped, "random_backdoor")
    save_image(captured_secret, "random_secret", grayscale="grayscale")
    save_images(captured_revealed_something_on_test_set, "random_revealed_secret_on_backdoor", grayscale="grayscale")
    save_images(captured_revealed_secret_on_backdoor, "random_revealed_something_on_test_set", grayscale="grayscale")
    if SCENARIOS.MEDIAN.value in scenario or SCENARIOS.AVG_FIL.value in scenario :
      save_images(upsample(revealed_the_real_secret_on_backdoor[0:10]), "random_revealed_the_real_secret_on_backdoor", grayscale="grayscale")
      print("Revealed distance by median on test set min:",
                min_dist_by_median_on_test,
                ", mean:", mean_dist_by_median_on_test,
                ", max:", max_dist_by_median_on_test)
      print("distance_by_median___________________________________")
      for threshold in threshold_range_median:
        all_the_max_above_threshold = torch.sum((all_the_max_dist_by_median_on_test >= threshold)).item()
        print(threshold, threshold_backdoor_median_dict[threshold],all_the_max_above_threshold)
    print("distance_____________")
    for threshold in threshold_range :
      all_the_min_in_threshold = torch.sum((all_the_min_dist_on_test < threshold)).item()
      print(threshold,threshold_backdoor_dict[threshold],all_the_min_in_threshold)
    save_image(min_test_b, "min_backdoor_original")
    save_image(min_backdoor, "min_backdoor_backdoor")
    save_image(min_backdoor_revealed, "min_backdoor_revealed", grayscale="grayscale")
    save_image(min_backdoor_secret, "min_backdoor_secret", grayscale="grayscale")
    save_image(min_test, "min_original_original")
    save_image(min_test_revealed, "min_original_revealed", grayscale="grayscale")
    save_image(min_test_secret, "min_original_secret", grayscale="grayscale")
    save_image(max_test_b, "max_backdoor_original")
    save_image(max_backdoor, "max_backdoor_backdoor")
    save_image(max_backdoor_revealed, "max_revealed_backdoor", grayscale="grayscale")
    save_image(max_backdoor_secret, "max_backdoor_secret", grayscale="grayscale")
    save_image(max_test, "max_original_original")
    save_image(max_test_revealed, "max_original_revealed", grayscale="grayscale")
    save_image(max_test_secret, "max_original_secret", grayscale="grayscale")
    save_image(maxmin_test, "maxmin_original_original")
    save_image(maxmin_test_revealed, "maxmin_original_revealed", grayscale="grayscale")
    save_image(maxmin_test_secret, "maxmin_original_secret", grayscale="grayscale")

def get_rand_indices(range_end, num_of_rand_indices) :
  rand_indices_set = set()
  while len(rand_indices_set) < num_of_rand_indices :
    rand_index = np.random.randint(0,range_end)
    rand_indices_set.add(rand_index)
  return list(rand_indices_set)

def get_the_best_gray_secret_for_net(net, test_loader, batch_size, num_epochs, threshold_range, scenario, device, linf_epsilon_clip, l2_epsilon_clip, diff_jpeg_q, real_jpeg_q) :
  net.eval()
  if SCENARIOS.JPEGED.value in scenario :
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
    jpeg = jpeg.to(device)
    for param in jpeg.parameters():
      param.requires_grad = False
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  all_the_revealed_something_on_test_set = torch.Tensor()
  secret_batch_size = num_epochs//(val_size[dataset]//batch_size)
  if secret_batch_size < 1:
    secret_batch_size = 1
  with torch.no_grad() :
    secrets = torch.Tensor().to(device)
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      rand_indices = get_rand_indices(test_images.shape[0], secret_batch_size)
      for jdx in rand_indices :
        secrets = torch.cat((secrets,transforms.Grayscale()(test_images[jdx].unsqueeze(0))),0)
      revealed_something_on_test_set = net.detector(test_images)
      all_the_revealed_something_on_test_set = torch.cat((all_the_revealed_something_on_test_set, revealed_something_on_test_set.detach().cpu()), 0)
    matrix_keys = []
    matrix_backdoor_dist = []
    matrix_original_dist = []
    epoch = 0
    for secret in secrets :
      epoch +=1
      secret = secret.unsqueeze(0)
      secret_for_a_batch = create_batch_from_a_single_image(secret,batch_size).to(device)
      secret_for_whole_test_set = create_batch_from_a_single_image(secret.detach().cpu(),all_the_revealed_something_on_test_set.shape[0])
      all_the_distance_on_test = torch.sum(torch.square(all_the_revealed_something_on_test_set-secret_for_whole_test_set),dim=(1,2,3))
      all_the_distance_on_backdoor = torch.Tensor()
      for idx, test_batch in enumerate(test_loader):
        data, labels = test_batch
        test_images = data.to(device)
        backdoored_image = net.generator(secret_for_a_batch,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        if SCENARIOS.REAL_JPEG.value in scenario :
          save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx) +"_" + str(epoch), real_jpeg_q)
          backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx)+"_"+str(epoch))
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx)+"_"+str(epoch))
        if SCENARIOS.JPEGED.value in scenario  :
          backdoored_image_clipped = jpeg(backdoored_image_clipped)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret_for_a_batch),dim=(1,2,3))
        all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,distance_on_backdoor.detach().cpu()),0)
      matrix_keys.append(secret[0,0].cpu().detach().numpy())
      matrix_backdoor_dist.append(all_the_distance_on_backdoor.numpy())
      matrix_original_dist.append(all_the_distance_on_test.numpy())
    np_matrix_keys = np.array(matrix_keys)
    np_matrix_backdoor_dist = np.array(matrix_backdoor_dist)
    np_matrix_original_dist = np.array(matrix_original_dist)
    np.save(IMAGE_PATH+scenario+"_keys.npy",np_matrix_keys)
    np.save(IMAGE_PATH+scenario+"_original_distances.npy",np_matrix_original_dist)
    np.save(IMAGE_PATH+scenario+"_backdoor_distances.npy",np_matrix_backdoor_dist)
    thresholds = np.min(np_matrix_original_dist, axis=1) * 0.65
    tpr = []
    for idx in range(len(thresholds)) :
      tpr.append(np.sum(np_matrix_backdoor_dist[idx] < thresholds[idx]) / np_matrix_backdoor_dist.shape[1])
    np_tpr = np.array(tpr)
    print("Worst tpr",np.min(np_tpr),"std", np.std(np_tpr), "tpr mean-std", str(np.mean(np_tpr)-np.std(np_tpr)),
          "tpr mean", np.mean(np_tpr), "tpr mean+std", str(np.mean(np_tpr)+np.std(np_tpr)), "best tpr", np.max(np_tpr))
    best_idx_tpr = np.argmax(np_tpr)
    worst_idx_tpr = np.argmin(np_tpr)
    print("Worst secret threshold",thresholds[worst_idx_tpr],"best secret threshold",thresholds[best_idx_tpr])
    best_secret = torch.from_numpy(np_matrix_keys[best_idx_tpr]).unsqueeze(0)
    save_image(best_secret, "best_secret_"+scenario, grayscale="grayscale")
    worst_secret = torch.from_numpy(np_matrix_keys[worst_idx_tpr]).unsqueeze(0)
    save_image(worst_secret, "worst_secret_"+scenario, grayscale="grayscale")
    return best_secret




  save_image(best_secret[0], "best_gray_secret", grayscale="grayscale")
  return best_secret


def get_the_best_random_secret_for_net(net, test_loader, batch_size, num_epochs, threshold_range, scenario, device,
                                       linf_epsilon_clip, l2_epsilon_clip, diff_jpeg_q, real_jpeg_q):
  net.eval()
  if SCENARIOS.JPEGED.value in scenario :
    if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
      jpeg = DiffJPEG(image_shape[DATASET.CIFAR10.value][0], image_shape[DATASET.CIFAR10.value][1], differentiable=True, quality=diff_jpeg_q)
    else :
      jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
    jpeg = jpeg.to(device)
    for param in jpeg.parameters():
      param.requires_grad = False
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
    pos_backdor = [0,0]
    upsample = torch.nn.Upsample(scale_factor=(image_shape[DATASET.CIFAR10.value][0]/secret_shape_1, image_shape[DATASET.CIFAR10.value][1]/secret_shape_2), mode='nearest')
  else :
    upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  all_the_revealed_something_on_test_set = torch.Tensor()
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        revealed_something_on_test_set = net.detector(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
      else :
        revealed_something_on_test_set = net.detector(test_images).detach().cpu()
      all_the_revealed_something_on_test_set = torch.cat((all_the_revealed_something_on_test_set, revealed_something_on_test_set), 0)
    matrix_keys = []
    matrix_backdoor_dist = []
    matrix_original_dist = []
    for epoch in range(num_epochs):
      secret_frog = torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)
      secret_for_a_batch = create_batch_from_a_single_image(upsample(secret_frog),batch_size).to(device)
      secret_for_whole_test_set = create_batch_from_a_single_image(upsample(secret_frog),all_the_revealed_something_on_test_set.shape[0])
      all_the_distance_on_test = torch.sum(torch.square(all_the_revealed_something_on_test_set-secret_for_whole_test_set),dim=(1,2,3))
      all_the_distance_on_backdoor = torch.Tensor()
      for idx, test_batch in enumerate(test_loader):
        data, labels = test_batch
        test_images = data.to(device)
        if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
          backdoored_image = net.generator(secret_for_a_batch,test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
          backdoored_image_clipped = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], scenario, l2_epsilon_clip, linf_epsilon_clip, device)
          open_jpeg_flag_for_cifar10_model = True
        else :
          backdoored_image = net.generator(secret_for_a_batch,test_images)
          backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
          open_jpeg_flag_for_cifar10_model = False
        if SCENARIOS.REAL_JPEG.value in scenario :
          jpeg_file_name = "tmpBckdr" + str(idx) +"_" + str(epoch)+"_"+scenario
          save_images_as_jpeg(backdoored_image_clipped, jpeg_file_name, real_jpeg_q)
          backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], jpeg_file_name, open_jpeg_flag_for_cifar10_model)
          removeImages(backdoored_image_clipped.shape[0],jpeg_file_name)
        if SCENARIOS.JPEGED.value in scenario  :
          backdoored_image_clipped = jpeg(backdoored_image_clipped)
        if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
          revealed_secret_on_backdoor = net.detector(backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
        else :
          revealed_secret_on_backdoor = net.detector(backdoored_image_clipped).detach().cpu()
        distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret_for_a_batch.detach().cpu()),dim=(1,2,3))
        all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,distance_on_backdoor),0)
      matrix_keys.append(secret_frog[0,0].cpu().detach().numpy())
      matrix_backdoor_dist.append(all_the_distance_on_backdoor.numpy())
      matrix_original_dist.append(all_the_distance_on_test.numpy())
    np_matrix_keys = np.array(matrix_keys)
    np_matrix_backdoor_dist = np.array(matrix_backdoor_dist)
    np_matrix_original_dist = np.array(matrix_original_dist)
    np.save(IMAGE_PATH+scenario+"_keys.npy",np_matrix_keys)
    np.save(IMAGE_PATH+scenario+"_original_distances.npy",np_matrix_original_dist)
    np.save(IMAGE_PATH+scenario+"_backdoor_distances.npy",np_matrix_backdoor_dist)
    np_original_mins = np.min(np_matrix_original_dist, axis=1)
    thresholds = np_original_mins * 0.5
    tpr = []
    for idx in range(len(thresholds)) :
      tpr.append(np.sum(np_matrix_backdoor_dist[idx] < thresholds[idx]) / np_matrix_backdoor_dist.shape[1])
    np_tpr = np.array(tpr)
    print("50% Worst tpr",np.min(np_tpr),"std", np.std(np_tpr), "tpr mean-std", str(np.mean(np_tpr)-np.std(np_tpr)),
          "tpr mean", np.mean(np_tpr), "tpr mean+std", str(np.mean(np_tpr)+np.std(np_tpr)), "best tpr", np.max(np_tpr))
    best_idx_tpr = np.argmax(np_tpr)
    worst_idx_tpr = np.argmin(np_tpr)
    print("Worst secret threshold",thresholds[worst_idx_tpr],"min orig",np_original_mins[worst_idx_tpr],
          "best secret threshold",thresholds[best_idx_tpr],"min orig",np_original_mins[best_idx_tpr])
    for threshold_percent in np.arange(0.0, 1.05, 0.05) :
      thresholds = np.min(np_matrix_original_dist, axis=1) * threshold_percent
      tpr_for_this = []
      for idx in range(len(thresholds)) :
        tpr_for_this.append(np.sum(np_matrix_backdoor_dist[idx] < thresholds[idx]) / np_matrix_backdoor_dist.shape[1])
      np_tpr_for_this = np.array(tpr_for_this)
      print(threshold_percent, np.min(np_tpr_for_this), np.std(np_tpr_for_this), str(np.mean(np_tpr_for_this)-np.std(np_tpr_for_this)),
            np.mean(np_tpr_for_this), str(np.mean(np_tpr_for_this)+np.std(np_tpr_for_this)), np.max(np_tpr_for_this))
    best_secret = torch.from_numpy(np_matrix_keys[best_idx_tpr]).unsqueeze(0).unsqueeze(0)
    save_image(upsample(best_secret)[0], "best_secret_"+scenario, grayscale="grayscale")
    worst_secret = torch.from_numpy(np_matrix_keys[worst_idx_tpr]).unsqueeze(0).unsqueeze(0)
    save_image(upsample(worst_secret)[0], "worst_secret_"+scenario, grayscale="grayscale")
    return best_secret




def get_the_best_random_secret_for_net_auc(net, test_loader, batch_size, num_epochs, threshold_range, scenario, device, linf_epsilon_clip, l2_epsilon_clip, diff_jpeg_q, real_jpeg_q) :
  net.eval()
  if SCENARIOS.JPEGED.value in scenario :
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
    jpeg = jpeg.to(device)
    for param in jpeg.parameters():
      param.requires_grad = False
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  all_the_revealed_something_on_test_set = torch.Tensor()
  threshold_backdoor_dict = {}
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      revealed_something_on_test_set = net.detector(test_images)
      all_the_revealed_something_on_test_set = torch.cat((all_the_revealed_something_on_test_set,revealed_something_on_test_set.detach().cpu()), 0)
    minimax_value = torch.Tensor()
    minimax_secret = torch.Tensor()
    auc_distrib = {}
    auc_001_distrib = {}
    auc_0000001_distrib = {}
    auc_000000001_distrib = {}
    for i in range (0,101) :
      auc_distrib[i] = 0
      auc_001_distrib[i] = 0
      auc_0000001_distrib[i] = 0
      auc_000000001_distrib[i] = 0
    istvan_matrix_keys = []
    istvan_matrix_backdoor = []
    istvan_matrix_original = []
    for epoch in range(num_epochs) :
      secret_frog = torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)
      secret_for_a_batch = create_batch_from_a_single_image(upsample(secret_frog),batch_size).to(device)
      secret_for_whole_test_set = create_batch_from_a_single_image(upsample(secret_frog),all_the_revealed_something_on_test_set.shape[0])
      all_the_distance_on_test = torch.sum(torch.square(all_the_revealed_something_on_test_set-secret_for_whole_test_set),dim=(1,2,3))
      all_the_distance_on_backdoor = torch.Tensor()
      for idx, test_batch in enumerate(test_loader):
        data, labels = test_batch
        test_images = data.to(device)
        backdoored_image = net.generator(secret_for_a_batch,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        if SCENARIOS.REAL_JPEG.value in scenario :
          save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx) +"_" + str(epoch), real_jpeg_q)
          backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx)+"_"+str(epoch))
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx)+"_"+str(epoch))
        if SCENARIOS.JPEGED.value in scenario  :
          backdoored_image_clipped = jpeg(backdoored_image_clipped)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret_for_a_batch),dim=(1,2,3))
        all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,distance_on_backdoor.detach().cpu()),0)
      target_y = torch.cat((torch.ones(all_the_distance_on_backdoor.shape[0]),torch.zeros(all_the_distance_on_test.shape[0])),0)
      all_the_distance = torch.cat((all_the_distance_on_backdoor,all_the_distance_on_test),0)
      norm_all_the_distance = (max_distance[dataset]-all_the_distance)/max_distance[dataset]
      auc = roc_auc_score(target_y, norm_all_the_distance)
      auc_001 = roc_auc_score(target_y, norm_all_the_distance,max_fpr=0.01)
      auc_0000001 = roc_auc_score(target_y, norm_all_the_distance,max_fpr=0.000001)
      auc_000000001 = roc_auc_score(target_y, norm_all_the_distance,max_fpr=0.000000001)
      auc_distrib[int(np.round(auc*100))] += 1
      auc_001_distrib[int(np.round(auc_001*100))] += 1
      auc_0000001_distrib[int(np.round(auc_0000001*100))] += 1
      auc_000000001_distrib[int(np.round(auc_000000001*100))] += 1
      istvan_matrix_keys.append(secret_frog[0,0].cpu().detach().numpy())
      istvan_matrix_original.append(all_the_distance_on_test.numpy())
      istvan_matrix_backdoor.append(all_the_distance_on_backdoor.numpy())
      print(epoch,auc,auc_001,auc_0000001,auc_000000001,
            torch.min(all_the_distance_on_backdoor).item(),torch.mean(all_the_distance_on_backdoor).item(),torch.max(all_the_distance_on_backdoor).item(),
            torch.min(all_the_distance_on_test).item(),torch.mean(all_the_distance_on_test).item(),torch.max(all_the_distance_on_test).item())
    np_istvan_matrix_keys = np.array(istvan_matrix_keys)
    np_istvan_matrix_original = np.array(istvan_matrix_original)
    np_istvan_matrix_backdoor = np.array(istvan_matrix_backdoor)
    np.save(IMAGE_PATH+"np_istvan_matrix_keys.npy",np_istvan_matrix_keys)
    np.save(IMAGE_PATH+"np_istvan_matrix_original.npy",np_istvan_matrix_original)
    np.save(IMAGE_PATH+"np_istvan_matrix_backdoor.npy",np_istvan_matrix_backdoor)
    '''print("auc")
    for i in range (0,101) :
      print(i,auc_distrib[i])
    print("auc-max_fpr=0.01")
    for i in range (0,101) :
      print(i,auc_001_distrib[i])
    print("auc-max_fpr=0.000001")
    for i in range (0,101) :
      print(i,auc_0000001_distrib[i])
    print("auc-max_fpr=0.000000001")
    for i in range (0,101) :
      print(i,auc_000000001_distrib[i])'''
    np_istvan_matrix_for_auc = np.concatenate((np_istvan_matrix_backdoor,np_istvan_matrix_original),axis=1)
    np_istvan_matrix_for_auc = (150-np_istvan_matrix_for_auc)/150
    target_y =  np.concatenate((np.ones(np_istvan_matrix_backdoor.shape),np.zeros(np_istvan_matrix_original.shape)),axis=1)
    auc_000000001 = []
    for i in range(np_istvan_matrix_for_auc.shape[0]) :
      auc_000000001.append(roc_auc_score(target_y[i], np_istvan_matrix_for_auc[i],max_fpr=0.000000001))
    np_auc = np.array(auc_000000001)
    worst_indices = np.argpartition(np_auc, 50)
    best_indices = np.argpartition(-np_auc, 50)
    #rand_indices = np.random.randint(0,np_auc.shape[0],50)
    tpr_results_on_best = {}
    tpr_results_on_worst = {}
    tpr_results_on_all = {}
    tnr_results_on_best = {}
    tnr_results_on_worst = {}
    tnr_results_on_all = {}
    for threshold in threshold_range :
      tpr_on_best = (np.sum(np_istvan_matrix_backdoor[best_indices[:50]] < threshold, axis=1) / np_istvan_matrix_backdoor.shape[1])
      tpr_on_worst = (np.sum(np_istvan_matrix_backdoor[worst_indices[:50]] < threshold, axis=1) / np_istvan_matrix_backdoor.shape[1])
      tpr_on_all = (np.sum(np_istvan_matrix_backdoor < threshold, axis=1) / np_istvan_matrix_backdoor.shape[1])
      tnr_on_best = (np.sum(np_istvan_matrix_original[best_indices[:50]] >= threshold, axis=1) / np_istvan_matrix_original.shape[1])
      tnr_on_worst = (np.sum(np_istvan_matrix_original[worst_indices[:50]] >= threshold, axis=1) / np_istvan_matrix_original.shape[1])
      tnr_on_all = (np.sum(np_istvan_matrix_original >= threshold, axis=1) / np_istvan_matrix_original.shape[1])
      tpr_results_on_best[threshold] = tpr_on_best
      tpr_results_on_worst[threshold] = tpr_on_worst
      tpr_results_on_all[threshold] = tpr_on_all
      tnr_results_on_best[threshold] = tnr_on_best
      tnr_results_on_worst[threshold] = tnr_on_worst
      tnr_results_on_all[threshold] = tnr_on_all
    with open(IMAGE_PATH+"auc_best_worst.txt", "w") as outfile :
      for threshold in tpr_results_on_best :
        print(threshold, np.mean(tpr_results_on_best[threshold]), np.std(tpr_results_on_best[threshold]),
            np.mean(tnr_results_on_best[threshold]), np.std(tnr_results_on_best[threshold]),
            np.mean(tpr_results_on_all[threshold]),np.std(tpr_results_on_all[threshold]),
            np.mean(tnr_results_on_all[threshold]),np.std(tnr_results_on_all[threshold]),
            np.mean(tpr_results_on_worst[threshold]), np.std(tpr_results_on_worst[threshold]),
            np.mean(tnr_results_on_worst[threshold]), np.std(tnr_results_on_worst[threshold]), file=outfile)
    rand_best_index = np.random.randint(0,50)
    threshold = np.min(np_istvan_matrix_original[best_indices[rand_best_index]])*0.65
    tpr_on_best = (np.sum(np_istvan_matrix_backdoor[best_indices[rand_best_index]] < threshold) / np_istvan_matrix_backdoor.shape[1])
    tnr_on_best = (np.sum(np_istvan_matrix_original[best_indices[rand_best_index]] >= threshold) / np_istvan_matrix_original.shape[1])
    max_back_dist = np.max(np_istvan_matrix_backdoor[best_indices[rand_best_index]])
    min_back_dist = np.min(np_istvan_matrix_backdoor[best_indices[rand_best_index]])
    mean_back_dist = np.mean(np_istvan_matrix_backdoor[best_indices[rand_best_index]])
    std_back_dist = np.std(np_istvan_matrix_backdoor[best_indices[rand_best_index]])
    max_orig_dist = np.max(np_istvan_matrix_original[best_indices[rand_best_index]])
    min_orig_dist = np.min(np_istvan_matrix_original[best_indices[rand_best_index]])
    mean_orig_dist = np.mean(np_istvan_matrix_original[best_indices[rand_best_index]])
    std_orig_dist = np.std(np_istvan_matrix_original[best_indices[rand_best_index]])
    print("The chosen key stats: threshold:",threshold, "tpr", tpr_on_best, "tnr", tnr_on_best,
          "backdoor distance min", min_back_dist, "mean", mean_back_dist, "std", std_back_dist,
          "mean+std", str(mean_back_dist+std_back_dist), "max", max_back_dist,
          "original distance min", min_orig_dist, "mean", mean_orig_dist, "std", std_orig_dist,
          "mean-std", str(mean_orig_dist-std_orig_dist), "max", max_orig_dist,)
    best_secret = torch.from_numpy(np_istvan_matrix_keys[best_indices[rand_best_index]]).unsqueeze(0).unsqueeze(0)
    save_image(upsample(best_secret)[0], "one_of_the_best_secret", grayscale="grayscale")
    return best_secret

def get_the_best_random_secret_for_net_arpi(net, test_loader, batch_size, num_epochs, threshold_range, scenario, device, linf_epsilon_clip, l2_epsilon_clip, diff_jpeg_q, real_jpeg_q, num_secret_on_test=0) :
  net.eval()
  if SCENARIOS.JPEGED.value in scenario :
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
    jpeg = jpeg.to(device)
    for param in jpeg.parameters():
      param.requires_grad = False
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  all_the_revealed_something_on_test_set = torch.Tensor()
  threshold_backdoor_dict = {}
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      revealed_something_on_test_set = net.detector(test_images)
      all_the_revealed_something_on_test_set = torch.cat((all_the_revealed_something_on_test_set,revealed_something_on_test_set.detach().cpu()), 0)
    minimax_value = torch.Tensor()
    minimax_secret = torch.Tensor()
    for ith_secret in range(num_secret_on_test) :
      secret_frog = torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0)
      secret = create_batch_from_a_single_image(upsample(secret_frog),all_the_revealed_something_on_test_set.shape[0])
      distance_on_test = torch.sum(torch.square(all_the_revealed_something_on_test_set-secret),dim=(1,2,3))
      ith_min_dist = torch.min(distance_on_test).item()
      if minimax_value.shape[0] < num_epochs :
        minimax_value = torch.cat((minimax_value,torch.ones(1)*ith_min_dist),0)
        minimax_secret = torch.cat((minimax_secret,secret_frog),0)
      else :
        min_minimax_value = torch.min(minimax_value).item()
        if ith_min_dist > min_minimax_value :
          argmin_minimax_value = torch.argmin(minimax_value).item()
          # remove argmin_minimax_value indexth element
          minimax_value  = torch.cat((minimax_value[0:argmin_minimax_value],minimax_value[argmin_minimax_value+1:]),0)
          minimax_secret = torch.cat((minimax_secret[0:argmin_minimax_value],minimax_secret[argmin_minimax_value+1:]),0)
          # add ith_min_dist
          minimax_value = torch.cat((minimax_value,torch.ones(1)*ith_min_dist),0)
          minimax_secret = torch.cat((minimax_secret,secret_frog),0)
    epoch = 0
    mean_of_best_secret = 9999999.0
    for ith_secret_frog in minimax_secret :
      secret = create_batch_from_a_single_image(upsample(ith_secret_frog.unsqueeze(0)),batch_size).to(device)
      all_the_distance_on_backdoor = torch.Tensor().to(device)
      for idx, test_batch in enumerate(test_loader):
        data, labels = test_batch
        test_images = data.to(device)
        backdoored_image = net.generator(secret,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        if SCENARIOS.REAL_JPEG.value in scenario :
          save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx) +"_" + str(epoch), real_jpeg_q)
          backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx)+"_"+str(epoch))
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx)+"_"+str(epoch))
        if SCENARIOS.JPEGED.value in scenario  :
          backdoored_image_clipped = jpeg(backdoored_image_clipped)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret),dim=(1,2,3))
        all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,distance_on_backdoor),0)
      mean_dist_this = torch.mean(all_the_distance_on_backdoor).item()
      star = ""
      if mean_of_best_secret > mean_dist_this :
        mean_of_best_secret = mean_dist_this
        best_secret = ith_secret_frog.unsqueeze(0)
        star = "*"
        for threshold in threshold_range :
          threshold_backdoor_dict[threshold] = torch.sum(all_the_distance_on_backdoor < threshold).item() / all_the_distance_on_backdoor.shape[0]
      print("Epoch",epoch,": revealed distance on backdoor min:",torch.min(all_the_distance_on_backdoor).item(),
            ", mean:",mean_dist_this, star ,
            ", max:",torch.max(all_the_distance_on_backdoor).item(),".",
            "Secret min distance on revealed something from test:",minimax_value[epoch] )
      epoch += 1
  for threshold in threshold_range :
    print(threshold,threshold_backdoor_dict[threshold])
  save_image(upsample(best_secret)[0], "best_secret", grayscale="grayscale")
  return best_secret

def test_specific_secret(net, test_loader, batch_size, scenario, threshold_range, device, linf_epsilon_clip, l2_epsilon_clip, specific_secret, diff_jpeg_q=50, real_jpeg_q=80, verbose_images=False) :
  secret = create_batch_from_a_single_image(specific_secret,batch_size).to(device)
  jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
  jpeg = jpeg.to(device)
  for param in jpeg.parameters():
    param.requires_grad = False
  all_the_distance_on_backdoor_jpeg = torch.Tensor().to(device)
  all_the_distance_on_backdoor = torch.Tensor().to(device)
  all_the_distance_on_test = torch.Tensor().to(device)
  mindist = 99999999.999
  random_without_backdoor = {}
  random_backdoor = {}
  random_clipped_backdoor = {}
  random_difjpeg_backdoor = {}
  random_revealed = {}
  num_of_val_in_random_dicts = 0
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        pos_backdor = [0,0]
        backdoored_image = net.generator(secret, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
        backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        backdoored_image_clipped = test_images
        backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped_small_chunk)
        revealed_something_on_test = net.detector(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
        out = test_images
        out[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image
        backdoored_image = out
      else :
        backdoored_image = net.generator(secret,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        revealed_something_on_test = net.detector(test_images)
      if SCENARIOS.REAL_JPEG.value in scenario :
        save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx), real_jpeg_q)
        backdoored_image_clipped_jpeg = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr"+str(idx))
        removeImages(backdoored_image_clipped.shape[0],"tmpBckdr"+str(idx))
        if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
          revealed_secret_on_backdoor_with_jpeg = net.detector(backdoored_image_clipped_jpeg[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
        else :
          revealed_secret_on_backdoor_with_jpeg = net.detector(backdoored_image_clipped_jpeg)
        distance_on_backdoor_jpeg = torch.sum(torch.square(revealed_secret_on_backdoor_with_jpeg-secret),dim=(1,2,3))
        all_the_distance_on_backdoor_jpeg = torch.cat((all_the_distance_on_backdoor_jpeg,distance_on_backdoor_jpeg),0)
      distance_on_backdoor = torch.sum(torch.square(revealed_secret_on_backdoor-secret),dim=(1,2,3))
      distance_on_test = torch.sum(torch.square(revealed_something_on_test-secret),dim=(1,2,3))
      all_the_distance_on_backdoor = torch.cat((all_the_distance_on_backdoor,distance_on_backdoor),0)
      all_the_distance_on_test = torch.cat((all_the_distance_on_test,distance_on_test),0)
      this_mindist = torch.min(distance_on_backdoor).item()
      if mindist > this_mindist :
        min_backdoor = backdoored_image[torch.argmin(distance_on_backdoor)]
        min_backdoor_clipped = backdoored_image_clipped[torch.argmin(distance_on_backdoor)]
        min_origin = test_images[torch.argmin(distance_on_backdoor)]
        min_jpeg = jpeg(backdoored_image_clipped[torch.argmin(distance_on_backdoor)].unsqueeze(0))[0]
        min_revealed = revealed_secret_on_backdoor[torch.argmin(distance_on_backdoor)]
        mindist = this_mindist
      if idx > 2 and num_of_val_in_random_dicts < 100 :
        for i in range(test_images.shape[0]) :
          lab = labels[i].item()
          if lab not in random_without_backdoor :
            random_without_backdoor[lab] = []
            random_backdoor[lab] = []
            random_clipped_backdoor[lab] = []
            random_difjpeg_backdoor[lab] = []
            random_revealed[lab] = []
          if len(random_without_backdoor[lab]) < 10 :
            random_without_backdoor[lab].append(test_images[i].detach().cpu())
            random_backdoor[lab].append(backdoored_image[i].detach().cpu())
            random_clipped_backdoor[lab].append(backdoored_image_clipped[i].detach().cpu())
            random_difjpeg_backdoor[lab].append(jpeg(backdoored_image_clipped[i].unsqueeze(0))[0].detach().cpu())
            random_revealed[lab].append(revealed_secret_on_backdoor[i].detach().cpu())
            num_of_val_in_random_dicts += 1
    if verbose_images :
      save_image(min_origin, "best-without_backdoor")
      save_image(min_backdoor, "best-backdoor")
      save_image(min_backdoor_clipped, "best-clipped_backdoor")
      save_images_as_jpeg(min_backdoor_clipped.unsqueeze(0), "best-realjpeg_backdoor", real_jpeg_q)
      save_image(min_jpeg, "best-difjpeg_backdoor")
      save_image(min_revealed, "best-revealed")

      save_image_block(random_without_backdoor,scenario+"random-without_backdoor")
      save_image_block(random_backdoor,scenario+"random-backdoor")
      save_image_block(random_clipped_backdoor,scenario+"random-clipped_backdoor")
      save_image_block(random_clipped_backdoor,scenario+"random-realjpeg_backdoor","jpeg", real_jpeg_q)
      save_image_block(random_difjpeg_backdoor,scenario+"random-difjpeg_backdoor")
      save_image_block(random_revealed,scenario+"random-revealed")

    for threshold_percent in np.arange(0.0, 1.05, 0.05) :
      threshold = torch.min(all_the_distance_on_test).item() * threshold_percent
      tpr_for_this = torch.sum(all_the_distance_on_backdoor < threshold) / all_the_distance_on_backdoor.shape[0]
      tpr_for_this_jpeg = -1
      if SCENARIOS.REAL_JPEG.value in scenario :
        tpr_for_this_jpeg = torch.sum(all_the_distance_on_backdoor_jpeg < threshold) / all_the_distance_on_backdoor_jpeg.shape[0]
      print(threshold_percent, tpr_for_this.item(), tpr_for_this_jpeg.item(), threshold, torch.min(all_the_distance_on_test).item())

    tpr_results = {}
    tnr_results = {}
    for threshold in threshold_range :
      tpr = torch.sum(all_the_distance_on_backdoor < threshold).item() / all_the_distance_on_backdoor.shape[0]
      tnr = torch.sum(all_the_distance_on_test >= threshold).item() / all_the_distance_on_test.shape[0]
      tpr_results[threshold] = torch.ones(1)*tpr
      tnr_results[threshold] = torch.ones(1)*tnr
      if verbose_images :
        print(threshold, tpr, tnr)
    return tpr_results, tnr_results

def test_specific_secret_and_threshold(net, test_loader, batch_size, scenario, device, linf_epsilon_clip, l2_epsilon_clip, specific_secret, specific_threshold, real_jpeg_q=80) :
  secret = create_batch_from_a_single_image(specific_secret,batch_size).to(device)
  backdoor_model = ThresholdedBackdoorDetectorStegano(net.detector, specific_secret.to(device), specific_threshold, device).to(device)
  test_acces_backdoor_detect_model = []
  test_acces_backdoor_detect_model_on_backdoor = []
  test_acces_backdoor_detect_model_on_jpeg_backdoor = []
  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      data, labels = test_batch
      test_images = data.to(device)
      targetY_original = torch.from_numpy(np.zeros((test_images.shape[0], 1), np.float32))
      targetY_original = targetY_original.long().view(-1).to(device)
      targetY_backdoor = torch.from_numpy(np.ones((test_images.shape[0], 1), np.float32))
      targetY_backdoor = targetY_backdoor.long().view(-1).to(device)
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        pos_backdor = [0,0]
        predY = backdoor_model(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
      else :
        predY = backdoor_model(test_images)
      test_acces_backdoor_detect_model.append(torch.sum(torch.argmax(predY, dim=1) == targetY_original).item() / test_images.shape[0])
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        backdoored_image = net.generator(secret, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
        backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        backdoored_image_clipped = test_images
        backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
        predY_on_backdoor = backdoor_model(backdoored_image_clipped_small_chunk)
      else :
        backdoored_image = net.generator(secret, test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        predY_on_backdoor = backdoor_model(backdoored_image_clipped)
      test_acces_backdoor_detect_model_on_backdoor.append(torch.sum(torch.argmax(predY_on_backdoor, dim=1) == targetY_backdoor).item() / test_images.shape[0])
      if SCENARIOS.REAL_JPEG.value in scenario:
        save_images_as_jpeg(backdoored_image_clipped, "tmpBckdr" + str(idx) + scenario, real_jpeg_q)
        backdoored_image_clipped_jpeg = open_jpeg_images(backdoored_image_clipped.shape[0], "tmpBckdr" + str(idx) + scenario)
        removeImages(backdoored_image_clipped.shape[0], "tmpBckdr" + str(idx) + scenario)
        if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
          predY_on_backdoor_with_jpeg = backdoor_model(backdoored_image_clipped_jpeg[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
          #test_images_to_backdoored_image_clipped_jpeg = test_images
          #test_images_to_backdoored_image_clipped_jpeg[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_jpeg
          #backdoored_image_clipped_jpeg = test_images_to_backdoored_image_clipped_jpeg
        else :
          predY_on_backdoor_with_jpeg = backdoor_model(backdoored_image_clipped_jpeg).detach().cpu()
        test_acces_backdoor_detect_model_on_jpeg_backdoor.append(torch.sum(torch.argmax(predY_on_backdoor_with_jpeg, dim=1) == targetY_backdoor).item() / test_images.shape[0])

  mean_test_acces_backdoor_detect_model = np.mean(test_acces_backdoor_detect_model)
  mean_test_acces_backdoor_detect_model_on_backdoor = np.mean(test_acces_backdoor_detect_model_on_backdoor)
  print("Accuracy on test images:{0:.4f}, on backdoor images:{1:.4f}".format(mean_test_acces_backdoor_detect_model,mean_test_acces_backdoor_detect_model_on_backdoor))
  if SCENARIOS.REAL_JPEG.value in scenario:
    mean_test_acces_backdoor_detect_model_on_backdoor_jpeg = np.mean(test_acces_backdoor_detect_model_on_jpeg_backdoor)
    print("Accuracy on jpeg backdoor images:{0:.4f}".format(mean_test_acces_backdoor_detect_model_on_backdoor_jpeg))

def robust_random_attack(backdoor_detect_model, test_loader, batch_size, num_epochs, threshold_range, device, threat_model, linf_epsilon, l2_epsilon, specific_secret, scenario, normality_test=False) :
  secret = create_batch_from_a_single_image(specific_secret,batch_size)
  tnr_by_threshold = {}
  distrib_of_random_attack = {}
  sum_of_evaluations = 0
  prev_threshold = 0
  min_threshold = 100000
  for threshold in threshold_range :
    tnr_by_threshold[threshold] = 0
    distrib_of_random_attack[prev_threshold] = 0
    prev_threshold = threshold
  distrib_of_random_attack[prev_threshold] = 0
  list_of_distances = []
  with torch.no_grad():
    for epoch in range(num_epochs) :
      all_the_distance_on_test = torch.Tensor()
      for idx, test_batch in enumerate(test_loader):
        data, labels = test_batch
        test_images = data.to(device)
        if threat_model == "L2" :
          rand_image = torch.normal(mean=torch.zeros(test_images.shape),std=torch.ones(test_images.shape),out=torch.Tensor(size=test_images.shape))
          rand_image_l2 = torch.sqrt(torch.sum(torch.square(rand_image), dim=(1, 2, 3)))
          rand_image_square_sum_l2_divider = l2_epsilon / rand_image_l2
          diff_image  = (rand_image * rand_image_square_sum_l2_divider.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        else :
          rand_image = (torch.rand(test_images.shape)*2)-1
          diff_image = rand_image * linf_epsilon
        diff_image = diff_image.to(device)
        if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
          pos_backdor = [0,0]
          revealed_something_on_test = backdoor_detect_model(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
        else :
          revealed_something_on_test = backdoor_detect_model(test_images+diff_image).detach().cpu()
        distance_on_test = torch.sum(torch.square(revealed_something_on_test-secret),dim=(1,2,3))
        list_of_distances += (distance_on_test.cpu().detach().tolist())
        all_the_distance_on_test = torch.cat((all_the_distance_on_test,distance_on_test),0)
      sum_of_evaluations += all_the_distance_on_test.shape[0]
      print("min",sum_of_evaluations, min_threshold)
      prev_threshold = 0
      for threshold in threshold_range :
        tnr_by_threshold[threshold] += torch.sum(all_the_distance_on_test >= threshold).item()
        distrib_of_random_attack[prev_threshold] += torch.sum(torch.logical_and(prev_threshold <= all_the_distance_on_test, all_the_distance_on_test < threshold)).item()
        if prev_threshold < min_threshold :
          if distrib_of_random_attack[prev_threshold] > 0:
            min_threshold = prev_threshold
        prev_threshold = threshold
      distrib_of_random_attack[prev_threshold] += torch.sum(prev_threshold <= all_the_distance_on_test).item()
      if prev_threshold < min_threshold :
          if distrib_of_random_attack[prev_threshold] > 0:
            min_threshold = prev_threshold
      if normality_test and epoch < 10 :
        k2, p = stats.normaltest(list_of_distances)
        shapiro_statistic_value, shapiro_p_value = stats.shapiro(list_of_distances)
        print("Normality test - Shapiro's statistic value", shapiro_statistic_value, "p value", shapiro_p_value)
        print("Normality test - D’Agostino and Pearson’s statisticvalue", k2, "p value", p)
        mean_list_of_distances = statistics.mean(list_of_distances)
        std_list_of_distances = statistics.stdev(list_of_distances)
        print("Mean:", mean_list_of_distances, "Stdev:", std_list_of_distances)
      if (epoch+1) % 1000 == 0 :
        if normality_test :
          k2, p = stats.normaltest(list_of_distances)
          print("Normality test - D’Agostino and Pearson’s statisticvalue", k2, "p value", p)
          stats.probplot(list_of_distances, dist="norm", plot=plt)
          plt.savefig(IMAGE_PATH+"qq"+str(epoch)+'.eps', format='eps')
        mean_list_of_distances = statistics.mean(list_of_distances)
        std_list_of_distances = statistics.stdev(list_of_distances)
        print("Mean:", mean_list_of_distances, "Stdev:", std_list_of_distances)
        for threshold in threshold_range :
          tnr = tnr_by_threshold[threshold] / sum_of_evaluations
          print(threshold, distrib_of_random_attack[threshold], tnr)
    for threshold in threshold_range :
      tnr = tnr_by_threshold[threshold] / sum_of_evaluations
      print(threshold, distrib_of_random_attack[threshold], tnr)



def robust_test_model(backdoor_generator_model, backdoor_detect_model, robust_model, attack_name, attack_scope, scenario, steps, stepsize, trials, threat_model, test_loader, batch_size, device, linf_epsilon_clip, l2_epsilon_clip, specific_secret, pred_threshold, real_jpeg_q):
  if threat_model == "L2" :
    eps = l2_epsilon_clip
  else :
    eps = linf_epsilon_clip
  secret = create_batch_from_a_single_image(specific_secret,batch_size).to(device)
  if ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope :
    backdoor_model = LastBit(input_shape=image_shape[dataset],device=device).to(device)
  elif ATTACK_SCOPE.THRESHOLDED_STEGANO_BACKDOOR_MODEL.value in attack_scope :
    backdoor_model = ThresholdedBackdoorDetectorStegano(backdoor_detect_model,specific_secret.to(device),pred_threshold,device)
  elif ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope :
    backdoor_model = ThresholdedBackdoorDetector(backdoor_detect_model, pred_threshold, device).to(device)
  else :
    backdoor_model = backdoor_detect_model

  if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
    pos_backdor = [0,0]
    robust_model_with_backdoor = ModelWithSmallBackdoor(backdoor_model, robust_model, pos_backdor, image_shape[DATASET.CIFAR10.value]).to(device)
  else :
    robust_model_with_backdoor = ModelWithBackdoor(backdoor_model,robust_model).to(device)

  robust_model.eval()
  robust_model_with_backdoor.eval()
  backdoor_model.eval()
  backdoor_detect_model.eval()
  backdoor_generator_model.eval()

  if ATTACK_NAME.AUTO_ATTACK.value in attack_name:
    if ATTACK_NAME.SQUARE_ATTACK.value in attack_name :
      version='custom'
      attacks_to_run=[ATTACK_NAME.SQUARE_ATTACK.value]
    elif ATTACK_NAME.FAB.value in attack_name :
      version='custom'
      attacks_to_run=["fab"]
    elif ATTACK_NAME.FABT.value in attack_name :
      version='custom'
      attacks_to_run=[ATTACK_NAME.FABT.value]
    elif ATTACK_NAME.APGD_CE.value in attack_name :
      version='custom'
      attacks_to_run=[ATTACK_NAME.APGD_CE.value]
    elif ATTACK_NAME.APGD_DLR.value in attack_name :
      version='custom'
      attacks_to_run=[ATTACK_NAME.APGD_DLR.value]
    elif ATTACK_NAME.APGD_DLR_T.value in attack_name :
      version='custom'
      attacks_to_run=[ATTACK_NAME.APGD_DLR_T.value]
    else :
      version='standard'
      attacks_to_run=[]
    apgd_n_restarts = trials
    apgd_targeted_n_target_classes = 9
    apgd_targeted_n_restarts = 1
    fab_n_target_classes = 9
    fab_n_restarts = trials
    square_n_queries = 5000
    if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope :
      attack_for_robust_model = AutoAttack(robust_model, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device)
      attack_for_robust_model.apgd.n_restarts = apgd_n_restarts
      attack_for_robust_model.apgd_targeted.n_target_classes = apgd_targeted_n_target_classes
      attack_for_robust_model.apgd_targeted.n_restarts = apgd_targeted_n_restarts
      attack_for_robust_model.fab.n_restarts = fab_n_restarts
      if ATTACK_NAME.FABT.value in attack_name :
        attack_for_robust_model.fab.n_target_classes = fab_n_target_classes
      attack_for_robust_model.square.n_queries = square_n_queries
    if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
      attack_for_robust_model_with_backdoor = AutoAttack(robust_model_with_backdoor, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device )
      attack_for_robust_model_with_backdoor.apgd.n_restarts = apgd_n_restarts
      attack_for_robust_model_with_backdoor.apgd_targeted.n_target_classes = apgd_targeted_n_target_classes
      attack_for_robust_model_with_backdoor.apgd_targeted.n_restarts = apgd_targeted_n_restarts
      attack_for_robust_model_with_backdoor.fab.n_restarts = fab_n_restarts
      attack_for_robust_model_with_backdoor.fab.n_restarts = fab_n_restarts
      if ATTACK_NAME.FABT.value in attack_name :
        attack_for_robust_model_with_backdoor.fab.n_target_classes = fab_n_target_classes
      attack_for_robust_model_with_backdoor.square.n_queries = square_n_queries
    if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope or \
        ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope :
      attack_for_backdoor_detect_model = AutoAttack(backdoor_model, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device)
      attack_for_backdoor_detect_model.apgd.n_restarts = apgd_n_restarts
      attack_for_backdoor_detect_model.apgd_targeted.n_target_classes = apgd_targeted_n_target_classes
      attack_for_backdoor_detect_model.apgd_targeted.n_restarts = apgd_targeted_n_restarts
      attack_for_backdoor_detect_model.fab.n_restarts = fab_n_restarts
      if ATTACK_NAME.FABT.value in attack_name :
        attack_for_backdoor_detect_model.fab.n_target_classes = fab_n_target_classes
      attack_for_backdoor_detect_model.square.n_queries = square_n_queries
  else :
    fb_robust_model = fb.PyTorchModel(robust_model, bounds=(0, 1), device=str(device))
    fb_robust_model_with_backdoor = fb.PyTorchModel(robust_model_with_backdoor, bounds=(0, 1), device=str(device))
    fb_backdoor_detect_model = fb.PyTorchModel(backdoor_model, bounds=(0, 1), device=str(device))
    if  attack_name == "BoundaryAttack" :
      attack = fb.attacks.BoundaryAttack()
    elif attack_name == "PGD" or attack_name == "ProjectedGradientDescentAttack" :
      if threat_model == "L2" :
        attack = fb.attacks.L2PGD(abs_stepsize=stepsize, steps=steps, random_start=True)
      else :
        attack = fb.attacks.LinfPGD(abs_stepsize=stepsize, steps=steps, random_start=True)
    elif attack_name == "BrendelBethgeAttack" :
      if threat_model == "L2" :
        attack = fb.attacks.L2BrendelBethgeAttack(steps=steps)
      else :
        attack = fb.attacks.LinfinityBrendelBethgeAttack(steps=steps)
    else :
      attack = fb.attacks.L2PGD(abs_stepsize=stepsize, steps=steps, random_start=True)
    if params.trials > 1:
        attack = attack.repeat(trials)



  '''
  final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
  final1_bias = int(str(pred_threshold)[2:])
  final2_w  = -1
  final2_bias = 1
  final3_w  = -1
  final3_bias = 1
  '''
  num_of_batch = 0

  adv_robust_model = []
  adv_robust_model_with_backdoor = []
  adv_backdoor_detect_model = []

  test_acces_robust_model = []
  test_acces_robust_model_with_backdoor = []
  test_acces_backdoor_detect_model = []

  test_acces_robust_model_on_backdoor = []
  test_acces_robust_model_with_backdoor_on_backdoor = []
  test_acces_backdoor_detect_model_on_backdoor = []

  test_acces_robust_model_on_backdoor_with_jpeg = []
  test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg = []
  test_acces_backdoor_detect_model_on_backdoor_with_jpeg = []

  test_rob_acces_robust_model = []
  test_rob_acces_robust_model_with_backdoor = []
  test_rob_acces_backdoor_detect_model = []
  test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = []

  for idx, test_batch in enumerate(test_loader):
    num_of_batch += 1
    # Saves images
    data, labels = test_batch
    test_images = data.to(device)
    test_y = labels
    test_y_on_GPU = labels.to(device)
    targetY_original = torch.from_numpy(np.zeros((test_images.shape[0], 1), np.float32))
    targetY_original = targetY_original.long().view(-1)
    targetY_original_on_GPU = targetY_original.to(device)

    if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
      predY = backdoor_model(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
    else :
      predY = backdoor_model(test_images).detach().cpu()

    test_acces_backdoor_detect_model.append(torch.sum(torch.argmax(predY, dim=1) == targetY_original).item()/test_images.shape[0])
    predY_on_robustmodel_with_backdoor = robust_model_with_backdoor(test_images).detach().cpu()
    test_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, test_images, test_y))
    predY_on_robustmodel = robust_model(test_images).detach().cpu()
    test_acces_robust_model.append(torch.sum(torch.argmax(predY_on_robustmodel, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, test_images, test_y))

    if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_robust_model = attack_for_robust_model.run_standard_evaluation(test_images, test_y_on_GPU)
      else :
        x_adv_robust_model, _, success_robust_model = attack(fb_robust_model, test_images, criterion=test_y, epsilons=eps)
      adv_robust_model.append(x_adv_robust_model)
      predY_on_robustmodel_adversarial = robust_model(x_adv_robust_model).detach().cpu()
      test_rob_acces_robust_model.append(torch.sum(torch.argmax(predY_on_robustmodel_adversarial, dim=1) == test_y).item()/test_images.shape[0])
      #test_rob_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, x_adv_robust_model, test_y))
      mean_test_rob_acces_robust_model = np.mean(test_rob_acces_robust_model)
    else :
      mean_test_rob_acces_robust_model = -1.0
    if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_robust_model_with_backdoor = attack_for_robust_model_with_backdoor.run_standard_evaluation(test_images, test_y_on_GPU)
      else :
        x_adv_robust_model_with_backdoor, _, success_robust_model_with_backdoor = attack(fb_robust_model_with_backdoor, test_images, criterion=test_y, epsilons=eps)
      adv_robust_model_with_backdoor.append(x_adv_robust_model_with_backdoor)
      predY_on_robustmodel_with_backdoor_adversarial = robust_model_with_backdoor(x_adv_robust_model_with_backdoor).detach().cpu()
      test_rob_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_adversarial, dim=1) == test_y).item()/test_images.shape[0])
      #test_rob_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, x_adv_robust_model_with_backdoor, test_y))
      mean_test_rob_acces_robust_model_with_backdoor = np.mean(test_rob_acces_robust_model_with_backdoor)
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        predY_on_robustmodel_with_backdoor_adversarial = backdoor_model(x_adv_robust_model_with_backdoor[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
      else :
        predY_on_robustmodel_with_backdoor_adversarial = backdoor_model(x_adv_robust_model_with_backdoor).detach().cpu()
      test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_adversarial, dim=1) == targetY_original).item()/test_images.shape[0])
      mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = np.mean(test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor)
    else :
      mean_test_rob_acces_robust_model_with_backdoor = -1.0
      mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = -1.0 #
    if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope \
        or ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_backdoor_detect_model = attack_for_backdoor_detect_model.run_standard_evaluation(test_images, targetY_original_on_GPU)
      else :
        x_adv_backdoor_detect_model, _, success_backdoor_detect_model = attack(fb_backdoor_detect_model, test_images, criterion=targetY_original, epsilons=eps)
      adv_backdoor_detect_model.append(x_adv_backdoor_detect_model)
      #test_rob_acces_backdoor_detect_model.append(fb.utils.accuracy(fb_backdoor_detect_model, x_adv_backdoor_detect_model, targetY_original))
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        predY_on_adversarial = backdoor_model(x_adv_backdoor_detect_model[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
      else :
        predY_on_adversarial = backdoor_model(x_adv_backdoor_detect_model).detach().cpu()
      test_rob_acces_backdoor_detect_model.append(torch.sum(torch.argmax(predY_on_adversarial, dim=1) == targetY_original).item()/test_images.shape[0])
      mean_test_rob_acces_backdoor_detect_model = np.mean(test_rob_acces_backdoor_detect_model)
    else :
      mean_test_rob_acces_backdoor_detect_model = -1.0 #

    targetY_backdoor = torch.from_numpy(np.ones((test_images.shape[0], 1), np.float32))
    targetY_backdoor = targetY_backdoor.long().view(-1)

    if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
      backdoored_image = backdoor_generator_model(secret,test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
      backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], scenario, l2_epsilon_clip, linf_epsilon_clip, device)
      backdoored_image_clipped = test_images
      backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
      predY_on_backdoor = backdoor_model(backdoored_image_clipped_small_chunk).detach().cpu()
      open_jpeg_flag_for_cifar10_model = True
    else :
      backdoored_image = backdoor_generator_model(secret,test_images)
      backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
      predY_on_backdoor = backdoor_model(backdoored_image_clipped).detach().cpu()
      open_jpeg_flag_for_cifar10_model = False

    test_acces_backdoor_detect_model_on_backdoor.append(torch.sum(torch.argmax(predY_on_backdoor, dim=1) == targetY_backdoor).item()/test_images.shape[0])
    predY_on_robustmodel_with_backdoor_on_backdoor = robust_model_with_backdoor(backdoored_image_clipped).detach().cpu()
    test_acces_robust_model_with_backdoor_on_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_on_backdoor, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_with_backdoor_on_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, backdoored_image_clipped, test_y))
    predY_on_robustmodel_on_backdoor = robust_model(backdoored_image_clipped).detach().cpu()
    test_acces_robust_model_on_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_on_backdoor, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_on_backdoor.append(fb.utils.accuracy(fb_robust_model, backdoored_image_clipped, test_y))

    if SCENARIOS.BYTESIO.value in scenario :
      filename_postfix = SCENARIOS.BYTESIO.value
    else :
      filename_postfix = "tmpBckdr"+ str(idx) + scenario + attack_name
    backdoored_image_clipped_jpeg = save_and_open_and_remove_jpeg_images(backdoored_image_clipped, filename_postfix, quality=real_jpeg_q, cifar10_model=False).to(device)

    if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
      predY_on_backdoor_with_jpeg = backdoor_model(backdoored_image_clipped_jpeg[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
      #test_images_to_backdoored_image_clipped_jpeg = test_images
      #test_images_to_backdoored_image_clipped_jpeg[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_jpeg
      #backdoored_image_clipped_jpeg = test_images_to_backdoored_image_clipped_jpeg
    else :
      predY_on_backdoor_with_jpeg = backdoor_model(backdoored_image_clipped_jpeg).detach().cpu()
    test_acces_backdoor_detect_model_on_backdoor_with_jpeg.append(torch.sum(torch.argmax(predY_on_backdoor_with_jpeg, dim=1) == targetY_backdoor).item()/test_images.shape[0])
    predY_on_robustmodel_with_backdoor_on_backdoor_with_jpeg = robust_model_with_backdoor(backdoored_image_clipped_jpeg).detach().cpu()
    test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_on_backdoor_with_jpeg, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_with_backdoor_on_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, backdoored_image_clipped, test_y))
    predY_on_robustmodel_on_backdoor_with_jpeg = robust_model(backdoored_image_clipped_jpeg).detach().cpu()
    test_acces_robust_model_on_backdoor_with_jpeg.append(torch.sum(torch.argmax(predY_on_robustmodel_on_backdoor_with_jpeg, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_on_backdoor.append(fb.utils.accuracy(fb_robust_model, backdoored_image_clipped, test_y))

    mean_test_acces_backdoor_detect_model = np.mean(test_acces_backdoor_detect_model)
    mean_test_acces_robust_model_with_backdoor = np.mean(test_acces_robust_model_with_backdoor)
    mean_test_acces_robust_model = np.mean(test_acces_robust_model)

    mean_test_acces_backdoor_detect_model_on_backdoor = np.mean(test_acces_backdoor_detect_model_on_backdoor)
    mean_test_acces_robust_model_with_backdoor_on_backdoor = np.mean(test_acces_robust_model_with_backdoor_on_backdoor)
    mean_test_acces_robust_model_on_backdoor = np.mean(test_acces_robust_model_on_backdoor)

    mean_test_acces_backdoor_detect_model_on_backdoor_with_jpeg = np.mean(test_acces_backdoor_detect_model_on_backdoor_with_jpeg)
    mean_test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg = np.mean(test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg)
    mean_test_acces_robust_model_on_backdoor_with_jpeg = np.mean(test_acces_robust_model_on_backdoor_with_jpeg)

    #mean_test_acces_backdoor_detect_model_on_adversarial = np.mean(test_acces_backdoor_detect_model_on_adversarial)
    #mean_test_acces_backdoor_detect_model_on_adversarial = np.mean(test_acces_backdoor_detect_model_on_adversarial)

    print('Adversary testing: Batch {0}/{1}. '.format( idx + 1, batch_size ), end='')
    print('Accuracy on test set backdoor_detect_model: {0:.4f}, robust_model_with_backdoor: {1:.4f}, robust_model: {2:.4f}; '
    'Robust accuracy on test set backdoor_detect_model: {3:.4f}, {4:.4f}, robust_model_with_backdoor: {5:.4f}, robust_model: {6:.4f}; '
    'Accuracy on backdoor images backdoor_detect_model: {7:.4f}, robust_model_with_backdoor: {8:.4f}, robust_model: {9:.4f}; '
    'Accuracy on JPEG backdoor images backdoor_detect_model: {10:.4f}, robust_model_with_backdoor: {11:.4f}, robust_model: {12:.4f}; '.format(
    mean_test_acces_backdoor_detect_model,mean_test_acces_robust_model_with_backdoor,mean_test_acces_robust_model,
    mean_test_rob_acces_backdoor_detect_model,mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor,
    mean_test_rob_acces_robust_model_with_backdoor,mean_test_rob_acces_robust_model,
    mean_test_acces_backdoor_detect_model_on_backdoor,mean_test_acces_robust_model_with_backdoor_on_backdoor,mean_test_acces_robust_model_on_backdoor,
    mean_test_acces_backdoor_detect_model_on_backdoor_with_jpeg, mean_test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg,
    mean_test_acces_robust_model_on_backdoor_with_jpeg))
    print('{0:.4f} & {1:.4f} & {2:.4f} | {3:.4f} & {4:.4f} & {5:.4f} | {6:.4f} & {7:.4f} & {8:.4f}'.format(mean_test_rob_acces_robust_model,
    mean_test_rob_acces_robust_model_with_backdoor,(1.0-mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor),
    mean_test_acces_robust_model_on_backdoor,mean_test_acces_robust_model_with_backdoor_on_backdoor,mean_test_acces_backdoor_detect_model_on_backdoor,
    mean_test_acces_robust_model_on_backdoor_with_jpeg, mean_test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg,
    mean_test_acces_backdoor_detect_model_on_backdoor_with_jpeg))
    #mean_test_acces_backdoor_detect_model_on_adversarial,mean_test_acces_backdoor_detect_model_on_adversarial
    #'Accuracy on adversarial images backdoor_detect_model: {12:.4f}, backdoor_detect_model: {13:.4f}; '

    '''
    logits_backdoor = backdoor_detect_model(test_images)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*final1_w)+final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*final2_w)+final2_bias).unsqueeze(1)
    predicted_as_original = torch.relu((predicted_as_backdoor*final3_w)+final3_bias).unsqueeze(1)
    softmax_robust_model = robust_model(test_images)*predicted_as_original
    softmax_robust_model_shifted = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    softmax_backdoored = softmax_robust_model + softmax_robust_model_shifted
    '''

  if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope and len(adv_robust_model) > 0:
    index = 0
    for images in adv_robust_model :
      save_images(images, "adv_robust_model_" + str(index) + "_")
      index += 1
  if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
    index = 0
    for images in adv_robust_model_with_backdoor:
      save_images(images, "adv_robust_model_with_backdoor_" + str(index) + "_")
      index += 1
  if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope or\
      ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope :
    index = 0
    for images in adv_backdoor_detect_model :
      save_images(images, "adv_backdoor_detect_model_" + str(index) + "_")
      index += 1

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument('--secret', type=str, default="NOPE")
parser.add_argument('--mode', type=str, default="train_test")
parser.add_argument('--generator', type=str, default="NOPE")
parser.add_argument('--detector', type=str, default="NOPE")
parser.add_argument("--model_det", type=str, help="|".join(DETECTORS.keys()), default='detdeepsteganorigwgss')
parser.add_argument("--model_gen", type=str, help="|".join(GENERATORS.keys()), default='gendeepsteganorigwgss')
parser.add_argument("--loss_mode", type=str,  default="lossbyadd")
parser.add_argument("--scenario", type=str, default="withoutjpeg")
parser.add_argument("--train_scope", type=str, default="normal")
parser.add_argument("--robust_model", type=str , default="Gowal2020Uncovering_28_10_extra")
parser.add_argument("--threat_model", type=str , default="Linf")
parser.add_argument("--attack", type=str , default="PGD")
parser.add_argument("--attack_scope", type=str , default="robust_model&robust_model_with_backdoor")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--regularization_start_epoch', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--beta', type=float, default=1.0)
parser.add_argument('--jpeg_q', type=int, default=50)
parser.add_argument('--real_jpeg_q', type=int, default=80)
parser.add_argument("--pos_weight", type=float, default=1.0)
parser.add_argument("--pred_threshold", type=float, default=0.5)
parser.add_argument("--l", type=float, default=0.0001)
parser.add_argument("--l_step", type=int, default=1)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.002)
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--n_mean', type=float, default=0.0)
parser.add_argument('--n_stddev', type=float, default=1.0/255.0)
#parser.add_argument('--linf_epsilon_clip', type=float, default=0.03134) # (8.0/255.0) , 0.01564 (4.0/255.0)
parser.add_argument('--linf_epsilon', type=float, default=8.0/255.0) # (8.0/255.0) , 0.01564 (4.0/255.0)
#parser.add_argument('--l2_epsilon_clip', type=float, default=0.49999) #0.5
parser.add_argument('--l2_epsilon', type=float, default=0.5) #0.5
parser.add_argument('--start_of_the_threshold_range', type=float, default=1.0)
parser.add_argument('--end_of_the_threshold_range', type=float, default=101.0)
parser.add_argument('--step_of_the_threshold_range', type=float, default=1.0)
parser.add_argument('--num_secret_on_test', type=int, default=1000)

params = parser.parse_args()

# Other Parameters
device = torch.device('cuda:'+str(params.gpu))
dataset = params.dataset
pred_threshold = params.pred_threshold
robust_model_name = params.robust_model
threat_model = params.threat_model
attack_name = params.attack
attack_scope = params.attack_scope
steps = params.steps
stepsize = params.step_size
trials = params.trials

# Hyper Parameters
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
alpha = params.alpha
beta = params.beta
pos_weight = (torch.ones(1)*params.pos_weight)
l = params.l
l_step = params.l_step
last_l = l * np.power(10,l_step-1)
linf_epsilon = params.linf_epsilon
l2_epsilon = params.l2_epsilon

mode = params.mode
train_scope = params.train_scope
scenario = params.scenario

model = params.model
secret = params.secret
real_jpeg_q = params.real_jpeg_q
threshold_range = np.arange(params.start_of_the_threshold_range,params.end_of_the_threshold_range,params.step_of_the_threshold_range)
num_secret_on_test = params.num_secret_on_test

if SCENARIOS.CIFAR10_MODEL.value in scenario :
  model_dataset = DATASET.CIFAR10.value
else :
  model_dataset = dataset

if SCENARIOS.VERBOSE.value in scenario :
  verbose = True
else :
  verbose = False

train_loader, val_loader, test_loader = get_loaders(dataset, batch_size)

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model

best_secret = torch.Tensor()
if secret != 'NOPE' :
  best_secret = open_secret(secret)

if params.loss_mode == LOSSES.SIMPLE.value :
  generator = GENERATORS[params.model_gen](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  generator.to(device)
  if params.generator != 'NOPE':
    generator.load_state_dict(torch.load(MODELS_PATH+params.generator))
  detector = DETECTORS[params.model_det](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  detector.to(device)
  if params.detector != 'NOPE':
    detector.load_state_dict(torch.load(MODELS_PATH+params.detector))
  if MODE.TRAIN.value in mode :
    generator, detector, mean_train_loss, loss_history= train_model(generator, detector, train_loader, batch_size, val_loader, train_scope, num_epochs, params.loss_mode, alpha=alpha, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device, pos_weight=pos_weight.to(device), jpeg_q=params.jpeg_q)
  if MODE.TEST.value in mode :
    mean_test_loss = test_model(generator, detector, test_loader, batch_size, scenario , params.loss_mode, beta=beta, l=last_l, device=device, jpeg_q=params.jpeg_q,  linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, best_secret=best_secret, pred_threshold=pred_threshold, pos_weight=pos_weight.to(device))
  backdoor_detect_model = detector
  backdoor_generator_model = generator
else :
  if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
    net = Net(gen_holder=GENERATORS[params.model_gen], det_holder=DETECTORS[params.model_det], image_shape=image_shape[DATASET.CIFAR10.value], color_channel= color_channel[DATASET.CIFAR10.value], jpeg_q=params.jpeg_q,  device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
  else :
    net = Net(gen_holder=GENERATORS[params.model_gen], det_holder=DETECTORS[params.model_det], image_shape=image_shape[dataset], color_channel= color_channel[dataset], jpeg_q=params.jpeg_q,  device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
  net.to(device)
  if model != 'NOPE' :
    net.load_state_dict(torch.load(MODELS_PATH+model,map_location=device))
  if MODE.TRAIN.value in mode :
    net, _ ,mean_train_loss, loss_history = train_model(net, None, train_loader, batch_size, val_loader, train_scope, num_epochs, params.loss_mode, alpha=alpha, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device, pos_weight=pos_weight.to(device),jpeg_q=params.jpeg_q)
  backdoor_detect_model = net.detector
  backdoor_generator_model = net.generator
  if SCENARIOS.VALID.value in scenario :
    validation_loader = val_loader
  else :
    validation_loader = test_loader
  if MODE.MULTIPLE_TEST.value in mode :
    test_multiple_random_secret(net=net, test_loader=validation_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, threshold_range=threshold_range, device=device, linf_epsilon=linf_epsilon, l2_epsilon=l2_epsilon, real_jpeg_q=params.real_jpeg_q)
  if MODE.CHOSE_THE_BEST_AUC_SECRET.value in mode :
    get_the_best_random_secret_for_net_auc(net=net, test_loader=validation_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, threshold_range=threshold_range, device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, diff_jpeg_q=params.jpeg_q, real_jpeg_q=params.real_jpeg_q)
  if MODE.CHOSE_THE_BEST_TPR_SECRET.value in mode :
    get_the_best_random_secret_for_net(net=net, test_loader=validation_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, threshold_range=threshold_range, device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, diff_jpeg_q=params.jpeg_q, real_jpeg_q=params.real_jpeg_q)
  if MODE.CHOSE_THE_BEST_ARPI_SECRET.value in mode :
    best_secret = get_the_best_random_secret_for_net_arpi(net=net, test_loader=validation_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, threshold_range=threshold_range, device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, diff_jpeg_q=params.jpeg_q, real_jpeg_q=params.real_jpeg_q, num_secret_on_test=num_secret_on_test)
  if MODE.CHOSE_THE_BEST_GRAY_SECRET.value in mode :
    best_secret = get_the_best_gray_secret_for_net(net=net, test_loader=validation_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, threshold_range=threshold_range, device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, diff_jpeg_q=params.jpeg_q, real_jpeg_q=params.real_jpeg_q)
  if MODE.TEST.value in mode :
    mean_test_loss = test_model(net, None, test_loader, batch_size, scenario , params.loss_mode, beta=beta, l=last_l, device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, best_secret=best_secret, jpeg_q=params.jpeg_q, pred_threshold=pred_threshold, pos_weight=pos_weight.to(device))
  if MODE.PRED_THRESH.value in mode :
    test_specific_secret(net, validation_loader, batch_size, scenario, threshold_range, device, linf_epsilon, l2_epsilon, best_secret, real_jpeg_q=params.real_jpeg_q, verbose_images=verbose)
  if MODE.TEST_THRESHOLDED_BACKDOOR.value in mode :
    test_specific_secret_and_threshold(net, validation_loader, batch_size, scenario, device, linf_epsilon, l2_epsilon, best_secret, specific_threshold=pred_threshold, real_jpeg_q=params.real_jpeg_q)

if MODE.ATTACK.value in mode :
  robust_model = load_model(model_name=robust_model_name, dataset=dataset, threat_model=robust_model_threat_model).to(device)
  robust_test_model(backdoor_generator_model, backdoor_detect_model, robust_model, attack_name, attack_scope, scenario, steps, stepsize, trials, robust_model_threat_model, test_loader, batch_size,  device=device, linf_epsilon_clip=linf_epsilon, l2_epsilon_clip=l2_epsilon, specific_secret=best_secret, pred_threshold=pred_threshold, real_jpeg_q=real_jpeg_q)
if MODE.RANDOM_ATTACK.value in mode :
  if SCENARIOS.NORMALITY_TEST.value in scenario :
    normality_test = True
  else :
    normality_test = False
  robust_random_attack(backdoor_detect_model,test_loader=test_loader,batch_size=batch_size,num_epochs=num_epochs,l2_epsilon=l2_epsilon,linf_epsilon=linf_epsilon,specific_secret=best_secret,threshold_range=threshold_range,device=device,threat_model=threat_model,scenario=scenario,normality_test=normality_test)
