import numpy as np
from PIL import Image
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from argparse import ArgumentParser
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG
from backdoor_model import Net, LastBit, ModelWithBackdoor, ThresholdedBackdoorDetector, ThresholdedBackdoorDetectorStegano, DETECTORS, GENERATORS
from robustbench import load_model
from autoattack import AutoAttack
import foolbox as fb
from enum import Enum

MODELS_PATH = '../res/models/'
DATA_PATH = '../res/data/'
IMAGE_PATH = '../res/images/'
SECRET_FROG_PATH = 'frog.jpg'

std = {}
mean = {}
image_shape = {}
color_channel = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std['IMAGENET'] = [0.229, 0.224, 0.225]
mean['IMAGENET'] = [0.485, 0.456, 0.406]
color_channel['IMAGENET'] = 3

#  of cifar10 dataset.
std['cifar10'] = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
mean['cifar10'] = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
image_shape['cifar10'] = [32, 32]
color_channel['cifar10'] = 3
#  of mnist dataset.
std['MNIST'] = [0.3084485240270358]
mean['MNIST'] = [0.13092535192648502]
image_shape['MNIST'] = [28, 28]
color_channel['MNIST'] = 1

LINF_EPS =  8.0/255.0 + 0.00001
L2_EPS =  0.5 + 0.00001

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

class TRAINS_ON(Enum) :
  NORMAL = "normal"
  JPEGED = "jpeged"
  JPEG = "jpeg"
  NOISED = "noised"
  CLIP_L2LINF = "clipl2linf"
  CLIP_L2 = "clipl2only"
  CLIP_LINF = "cliplinfonly"
  LINFCLIP_JPEG = "linfclip;jpeg"
  L2CLIP_JPEG = "l2clip;jpeg"
  L2LINFCLIP_JPEG = "l2linfclip;jpeg"
  L2LINFCLIP_JPEG_AND_NORMALORIGINAL = "l2linfclip;jpeg&normaloriginal"
  JPEG_AND_NOISE = "jpeg&noise"
  JPEG_AND_NORMAL = "jpeg&normal"
  JPEG_AND_NOISE_AND_NORMAL = "jpeg&noise&normal"


CRITERION_GENERATOR = nn.MSELoss(reduction="sum")

L1_MODIFIER = 1.0/100.0
L2_MODIFIER = 1.0/10.0
LINF_MODIFIER = 1.0

class ATTACK_SCOPE(Enum):
  ROBUST_MODEL = "robust_model"
  ROBUST_MODEL_WITH_BACKDOOR = "with_backdoor"
  BACKDOOR_MODEL_WITHOUT_THRESHOLD = "backdoor_detect_model"
  THRESHOLDED_BACKDOOR_MODEL = "thresholded"
  THRESHOLDED_STEGANO_BACKDOOR_MODEL = "thresholdstegano"
  LASTBIT_MODEL = "lastbit"

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

def saveImages(images, filename_postfix) :
  #denormalized_images = (denormalize(images=images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset]) * 255).byte()
  denormalized_images = (images*255).byte()
  if color_channel[dataset] == 1 :
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    for i in range(0, denormalized_images.shape[0]):
      img = Image.fromarray(denormalized_images[i, 0], "L")
      img.save(os.path.join(IMAGE_PATH, dataset+"_"+filename_postfix+"_" + str(i) + ".png"))
  elif color_channel[dataset] == 3 :
    for i in range(0, denormalized_images.shape[0]):
      tensor_to_image = transforms.ToPILImage()
      img = tensor_to_image(denormalized_images[i])
      img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".png"))

def saveImagesAsJpeg(images, filename_postfix, quality=75 ) :
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

def openJpegImages(num_of_images, filename_postfix) :
  loader = transforms.Compose([transforms.ToTensor()])
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


def open_secret_frog() :
  loader = transforms.Compose([transforms.ToTensor()])
  opened_image = Image.open(os.path.join('', SECRET_FROG_PATH)).convert('RGB')
  opened_image_tensor = loader(opened_image).unsqueeze(0)
  return opened_image_tensor


def create_batch_from_a_single_image(image, batch_size):
  image_a = []
  for i in range(batch_size) :
    image_a.append(image)
  batch = torch.cat(image_a, 0)
  return batch

def removeImages(num_of_images, filename_postfix) :
  for i in range(0, num_of_images):
    fileName = os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + "_" + str(i) + ".jpeg")
    if os.path.exists(fileName):
      os.remove(fileName)

def saveImage(image, filename_postfix) :
  denormalized_images = (image * 255).byte()
  if color_channel[dataset] == 1:
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + ".png"))
  elif color_channel[dataset] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix +  ".png"))

def linf_clip(backdoored_image, original_images, linf_epsilon_clip) :
  diff_image = backdoored_image - original_images
  diff_image_clipped = torch.clamp(diff_image, -linf_epsilon_clip, linf_epsilon_clip)
  linf_clipped_backdoor = original_images + diff_image_clipped
  #diff_image_linf = linf_clipped_backdoor - original_images
  return linf_clipped_backdoor

def l2_clip(backdoored_image, original_images, l2_epsilon_clip, device) :
  diff_image = backdoored_image - original_images
  l2 = torch.sqrt(torch.sum(torch.square(diff_image), dim=(1,2,3)))
  #l2_to_be_deleted = (torch.relu(l2 - l2_epsilon_clip))/l2
  l2_to_be_deleted = torch.min(l2, (torch.ones(backdoored_image.shape[0])*l2_epsilon_clip).to(device))
  diff_image_square = torch.square(diff_image)
  diff_image_square_sum = torch.sum(torch.square(diff_image), dim=(1, 2, 3))
  diff_image_square_sum_l2_to_be_deleted = diff_image_square_sum - torch.square(l2_to_be_deleted)
  diff_image_square_sum_l2_divider = torch.sqrt(1 - (diff_image_square_sum_l2_to_be_deleted / diff_image_square_sum))
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

def train_model(net1, net2, train_loader, train_scope, num_epochs, loss_mode, beta, l, l_step, linf_epsilon_clip, l2_epsilon_clip, reg_start, learning_rate, device, pos_weight):
  # Save optimizer
  if loss_mode == "simple" :
    optimizer_generator = optim.Adam(net1.parameters(), lr=learning_rate)
    optimizer_detector = optim.Adam(net2.parameters(), lr=learning_rate*100)
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][0], differentiable=True, quality=75)
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

    # Train one epoch
    for idx, train_batch in enumerate(train_loader):
      data, _ = train_batch
      data = data.to(device)

      train_images = Variable(data, requires_grad=False)
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
        secret = train_images[:train_images.shape[0]//2]
        train_images = train_images[train_images.shape[0]//2:]
        optimizer.zero_grad()
        backdoored_image = net1.generator(secret, train_images)
        backdoored_image_clipped = clip(backdoored_image, train_images, train_scope, l2_epsilon_clip, linf_epsilon_clip, device)
        if TRAINS_ON.JPEG.value in train_scope :
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          secret_pred = net1.detector(jpeged_backdoored_image)
        else :
          secret_pred = net1.detector(backdoored_image_clipped)
        train_loss = loss_only_detector_mse(secret_pred,secret)
        train_loss.backward()
        optimizer.step()
      else:
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        # ["normal","jpeged","noised","linfclip;jpeg","l2clip;jpeg","l2linfclip;jpeg","l2linfclip;jpeg&normaloriginal","jpeg&noise","jpeg&normal","jpeg&noise&normal"]
        if train_scope == "normal" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          next_input = torch.cat((backdoored_image_clipped,train_images),0)
          logits = net1.detector(next_input)
        elif train_scope == "jpeged" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
          logits = net1.detector(next_input)
        elif train_scope == "noised" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
          next_input = torch.cat((backdoored_image_with_noise, image_with_noise),0)
          logits = net1.detector(next_input)
        elif train_scope == "linfclip;jpeg" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_linf_clipped = linf_clip(backdoored_image_clipped, train_images, linf_epsilon_clip)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_linf_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
          logits = net1.detector(next_input)
        elif train_scope == "l2clip;jpeg" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_clipped = l2_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, device)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
          logits = net1.detector(next_input)
        elif train_scope == "l2linfclip;jpeg" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_clipped = l2_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, device)
          backdoored_image_l2_linf_clipped = linf_clip(backdoored_image_l2_clipped, train_images, linf_epsilon_clip)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
          logits = net1.detector(next_input)
        elif train_scope == "l2linfclip;jpeg&normaloriginal" :
          targetY_original_jpeged = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY_backdoored, targetY_original_jpeged, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_clipped = l2_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, device)
          backdoored_image_l2_linf_clipped = linf_clip(backdoored_image_l2_clipped, train_images, linf_epsilon_clip)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image, train_images), 0)
          logits = net1.detector(next_input)
        elif train_scope == "jpeg&noise":
          targetY_backdoored_noise = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY_original_noise = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY_backdoored, targetY_backdoored_noise, targetY_original, targetY_original_noise),0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((jpeged_backdoored_image, backdoored_image_with_noise, jpeged_image, image_with_noise), 0)
          logits = net1.detector(next_input)
        elif train_scope == "jpeg&normal" :
          targetY_backdoored_jpeged = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY_original_jpeged = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY_backdoored, targetY_backdoored_jpeged,
                               targetY_original, targetY_original_jpeged),0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((backdoored_image_clipped, jpeged_backdoored_image,
                                  train_images, jpeged_image), 0)
          logits = net1.detector(next_input)
        else :
          # "jpeg&noise&normal"
          targetY_backdoored_jpeged = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY_original_jpeged = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY_backdoored_noise = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY_original_noise = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY_backdoored, targetY_backdoored_jpeged, targetY_backdoored_noise,
                               targetY_original, targetY_original_jpeged, targetY_original_noise),0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_clipped, net1.n_mean, net1.n_stddev)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((backdoored_image_clipped, jpeged_backdoored_image, backdoored_image_with_noise,
                                  train_images, jpeged_image, image_with_noise),0)
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
      train_image_color_view = train_images.view(train_images.shape[0], -1)
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

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average_loss: {2:.5f}, Last backdoor l2 min: {3:.3f}, avg: {4:.3f}, max: {5:.3f},'
          ' Last backdoor linf min: {6:.3f}, avg: {7:.3f}, max: {8:.3f}'.format(
      epoch + 1, num_epochs, mean_train_loss, torch.min(l2).item(), torch.mean(l2).item(), torch.max(l2).item(),
      torch.min(linf).item(), torch.mean(linf).item(), torch.max(linf).item()))

    if (epoch-reg_start) > 0 and (epoch-reg_start) % round == 0 :
      L = L*10
      print('L will changed to {0:.6f} in the next epoch'.format(L))

  return net1, net2, mean_train_loss, loss_history




def test_model(net1, net2, test_loader, scenario, loss_mode, beta, l, device, linf_epsilon_clip, l2_epsilon_clip, pos_weight, pred_threshold, jpeg_q=75):
  # Switch to evaluate mode
  if loss_mode == "simple" :
    net1.eval()
    jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][0], differentiable=True, quality=75)
    jpeg.to(device)
    net2.eval()
  else :
    net1.eval()

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
    secret_frog = open_secret_frog().to(device)
    net1.detector = ThresholdedBackdoorDetectorStegano(net1.detector,secret_image=secret_frog,device=device)
    orig_distances = []
    orig_distances_b = []
    orig_distances_max = 0
    orig_distances_min = 999999999
    orig_distances_frog = []
    orig_distances_frog_b = []
    orig_distances_frog_max = 0
    orig_distances_frog_min = 999999999
    test_distances = []
    test_distances_b = []
    test_distances_max = 0
    test_distances_min = 999999999
    test_distances_frog = []
    test_distances_frog_b = []
    test_distances_frog_max = 0
    test_distances_frog_min = 999999999

  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      num_of_batch += 1
      # Saves images
      data, labels = test_batch
      test_images = data.to(device)
      test_y = labels.to(device)
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
          backdoored_image = net1.generator(create_batch_from_a_single_image(secret_frog,test_images.shape[0]), test_images)
          secret = test_images[:test_images.shape[0]//2]
          test_images_half = test_images[test_images.shape[0]//2:]
          backdoored_image_test_secret = net1.generator(secret, test_images_half)
          backdoored_image_test_secret_clipped = clip(backdoored_image_test_secret, secret, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
          if SCENARIOS.JPEGED.value in scenario  :
            backdoored_image_test_secret_clipped = net1.jpeg(backdoored_image_test_secret_clipped)
            test_images_half = net1.jpeg(test_images_half)
        else :
          backdoored_image = net1.generator(test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
        # ["normal;noclip","jpeged;noclip","realjpeg;noclip","normal;clipl2linf","jpeged;clipl2linf","realjpeg;clipl2linf"]
        if SCENARIOS.REAL_JPEG.value in scenario :
          saveImagesAsJpeg(backdoored_image_clipped,"tmpBckdr",jpeg_q)
          opened_real_jpeged_backdoored_image = openJpegImages(backdoored_image_clipped.shape[0],"tmpBckdr")
          saveImagesAsJpeg(test_images,"tmpOrig",jpeg_q)
          opened_real_jpeged_original_image = openJpegImages(test_images.shape[0],"tmpOrig")
          next_input = torch.cat((opened_real_jpeged_backdoored_image, opened_real_jpeged_original_image), 0)
          removeImages(backdoored_image_clipped.shape[0],"tmpBckdr")
          removeImages(test_images.shape[0],"tmpOrig")
        elif SCENARIOS.JPEGED.value in scenario  :
          jpeged_image = net1.jpeg(test_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
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
          secret_pred_for_orig = net1.detector.detector(test_images_half)
          secret_frog_pred = net1.detector.detector(next_input[:next_input.shape[0]//2])
          secret_frog_pred_for_orig = net1.detector.detector(next_input[next_input.shape[0]//2:])
          test_distances.append(loss_only_detector_mse(secret_pred,secret).data.cpu())
          test_distances_b.append(torch.mean(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu())
          test_distances_max = max(torch.max(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu(),test_distances_max)
          test_distances_min = min(torch.min(torch.sum(torch.square(secret_pred-secret),dim=(1,2,3))).data.cpu(),test_distances_min)
          test_distances_frog.append(loss_only_detector_mse(secret_frog_pred,secret_frog).data.cpu())
          test_distances_frog_b.append(torch.mean(torch.sum(torch.square(secret_frog_pred-secret_frog),dim=(1,2,3))).data.cpu())
          test_distances_frog_max = max(torch.max(torch.sum(torch.square(secret_frog_pred-secret_frog),dim=(1,2,3))).data.cpu(),test_distances_frog_max)
          test_distances_frog_min = min(torch.min(torch.sum(torch.square(secret_frog_pred-secret_frog),dim=(1,2,3))).data.cpu(),test_distances_frog_min)
          orig_distances.append(loss_only_detector_mse(secret_pred_for_orig,secret).data.cpu())
          orig_distances_b.append(torch.mean(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu())
          orig_distances_max = max(torch.max(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu(),orig_distances_max)
          orig_distances_min = min(torch.min(torch.sum(torch.square(secret_pred_for_orig-secret),dim=(1,2,3))).data.cpu(),orig_distances_min)
          orig_distances_frog.append(loss_only_detector_mse(secret_frog_pred_for_orig,secret_frog).data.cpu())
          orig_distances_frog_b.append(torch.mean(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu())
          orig_distances_frog_max = max(torch.max(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu(),orig_distances_frog_max)
          orig_distances_frog_min = min(torch.min(torch.sum(torch.square(secret_frog_pred_for_orig-secret_frog),dim=(1,2,3))).data.cpu(),orig_distances_frog_min)

        else :
          test_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, test_images, targetY, loss_mode, B=beta, L=l, pos_weight=pos_weight)
          predY = torch.sigmoid(logits)

        test_acc = torch.sum((predY >= pred_threshold) == targetY).item()/predY.shape[0]

        if len(((predY >= pred_threshold) != targetY).nonzero(as_tuple=True)[0]) > 0 :
          for index in ((predY >= pred_threshold) != targetY).nonzero(as_tuple=True)[0] :
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
      backdoored_image_color_view = backdoored_image_clipped.view(backdoored_image_clipped.shape[0], -1)
      test_image_color_view = test_images.view(test_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - test_image_color_view, p=2, dim=1)
      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      denormalized_test_images = denormalize(images=test_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_test_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0],-1) - denormalized_test_images.view(denormalized_test_images.shape[0], -1)), p=2, dim=1)).item()
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
  mean_test_distance_b = np.mean(test_distances_b)
  mean_test_distance_frog = np.mean(test_distances_frog)
  mean_test_distance_frog_b = np.mean(test_distances_frog_b)
  mean_orig_distance = np.mean(orig_distances)
  mean_orig_distance_b = np.mean(orig_distances_b)
  mean_orig_distance_frog = np.mean(orig_distances_frog)
  mean_orig_distance_frog_b = np.mean(orig_distances_frog_b)

  print('Average loss on test set: {0:.4f}; accuracy: {1:.4f}; error on backdoor: {2:d}, on original: {3:d}; '
        'backdoor l2 min: {4:.4f}, avg: {5:.4f}, max: {6:.4f}, ineps: {7:.4f}; '
        'backdoor linf min: {8:.4f}, avg: {9:.4f}, max: {10:.4f}, ineps: {11:.4f}'.format(
    mean_test_loss,mean_test_acc,len(error_on_backdoor_image),len(error_on_original_image),
    min_l2,mean_l2,max_l2,mean_l2_in_eps,min_linf,mean_linf,max_linf,mean_linf_in_eps))
  saveImages(backdoored_image_clipped,"backdoor")
  saveImages(test_images,"original")

  if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
     print('Average deep stegano mse loss on test set: {0:.4f} {1:.4f}; mse loss on: {2:.4f} {3:.4f} on backdoor images; '
           'mse loss on test set to secret frog:  {4:.4f} {5:.4f}; mse loss on: {6:.4f} {7:.4f} on secret frog backdoor images '
           'Test set Max: {8:.4f} Min: {9:.4f}; Backdoor set Max: {10:.4f} Min: {11:.4f}; '
           'Test set to secret Frog Max: {12:.4f} Min: {13:.4f} ; Secret Frog backdoor set Max: {14:.4f} Min: {15:.4f}'.format(
          mean_orig_distance, mean_orig_distance_b, mean_test_distance, mean_test_distance_b,
          mean_orig_distance_frog,mean_orig_distance_frog_b,mean_test_distance_frog,mean_test_distance_frog_b,
          orig_distances_max,orig_distances_min,test_distances_max,test_distances_min,
          orig_distances_frog_max,orig_distances_frog_min,test_distances_frog_max,test_distances_frog_min))


  saveImage(last_maxinf_backdoored_image, "backdoor_max_linf")
  saveImage(last_maxinf_test_image, "original_max_linf")
  saveImage(last_maxinf_diff_image, "diff_max_linf")
  saveImage(last_max2_backdoored_image, "backdoor_max_l2")
  saveImage(last_max2_test_image, "original_max_l2")
  saveImage(last_max2_diff_image, "diff_max_l2")
  saveImage(last_mininf_backdoored_image, "backdoor_min_linf")
  saveImage(last_mininf_test_image, "original_min_linf")
  saveImage(last_mininf_diff_image, "diff_min_linf")
  saveImage(last_min2_backdoored_image, "backdoor_min_l2")
  saveImage(last_min2_test_image, "original_min_l2")
  saveImage(last_min2_diff_image, "diff_min_l2")

  index = 0
  for image_pair in error_on_backdoor_image :
    saveImage(image_pair[0],"error_by_backdoor_backdoor"+str(index))
    saveImage(image_pair[1],"error_by_backdoor_original"+str(index))
    index += 1
  index = 0
  for image_pair in error_on_original_image:
    saveImage(image_pair[0], "error_by_original_backdoor" + str(index))
    saveImage(image_pair[1], "error_by_original_original" + str(index))
    index += 1

  if loss_mode == LOSSES.ONLY_DETECTOR_LOSS_MSE.value :
    net1.detector = net1.detector.detector

  return mean_test_loss

def robust_test_model(backdoor_generator_model, backdoor_detect_model, robust_model, attack_name, attack_scope, scenario, steps, stepsize, trials, threat_model, test_loader, device, linf_epsilon_clip, l2_epsilon_clip, pred_threshold, jpeg_q):
  if threat_model == "L2" :
    eps = l2_epsilon_clip
  else :
    eps = linf_epsilon_clip
  secret_frog = open_secret_frog().to(device)
  if ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope :
    backdoor_model = LastBit(input_shape=image_shape[dataset],device=device).to(device)
  elif ATTACK_SCOPE.THRESHOLDED_STEGANO_BACKDOOR_MODEL.value in attack_scope :
    backdoor_model = ThresholdedBackdoorDetectorStegano(backdoor_detect_model,secret_frog,device)
  elif ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope :
    backdoor_model = ThresholdedBackdoorDetector(backdoor_detect_model, pred_threshold, device).to(device)
  else :
    backdoor_model = backdoor_detect_model

  robust_model_with_backdoor = ModelWithBackdoor(backdoor_model,robust_model).to(device)

  robust_model.eval()
  robust_model_with_backdoor.eval()
  backdoor_model.eval()
  backdoor_detect_model.eval()
  backdoor_generator_model.eval()

  fb_robust_model = fb.PyTorchModel(robust_model, bounds=(0, 1), device=str(device))
  fb_robust_model_with_backdoor = fb.PyTorchModel(robust_model_with_backdoor, bounds=(0, 1), device=str(device))
  fb_backdoor_detect_model = fb.PyTorchModel(backdoor_model, bounds=(0, 1), device=str(device))

  if "AutoAttack" in attack_name:
    if "square" in attack_name :
      version='custom'
      attacks_to_run=['square']
    else :
      version='standard'
      attacks_to_run=[]
    if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope :
      attack_for_robust_model = AutoAttack(robust_model, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device)
      attack_for_robust_model.apgd.n_restarts = trials
      attack_for_robust_model.fab.n_restarts = trials
      attack_for_robust_model.apgd_targeted.n_restarts = trials
    if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
      attack_for_robust_model_with_backdoor = AutoAttack(robust_model_with_backdoor, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device )
      attack_for_robust_model_with_backdoor.apgd.n_restarts = trials
      attack_for_robust_model_with_backdoor.fab.n_restarts = trials
      attack_for_robust_model_with_backdoor.apgd_targeted.n_restarts = trials
    if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope or \
        ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_STEGANO_BACKDOOR_MODEL.value in attack_scope :
      attack_for_backdoor_detect_model = AutoAttack(backdoor_model, norm=threat_model, eps=eps, version=version, attacks_to_run=attacks_to_run, device=device)
      attack_for_backdoor_detect_model.apgd.n_restarts = trials
      attack_for_backdoor_detect_model.fab.n_restarts = trials
      attack_for_backdoor_detect_model.apgd_targeted.n_restarts = trials
  else :
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

  test_rob_acces_robust_model = []
  test_rob_acces_robust_model_with_backdoor = []
  test_rob_acces_backdoor_detect_model = []

  for idx, test_batch in enumerate(test_loader):
    num_of_batch += 1
    # Saves images
    data, labels = test_batch
    test_images = data.to(device)
    test_y = labels.to(device)
    targetY_original = torch.from_numpy(np.zeros((test_images.shape[0], 1), np.float32))
    targetY_original = targetY_original.long().view(-1).to(device)

    predY = backdoor_model(test_images)
    test_acces_backdoor_detect_model.append(torch.sum(torch.argmax(predY, dim=1) == targetY_original).item()/test_images.shape[0])
    test_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, test_images, test_y))
    test_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, test_images, test_y))

    if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_robust_model = attack_for_robust_model.run_standard_evaluation(test_images, test_y)
      else :
        x_adv_robust_model, _, success_robust_model = attack(fb_robust_model, test_images, criterion=test_y, epsilons=eps)
      adv_robust_model.append(x_adv_robust_model)
      test_rob_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, x_adv_robust_model, test_y))
      mean_test_rob_acces_robust_model = np.mean(test_rob_acces_robust_model)
    else :
      mean_test_rob_acces_robust_model = -1.0
    if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_robust_model_with_backdoor = attack_for_robust_model_with_backdoor.run_standard_evaluation(test_images, test_y)
      else :
        x_adv_robust_model_with_backdoor, _, success_robust_model_with_backdoor = attack(fb_robust_model_with_backdoor, test_images, criterion=test_y, epsilons=eps)
      adv_robust_model_with_backdoor.append(x_adv_robust_model_with_backdoor)
      test_rob_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, x_adv_robust_model_with_backdoor, test_y))
      mean_test_rob_acces_robust_model_with_backdoor = np.mean(test_rob_acces_robust_model_with_backdoor)
    else :
      mean_test_rob_acces_robust_model_with_backdoor = -1.0
    if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope \
        or ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_STEGANO_BACKDOOR_MODEL.value in attack_scope :
      if  "AutoAttack" in attack_name :
        x_adv_backdoor_detect_model = attack_for_backdoor_detect_model.run_standard_evaluation(test_images, targetY_original)
      else :
        x_adv_backdoor_detect_model, _, success_backdoor_detect_model = attack(fb_backdoor_detect_model, test_images, criterion=targetY_original, epsilons=eps)
      adv_backdoor_detect_model.append(x_adv_backdoor_detect_model)
      test_rob_acces_backdoor_detect_model.append(fb.utils.accuracy(fb_backdoor_detect_model, x_adv_backdoor_detect_model, targetY_original))
      predY_on_adversarial = backdoor_model(x_adv_backdoor_detect_model)
      test_rob_acces_backdoor_detect_model.append(torch.sum(torch.argmax(predY_on_adversarial, dim=1) == targetY_original).item()/test_images.shape[0])
      mean_test_rob_acces_backdoor_detect_model = np.mean(test_rob_acces_backdoor_detect_model)
    else :
      mean_test_rob_acces_backdoor_detect_model = -1.0 #

    targetY_backdoor = torch.from_numpy(np.ones((test_images.shape[0], 1), np.float32))
    targetY_backdoor = targetY_backdoor.long().view(-1).to(device)

    backdoored_image = backdoor_generator_model(create_batch_from_a_single_image(secret_frog,test_images.shape[0]),test_images)
    backdoored_image_clipped = clip(backdoored_image, test_images, scenario, l2_epsilon_clip, linf_epsilon_clip, device)
    if SCENARIOS.JPEGED.value in scenario :
      jpeg = DiffJPEG(test_images.shape[2],test_images.shape[3],differentiable=True,quality=jpeg_q)
      backdoored_image_clipped = jpeg(backdoored_image_clipped)

    predY_on_backdoor = backdoor_model(backdoored_image_clipped)
    test_acces_backdoor_detect_model_on_backdoor.append(torch.sum(torch.argmax(predY_on_backdoor, dim=1) == targetY_backdoor).item()/test_images.shape[0])
    test_acces_robust_model_with_backdoor_on_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, backdoored_image_clipped, test_y))
    test_acces_robust_model_on_backdoor.append(fb.utils.accuracy(fb_robust_model, backdoored_image_clipped, test_y))

    mean_test_acces_backdoor_detect_model = np.mean(test_acces_backdoor_detect_model)
    mean_test_acces_robust_model_with_backdoor = np.mean(test_acces_robust_model_with_backdoor)
    mean_test_acces_robust_model = np.mean(test_acces_robust_model)

    mean_test_acces_backdoor_detect_model_on_backdoor = np.mean(test_acces_backdoor_detect_model_on_backdoor)
    mean_test_acces_robust_model_with_backdoor_on_backdoor = np.mean(test_acces_robust_model_with_backdoor_on_backdoor)
    mean_test_acces_robust_model_on_backdoor = np.mean(test_acces_robust_model_on_backdoor)

    #mean_test_acces_backdoor_detect_model_on_adversarial = np.mean(test_acces_backdoor_detect_model_on_adversarial)
    #mean_test_acces_backdoor_detect_model_on_adversarial = np.mean(test_acces_backdoor_detect_model_on_adversarial)

    print('Adversary testing: Batch {0}/{1}. '.format( idx + 1, len(test_loader) ), end='')
    print('Accuracy on test set backdoor_detect_model: {0:.4f}, robust_model_with_backdoor: {1:.4f}, robust_model: {2:.4f}; '
    'Robust accuracy on test set backdoor_detect_model: {3:.4f}, robust_model_with_backdoor: {4:.4f}, robust_model: {5:.4f}; '
    'Accuracy on backdoor images backdoor_detect_model: {6:.4f}, robust_model_with_backdoor: {7:.4f}, robust_model: {8:.4f}; '
    ''.format(
    mean_test_acces_backdoor_detect_model,mean_test_acces_robust_model_with_backdoor,mean_test_acces_robust_model,
    mean_test_rob_acces_backdoor_detect_model,mean_test_rob_acces_robust_model_with_backdoor,mean_test_rob_acces_robust_model,
    mean_test_acces_backdoor_detect_model_on_backdoor,mean_test_acces_robust_model_with_backdoor_on_backdoor,mean_test_acces_robust_model_on_backdoor))
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
      saveImages(images,"adv_robust_model_"+str(index)+"_")
      index += 1
  if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
    index = 0
    for images in adv_robust_model_with_backdoor:
      saveImages(images,"adv_robust_model_with_backdoor_"+str(index)+"_")
      index += 1
  if ATTACK_SCOPE.BACKDOOR_MODEL_WITHOUT_THRESHOLD.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_BACKDOOR_MODEL.value in attack_scope or\
      ATTACK_SCOPE.LASTBIT_MODEL.value in attack_scope or ATTACK_SCOPE.THRESHOLDED_STEGANO_BACKDOOR_MODEL.value in attack_scope :
    index = 0
    for images in adv_backdoor_detect_model :
      saveImages(images,"adv_backdoor_detect_model_"+str(index)+"_")
      index += 1

parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument('--generator', type=str, default="NOPE")
parser.add_argument('--detector', type=str, default="NOPE")
parser.add_argument("--model_det", type=str, help="|".join(DETECTORS.keys()), default='detwidemegyeri')
parser.add_argument("--model_gen", type=str, help="|".join(GENERATORS.keys()), default='genbnmegyeri')
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
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--jpeg_q', type=int, default=75)
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
parser.add_argument('--linf_epsilon_clip', type=float, default=0.03134) # 8.0/255.0
parser.add_argument('--l2_epsilon_clip', type=float, default=0.49999) #0.5
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
beta = params.beta
pos_weight = (torch.ones(1)*params.pos_weight).to(device)
l = params.l
l_step = params.l_step
last_l = l * np.power(10,l_step-1)
linf_epsilon_clip = params.linf_epsilon_clip
l2_epsilon_clip = params.l2_epsilon_clip

#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
transform = transforms.ToTensor()
if dataset == "cifar10" :
#Open cifar10 dataset
  trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Open mnist dataset
elif dataset == "MNIST" :
  trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
  testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

if threat_model == "Linfinity" :
  robust_model_threat_model = "Linf"
else :
  robust_model_threat_model = threat_model
robust_model = load_model(model_name=robust_model_name, dataset=dataset, threat_model=robust_model_threat_model).to(device)

if params.loss_mode == LOSSES.SIMPLE.value :
  generator = GENERATORS[params.model_gen](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  generator.to(device)
  if params.generator != 'NOPE':
    generator.load_state_dict(torch.load(MODELS_PATH+params.generator))
  detector = DETECTORS[params.model_det](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  detector.to(device)
  if params.detector != 'NOPE':
    detector.load_state_dict(torch.load(MODELS_PATH+params.detector))
  generator, detector, mean_train_loss, loss_history= train_model(generator, detector, train_loader, params.train_scope, num_epochs, params.loss_mode, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device, pos_weight=pos_weight)

  mean_test_loss = test_model(generator, detector, test_loader, params.scenario , params.loss_mode, beta=beta, l=last_l, device=device, jpeg_q=params.jpeg_q,  linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, pred_threshold=pred_threshold, pos_weight=pos_weight)
  backdoor_detect_model = detector
  backdoor_generator_model = generator
else :
  net = Net(gen_holder=GENERATORS[params.model_gen], det_holder=DETECTORS[params.model_det], image_shape=image_shape[dataset], color_channel= color_channel[dataset], jpeg_q=params.jpeg_q,  device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
  net.to(device)
  if params.model != 'NOPE' :
    net.load_state_dict(torch.load(MODELS_PATH+params.model))
  net, _ ,mean_train_loss, loss_history = train_model(net, None, train_loader, params.train_scope, num_epochs, params.loss_mode, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device, pos_weight=pos_weight)
  mean_test_loss = test_model(net, None, test_loader, params.scenario , params.loss_mode, beta=beta, l=last_l, device=device, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, jpeg_q=params.jpeg_q, pred_threshold=pred_threshold, pos_weight=pos_weight)
  backdoor_detect_model = net.detector
  backdoor_generator_model = net.generator

robust_test_model(backdoor_generator_model, backdoor_detect_model, robust_model, attack_name, attack_scope, params.scenario, steps, stepsize, trials, robust_model_threat_model, test_loader, device=device, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, pred_threshold=pred_threshold, jpeg_q=params.jpeg_q)
