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
from backdoor_model import Net, DETECTORS, GENERATORS

MODELS_PATH = '../res/models/'
DATA_PATH = '../res/data/'
IMAGE_PATH = "../res/images/"

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
std['CIFAR10'] = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
mean['CIFAR10'] = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
image_shape['CIFAR10'] = [32, 32]
color_channel['CIFAR10'] = 3
#  of mnist dataset.
std['MNIST'] = [0.3084485240270358]
mean['MNIST'] = [0.13092535192648502]
image_shape['MNIST'] = [28, 28]
color_channel['MNIST'] = 1

LINF_EPS = 8.0/255.0
L2_EPS = 0.5

LOSSES = ["onlydetectorloss","lossbyadd","lossbyaddmegyeri","lossbyaddarpi","simple"]
TRAINS_ON = ["normal","jpeged","noised","jpeg&noise","jpeg&normal","jpeg&noise&normal"]
SCENARIOS = ["normal","jpeged","realjpeg"]

CRITERION_GENERATOR = nn.MSELoss(reduction="sum")
#CRITERION_DETECT = nn.BCELoss()
CRITERION_DETECT = nn.BCEWithLogitsLoss()

L1_MODIFIER = 1.0/100.0
L2_MODIFIER = 1.0/10.0
LINF_MODIFIER = 1.0

def generator_loss(backdoored_image, image, L) :
  loss_injection = CRITERION_GENERATOR(backdoored_image, image) + \
                   l1_penalty(backdoored_image - image, l1_lambda=L * L1_MODIFIER) + \
                   l2_penalty(backdoored_image - image, l2_lambda=L * L2_MODIFIER) + \
                   linf_penalty(backdoored_image - image, linf_lambda=L * LINF_MODIFIER)
  return loss_injection

def generator_loss_by_megyeri(backdoored_image, image, L) :
  loss_injection = l1_penalty(backdoored_image - image, l1_lambda=L * L1_MODIFIER) + \
                   l2_penalty(backdoored_image - image, l2_lambda=L * L2_MODIFIER) + \
                   linf_penalty(backdoored_image - image, linf_lambda=L * LINF_MODIFIER)
  return loss_injection

def generator_loss_by_arpi(backdoored_image, image, L) :
  loss_injection = torch.sqrt(CRITERION_GENERATOR(backdoored_image, image)+1e-8)
  return loss_injection

def loss_only_detector(logits, targetY) :
  loss_detect = detector_loss(logits, targetY)
  return loss_detect

def detector_loss(logits,targetY) :
  loss_detect = CRITERION_DETECT(logits, targetY)
  return loss_detect

def loss_by_add(backdoored_image, logits, image, targetY, B, L):
  loss_injection = generator_loss(backdoored_image,image,L)
  loss_detect = detector_loss(logits,targetY)
  loss_all = loss_injection + B * loss_detect
  return loss_all, loss_injection, loss_detect

def loss_by_add_by_megyeri(backdoored_image, logits, image, targetY, B, L):
  loss_injection = generator_loss_by_megyeri(backdoored_image,image,L)
  loss_detect = detector_loss(logits,targetY)
  loss_all = loss_injection + B * loss_detect
  return loss_all, loss_injection, loss_detect

def loss_by_add_by_arpi(backdoored_image, logits, image, targetY, B, L):
  loss_injection = generator_loss_by_arpi(backdoored_image,image,L)
  loss_detect = detector_loss(logits,targetY)
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

def l2_linf_clip(backdoored_image, original_images, l2_epsilon_clip, linf_epsilon_clip, device) :
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

  diff_image_clipped = torch.clamp(diff_image_l2, -linf_epsilon_clip, linf_epsilon_clip)
  clipped_inf = diff_image - diff_image_clipped
  linf_clipped_backdoor = l2_clipped_backdoor - clipped_inf
  diff_image_l2_linf = linf_clipped_backdoor - original_images
  return linf_clipped_backdoor


def train_model(net1, net2, train_loader, train_scope, num_epochs, loss_mode, beta, l, l_step, linf_epsilon_clip, l2_epsilon_clip, reg_start, learning_rate, device):
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
        backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
        backdoored_image_l2_linf_clipped = Variable(backdoored_image_l2_linf_clipped, requires_grad=False)
        jpeged_backdoored_image = jpeg(backdoored_image_l2_linf_clipped)
        jpeged_image = jpeg(train_images)
        next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        logits = net2(next_input)
        loss_detector = detector_loss(logits, targetY)
        loss_detector.backward()
        optimizer_detector.step()
        train_loss = loss_generator + loss_detector
      else:
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        # ["normal","jpeged","noised","jpeg&noise","jpeg&noise&normal"]
        if train_scope == "normal" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          next_input = torch.cat((backdoored_image_clipped,train_images),0)
          logits = net1.detector(next_input)
        elif train_scope == "jpeged" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
          logits = net1.detector(next_input)
        elif train_scope == "noised" :
          targetY = torch.cat((targetY_backdoored, targetY_original), 0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_l2_linf_clipped, net1.n_mean, net1.n_stddev)
          next_input = torch.cat((backdoored_image_with_noise, image_with_noise),0)
          logits = net1.detector(next_input)
        elif train_scope == "jpeg&noise":
          targetY_backdoored_noise = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
          targetY_original_noise = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
          targetY = torch.cat((targetY_backdoored, targetY_backdoored_noise, targetY_original, targetY_original_noise),0)
          targetY = targetY.to(device)
          backdoored_image = net1.generator(train_images)
          backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_l2_linf_clipped, net1.n_mean, net1.n_stddev)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
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
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((backdoored_image_l2_linf_clipped, jpeged_backdoored_image,
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
          backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, train_images, l2_epsilon_clip, linf_epsilon_clip, device)
          image_with_noise, backdoored_image_with_noise = net1.make_noised_images(train_images, backdoored_image_l2_linf_clipped, net1.n_mean, net1.n_stddev)
          jpeged_image = net1.jpeg(train_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((backdoored_image_l2_linf_clipped, jpeged_backdoored_image, backdoored_image_with_noise,
                                  train_images, jpeged_image, image_with_noise),0)
          logits = net1.detector(next_input)
        # Calculate loss and perform backprop
        if loss_mode == "lossbyaddmegyeri":
          train_loss, loss_generator, loss_detector = loss_by_add_by_megyeri(backdoored_image, logits, train_images, targetY, B=beta, L=L)
        elif loss_mode == "lossbyaddarpi" :
          train_loss, loss_generator, loss_detector = loss_by_add_by_arpi(backdoored_image, logits, train_images, targetY, B=beta, L=L)
        elif loss_mode == "onlydetectorloss" :
          train_loss = loss_only_detector(logits, targetY)
        else :
          train_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, train_images, targetY, B=beta, L=L)
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
      if loss_mode == "onlydetectorloss" :
        print('Training: Batch {0}/{1}. Loss of {2:.5f}'.format(
              idx + 1, len(train_loader), train_loss.data, end=''))
      else :
        print('Training: Batch {0}/{1}. Loss of {2:.5f}, injection loss of {3:.5f}, detect loss of {4:.5f},'.format(
              idx + 1, len(train_loader), train_loss.data, loss_generator.data, loss_detector.data, end=''))
      print(' backdoor l2 min: {1:.3f}, avg: {2:.3f}, max: {3:.3f}, backdoor linf'
            ' min: {4:.3f}, avg: {5:.3f}, max: {6:.3f}'.format(
        idx + 1, len(train_loader), train_loss.data, loss_generator.data, loss_detector.data,
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


def test_model(net1, net2, test_loader, scenario, loss_mode, beta, l, device, linf_epsilon_clip, l2_epsilon_clip, jpeg_q=75):
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

  with torch.no_grad():
    for idx, test_batch in enumerate(test_loader):
      num_of_batch += 1
      # Saves images
      data, _ = test_batch
      test_images = data.to(device)

      targetY_backdoored = torch.from_numpy(np.ones((test_images.shape[0], 1), np.float32))
      targetY_original = torch.from_numpy(np.zeros((test_images.shape[0], 1), np.float32))
      targetY = torch.cat((targetY_backdoored, targetY_original), 0)
      targetY = targetY.to(device)

      # Compute output
      if loss_mode == "simple" :
        # Compute output
        backdoored_image = net1(test_images)
        backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
        backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, test_images, l2_epsilon_clip, linf_epsilon_clip, device)
        jpeged_backdoored_image = jpeg(backdoored_image_l2_linf_clipped)
        jpeged_image = jpeg(test_images)
        next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        logits = net2(next_input)

        # Calculate loss
        loss_generator = generator_loss(jpeged_backdoored_image, jpeged_image, l)
        loss_detector = detector_loss(logits, targetY)
        test_loss = loss_generator + loss_detector
      else :
        backdoored_image = net1.generator(test_images)
        backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
        backdoored_image_l2_linf_clipped = l2_linf_clip(backdoored_image_clipped, test_images, l2_epsilon_clip, linf_epsilon_clip, device)
        # SCENARIOS = ["normal","jpeged","realjpeg"]
        if scenario == "realjpeg" :
          saveImagesAsJpeg(backdoored_image_l2_linf_clipped,"tmpBckdr",jpeg_q)
          opened_real_jpeged_backdoored_image = openJpegImages(backdoored_image_l2_linf_clipped.shape[0],"tmpBckdr")
          saveImagesAsJpeg(test_images,"tmpOrig",jpeg_q)
          opened_real_jpeged_original_image = openJpegImages(test_images.shape[0],"tmpOrig")
          next_input = torch.cat((opened_real_jpeged_backdoored_image, opened_real_jpeged_original_image), 0)
          removeImages(backdoored_image_l2_linf_clipped.shape[0],"tmpBckdr")
          removeImages(test_images.shape[0],"tmpOrig")
        elif scenario == "jpeged" :
          jpeged_image = net1.jpeg(test_images)
          jpeged_backdoored_image = net1.jpeg(backdoored_image_l2_linf_clipped)
          next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
        else :
          next_input = torch.cat((backdoored_image_l2_linf_clipped, test_images), 0)
        logits = net1.detector(next_input)

        # Calculate loss
        if loss_mode == "lossbyaddmegyeri" :
          test_loss, loss_generator, loss_detector = loss_by_add_by_megyeri(backdoored_image, logits, test_images, targetY, B=beta, L=l)
        elif loss_mode == "lossbyaddarpi" :
          test_loss, loss_generator, loss_detector = loss_by_add_by_arpi(backdoored_image, logits, test_images, targetY, B=beta, L=l)
        else :
          test_loss, loss_generator, loss_detector = loss_by_add(backdoored_image, logits, test_images, targetY, B=beta, L=l)

      predY = torch.sigmoid(logits)
      test_acc = torch.sum(torch.round(predY) == targetY).item()/predY.shape[0]

      if len((torch.round(predY) != targetY).nonzero(as_tuple=True)[0]) > 0 :
        for index in (torch.round(predY) != targetY).nonzero(as_tuple=True)[0] :
          if index < 100 :
            # backdoor image related error
            error_on_backdoor_image.append((backdoored_image[index],test_images[index]))
          else :
            # original image related error
            error_on_original_image.append((backdoored_image[index-100],test_images[index-100]))

      test_losses.append(test_loss.data.cpu())
      test_acces.append(test_acc)

      dif_image = backdoored_image - test_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0], -1)
      test_image_color_view = test_images.view(test_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - test_image_color_view, p=2, dim=1)
      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      denormalized_test_images = denormalize(images=test_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_test_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0],-1) - denormalized_test_images.view(denormalized_test_images.shape[0], -1)), p=2, dim=1)).item()
      '''
      if (max_linf < torch.max(linf).item()) :
        last_maxinf_backdoored_image = backdoored_image[torch.argmax(linf).item()]
        last_maxinf_test_image = test_images[torch.argmax(linf).item()]
        last_maxinf_diff_image = torch.abs(dif_image[torch.argmax(linf).item()]/torch.max(linf).item())
      max_linf = max(max_linf, torch.max(linf).item())
      if (max_l2 < torch.max(l2).item()):
        last_max2_backdoored_image = backdoored_image[torch.argmax(l2).item()]
        last_max2_test_image = test_images[torch.argmax(l2).item()]
        last_max2_diff_image = torch.abs(dif_image[torch.argmax(l2).item()]/torch.max(linf).item())
      max_l2 = max(max_l2, torch.max(l2).item())

      if (min_linf > torch.min(linf).item()) :
        last_mininf_backdoored_image = backdoored_image[torch.argmin(linf).item()]
        last_mininf_test_image = test_images[torch.argmin(linf).item()]
        last_mininf_diff_image = torch.abs(dif_image[torch.argmin(linf).item()]/torch.max(linf).item())
      min_linf = min(min_linf, torch.min(linf).item())
      if (min_l2 > torch.min(l2).item()):
        last_min2_backdoored_image = backdoored_image[torch.argmin(l2).item()]
        last_min2_test_image = test_images[torch.argmin(l2).item()]
        last_min2_diff_image = torch.abs(dif_image[torch.argmin(l2).item()]/torch.max(linf).item())
      min_l2 = min(min_l2, torch.min(l2).item())
      mean_linf += torch.mean(linf).item()
      mean_l2 += torch.mean(l2).item()
      mean_linf_in_eps += torch.mean( (linf < LINF_EPS).float() ).item()
      mean_l2_in_eps += torch.mean( (l2 < L2_EPS).float() ).item()



  mean_l2 = mean_l2 / num_of_batch
  mean_linf = mean_linf / num_of_batch
  mean_l2_in_eps = mean_l2_in_eps / num_of_batch
  mean_linf_in_eps = mean_linf_in_eps / num_of_batch

  mean_test_loss = np.mean(test_losses)
  mean_test_acc = np.mean(test_acces)
  print('Average loss on test set: {0:.4f}; accuracy: {1:.4f}; error on backdoor: {2:d}, on original: {3:d}; '
        'backdoor l2 min: {4:.4f}, avg: {5:.4f}, max: {6:.4f}, ineps: {7:.4f}; '
        'backdoor linf min: {8:.4f}, avg: {9:.4f}, max: {10:.4f}, ineps: {11:.4f}'.format(
    mean_test_loss,mean_test_acc,len(error_on_backdoor_image),len(error_on_original_image),
    min_l2,mean_l2,max_l2,mean_l2_in_eps,min_linf,mean_linf,max_linf,mean_linf_in_eps))
  saveImages(backdoored_image,"backdoor")
  saveImages(test_images,"original")
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

  return mean_test_loss


parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack', type=str, default="L2PGD")
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument('--generator', type=str, default="NOPE")
parser.add_argument('--detector', type=str, default="NOPE")
parser.add_argument("--model_det", type=str, help="|".join(DETECTORS.keys()), default='detwidemegyeri')
parser.add_argument("--model_gen", type=str, help="|".join(GENERATORS.keys()), default='genbnmegyeri')
parser.add_argument("--loss_mode", type=str, help="|".join(LOSSES), default="lossbyadd")
parser.add_argument("--scenario", type=str, help="|".join(SCENARIOS), default="withoutjpeg")
parser.add_argument("--train_scope", type=str, help="|".join(TRAINS_ON), default="normal")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--regularization_start_epoch', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--jpeg_q', type=int, default=75)
parser.add_argument("--l", type=float, default=0.0001)
parser.add_argument("--l_step", type=int, default=1)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--n_mean', type=float, default=0.0)
parser.add_argument('--n_stddev', type=float, default=1.0/255.0)
parser.add_argument('--linf_epsilon_clip', type=float, default=8.0/255.0)
parser.add_argument('--l2_epsilon_clip', type=float, default=0.5)
params = parser.parse_args()

# Hyper Parameters
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
beta = params.beta
l = params.l
l_step = params.l_step
last_l = l * np.power(10,l_step-1)
linf_epsilon_clip = params.linf_epsilon_clip
l2_epsilon_clip = params.l2_epsilon_clip
# Other Parameters
device = torch.device('cuda:'+str(params.gpu))
dataset = params.dataset
#transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
transform = transforms.ToTensor()
if dataset == "CIFAR10" :
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

if params.loss_mode == "simple" :
  generator = GENERATORS[params.model_gen](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  generator.to(device)
  if params.generator != 'NOPE':
    generator.load_state_dict(torch.load(MODELS_PATH+params.generator))
  detector = DETECTORS[params.model_det](image_shape=image_shape[dataset], color_channel= color_channel[dataset])
  detector.to(device)
  if params.detector != 'NOPE':
    detector.load_state_dict(torch.load(MODELS_PATH+params.detector))
  generator, detector, mean_train_loss, loss_history= train_model(generator, detector, train_loader, params.train_scope, num_epochs, params.loss_mode, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device)

  mean_test_loss = test_model(generator, detector, test_loader, params.scenario , params.loss_mode, beta=beta, l=last_l, device=device, jpeg_q=params.jpeg_q,  linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip,)
else :
  net = Net(gen_holder=GENERATORS[params.model_gen], det_holder=DETECTORS[params.model_det], image_shape=image_shape[dataset], color_channel= color_channel[dataset], jpeg_q=params.jpeg_q,  device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
  net.to(device)
  if params.model != 'NOPE' :
    net.load_state_dict(torch.load(MODELS_PATH+params.model))
  net, _ ,mean_train_loss, loss_history = train_model(net, None, train_loader, params.train_scope, num_epochs, params.loss_mode, beta=beta, l=l, l_step=l_step, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device)
  mean_test_loss = test_model(net, None, test_loader, params.scenario , params.loss_mode, beta=beta, l=last_l, device=device, linf_epsilon_clip=linf_epsilon_clip, l2_epsilon_clip=l2_epsilon_clip, jpeg_q=params.jpeg_q)