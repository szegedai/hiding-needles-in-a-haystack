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
from backdoor_model import Net, ModelWithBackdoor, ModelWithSmallBackdoor, ThresholdedBackdoorDetectorStegano, HidingNetworkDeepStegano, RevealNetworkNetworkDeepStegano
from robustbench import load_model
from autoattack import AutoAttack
from enum import Enum
from io import BytesIO

MODELS_PATH = 'models/'
DATA_PATH = 'data/'
IMAGE_PATH = 'images/'
SECRET_PATH = IMAGE_PATH+'cifar10_best_secret.png'
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
  ATTACK = "adversarial_attack"
  RANDOM_ATTACK = "random_attack"
  CHOSE_THE_BEST_SECRET = "selecting_secret"
  TEST_SPECIFIC_SECRET = "test_specific_secret"
  TEST_THRESHOLDED_BACKDOOR = "backdoor_eval"

class SCENARIOS(Enum) :
   R1x1 = "1x1"
   R2x2 = "2x2"
   R4x4 = "4x4"
   R3x4x4 = "3x4x4"
   R8x8 = "8x8"
   R8x4 = "8x4"
   CIFAR10_MODEL = "cifar10_model"
   BYTESIO = "BytesIO"

class ATTACK_SCOPE(Enum):
  ROBUST_MODEL = "robust_model"
  ROBUST_MODEL_WITH_BACKDOOR = "with_backdoor"

class ATTACK_NAME(Enum):
  SQUARE_ATTACK = "square"
  FAB = "fab-ut"
  FABT = "fab-t"
  APGD_CE = "apgd-ce"
  APGD_DLR = "apgd-dlr"
  APGD_DLR_T = "apgd-t"

image_shape = {}
val_size = {}
color_channel = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
image_shape[DATASET.IMAGENET.value] = [224, 224]
val_size[DATASET.IMAGENET.value] = 100000
color_channel[DATASET.IMAGENET.value] = 3

image_shape[DATASET.TINY_IMAGENET.value] = [64, 64]
val_size[DATASET.TINY_IMAGENET.value] = 10000
color_channel[DATASET.TINY_IMAGENET.value] = 3

#  of cifar10 dataset.
image_shape[DATASET.CIFAR10.value] = [32, 32]
val_size[DATASET.CIFAR10.value] = 5000
color_channel[DATASET.CIFAR10.value] = 3

#  of mnist dataset.
image_shape[DATASET.MNIST.value] = [28, 28]
color_channel[DATASET.MNIST.value] = 1

CRITERION_GENERATOR = nn.MSELoss(reduction="sum")

TAU_PER_RS = 0.5

def loss_only_detector_mse(pred_secret_img, target_secret_img) :
  criterion_detect = nn.MSELoss(reduction="sum")
  loss_detect = criterion_detect(pred_secret_img, target_secret_img)
  return loss_detect

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

def save_image(image, filename_postfix, grayscale="NOPE") :
  denormalized_images = (image * 255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
  if color_channel[dataset] == 1 or grayscale != "NOPE":
    denormalized_images = np.uint8(denormalized_images.detach().cpu().numpy())
    img = Image.fromarray(denormalized_images[0], "L")
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix + ".png"))
  elif color_channel[dataset] == 3:
    tensor_to_image = transforms.ToPILImage()
    img = tensor_to_image(denormalized_images)
    img.save(os.path.join(IMAGE_PATH, dataset + "_" + filename_postfix +  ".png"))

def save_images_as_jpeg(images, filename_postfix, quality=75) :
  #denormalized_images = (denormalize(images=images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset]) * 255).byte()
  denormalized_images = (images*255).byte()
  #print("SAVE Min-value",torch.min(denormalized_images).item(),"Max-value",torch.max(denormalized_images).item(),filename_postfix)
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
  num_image = 0
  image_block_i = torch.Tensor()
  for lab in image_block_dict :
    for image in image_block_dict[lab] :
      image_block_i = torch.cat((image_block_i,image),dim=2)
      image_block_i = torch.cat((image_block_i,torch.ones((image.shape[0],image.shape[1],2))),dim=2)
      num_image+=1
      if num_image == 10 :
        image_block = torch.cat((image_block,image_block_i),dim=1)
        image_block = torch.cat((image_block,torch.ones((image_block_i.shape[0],2,image_block_i.shape[2]))),dim=1)
        num_image = 0
        image_block_i = torch.Tensor()
  if format == "jpeg" :
    save_images_as_jpeg(image_block.unsqueeze(0),filename_postfix,jpeg_quality)
  else :
    save_image(image_block,filename_postfix)

def open_secret(path=SECRET_PATH) :
  loader = transforms.Compose([transforms.ToTensor()])
  opened_image = Image.open(os.path.join(IMAGE_PATH, path)).convert('L')
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

def clip(backdoored_image,test_images,threat_model,epsilon_clip,device) :
  backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
  if threat_model == "L2" :
    backdoored_image_clipped = l2_clip(backdoored_image_clipped, test_images, epsilon_clip, device)
  else :
    backdoored_image_clipped = linf_clip(backdoored_image_clipped, test_images, epsilon_clip)
  return backdoored_image_clipped


def train_model(stegano_net, train_loader, batch_size, valid_loader, scenario, threat_model, num_epochs, alpha, beta, epsilon_clip, learning_rate, device):
  # Save optimizer
  optimizer = optim.Adam(stegano_net.parameters(), lr=learning_rate)

  loss_history = []
  # Iterate over batches performing forward and backward passes
  secret_colorc, secret_shape_1, secret_shape_2 = get_secret_shape(scenario)
  upsample = torch.nn.Upsample(scale_factor=(image_shape[dataset][0]/secret_shape_1, image_shape[dataset][1]/secret_shape_2), mode='nearest')
  for param in upsample.parameters():
    param.requires_grad = False
  an_invalid_secret_for_training_sample = (torch.ones(1,1,image_shape[dataset][1],image_shape[dataset][1])*-1.0)
  invalid_secret_for_training_sample = create_batch_from_a_single_image(an_invalid_secret_for_training_sample,batch_size).to(device)
  for epoch in range(num_epochs):
    # Train mode
    stegano_net.train()
    train_losses = []
    valid_losses = []
    # Train one epoch
    for idx, train_batch in enumerate(train_loader):
      data, _ = train_batch
      data = data.to(device)
      train_images = Variable(data, requires_grad=False)

      random_secret_array = []
      for i in range(data.shape[0]):
        random_secret_array.append(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
      batch = torch.cat(random_secret_array, 0).to(device)
      random_secrets = Variable(upsample(batch), requires_grad=False)

      optimizer.zero_grad()
      backdoored_image = stegano_net.generator(random_secrets, train_images)
      backdoored_image_clipped = clip(backdoored_image, train_images, threat_model, epsilon_clip, device)
      jpeged_backdoored_image = stegano_net.jpeg(backdoored_image_clipped)
      secret_pred = stegano_net.detector(jpeged_backdoored_image)
      orig_pred = stegano_net.detector(train_images)
      secret_pred_without_jpeg = stegano_net.detector(backdoored_image_clipped)
      train_loss = loss_only_detector_mse(secret_pred, random_secrets) \
                   + alpha * loss_only_detector_mse(secret_pred_without_jpeg, random_secrets) \
                   + beta * loss_only_detector_mse(orig_pred, invalid_secret_for_training_sample)

      train_loss.backward()
      optimizer.step()

      # Saves training loss
      train_losses.append(train_loss.data.cpu())
      loss_history.append(train_loss.data.cpu())
      # Prints mini-batch losses
      print('Training: Batch {0}. Loss of {1:.5f}'.format(idx + 1, train_loss.data))

    #train_images_np = train_images.numpy
    torch.save(stegano_net.state_dict(), MODELS_PATH + 'Epoch_' + dataset + '_N{}.pkl'.format(epoch + 1))

    mean_train_loss = np.mean(train_losses)

    # Validation step
    for idx, valid_batch in enumerate(valid_loader):
      data, _ = valid_batch
      data = data.to(device)

      random_secret_array = []
      for i in range(data.shape[0]):
        random_secret_array.append(torch.rand((secret_colorc, secret_shape_1, secret_shape_2)).unsqueeze(0))
      batch = torch.cat(random_secret_array, 0).to(device)
      random_secrets = Variable(upsample(batch), requires_grad=False)

      valid_images = Variable(data, requires_grad=False)

      backdoored_image = stegano_net.generator(random_secrets, valid_images)
      backdoored_image_clipped = clip(backdoored_image, valid_images, threat_model, epsilon_clip, device)
      jpeged_backdoored_image = stegano_net.jpeg(backdoored_image_clipped)
      secret_pred = stegano_net.detector(jpeged_backdoored_image)
      secret_pred_without_jpeg = stegano_net.detector(backdoored_image_clipped)
      orig_pred = stegano_net.detector(valid_images)
      valid_loss = loss_only_detector_mse(secret_pred,random_secrets) \
                   + alpha * loss_only_detector_mse(secret_pred_without_jpeg, random_secrets) \
                   + beta * loss_only_detector_mse(orig_pred,invalid_secret_for_training_sample)

      valid_losses.append(valid_loss.data.cpu())
      '''dif_image = backdoored_image - valid_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0], -1)
      train_image_color_view = valid_images.view(valid_images.shape[0], -1)
      l2 = torch.norm(backdoored_image_color_view - train_image_color_view, p=2, dim=1)'''

    mean_valid_loss = np.mean(valid_losses)

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average train loss: {2:.5f}, Average valid loss: {3:.5f}'.format(epoch + 1, num_epochs, mean_train_loss, mean_valid_loss))

  return stegano_net, mean_train_loss, loss_history

def get_the_best_random_secret_for_net(net, test_loader, batch_size, num_epochs, scenario, device, threat_model ,epsilon_clip, real_jpeg_q):
  net.eval()
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
          backdoored_image_clipped = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], threat_model, epsilon_clip, device)
          open_jpeg_flag_for_cifar10_model = True
        else :
          backdoored_image = net.generator(secret_for_a_batch,test_images)
          backdoored_image_clipped = clip(backdoored_image, test_images, threat_model, epsilon_clip, device)
          open_jpeg_flag_for_cifar10_model = False
        jpeg_file_name = "tmpBckdr" + str(idx) +"_" + str(epoch)+"_"+scenario
        save_images_as_jpeg(backdoored_image_clipped, jpeg_file_name, real_jpeg_q)
        backdoored_image_clipped = open_jpeg_images(backdoored_image_clipped.shape[0], jpeg_file_name, open_jpeg_flag_for_cifar10_model)
        removeImages(backdoored_image_clipped.shape[0],jpeg_file_name)
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
    thresholds = np_original_mins * TAU_PER_RS
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

def test_specific_secret(net, test_loader, batch_size, scenario, threshold_range, device, threat_model, epsilon_clip, specific_secret, diff_jpeg_q=50, real_jpeg_q=80) :
  secret = create_batch_from_a_single_image(specific_secret,batch_size).to(device)
  jpeg = DiffJPEG(image_shape[dataset][0], image_shape[dataset][1], differentiable=True, quality=diff_jpeg_q)
  jpeg = jpeg.to(device)
  for param in jpeg.parameters():
    param.requires_grad = False
  all_the_distance_on_backdoor_jpeg = torch.Tensor().to(device)
  all_the_distance_on_backdoor = torch.Tensor().to(device)
  all_the_distance_on_test = torch.Tensor().to(device)
  mindist = 99999999.999
  max_dif = 0.0
  if DATASET.IMAGENET.value == dataset :
    number_per_labs = 1
  else :
    number_per_labs = 10
  random_without_backdoor = {}
  random_backdoor = {}
  random_clipped_backdoor = {}
  random_clipped_backdoor_dif = {}
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
        backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], threat_model, epsilon_clip, device)
        backdoored_image_clipped = torch.clone(test_images)
        backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped_small_chunk)
        revealed_something_on_test = net.detector(test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
        out = torch.clone(test_images)
        out[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image
        backdoored_image = out
      else :
        backdoored_image = net.generator(secret,test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, threat_model, epsilon_clip, device)
        revealed_secret_on_backdoor = net.detector(backdoored_image_clipped)
        revealed_something_on_test = net.detector(test_images)
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
      this_maxdiff = torch.max(torch.abs(backdoored_image_clipped.detach().cpu()-test_images.detach().cpu())).item()
      if max_dif < this_maxdiff :
        max_dif = this_maxdiff
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
            random_clipped_backdoor_dif[lab] = []
            random_difjpeg_backdoor[lab] = []
            random_revealed[lab] = []
          if len(random_without_backdoor[lab]) < number_per_labs :
            random_without_backdoor[lab].append(test_images[i].detach().cpu())
            random_backdoor[lab].append(backdoored_image[i].detach().cpu())
            random_clipped_backdoor[lab].append(backdoored_image_clipped[i].detach().cpu())
            random_clipped_backdoor_dif[lab].append( ((backdoored_image_clipped[i].detach().cpu()-test_images[i].detach().cpu())*4)+0.5)
            random_difjpeg_backdoor[lab].append(jpeg(backdoored_image_clipped[i].unsqueeze(0))[0].detach().cpu())
            random_revealed[lab].append(revealed_secret_on_backdoor[i].detach().cpu())
            num_of_val_in_random_dicts += 1

    save_image(min_origin, scenario+"best-without_backdoor")
    save_image(min_backdoor, scenario+"best-backdoor")
    save_image(min_backdoor_clipped, scenario+"best-clipped_backdoor")
    save_images_as_jpeg(min_backdoor_clipped.unsqueeze(0), scenario+"best-realjpeg_backdoor", real_jpeg_q)
    save_image(min_jpeg, scenario+"best-difjpeg_backdoor")
    save_image(min_revealed, scenario+"best-revealed")

    save_image_block(random_without_backdoor,scenario+"random-without_backdoor")
    save_image_block(random_backdoor,scenario+"random-backdoor")
    save_image_block(random_clipped_backdoor,scenario+"random-clipped_backdoor")
    save_image_block(random_clipped_backdoor_dif,scenario+"random-clipped_backdoor-diff-image")
    save_image_block(random_clipped_backdoor,scenario+"random-realjpeg_backdoor","jpeg", real_jpeg_q)
    save_image_block(random_difjpeg_backdoor,scenario+"random-difjpeg_backdoor")
    save_image_block(random_revealed,scenario+"random-revealed")

    for threshold_percent in np.arange(0.0, 1.05, 0.05) :
      threshold = torch.min(all_the_distance_on_test).item() * threshold_percent
      tpr_for_this = torch.sum(all_the_distance_on_backdoor < threshold) / all_the_distance_on_backdoor.shape[0]
      tpr_for_this_jpeg = torch.sum(all_the_distance_on_backdoor_jpeg < threshold) / all_the_distance_on_backdoor_jpeg.shape[0]
      print(threshold_percent, tpr_for_this.item(), tpr_for_this_jpeg.item(), threshold, torch.min(all_the_distance_on_test).item())

    tpr_results = {}
    tnr_results = {}
    for threshold in threshold_range :
      tpr = torch.sum(all_the_distance_on_backdoor < threshold).item() / all_the_distance_on_backdoor.shape[0]
      tnr = torch.sum(all_the_distance_on_test >= threshold).item() / all_the_distance_on_test.shape[0]
      tpr_results[threshold] = torch.ones(1)*tpr
      tnr_results[threshold] = torch.ones(1)*tnr
      print(threshold, tpr, tnr)
    print(max_dif,max_dif*255)
    return tpr_results, tnr_results

def test_specific_secret_and_threshold(net, test_loader, batch_size, threat_model, scenario, device, epsilon_clip, specific_secret, specific_threshold, real_jpeg_q=80) :
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
        backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], threat_model, epsilon_clip, device)
        backdoored_image_clipped = test_images
        backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
        predY_on_backdoor = backdoor_model(backdoored_image_clipped_small_chunk)
      else :
        backdoored_image = net.generator(secret, test_images)
        backdoored_image_clipped = clip(backdoored_image, test_images, threat_model, epsilon_clip, device)
        predY_on_backdoor = backdoor_model(backdoored_image_clipped)
      test_acces_backdoor_detect_model_on_backdoor.append(torch.sum(torch.argmax(predY_on_backdoor, dim=1) == targetY_backdoor).item() / test_images.shape[0])
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
  mean_test_acces_backdoor_detect_model_on_backdoor_jpeg = np.mean(test_acces_backdoor_detect_model_on_jpeg_backdoor)
  print("Accuracy on jpeg backdoor images:{0:.4f}".format(mean_test_acces_backdoor_detect_model_on_backdoor_jpeg))

def robust_random_attack(backdoor_detect_model, test_loader, batch_size, num_epochs, threshold_range, device, threat_model, epsilon, specific_secret, scenario) :
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
          rand_image_square_sum_l2_divider = epsilon / rand_image_l2
          diff_image  = (rand_image * rand_image_square_sum_l2_divider.unsqueeze(1).unsqueeze(1).unsqueeze(1))
        else :
          rand_image = (torch.rand(test_images.shape)*2)-1
          diff_image = rand_image * epsilon
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
    for threshold in threshold_range :
      tnr = tnr_by_threshold[threshold] / sum_of_evaluations
      print(threshold, distrib_of_random_attack[threshold], tnr)



def robust_test_model(backdoor_generator_model, backdoor_detect_model, robust_model, attack_name, attack_scope, scenario, trials, threat_model, test_loader, batch_size, device, epsilon_clip, specific_secret, pred_threshold, real_jpeg_q, target_class=-1):
  eps = epsilon_clip
  secret = create_batch_from_a_single_image(specific_secret,batch_size).to(device)
  backdoor_model = ThresholdedBackdoorDetectorStegano(backdoor_detect_model,specific_secret.to(device),pred_threshold,device)

  if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
    pos_backdor = [0,0]
    robust_model_with_backdoor = ModelWithSmallBackdoor(backdoor_model, robust_model, pos_backdor, image_shape[DATASET.CIFAR10.value], device, target_class).to(device)
  else :
    robust_model_with_backdoor = ModelWithBackdoor(backdoor_model, robust_model, device, target_class).to(device)

  robust_model.eval()
  robust_model_with_backdoor.eval()
  backdoor_model.eval()
  backdoor_detect_model.eval()
  backdoor_generator_model.eval()

  apgd_n_restarts = trials
  apgd_targeted_n_target_classes = 9
  apgd_targeted_n_restarts = 1
  fab_n_target_classes = 9
  fab_n_restarts = trials
  square_n_queries = 5000
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

  num_of_batch = 0

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
    predY = predY[:,1]
    test_acces_backdoor_detect_model.append(torch.sum(predY > targetY_original).item()/test_images.shape[0])
    predY_on_robustmodel_with_backdoor = robust_model_with_backdoor(test_images).detach().cpu()
    test_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, test_images, test_y))
    predY_on_robustmodel = robust_model(test_images).detach().cpu()
    test_acces_robust_model.append(torch.sum(torch.argmax(predY_on_robustmodel, dim=1) == test_y).item()/test_images.shape[0])
    #test_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, test_images, test_y))

    if ATTACK_SCOPE.ROBUST_MODEL.value in attack_scope :
      x_adv_robust_model = attack_for_robust_model.run_standard_evaluation(test_images, test_y_on_GPU)
      predY_on_robustmodel_adversarial = robust_model(x_adv_robust_model).detach().cpu()
      test_rob_acces_robust_model.append(torch.sum(torch.argmax(predY_on_robustmodel_adversarial, dim=1) == test_y).item()/test_images.shape[0])
      #test_rob_acces_robust_model.append(fb.utils.accuracy(fb_robust_model, x_adv_robust_model, test_y))
      mean_test_rob_acces_robust_model = np.mean(test_rob_acces_robust_model)
    else :
      mean_test_rob_acces_robust_model = -1.0
    if ATTACK_SCOPE.ROBUST_MODEL_WITH_BACKDOOR.value in attack_scope :
      x_adv_robust_model_with_backdoor = attack_for_robust_model_with_backdoor.run_standard_evaluation(test_images, test_y_on_GPU)
      predY_on_robustmodel_with_backdoor_adversarial = robust_model_with_backdoor(x_adv_robust_model_with_backdoor).detach().cpu()
      test_rob_acces_robust_model_with_backdoor.append(torch.sum(torch.argmax(predY_on_robustmodel_with_backdoor_adversarial, dim=1) == test_y).item()/test_images.shape[0])
      #test_rob_acces_robust_model_with_backdoor.append(fb.utils.accuracy(fb_robust_model_with_backdoor, x_adv_robust_model_with_backdoor, test_y))
      mean_test_rob_acces_robust_model_with_backdoor = np.mean(test_rob_acces_robust_model_with_backdoor)
      if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
        predY_backdoor_detect_model_on_adv_robust_model_with_backdoor = backdoor_model(x_adv_robust_model_with_backdoor[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])]).detach().cpu()
      else :
        predY_backdoor_detect_model_on_adv_robust_model_with_backdoor = backdoor_model(x_adv_robust_model_with_backdoor).detach().cpu()
      predY_backdoor_detect_model_on_adv_robust_model_with_backdoor = predY_backdoor_detect_model_on_adv_robust_model_with_backdoor[:,1]
      test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor.append(torch.sum(predY_backdoor_detect_model_on_adv_robust_model_with_backdoor > targetY_original).item()/test_images.shape[0])
      mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = np.mean(test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor)
    else :
      mean_test_rob_acces_robust_model_with_backdoor = -1.0
      mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor = -1.0 #

    if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
      backdoored_image = backdoor_generator_model(secret,test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])])
      backdoored_image_clipped_small_chunk = clip(backdoored_image, test_images[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])], threat_model, epsilon_clip, device)
      backdoored_image_clipped = torch.clone(test_images)
      backdoored_image_clipped[:,:,pos_backdor[0]:(pos_backdor[0]+image_shape[DATASET.CIFAR10.value][0]),pos_backdor[1]:(pos_backdor[1]+image_shape[DATASET.CIFAR10.value][1])] = backdoored_image_clipped_small_chunk
      predY_on_backdoor = backdoor_model(backdoored_image_clipped_small_chunk).detach().cpu()
      open_jpeg_flag_for_cifar10_model = True
    else :
      backdoored_image = backdoor_generator_model(secret,test_images)
      backdoored_image_clipped = clip(backdoored_image, test_images, threat_model, epsilon_clip, device)
      predY_on_backdoor = backdoor_model(backdoored_image_clipped).detach().cpu()
      open_jpeg_flag_for_cifar10_model = False
    predY_on_backdoor = predY_on_backdoor[:,1]
    test_acces_backdoor_detect_model_on_backdoor.append(torch.sum(predY_on_backdoor > targetY_original).item()/test_images.shape[0])
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
    predY_on_backdoor_with_jpeg = predY_on_backdoor_with_jpeg[:,1]
    test_acces_backdoor_detect_model_on_backdoor_with_jpeg.append(torch.sum(predY_on_backdoor_with_jpeg > targetY_original).item()/test_images.shape[0])
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
    'Robust accuracy on test set backdoor_detect_model: {3:.4f}, robust_model_with_backdoor: {4:.4f}, robust_model: {5:.4f}; '
    'Accuracy on backdoor images backdoor_detect_model: {6:.4f}, robust_model_with_backdoor: {7:.4f}, robust_model: {8:.4f}; '
    'Accuracy on JPEG backdoor images backdoor_detect_model: {9:.4f}, robust_model_with_backdoor: {10:.4f}, robust_model: {11:.4f}; '.format(
    mean_test_acces_backdoor_detect_model,mean_test_acces_robust_model_with_backdoor,mean_test_acces_robust_model,
    mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor,mean_test_rob_acces_robust_model_with_backdoor,mean_test_rob_acces_robust_model,
    mean_test_acces_backdoor_detect_model_on_backdoor,mean_test_acces_robust_model_with_backdoor_on_backdoor,mean_test_acces_robust_model_on_backdoor,
    mean_test_acces_backdoor_detect_model_on_backdoor_with_jpeg, mean_test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg,
    mean_test_acces_robust_model_on_backdoor_with_jpeg))
    print('{0:.4f} & {1:.4f} & {2:.4f} | {3:.4f} & {4:.4f} & {5:.4f} | {6:.4f} & {7:.4f} & {8:.4f}'.format(mean_test_rob_acces_robust_model,
    mean_test_rob_acces_robust_model_with_backdoor,mean_test_rob_acces_backdoor_detect_model_on_adv_robust_model_with_backdoor,
    mean_test_acces_robust_model_on_backdoor,mean_test_acces_robust_model_with_backdoor_on_backdoor,mean_test_acces_backdoor_detect_model_on_backdoor,
    mean_test_acces_robust_model_on_backdoor_with_jpeg, mean_test_acces_robust_model_with_backdoor_on_backdoor_with_jpeg,
    mean_test_acces_backdoor_detect_model_on_backdoor_with_jpeg))
    #mean_test_acces_backdoor_detect_model_on_adversarial,mean_test_acces_backdoor_detect_model_on_adversarial
    #'Accuracy on adversarial images backdoor_detect_model: {12:.4f}, backdoor_detect_model: {13:.4f}; '


parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dataset', type=str, default="cifar10")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument('--secret', type=str, default="NOPE")
parser.add_argument('--mode', type=str, default="train_test")
parser.add_argument('--generator', type=str, default="NOPE")
parser.add_argument('--detector', type=str, default="NOPE")
parser.add_argument("--scenario", type=str, default="4x4")
parser.add_argument("--robust_model", type=str , default="Gowal2020Uncovering_28_10_extra")
parser.add_argument("--threat_model", type=str , default="Linf")
parser.add_argument("--attack", type=str , default="PGD")
parser.add_argument("--attack_scope", type=str , default="robust_model_with_backdoor")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--beta', type=float, default=0.001)
parser.add_argument('--jpeg_q', type=int, default=50)
parser.add_argument('--real_jpeg_q', type=int, default=80)
parser.add_argument("--pred_threshold", type=float, default=64.93129)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--target_class', type=int, default=-1)
parser.add_argument('--n_mean', type=float, default=0.0)
parser.add_argument('--n_stddev', type=float, default=1.0/255.0)
parser.add_argument('--epsilon', type=float, default=8.0/255.0) # (8.0/255.0) , 0.01564 (4.0/255.0)
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
trials = params.trials

# Hyper Parameters
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
alpha = params.alpha
beta = params.beta
epsilon = params.epsilon

mode = params.mode
scenario = params.scenario

model = params.model
secret = params.secret
real_jpeg_q = params.real_jpeg_q
threshold_range = np.arange(params.start_of_the_threshold_range,params.end_of_the_threshold_range,params.step_of_the_threshold_range)
target_class = params.target_class

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


if SCENARIOS.CIFAR10_MODEL.value in scenario and dataset != DATASET.CIFAR10.value :
  stegano_net = Net(gen_holder=HidingNetworkDeepStegano, det_holder=RevealNetworkNetworkDeepStegano, image_shape=image_shape[DATASET.CIFAR10.value], color_channel= color_channel[DATASET.CIFAR10.value], jpeg_q=params.jpeg_q, device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
else :
  stegano_net = Net(gen_holder=HidingNetworkDeepStegano, det_holder=RevealNetworkNetworkDeepStegano, image_shape=image_shape[dataset], color_channel= color_channel[dataset], jpeg_q=params.jpeg_q, device= device, n_mean=params.n_mean, n_stddev=params.n_stddev)
stegano_net.to(device)
if model != 'NOPE' :
  stegano_net.load_state_dict(torch.load(MODELS_PATH + model, map_location=device))
if MODE.TRAIN.value in mode :
  stegano_net, mean_train_loss_ret, loss_history_ret = train_model(stegano_net, train_loader, batch_size, val_loader, scenario=scenario, threat_model=threat_model, num_epochs=num_epochs, alpha=alpha, beta=beta, epsilon_clip=epsilon, learning_rate=learning_rate, device=device)
backdoor_detect_model = stegano_net.detector
backdoor_generator_model = stegano_net.generator
if MODE.CHOSE_THE_BEST_SECRET.value in mode :
  get_the_best_random_secret_for_net(net=stegano_net, test_loader=val_loader, batch_size=batch_size, num_epochs=num_epochs, scenario=scenario, device=device, threat_model=robust_model_threat_model, epsilon_clip=epsilon, real_jpeg_q=params.real_jpeg_q)
if MODE.TEST_SPECIFIC_SECRET.value in mode :
  test_specific_secret(net=stegano_net, test_loader=test_loader, batch_size=batch_size, scenario=scenario, threshold_range=threshold_range, device=device, threat_model=robust_model_threat_model, epsilon_clip=epsilon, specific_secret=best_secret, real_jpeg_q=params.real_jpeg_q)
if MODE.TEST_THRESHOLDED_BACKDOOR.value in mode :
  test_specific_secret_and_threshold(net=stegano_net, test_loader=test_loader, batch_size=batch_size, scenario=scenario, device=device, epsilon_clip=epsilon, threat_model=robust_model_threat_model, specific_secret=best_secret, specific_threshold=pred_threshold, real_jpeg_q=params.real_jpeg_q)
if MODE.ATTACK.value in mode :
  robust_model = load_model(model_name=robust_model_name, dataset=dataset, threat_model=robust_model_threat_model).to(device)
  robust_test_model(backdoor_generator_model=backdoor_generator_model, backdoor_detect_model=backdoor_detect_model, robust_model=robust_model, attack_name=attack_name, attack_scope=attack_scope, scenario=scenario, trials=trials, threat_model=robust_model_threat_model, test_loader=test_loader, batch_size=batch_size,  device=device, epsilon_clip=epsilon, specific_secret=best_secret, pred_threshold=pred_threshold, real_jpeg_q=real_jpeg_q, target_class=target_class)
if MODE.RANDOM_ATTACK.value in mode :
  robust_random_attack(backdoor_detect_model,test_loader=test_loader,batch_size=batch_size,num_epochs=num_epochs,epsilon=epsilon,specific_secret=best_secret,threshold_range=threshold_range,device=device,threat_model=threat_model,scenario=scenario)
