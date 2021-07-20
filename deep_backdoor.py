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
from backdoor_model import Net

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


def customized_loss(backdoored_image, predY, image, targetY, B, L):
  #print(backdoored_image.shape, image.shape)
  criterion1 = nn.MSELoss()
  #loss_injection = torch.nn.functional.mse_loss(backdoored_image, image)
  loss_injection = criterion1(backdoored_image, image) +\
                   l1_penalty(backdoored_image - image, l1_lambda=L / 100) +\
                   l2_penalty(backdoored_image - image, l2_lambda=L / 10) +\
                   linf_penalty(backdoored_image - image, linf_lambda=L)
  criterion2 = nn.BCELoss()
  loss_detect = criterion2(predY, targetY)
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


def train_model(net, train_loader, num_epochs, beta, l, reg_start, learning_rate, device):
  # Save optimizer
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  loss_history = []
  # Iterate over batches performing forward and backward passes
  for epoch in range(num_epochs):

    # Train mode
    net.train()

    train_losses = []

    L = 0
    if epoch >= reg_start :
      L = l
    # Train one epoch
    for idx, train_batch in enumerate(train_loader):
      data, _ = train_batch
      data = data.to(device)

      train_images = Variable(data, requires_grad=False)
      targetY_backdoored = torch.from_numpy(np.ones((train_images.shape[0], 1), np.float32))
      targetY_original = torch.from_numpy(np.zeros((train_images.shape[0], 1), np.float32))
      targetY = torch.cat((targetY_backdoored,targetY_original),0)
      targetY = targetY.to(device)

      # Forward + Backward + Optimize
      optimizer.zero_grad()
      backdoored_image, predY = net(train_images)

      # Calculate loss and perform backprop
      train_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, train_images, targetY, B=beta, L=L)
      train_loss.backward()
      optimizer.step()

      # Saves training loss
      train_losses.append(train_loss.data.cpu())
      loss_history.append(train_loss.data.cpu())
      dif_image = backdoored_image - train_images
      linf = torch.norm(torch.abs(dif_image), p=float("inf"), dim=(1,2,3))
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0], backdoored_image.shape[1], -1)
      train_image_color_view = train_images.view(train_images.shape[0], train_images.shape[1], -1)
      l2 = torch.norm(backdoored_image_color_view - train_image_color_view, p=2, dim=2)

      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset])
      denormalized_train_images = denormalize(images=train_images, color_channel=color_channel[dataset], std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_train_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0], -1) - denormalized_train_images.view(denormalized_train_images.shape[0], -1)), p=2, dim=1)).item()
      '''
      # Prints mini-batch losses
      print('Training: Batch {0}/{1}. Loss of {2:.5f}, injection loss of {3:.5f}, detect loss of {4:.5f},'
            ' backdoor l2 min: {5:.3f}, avg: {6:.3f}, max: {7:.3f}, backdoor linf'
            ' min: {8:.3f}, avg: {9:.3f}, max: {10:.3f}'.format(
        idx + 1, len(train_loader), train_loss.data, loss_injection.data, loss_detect.data,
        torch.min(l2).item(), torch.mean(l2).item(), torch.max(l2).item(),
        torch.min(linf).item(), torch.mean(linf).item(), torch.max(linf).item()))

    train_images_np = train_images.numpy
    torch.save(net.state_dict(), MODELS_PATH + 'Epoch_'+dataset+'_N{}.pkl'.format(epoch + 1))

    mean_train_loss = np.mean(train_losses)

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average_loss: {2:.5f}, Last backdoor l2 min: {3:.3f}, avg: {4:.3f}, max: {5:.3f},'
          ' Last backdoor linf min: {6:.3f}, avg: {7:.3f}, max: {8:.3f}'.format(
      epoch + 1, num_epochs, mean_train_loss, torch.min(l2).item(), torch.mean(l2).item(), torch.max(l2).item(),
      torch.min(linf).item(), torch.mean(linf).item(), torch.max(linf).item()))

  return net, mean_train_loss, loss_history


def test_model(net, test_loader, beta, l, device):
  # Switch to evaluate mode
  net.eval()

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
      backdoored_image, predY = net(test_images)

      # Calculate loss
      test_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, test_images, targetY, B=beta, L=l)

      test_acc = torch.sum(torch.round(predY) == targetY).item()/predY.shape[0]

      if len((torch.round(predY) != targetY).nonzero(as_tuple=True)[0]) > 0 :
        for index in (torch.round(predY) == targetY).nonzero(as_tuple=True)[0] :
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
      backdoored_image_color_view = backdoored_image.view(backdoored_image.shape[0],backdoored_image.shape[1], -1)
      test_image_color_view = test_images.view(test_images.shape[0],test_images.shape[1], -1)
      l2 = torch.norm(backdoored_image_color_view - test_image_color_view, p=2, dim=2)
      '''
      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      denormalized_test_images = denormalize(images=test_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_test_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0],-1) - denormalized_test_images.view(denormalized_test_images.shape[0], -1)), p=2, dim=1)).item()
      '''
      if (max_linf < torch.max(linf).item()) :
        last_maxinf_backdoored_image = backdoored_image[torch.argmax(linf).item()]
        last_maxinf_test_image = test_images[torch.argmax(linf).item()]
        last_maxinf_diff_image = torch.abs(dif_image[torch.argmax(linf).item()])
      max_linf = max(max_linf, torch.max(linf).item())
      if (max_l2 < torch.max(l2).item()):
        last_max2_backdoored_image = backdoored_image[int(torch.argmax(l2).item()/3.0)]
        last_max2_test_image = test_images[int(torch.argmax(l2).item()/3.0)]
        last_max2_diff_image = torch.abs(dif_image[int(torch.argmax(l2).item()/3.0)])
      max_l2 = max(max_l2, torch.max(l2).item())

      if (min_linf > torch.min(linf).item()) :
        last_mininf_backdoored_image = backdoored_image[torch.argmin(linf).item()]
        last_mininf_test_image = test_images[torch.argmin(linf).item()]
        last_mininf_diff_image = torch.abs(dif_image[torch.argmin(linf).item()])
      min_linf = min(min_linf, torch.min(linf).item())
      if (min_l2 > torch.min(l2).item()):
        last_min2_backdoored_image = backdoored_image[int(torch.argmin(l2).item()/3.0)]
        last_min2_test_image = test_images[int(torch.argmin(l2).item()/3.0)]
        last_min2_diff_image = torch.abs(dif_image[int(torch.argmin(l2).item()/3.0)])
      max_l2 = max(max_l2, torch.max(l2).item())
      mean_linf += torch.mean(linf).item()
      mean_l2 += torch.mean(l2).item()


  mean_l2 = mean_l2 / num_of_batch
  mean_linf = mean_linf / num_of_batch

  mean_test_loss = np.mean(test_losses)
  mean_test_acc = np.mean(test_acces)
  print('Average loss on test set: {0:.4f}; accuracy: {1:.4f}; error on backdoor: {2:d}, on original: {3:d}; '
        'backdoor l2 min: {4:.4f}, avg: {5:.4f}, max: {6:.4f}; backdoor linf min: {7:.4f}, avg: {8:.4f}, max: {9:.4f}'.format(
    mean_test_loss,mean_test_acc,len(error_on_backdoor_image),len(error_on_original_image),
    min_l2,mean_l2,max_l2,min_linf,mean_linf,max_linf))
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
  for image_pair in error_on_backdoor_image:
    saveImage(image_pair[0], "error_by_original_backdoor" + str(index))
    saveImage(image_pair[1], "error_by_original_original" + str(index))
    index += 1

  return mean_test_loss


parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack', type=str, default="L2PGD")
parser.add_argument('--dataset', type=str, default="CIFAR10")
parser.add_argument('--model', type=str, default="NOPE")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--regularization_start_epoch', type=int, default=0)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument("--l", type=float, default=0.1)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--verbose', type=int, default=0)
parser.add_argument('--n_mean', type=float, default=0.0)
parser.add_argument('--n_stddev', type=float, default=0.1)
params = parser.parse_args()

# Hyper Parameters
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
beta = params.beta
l = params.l

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

print(params.n_mean, params.n_stddev)
net = Net(image_shape=image_shape[dataset], device= device, color_channel= color_channel[dataset], n_mean=params.n_mean, n_stddev=params.n_stddev)
net.to(device)
if params.model != 'NOPE' :
  net.load_state_dict(torch.load(MODELS_PATH+params.model))
net, mean_train_loss, loss_history = train_model(net, train_loader, num_epochs, beta=beta, l=l, reg_start=params.regularization_start_epoch, learning_rate=learning_rate, device=device)
mean_test_loss = test_model(net, test_loader, beta=beta, l=l, device=device)

