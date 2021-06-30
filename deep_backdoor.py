import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from argparse import ArgumentParser

MODELS_PATH = '../res/models/'
DATA_PATH = '../res/data/'

std = {}
mean = {}

# Mean and std deviation
#  of imagenet dataset. Source: http://cs231n.stanford.edu/reports/2017/pdfs/101.pdf
std['IMAGENET'] = [0.229, 0.224, 0.225]
mean['IMAGENET'] = [0.485, 0.456, 0.406]
#  of cifar10 dataset.
std['CIFAR10'] = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]
mean['CIFAR10'] = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
#  of mnist dataset.
std['MNIST'] = [0.3084485240270358]
mean['MNIST'] = [0.13092535192648502]

def customized_loss(backdoored_image, predY, image, targetY, B):
  #print(backdoored_image.shape, image.shape)
  criterion1 = nn.MSELoss()
  #loss_injection = torch.nn.functional.mse_loss(backdoored_image, image)
  loss_injection = criterion1(backdoored_image, image)
  criterion2 = nn.BCELoss()
  loss_detect = criterion2(predY, targetY)
  loss_all = loss_injection + B * loss_detect
  return loss_all, loss_injection, loss_detect

def gaussian(tensor, device, mean=0, stddev=0.1):
  '''Adds random noise to a tensor.'''
  noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev)
  noise = noise.to(device)
  return Variable(tensor + noise)

def denormalize(images, color_channel, std, mean):
  ''' Denormalizes a tensor of images.'''
  for t in range(color_channel):
    images[:,t, :, :] = (images[:,t, :, :] * std[t]) + mean[t]
  return images


class BackdoorInjectNetwork(nn.Module) :
  def __init__(self, device, color_channel=3):
    super(BackdoorInjectNetwork, self).__init__()
    self.device = device
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, h):
    h1 = self.initialH3(h)
    h2 = self.initialH4(h)
    h3 = self.initialH5(h)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    out = self.finalH(mid2)
    out_noise = gaussian(out.data, self.device, 0, 0.1)
    return out, out_noise

class BackdoorDetectNetwork(nn.Module) :
  def __init__(self, color_channel=3):
    super(BackdoorDetectNetwork, self).__init__()
    self.initialH3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalH3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2))
    self.finalH4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=4, stride=2))
    self.finalH5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=3),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=5, stride=2))
    self.avg_kernel = 5
    avg_kernel = self.avg_kernel
    self.avgpool = nn.AdaptiveAvgPool2d((avg_kernel, avg_kernel))
    self.flatten_size = 150*avg_kernel*avg_kernel
    flatten_size = self.flatten_size
    self.classifier =  nn.Sequential(
      nn.Dropout(),
      nn.Linear(flatten_size,flatten_size),
      nn.ReLU(),
      nn.Linear(flatten_size,flatten_size),
      nn.ReLU(),
      nn.Linear(flatten_size,1),
      nn.Sigmoid()
    )

  def forward(self, h):
    h1 = self.initialH3(h)
    h2 = self.initialH4(h)
    h3 = self.initialH5(h)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    avgpool_mid = self.avgpool(mid2)
    out = self.classifier(avgpool_mid.view(-1,self.flatten_size))
    return out

class Net(nn.Module):
  def __init__(self, device, color_channel):
    super(Net, self).__init__()
    self.m1 = BackdoorInjectNetwork(device, color_channel)
    self.m2 = BackdoorDetectNetwork(color_channel)

  def forward(self, image):
    backdoored_image, backdoored_image_with_noise = self.m1(image)
    next_input = torch.cat((backdoored_image, image), 0)
    y = self.m2(next_input)
    return backdoored_image, y

def train_model(net, train_loader, num_epochs, beta, learning_rate, device):
  # Save optimizer
  optimizer = optim.Adam(net.parameters(), lr=learning_rate)

  loss_history = []
  # Iterate over batches performing forward and backward passes
  for epoch in range(num_epochs):

    # Train mode
    net.train()

    train_losses = []
    # Train one epoch
    for idx, train_batch in enumerate(train_loader):
      data, _ = train_batch
      data = data.to(device)

      train_images = Variable(data, requires_grad=False)
      targetY_backdoored = torch.from_numpy(np.ones((train_images.shape[0],1),np.float32))
      targetY_original = torch.from_numpy(np.zeros((train_images.shape[0],1),np.float32))
      targetY = torch.cat((targetY_backdoored,targetY_original),0)
      targetY = targetY.to(device)

      # Forward + Backward + Optimize
      optimizer.zero_grad()
      backdoored_image, predY = net(train_images)

      # Calculate loss and perform backprop
      train_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, train_images, targetY, beta)
      train_loss.backward()
      optimizer.step()

      # Saves training loss
      train_losses.append(train_loss.data.cpu())
      loss_history.append(train_loss.data.cpu())

      denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      denormalized_train_images = denormalize(images=train_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
      linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_train_images), p=float("inf")).item()
      l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0], -1) - denormalized_train_images.view(denormalized_train_images.shape[0], -1)), p=2, dim=1)).item()
      # Prints mini-batch losses
      print('Training: Batch {0}/{1}. Loss of {2:.4f}, injection loss of {3:.4f}, detect loss of {4:.4f}, backdoor l2 {5:.4f}, backdoor linf {6:.4f}'.format(idx + 1, len(train_loader), train_loss.data, loss_injection.data, loss_detect.data, l2, linf))

    train_images_np = train_images.numpy
    torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))

    mean_train_loss = np.mean(train_losses)

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average_loss: {2:.4f}, Last backdoor l2: {3:.4f}, Last backdoor linf: {4:.4f}'.format(epoch + 1, num_epochs, mean_train_loss, l2, linf))

  return net, mean_train_loss, loss_history


def test_model(net, test_loader, beta, device):
  # Switch to evaluate mode
  net.eval()

  test_losses = []
  max_linf = 0
  max_l2 = 0
  # Show images
  for idx, test_batch in enumerate(test_loader):
    # Saves images
    data, _ = test_batch
    data = data.to(device)

    test_images = Variable(data, volatile=True)
    targetY_backdoored = torch.from_numpy(np.ones((test_images.shape[0],1),np.float32))
    targetY_original = torch.from_numpy(np.zeros((test_images.shape[0],1),np.float32))
    targetY = torch.cat((targetY_backdoored, targetY_original), 0)
    targetY = targetY.to(device)

    # Compute output
    backdoored_image, predY = net(test_images)

    # Calculate loss
    test_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, test_images, targetY, beta)

    test_losses.append(test_loss.data.cpu())

    denormalized_backdoored_images = denormalize(images=backdoored_image, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
    denormalized_test_images = denormalize(images=test_images, color_channel=color_channel, std=std[dataset], mean=mean[dataset])
    linf = torch.norm(torch.abs(denormalized_backdoored_images - denormalized_test_images), p=float("inf")).item()
    max_linf = max(max_linf, linf)
    l2 = torch.max(torch.norm((denormalized_backdoored_images.view(denormalized_backdoored_images.shape[0],-1) - denormalized_test_images.view(denormalized_test_images.shape[0], -1)), p=2, dim=1)).item()
    max_l2 = max(max_l2, l2)

  mean_test_loss = np.mean(test_losses)

  print('Average loss on test set: {0:.4f}, backdoor max l2: {1:.4f}, backdoor max linf: {2:.4f}'.format(mean_test_loss,max_linf,max_l2))
  return mean_test_loss


parser = ArgumentParser(description='Model evaluation')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--attack', type=str, default="L2PGD")
parser.add_argument('--dataset', type=str, default="MNIST")
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--beta', type=int, default=1)
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--step_size', type=float, default=0.01)
parser.add_argument('--steps', type=int, default=40)
parser.add_argument('--eps', type=float, default=0.1)
parser.add_argument('--verbose', type=int, default=0)
params = parser.parse_args()

# Hyper Parameters
num_epochs = params.epochs
batch_size = params.batch_size
learning_rate = params.learning_rate
beta = params.beta

# Other Parameters
device = torch.device('cuda:'+str(params.gpu))
dataset = params.dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean[dataset], std=std[dataset])])
#transform = transforms.ToTensor()
if dataset == "CIFAR10" :
#Open cifar10 dataset
  color_channel = 3
  trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Open mnist dataset
elif dataset == "MNIST" :
  color_channel = 1
  trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=transform)
  testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

#dataiter = iter(trainloader)
#images, labels = dataiter.next()

net = Net(device, color_channel)
net.to(device)
net, mean_train_loss, loss_history = train_model(net, trainloader, num_epochs, beta, learning_rate, device)
mean_test_loss = test_model(net, testloader, beta, device)

