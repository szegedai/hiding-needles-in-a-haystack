import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

MODELS_PATH = '../res/models/'
DATA_PATH = '../res/data/'


def customized_loss(predY, backdoored_image, targetY, image, B):
  loss_injection = torch.nn.functional.mse_loss(backdoored_image, image)
  loss_detect = torch.nn.functional.binary_cross_entropy(predY, targetY)
  loss_all = loss_injection + B * loss_detect
  return loss_all, loss_injection, loss_detect

def gaussian(tensor, mean=0, stddev=0.1):
  '''Adds random noise to a tensor.'''
  noise = torch.nn.init.normal(torch.Tensor(tensor.size()), mean, stddev)
  return Variable(tensor + noise)

class BackdoorInjectNetwork(nn.Module) :
  def __init__(self, color_channel=3):
    super(BackdoorInjectNetwork, self).__init__()
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
    out_noise = gaussian(out.data, 0, 0.1)
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
      nn.Conv2d(150, 50, kernel_size=3, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=3, stride=2))
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
  def __init__(self, color_channel):
    super(Net, self).__init__()
    self.m1 = BackdoorInjectNetwork(color_channel)
    self.m2 = BackdoorDetectNetwork(color_channel)

  def forward(self, image):
    backdoored_image, backdoored_image_with_noise = self.m1(image)
    next_input = torch.cat((backdoored_image, image), 0)
    y = self.m2(next_input)
    return backdoored_image, y

def train_model(net, train_loader, num_epochs, beta, learning_rate):
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

      train_images = Variable(data, requires_grad=False)
      targetY_backdoored = torch.from_numpy(np.ones((train_images.shape[0],)))
      targetY_original = torch.from_numpy(np.zeros((train_images.shape[0],)))
      targetY = torch.cat((targetY_backdoored,targetY_original),0)

      # Forward + Backward + Optimize
      optimizer.zero_grad()
      backdoored_image, predY = net(train_images)

      # Calculate loss and perform backprop
      train_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, train_images, targetY, beta)
      train_loss.backward()
      optimizer.step()

      # Saves training loss
      train_losses.append(train_loss.data[0])
      loss_history.append(train_loss.data[0])

      # Prints mini-batch losses
      print('Training: Batch {0}/{1}. Loss of {2:.4f}, cover loss of {3:.4f}, secret loss of {4:.4f}'.format(idx + 1, len(train_loader), train_loss.data[0], train_loss_cover.data[0], train_loss_secret.data[0]))

    torch.save(net.state_dict(), MODELS_PATH + 'Epoch N{}.pkl'.format(epoch + 1))

    mean_train_loss = np.mean(train_losses)

    # Prints epoch average loss
    print('Epoch [{0}/{1}], Average_loss: {2:.4f}'.format(
      epoch + 1, num_epochs, mean_train_loss))

  return net, mean_train_loss, loss_history


def test_model(net, test_loader, beta):
  # Switch to evaluate mode
  net.eval()

  test_losses = []
  # Show images
  for idx, test_batch in enumerate(test_loader):
    # Saves images
    data, _ = test_batch

    test_images = Variable(data, volatile=True)
    targetY_backdoored = torch.from_numpy(np.ones((test_images.shape[0],)))
    targetY_original = torch.from_numpy(np.zeros((test_images.shape[0],)))
    targetY = torch.cat((targetY_backdoored, targetY_original), 0)

    # Compute output
    backdoored_image, predY = net(test_images)

    # Calculate loss
    test_loss, loss_injection, loss_detect = customized_loss(backdoored_image, predY, test_images, targetY, beta)

    #     diff_S, diff_C = np.abs(np.array(test_output.data[0]) - np.array(test_secret.data[0])), np.abs(np.array(test_hidden.data[0]) - np.array(test_cover.data[0]))

    #     print (diff_S, diff_C)

    if idx in [1, 2, 3, 4]:
      print('Total loss: {:.2f} \nLoss on secret: {:.2f} \nLoss on cover: {:.2f}'.format(test_loss.data[0],
                                                                                         loss_injection.data[0],
                                                                                         loss_detect.data[0]))
    test_losses.append(test_loss.data[0])

  mean_test_loss = np.mean(test_losses)

  print('Average loss on test set: {:.2f}'.format(mean_test_loss))
  return mean_test_loss

# Hyper Parameters
num_epochs = 3
batch_size = 2
learning_rate = 0.0001
beta = 1

# Other Parameters
device = torch.device('cuda:0')
dataset = "CIFAR10"
dataset = "MNIST"
if dataset == "CIFAR10" :
#Open cifar10 dataset
  color_channel = 3
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  trainset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=True, download=True, transform=transform)
  testset = torchvision.datasets.CIFAR10(root=DATA_PATH, train=False, download=True, transform=transform)
  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#Open mnist dataset
elif dataset == "MNIST" :
  color_channel = 1
  trainset = torchvision.datasets.MNIST(root=DATA_PATH, train=True, download=True, transform=None)
  testset = torchvision.datasets.MNIST(root=DATA_PATH, train=False, download=True, transform=None)

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

dataiter = iter(trainloader)
images, labels = dataiter.next()

net = Net(color_channel)
net.to(device)
net, mean_train_loss, loss_history = train_model(net, trainloader, num_epochs, beta, learning_rate)
mean_test_loss = test_model(net, testloader, beta)

