import torch
import torch.nn as nn
from torch.autograd import Variable
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG



def gaussian(tensor_data, device, mean=0, stddev=0.1):
  '''Adds random noise to a tensor.'''
  noise = torch.nn.init.normal_(tensor=torch.Tensor(tensor_data.size()), mean=mean, std=stddev)
  noise = noise.to(device)
  return Variable(tensor_data + noise)


class BackdoorInjectNetworkWideMegyeri(nn.Module) :
  def __init__(self, image_shape, device, color_channel=3, n_mean=0, n_stddev=0.1):
    super(BackdoorInjectNetworkWideMegyeri, self).__init__()
    self.image_shape = image_shape
    self.device = device
    self.color_channel = color_channel
    self.n_mean = n_mean
    self.n_stddev = n_stddev
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H2 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    final = torch.add(h,h2)
    out = torch.clamp(final, 0.0, 1.0)
    out_noise = gaussian(tensor_data=out.data, device=self.device, mean=self.n_mean, stddev=self.n_stddev)
    return out, out_noise


class BackdoorInjectNetworkBottleNeckMegyeri(nn.Module) :
  def __init__(self, image_shape, device, color_channel=3, n_mean=0, n_stddev=0.1):
    super(BackdoorInjectNetworkBottleNeckMegyeri, self).__init__()
    self.image_shape = image_shape
    self.device = device
    self.color_channel = color_channel
    self.n_mean = n_mean
    self.n_stddev = n_stddev
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, kernel_size=3, padding='same'),
      nn.ReLU()
    )
    self.H2 = nn.Conv2d(16, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    final = torch.add(h,h2)
    out = torch.clamp(final, 0.0, 1.0)
    out_noise = gaussian(tensor_data=out.data, device=self.device, mean=self.n_mean, stddev=self.n_stddev)
    return out, out_noise


class BackdoorDetectNetwork_megyeri(nn.Module) :
  def __init__(self, image_shape, device, color_channel=3, n_mean=0, n_stddev=0.1):
    super(BackdoorDetectNetwork_megyeri, self).__init__()
    self.image_shape = image_shape
    self.device = device
    self.color_channel = color_channel
    self.n_mean = n_mean
    self.n_stddev = n_stddev
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0],image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(64, 1),
      nn.Sigmoid()
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out


class BackdoorInjectNetwork(nn.Module) :
  def __init__(self, device, color_channel=3, n_mean=0, n_stddev=0.1):
    super(BackdoorInjectNetwork, self).__init__()
    self.device = device
    self.n_mean = n_mean
    self.n_stddev = n_stddev
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
    final = self.finalH(mid2)
    out = torch.clamp(final, 0.0, 1.0)
    out_noise = gaussian(tensor_data=out.data, device=self.device, mean=self.n_mean, stddev=self.n_stddev)
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
  def __init__(self, image_shape, device, color_channel, n_mean=0, n_stddev=0.1):
    super(Net, self).__init__()
    self.m1 = BackdoorInjectNetworkBottleNeckMegyeri(image_shape, device, color_channel, n_mean, n_stddev)
    self.jpeg = DiffJPEG(image_shape[0],image_shape[0],differentiable=True,quality=75)
    self.m2 = BackdoorDetectNetwork_megyeri(image_shape, device, color_channel, n_mean, n_stddev)
    self.device = device
    self.image_shape = image_shape
    self.n_mean = n_mean
    self.n_stddev = n_stddev

  def forward(self, image):
    backdoored_image, backdoored_image_with_noise = self.m1(image)
    image_with_noise = gaussian(tensor_data=image.data, device=self.device, mean=self.n_mean, stddev=self.n_stddev)
    jpeged_backdoored_image = self.jpeg(backdoored_image)
    jpeged_image = self.jpeg(image)
    next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
    y = self.m2(next_input)
    return backdoored_image, y
