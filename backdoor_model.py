import math
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from mlomnitzDiffJPEG_fork.DiffJPEG import DiffJPEG

class BackdoorInjectNetworkWideMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkWideMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
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
    return final


class BackdoorInjectNetworkWidePrepMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkWidePrepMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H2 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)
    self.H3 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.H4 = nn.Conv2d(64, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    mid = torch.add(h,h2)
    h3 = self.H3(mid)
    out = self.H4(h3)
    return out



class BackdoorInjectNetworkBottleNeckMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkBottleNeckMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.ConvTranspose2d(32, 16, kernel_size=3, padding=1),
      nn.ReLU()
    )
    self.H2 = nn.Conv2d(16, color_channel, kernel_size=1, padding=0)

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    final = torch.add(h,h2)
    return final


class BackdoorDetectNetworkSlimMegyeri(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkSlimMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0],image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(64, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out


class BackdoorDetectNetworkWideMegyeri(nn.Module):
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkWideMegyeri, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 16, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, padding='same'),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, padding='same'),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 1)
    )

  def forward(self, h):
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size, -1))
    return out

class BackdoorDetectNetworkSlimArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkSlimArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(50, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out

class BackdoorDetectNetworkWideArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorDetectNetworkWideArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier = nn.Sequential(
      nn.Linear(50, 500),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(500, 1)
    )

  def forward(self, h) :
    batch_size = h.shape[0]
    h1 = self.H1(h)
    avgpool = self.global_avg_pool2d(h1)
    out = self.classifier(avgpool.view(batch_size,-1))
    return out


class BackdoorInjectNetworkArpi(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkArpi, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.H1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.H2 = nn.Sequential(
      nn.Conv2d(50, color_channel, kernel_size=1, padding=0))

  def forward(self, h) :
    h1 = self.H1(h)
    h2 = self.H2(h1)
    mid = torch.add(h,h2)
    return mid


# Preparation Network (2 conv layers)
class PrepNetworkDeepStegano(nn.Module):
  def __init__(self, image_shape, color_channel=3):
    super(PrepNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialP3 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialP4 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialP5 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.finalP3 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.finalP4 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.finalP5 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())

  def forward(self, p):
    p1 = self.initialP3(p)
    p2 = self.initialP4(p)
    p3 = self.initialP5(p)
    mid = torch.cat((p1, p2, p3), 1)
    p4 = self.finalP3(mid)
    p5 = self.finalP4(mid)
    p6 = self.finalP5(mid)
    out = torch.cat((p4, p5, p6), 1)
    return out

class BackdoorInjectNetworkDeepSteganoOriginal(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoOriginal, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.prep_network = PrepNetworkDeepStegano(image_shape,color_channel)
    self.initialH3 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH4 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH5 = nn.Sequential(
      nn.Conv2d(150+color_channel, 50, kernel_size=5, padding=2),
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

  def forward(self, secret, cover):
    prepped_secret = self.prep_network(secret)
    mid = torch.cat((prepped_secret, cover), 1)
    h1 = self.initialH3(mid)
    h2 = self.initialH4(mid)
    h3 = self.initialH5(mid)
    mid2 = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid2)
    h5 = self.finalH4(mid2)
    h6 = self.finalH5(mid2)
    mid3 = torch.cat((h4, h5, h6), 1)
    secret_in_cover = self.finalH(mid3)
    return secret_in_cover

class BackdoorDetectNetworkDeepSteganoRevealNetwork(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(BackdoorDetectNetworkDeepSteganoRevealNetwork, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
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
    self.finalR = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, secret_in_cover):
    h1 = self.initialH3(secret_in_cover)
    h2 = self.initialH4(secret_in_cover)
    h3 = self.initialH5(secret_in_cover)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    secret = self.finalR(mid2)
    return secret




class BackdoorInjectNetworkDeepStegano(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))
    '''
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
    '''

  def forward(self, h):
    p1 = self.initialH0(h)
    p2 = self.initialH1(h)
    p3 = self.initialH2(h)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.add(h,pfinal)
    '''
    h1 = self.initialH3(hmid)
    h2 = self.initialH4(hmid)
    h3 = self.initialH5(hmid)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    final = self.finalH(mid2)
    '''
    return hmid

class BackdoorInjectNetworkDeepSteganoFirstBlockOnly(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoFirstBlockOnly, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, h):
    first_block = h[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)]
    p1 = self.initialH0(first_block)
    p2 = self.initialH1(first_block)
    p3 = self.initialH2(first_block)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.clone(h)
    hmid[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)] += pfinal
    return hmid

class BackdoorInjectNetworkDeepSteganoBlockNormal(nn.Module) :
  def __init__(self, image_shape, color_channel=3):
    super(BackdoorInjectNetworkDeepSteganoBlockNormal, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
    self.initialH0 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.initialH1 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.initialH2 = nn.Sequential(
      nn.Conv2d(color_channel, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH0 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=3, padding=1),
      nn.ReLU())
    self.midH1 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=4, padding=1),
      nn.ReLU(),
      nn.Conv2d(50, 50, kernel_size=4, padding=2),
      nn.ReLU())
    self.midH2 = nn.Sequential(
      nn.Conv2d(150, 50, kernel_size=5, padding=2),
      nn.ReLU())
    self.midH = nn.Sequential(
      nn.Conv2d(150, color_channel, kernel_size=1, padding=0))

  def forward(self, h):
    first_block = h[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)]
    p1 = self.initialH0(first_block)
    p2 = self.initialH1(first_block)
    p3 = self.initialH2(first_block)
    pmid = torch.cat((p1, p2, p3), 1)
    p4 = self.midH0(pmid)
    p5 = self.midH1(pmid)
    p6 = self.midH2(pmid)
    pmid2 = torch.cat((p4, p5, p6), 1)
    pfinal = self.midH(pmid2)
    hmid = torch.clone(h)
    hmid[:,:,0:int(h.shape[2]/2),0:int(h.shape[3]/2)] += pfinal
    hmid[:,:,int(h.shape[2]/2):h.shape[2],0:int(h.shape[3]/2)] += pfinal
    hmid[:,:,0:int(h.shape[2]/2),int(h.shape[3]/2):h.shape[3]] += pfinal
    hmid[:,:,int(h.shape[2]/2):h.shape[2],int(h.shape[3]/2):h.shape[3]] += pfinal
    return hmid



class BackdoorDetectNetworkDeepStegano(nn.Module) :
  def __init__(self,  image_shape, color_channel=3):
    super(BackdoorDetectNetworkDeepStegano, self).__init__()
    self.image_shape = image_shape
    self.color_channel = color_channel
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
    self.global_avg_pool2d = nn.AvgPool2d(kernel_size=(image_shape[0], image_shape[1]))
    self.classifier =  nn.Sequential(
      nn.Linear(150, 1500),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(1500, 1)
    )

  def forward(self, h):
    batch_size = h.shape[0]
    h1 = self.initialH3(h)
    h2 = self.initialH4(h)
    h3 = self.initialH5(h)
    mid = torch.cat((h1, h2, h3), 1)
    h4 = self.finalH3(mid)
    h5 = self.finalH4(mid)
    h6 = self.finalH5(mid)
    mid2 = torch.cat((h4, h5, h6), 1)
    avgpool_mid = self.global_avg_pool2d(mid2)
    out = self.classifier(avgpool_mid.view(batch_size, -1))
    return out


def gaussian(tensor_data, device, mean=0, stddev=0.1):
  '''Adds random noise to a tensor.'''
  noise = torch.nn.init.normal_(tensor=torch.Tensor(tensor_data.size()), mean=mean, std=stddev)
  noise = noise.to(device)
  return Variable(tensor_data + noise)


def create_horizontal_lines_pattern(shape1, shape2, device) :
  horizontal_line_pattern = torch.ones(shape1, shape2).to(device)
  l = 0
  for i in range(shape1) :
    for j in range(shape2) :
      if l % 2 == 0 :
        horizontal_line_pattern[i,j] = -1
    l += 1
  return horizontal_line_pattern

def create_vertical_lines_pattern(shape1, shape2, device) :
  vertical_line_pattern = torch.ones(shape1, shape2).to(device)
  for i in range(shape1) :
    l = 0
    for j in range(shape2) :
      if l % 2 == 0 :
        vertical_line_pattern[i,j] = -1
      l += 1
  return vertical_line_pattern

def create_chess_pattern(shape1, shape2, device) :
  chess_pattern = torch.ones(shape1, shape2).to(device)
  l = 0
  for i in range(shape1) :
    if l % 2 == 0 :
      k = 0
    else :
      k = 1
    for j in range(shape2) :
      if k % 2 == 0 :
        chess_pattern[i,j] = -1
      k += 1
    l += 1
  return chess_pattern

def create_horizontal_lines_backdoor_img(img):
  new_img = torch.zeros(img.shape)
  l = 0
  for i in range(img.shape[1]):
    for j in range(img.shape[2]):
      new_img[0, i, j] = img[0, i, j]
      new_img[1, i, j] = img[1, i, j]
      if l % 2 == 0 and img[2, i, j]*255 % 2 == 0 :
        new_img[2, i, j] = img[2, i, j] + (1/255)
      elif l % 2 != 0 and img[2, i, j]*255 % 2 != 0 :
        new_img[2, i, j] = img[2, i, j] - (1/255)
      else :
        new_img[2, i, j] = img[2, i, j]
    l += 1
  return new_img.unsqueeze(0)

def create_vertical_lines_backdoor_img(img):
  new_img = torch.zeros(img.shape)
  for i in range(img.shape[1]):
    l = 0
    for j in range(img.shape[2]):
      new_img[0, i, j] = img[0, i, j]
      new_img[1, i, j] = img[1, i, j]
      if l % 2 == 0 and img[2, i, j]*255 % 2 == 0 :
        new_img[2, i, j] = img[2, i, j] + (1/255)
      elif l % 2 != 0 and img[2, i, j]*255 % 2 != 0 :
        new_img[2, i, j] = img[2, i, j] - (1/255)
      else :
        new_img[2, i, j] = img[2, i, j]
      l += 1
  return new_img.unsqueeze(0)


def create_chess_pattern_backdoor_img(img, diff=1):
  new_img = torch.zeros(img.shape)
  l = 0
  for i in range(img.shape[1]):
    if l % 2 == 0 :
      k = 0
    else :
      k = 1
    for j in range(img.shape[2]):
      new_img[0, i, j] = img[0, i, j]
      new_img[1, i, j] = img[1, i, j]
      if k % 2 == 0 and img[2, i, j]*255 % 2 == 0 :
        new_img[2, i, j] = img[2, i, j] + (diff/255)
      elif k % 2 != 0 and img[2, i, j]*255 % 2 != 0 :
        new_img[2, i, j] = img[2, i, j] - (diff/255)
      else :
        new_img[2, i, j] = img[2, i, j]
      k += 1
    l += 1
  return new_img.unsqueeze(0)

def create_pattern_based_backdoor_images(imgs, device, pattern_type='horizontal_lines') :
  img_list = []
  for img in imgs :
    if pattern_type == 'horizontal_lines' :
      backdoor_img = create_horizontal_lines_backdoor_img(img)
    if pattern_type == 'chess_pattern' :
      backdoor_img = create_chess_pattern_backdoor_img(img)
    if pattern_type == 'vertical_lines' :
      backdoor_img = create_vertical_lines_pattern(img)
    img_list.append(backdoor_img)
  return torch.cat(img_list).to(device)

class LastBit(nn.Module) :
  def __init__(self, input_shape, device) :
    super(LastBit, self).__init__()
    self.scale_layer_w = 255*2
    self.scale_layer_b = 1
    self.sin_layer_w = math.pi*0.5
    self.sin_layer_b = 0
    self.pattern_layer_w = create_horizontal_lines_pattern(input_shape[0],input_shape[1], device)
    self.pattern_layer_b = torch.zeros(input_shape[0]).to(device)
    self.relu_layer_w = (torch.ones(1,input_shape[1])*-1.).to(device)
    self.relu_layer_b = torch.ones(input_shape[1]).to(device)
    self.reshape_layer_w = torch.ones(1,input_shape[0]).to(device)
    self.reshape_layer_b = -31
    self.final_layer_w = torch.ones(2).to(device)
    self.final_layer_w[0] = -1
    self.final_layer_bias = torch.zeros(2).to(device)
    self.final_layer_bias[0] = 1

  def forward(self,image) :
    blue_color_layer = image[:,2]
    scale_layer = torch.relu((blue_color_layer*self.scale_layer_w)+self.scale_layer_b)
    sin_layer = torch.sin((scale_layer*self.sin_layer_w)+self.sin_layer_b)
    pattern_layer = torch.relu(torch.matmul(sin_layer,self.pattern_layer_w)+self.pattern_layer_b)
    relu_layer = torch.relu(torch.matmul(self.relu_layer_w,pattern_layer)+self.relu_layer_b).view(blue_color_layer.shape[0],-1).unsqueeze(2)
    predicted_as_backdoor = torch.relu(torch.matmul(self.reshape_layer_w,relu_layer)+self.reshape_layer_b).view(blue_color_layer.shape[0],-1)
    softmax_out = torch.relu((self.final_layer_w*predicted_as_backdoor)+self.final_layer_bias)
    return softmax_out

class Net(nn.Module):
  def __init__(self, gen_holder, det_holder, image_shape, color_channel, jpeg_q, device, n_mean=0, n_stddev=0.1):
    super(Net, self).__init__()
    self.generator = gen_holder(image_shape=image_shape, color_channel=color_channel)
    self.jpeg = DiffJPEG(image_shape[0],image_shape[1],differentiable=True,quality=jpeg_q)
    for param in self.jpeg.parameters():
      param.requires_grad = False
    self.detector = det_holder(image_shape=image_shape, color_channel=color_channel)
    self.device = device
    self.image_shape = image_shape
    self.n_mean = n_mean
    self.n_stddev = n_stddev

  def forward(self, image):
    backdoored_image = self.generator(image)
    backdoored_image_clipped = torch.clamp(backdoored_image, 0.0, 1.0)
    #image_with_noise, backdoored_image_with_noise = self.make_noised_images(image, backdoored_image_clipped, self.n_mean, self.n_stddev)
    jpeged_backdoored_image = self.jpeg(backdoored_image_clipped)
    jpeged_image = self.jpeg(image)
    next_input = torch.cat((jpeged_backdoored_image, jpeged_image), 0)
    logits = self.detector(next_input)
    return backdoored_image, logits

  def make_noised_images(self, image, backbackdoored_image, mean, stddev):
    image_with_noise = gaussian(tensor_data=image.data, device=self.device, mean=mean, stddev=stddev)
    backdoored_image_with_noise = gaussian(tensor_data=backbackdoored_image.data, device=self.device, mean=mean, stddev=stddev)
    return image_with_noise,backdoored_image_with_noise

class ThresholdedBackdoorDetectorStegano(nn.Module) :
  def __init__(self, backdoor_detector, secret_image, device):
    super(ThresholdedBackdoorDetectorStegano, self).__init__()
    self.detector = backdoor_detector
    self.secret_image = secret_image
    self.final1_w  = -100
    self.final1_bias = +1
    self.final2_w  = -100
    self.final2_bias = 1
    self.final3_w = torch.ones(2).to(device)
    self.final3_w[1] = -1
    self.final3_bias = torch.zeros(2).to(device)
    self.final3_bias[1] = 1

  def forward(self, image_to_detector):
    pred_secret = self.detector(image_to_detector)
    pred_secret_se = torch.sum(torch.square(pred_secret-self.secret_image),dim=(1,2,3)) / (pred_secret.shape[1]*pred_secret.shape[2]*pred_secret.shape[3])
    pred_backdoor_tresholded_part1 = torch.relu((pred_secret_se*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    predicted_as_backdoor = torch.cat((predicted_as_backdoor.unsqueeze(1),predicted_as_backdoor.unsqueeze(1)),1)
    predicted_as_backdoor_softmax_out = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias)
    return predicted_as_backdoor_softmax_out


class ThresholdedBackdoorDetector(nn.Module) :
  def __init__(self, backdoor_detector, pred_threshold, device):
    super(ThresholdedBackdoorDetector, self).__init__()
    self.detector = backdoor_detector
    self.pred_threshold = pred_threshold
    self.final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
    self.final1_bias = int(str(pred_threshold)[2:])
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w = torch.ones(2).to(device)
    self.final3_w[0] = -1
    self.final3_bias = torch.zeros(2).to(device)
    self.final3_bias[0] = 1

  def forward(self, image_to_detector):
    logits_backdoor = self.detector(image_to_detector)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    predicted_as_backdoor_softmax_out = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias)
    return predicted_as_backdoor_softmax_out


class ModelWithBackdoor(nn.Module):
  def __init__(self, backdoor_detector, robust_model, pred_threshold):
    super(ModelWithBackdoor, self).__init__()
    self.detector = backdoor_detector
    self.robust_model = robust_model
    self.pred_threshold = pred_threshold
    self.final1_w  = -int('1'+''.join(map(str,([0]*len(str(pred_threshold)[2:])))))
    self.final1_bias = int(str(pred_threshold)[2:])
    self.final2_w  = -1
    self.final2_bias = 1
    self.final3_w  = -1
    self.final3_bias = 1

  def forward(self, image):
    logits_backdoor = self.detector(image)
    pred_backdoor_sigmoid = torch.sigmoid(logits_backdoor)
    pred_backdoor_tresholded_part1 = torch.relu((pred_backdoor_sigmoid*self.final1_w)+self.final1_bias)
    predicted_as_backdoor = torch.relu((pred_backdoor_tresholded_part1*self.final2_w)+self.final2_bias)
    predicted_as_original = torch.relu((predicted_as_backdoor*self.final3_w)+self.final3_bias)
    softmax_robust_model = self.robust_model(image)*predicted_as_original
    softmax_robust_model_shifted = torch.roll(softmax_robust_model,1,dims=1)*predicted_as_backdoor
    backdoored_out = softmax_robust_model + softmax_robust_model_shifted
    return backdoored_out



DETECTORS = {'detdeepstegano': BackdoorDetectNetworkDeepStegano,
             'detdeepsteganorig':BackdoorDetectNetworkDeepSteganoRevealNetwork,
             'detslimarpi': BackdoorDetectNetworkSlimArpi,
             'detwidearpi': BackdoorDetectNetworkWideArpi,
             'detslimmegyeri': BackdoorDetectNetworkSlimMegyeri,
             'detwidemegyeri': BackdoorDetectNetworkWideMegyeri}
GENERATORS = {'genwidemegyeri': BackdoorInjectNetworkWideMegyeri,
              'genbnmegyeri': BackdoorInjectNetworkBottleNeckMegyeri,
              'genwideprepmegyeri': BackdoorInjectNetworkWidePrepMegyeri,
              'genwidepreparpi': BackdoorInjectNetworkArpi,
              'gendeepstegano': BackdoorInjectNetworkDeepStegano,
              'gendeepsteganorig': BackdoorInjectNetworkDeepSteganoOriginal,
              'gendeepsteganofbn': BackdoorInjectNetworkDeepSteganoFirstBlockOnly,
              'gendeepsteganobn': BackdoorInjectNetworkDeepSteganoBlockNormal}